#!/usr/bin/env python3
"""Run ReassembleNet on one preprocessed .pt sample and export relative relations.

This is the first model-backed adapter:
- Loads a RePAIR_dataset/jsons_test data_*.pt sample through the existing dataset code.
- Loads the ReassembleNet diffusion checkpoint and keypoint selector checkpoint.
- Runs one prediction.
- Aggregates keypoint-level predicted poses into per-piece poses.
- Exports a relation JSON compatible with mirobot_gazebo_demo.

The output scale is the model's normalized coordinate system times --model_unit_to_meter.
Tune that scale before using with a real table/robot.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from reassemblenet import dist_util
from reassemblenet.k_point_selector import K_Point_Selector
from reassemblenet.repair import repair
from reassemblenet.script_util import create_model_and_diffusion
from reassemblenet.train_util import TrainLoop


DEFAULT_CHECKPOINT = Path("ReassembleNet_ckpts/Exp_NCC_geoKP_10000/model_epoch2699.pt")
DEFAULT_KP_CHECKPOINT = Path("ReassembleNet_ckpts/Exp_NCC_geoKP_10000/model_epoch2699_kp.pth")
DEFAULT_DATASET_PATH = Path("RePAIR_dataset")
DEFAULT_OUTPUT = Path("/tmp/reassemblenet_pred_relations.json")


def normalize_deg(angle):
    while angle > 180.0:
        angle -= 360.0
    while angle <= -180.0:
        angle += 360.0
    return angle


def rotation_vec_to_deg(vec):
    # Training stores the first row of a 2D rotation matrix: [cos(theta), -sin(theta)].
    c = float(vec[0])
    neg_s = float(vec[1])
    return math.degrees(math.atan2(-neg_s, c))


def parse_args():
    parser = argparse.ArgumentParser(description="Export ReassembleNet predicted relative relations.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--kp_checkpoint", type=Path, default=DEFAULT_KP_CHECKPOINT)
    parser.add_argument("--dataset_path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument(
        "--sample_pt",
        type=Path,
        default=None,
        help="Optional direct RePAIR/ReassembleNet preprocessed .pt sample. Overrides --sample_index.",
    )
    parser.add_argument("--reference_piece", type=int, default=2)
    parser.add_argument("--output_json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model_unit_to_meter", type=float, default=0.08)
    parser.add_argument("--max_num_points", type=int, default=3300)
    parser.add_argument("--max_pieces", type=int, default=33)
    parser.add_argument("--one_hot_dim", type=int, default=128)
    parser.add_argument("--diffusion_steps", type=int, default=600)
    parser.add_argument(
        "--timestep_respacing",
        type=str,
        default="",
        help="Optional faster sampler spacing, e.g. 50. Empty uses all diffusion_steps.",
    )
    parser.add_argument("--position_tolerance_m", type=float, default=0.008)
    parser.add_argument("--yaw_tolerance_deg", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve(path):
    return path if path.is_absolute() else Path.cwd() / path




def numeric_piece_keys(sample):
    keys = []
    for key in sample.keys():
        text = str(key)
        if text.isdigit() and text != "0":
            keys.append(int(text))
    return sorted(keys)


def point_rows(piece_payload):
    rows = []
    for row in piece_payload:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        if isinstance(row[0], str):
            continue
        if not isinstance(row[0], (list, tuple)) or len(row[0]) != 2:
            continue
        rows.append(row)
    return rows


def one_hot(index, size):
    arr = np.zeros(size, dtype=np.float64)
    if 0 <= index < size:
        arr[index] = 1.0
    return arr


def rotate_points_with_identity(points, piece_indices):
    # Prediction-time sample_pt adapter keeps input fragment orientation unchanged.
    return points, np.tile(np.array([[1.0, 0.0]], dtype=np.float64), (points.shape[0], 1))


def build_arrays_from_pt_sample(sample, args):
    sizes = {
        "center": 2,
        "angles": 2,
        "poly": 2,
        "corner_index": args.one_hot_dim,
        "piece_index": args.one_hot_dim,
        "padding_mask": 1,
        "connections": 2,
    }
    indices = np.cumsum([0] + list(sizes.values()))
    start_idx = {key: int(indices[i]) for i, key in enumerate(sizes)}
    end_idx = {key: int(indices[i + 1]) for i, key in enumerate(sizes)}

    piece_keys = numeric_piece_keys(sample)
    puzzle_rows = []
    corner_bounds = []
    num_points = 0
    piece_names = {}
    for piece_order, piece_id in enumerate(piece_keys, start=1):
        payload = sample[str(piece_id)]
        rows = point_rows(payload)[:100]
        if len(rows) < 3:
            continue
        points = np.array([row[0] for row in rows], dtype=np.float64)
        image_size = np.array(sample.get("0", [2000, 2000]), dtype=np.float64)
        poly_abs = np.column_stack([points[:, 0] / image_size[0], points[:, 1] / image_size[1]])
        center_value = poly_abs.mean(axis=0)
        poly = poly_abs - center_value
        piece_indices = np.repeat(one_hot(piece_order, args.one_hot_dim)[None, :], len(rows), axis=0)
        poly, angles = rotate_points_with_identity(poly, piece_indices)
        center = np.repeat(center_value[None, :], len(rows), axis=0)
        corner_indices = np.array([one_hot(i, args.one_hot_dim) for i in range(len(rows))], dtype=np.float64)
        padding_mask = np.ones((len(rows), 1), dtype=np.float64)
        connections = np.array([[i, (i + 1) % len(rows)] for i in range(len(rows))], dtype=np.float64) + num_points
        geom_feats = np.array([row[1] for row in rows], dtype=np.float64)
        piece_block = np.concatenate(
            [center, angles, poly, corner_indices, piece_indices, padding_mask, connections, geom_feats],
            axis=1,
        )
        puzzle_rows.append(piece_block)
        corner_bounds.append([num_points, num_points + len(rows)])
        num_points += len(rows)
        name = ""
        for row in payload:
            if isinstance(row, (list, tuple)) and len(row) >= 2 and row[0] == "piece_name":
                name = str(row[1])
        piece_names[piece_order] = name or f"piece_{piece_order}"

    if not puzzle_rows:
        raise ValueError("sample_pt did not contain usable piece rows")
    puzzle_layout = np.concatenate(puzzle_rows, axis=0)
    if len(puzzle_layout) > args.max_num_points:
        raise ValueError(f"sample has {len(puzzle_layout)} points, above max_num_points={args.max_num_points}")
    horizontal_dim = puzzle_layout.shape[1]
    padding = np.zeros((args.max_num_points - len(puzzle_layout), horizontal_dim), dtype=np.float64)
    puzzle_layout = np.concatenate([puzzle_layout, padding], axis=0)

    gen_mask = np.ones((args.max_num_points, args.max_num_points), dtype=np.float64)
    gen_mask[:num_points, :num_points] = 0.0
    self_mask = np.ones((args.max_num_points, args.max_num_points), dtype=np.float64)
    for start, end in corner_bounds:
        self_mask[start:end, start:end] = 0.0
    rels = np.zeros((args.max_num_points, 2), dtype=np.float64)

    batch = puzzle_layout[:, :4].T
    cond = {
        "self_mask": self_mask,
        "gen_mask": gen_mask,
        "poly": puzzle_layout[:, start_idx["poly"]:end_idx["poly"]],
        "corner_indices": puzzle_layout[:, start_idx["corner_index"]:end_idx["corner_index"]],
        "room_indices": puzzle_layout[:, start_idx["piece_index"]:end_idx["piece_index"]],
        "src_key_padding_mask": 1.0 - puzzle_layout[:, start_idx["padding_mask"]],
        "connections": puzzle_layout[:, start_idx["connections"]:end_idx["connections"]],
        "rels": rels,
        "trans": puzzle_layout[:, 0:2],
        "rots": puzzle_layout[:, 2:4],
        "trans_rot_gt": puzzle_layout[:, 0:4],
        "num_pieces": len(piece_names),
        "geom_feats": puzzle_layout[:, -2:],
    }
    meta = {
        "puzzle_name": sample.get("puzzle_name", "sample_pt"),
        "image_size": sample.get("0"),
        "piece_count": len(piece_names),
        "piece_names": piece_names,
        "point_count": int(num_points),
    }
    return batch, cond, meta


def prepare_sample_pt(sample_pt, args, device):
    sample_path = resolve(sample_pt)
    sample = torch.load(sample_path, map_location="cpu", weights_only=False)
    batch_np, cond_np, meta = build_arrays_from_pt_sample(sample, args)
    batch = torch.as_tensor(batch_np, dtype=torch.float64, device=device).unsqueeze(0)
    cond = {}
    for key, value in cond_np.items():
        if np.isscalar(value):
            cond[key] = torch.as_tensor([value], device=device)
        else:
            cond[key] = torch.as_tensor(value, dtype=torch.float64, device=device).unsqueeze(0)
    meta["sample_pt"] = str(sample_path)
    return batch, cond, meta

def make_dataset(args, device):
    return repair(
        set_name="test",
        rotation=True,
        dataset_path=str(resolve(args.dataset_path)),
        max_num_points=args.max_num_points,
        maxcount=args.max_pieces,
        number_larger_than_pieces_and_points_in_puzzle=args.one_hot_dim,
        device=str(device),
        loader_num_workers=0,
        rank=0,
        use_geometry_only=True,
        use_global_texture_only=False,
        images_folder_path=None,
        use_local_texture_only=False,
        use_geometry_and_global_texture=False,
        use_geometry_and_local_texture=False,
        use_local_and_global_texture=False,
        use_geometry_global_local_texture=False,
        use_learnable_kp_selection=True,
    )


def make_model(args, device):
    model, diffusion = create_model_and_diffusion(
        input_channels=4,
        condition_channels=258,
        num_channels=256,
        out_channels=4,
        use_checkpoint=False,
        learn_sigma=False,
        diffusion_steps=args.diffusion_steps,
        noise_schedule="cosine",
        timestep_respacing=args.timestep_respacing,
        use_kl=False,
        predict_xstart=True,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        dataset="repair",
        set_name="test",
        rotation=True,
        exp_name="predict_export",
        use_image_features=False,
        use_geometry_only=True,
        use_global_texture_only=False,
        use_local_texture_only=False,
        use_geometry_and_global_texture=False,
        use_geometry_and_local_texture=False,
        use_local_and_global_texture=False,
        use_geometry_global_local_texture=False,
    )
    state_dict = torch.load(resolve(args.checkpoint), map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    kp_model = K_Point_Selector(
        max_num_pieces=args.max_pieces,
        input_channels=4,
        hidden_channels=32,
        output_channels=2,
        ratio=20,
        min_score=None,
        freeze_backbone=True,
        backbone_weights_path=None,
    ).to(device).eval()
    kp_state = torch.load(resolve(args.kp_checkpoint), map_location=device, weights_only=False)
    kp_model.nets.load_state_dict(kp_state)
    print(f"Loaded keypoint selector weights from {resolve(args.kp_checkpoint)}", flush=True)
    return model, diffusion, kp_model


def prepare_sample(dataset, sample_index, device):
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(f"sample_index {sample_index} outside dataset length {len(dataset)}")
    batch, cond = dataset[sample_index]
    batch = torch.as_tensor(batch, dtype=torch.float64, device=device).unsqueeze(0)
    cond_t = {}
    for key, value in cond.items():
        if isinstance(value, np.ndarray):
            cond_t[key] = torch.as_tensor(value, dtype=torch.float64, device=device).unsqueeze(0)
        elif np.isscalar(value):
            cond_t[key] = torch.as_tensor([value], device=device)
        else:
            cond_t[key] = torch.as_tensor(value, device=device).unsqueeze(0)
    return batch, cond_t


def process_keypoints(batch, cond, model, kp_model, diffusion, args, device):
    # Reuse TrainLoop's keypoint-selection preprocessing without constructing optimizers.
    helper = object.__new__(TrainLoop)
    helper.model = model
    helper.kp_selection_model = kp_model
    helper.diffusion = diffusion
    helper.device = device
    helper.mpkdim = args.max_pieces
    helper.number_larger_than_pieces_and_points_in_puzzle = args.one_hot_dim
    helper.use_learnable_kp_selection = True
    return TrainLoop.process_learnable_kp_selection(helper, batch, cond)


def piece_ids_from_room_indices(room_indices):
    # room_indices is [N, one_hot_dim], one-hot index 1..piece_count for real pieces.
    arr = room_indices.detach().cpu().numpy()
    ids = arr.argmax(axis=1).astype(int)
    valid = arr.max(axis=1) > 0.5
    return ids, valid


def aggregate_piece_poses(sample, cond, scale):
    # sample: [1, 4, N]. cond room_indices: [1, N, one_hot_dim].
    pred = sample[0].detach().cpu().numpy().T  # [N, 4]
    ids, valid = piece_ids_from_room_indices(cond["room_indices"][0])
    poses = {}
    for piece_id in sorted(set(ids[valid])):
        if piece_id <= 0:
            continue
        mask = valid & (ids == piece_id)
        if not mask.any():
            continue
        rows = pred[mask]
        xy = rows[:, :2].mean(axis=0)
        rot = rows[:, 2:4].mean(axis=0)
        poses[int(piece_id)] = {
            "x": float(xy[0] * scale),
            "y": float(-xy[1] * scale),
            "model_x": float(xy[0]),
            "model_y": float(xy[1]),
            "yaw_deg": rotation_vec_to_deg(rot),
            "keypoint_count": int(mask.sum()),
        }
    return poses


def relations_from_poses(poses, reference_piece, args):
    if reference_piece not in poses:
        reference_piece = sorted(poses)[0]
    ref = poses[reference_piece]
    relations = []
    for piece_id in sorted(poses):
        if piece_id == reference_piece:
            continue
        pose = poses[piece_id]
        relations.append({
            "task_name": f"reassemblenet_pred_piece_{piece_id}_relative_to_piece_{reference_piece}",
            "move_piece": piece_id,
            "reference_piece": reference_piece,
            "relative_dx": pose["x"] - ref["x"],
            "relative_dy": pose["y"] - ref["y"],
            "relative_dyaw_deg": normalize_deg(pose["yaw_deg"] - ref["yaw_deg"]),
        })
    return reference_piece, relations


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    sample_meta = {}
    if args.sample_pt is not None:
        batch, cond, sample_meta = prepare_sample_pt(args.sample_pt, args, device)
        print(f"sample_pt: {sample_meta['sample_pt']}", flush=True)
        print(
            f"puzzle_name: {sample_meta.get('puzzle_name')} "
            f"piece_count: {sample_meta.get('piece_count')} "
            f"point_count: {sample_meta.get('point_count')}",
            flush=True,
        )
    else:
        dataset = make_dataset(args, device)
        print(f"dataset length: {len(dataset)}", flush=True)
        batch, cond = prepare_sample(dataset, args.sample_index, device)
        sample_meta = {"sample_index": args.sample_index, "dataset_path": str(resolve(args.dataset_path))}
        print(f"sample_index: {args.sample_index} batch_shape={tuple(batch.shape)}", flush=True)

    model, diffusion, kp_model = make_model(args, device)
    batch, cond = process_keypoints(batch, cond, model, kp_model, diffusion, args, device)
    print(f"after keypoint selection batch_shape={tuple(batch.shape)}", flush=True)

    sample_fn = diffusion.p_sample_loop
    with torch.no_grad():
        sample = sample_fn(
            model,
            batch.shape,
            clip_denoised=True,
            model_kwargs=cond,
        )

    poses = aggregate_piece_poses(sample, cond, args.model_unit_to_meter)
    reference_piece, relations = relations_from_poses(poses, args.reference_piece, args)

    output = {
        "source": "ReassembleNet adapter v2: model prediction to relative piece relations",
        "mode": "relative_piece_relations",
        "adapter_stage": "preprocessed_pt_model_prediction_to_relative_relations",
        "checkpoint": str(resolve(args.checkpoint)),
        "kp_checkpoint": str(resolve(args.kp_checkpoint)),
        "dataset_path": str(resolve(args.dataset_path)),
        "sample_index": args.sample_index,
        "sample_pt": str(resolve(args.sample_pt)) if args.sample_pt is not None else None,
        "sample_meta": sample_meta,
        "reference_piece": reference_piece,
        "model_unit_to_meter": args.model_unit_to_meter,
        "diffusion_steps": args.diffusion_steps,
        "timestep_respacing": args.timestep_respacing,
        "note": "Predicted poses are aggregated from selected keypoint-level outputs. Scale/calibration must be tuned before robot execution.",
        "units": {
            "relative_dx": "meters after model_unit_to_meter scaling",
            "relative_dy": "meters after model_unit_to_meter scaling and Y inversion",
            "relative_dyaw_deg": "degrees relative to reference piece yaw",
        },
        "tolerances": {
            "position_m": args.position_tolerance_m,
            "yaw_deg": args.yaw_tolerance_deg,
        },
        "piece_names": sample_meta.get("piece_names", {}),
        "piece_poses": poses,
        "relations": relations,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        f.write("\n")

    print(f"wrote {args.output_json}")
    print(f"reference_piece: {reference_piece}")
    for relation in relations:
        print(
            f"{relation['task_name']}: "
            f"dx={relation['relative_dx']:.4f}m "
            f"dy={relation['relative_dy']:.4f}m "
            f"dyaw={relation['relative_dyaw_deg']:.1f}deg"
        )


if __name__ == "__main__":
    main()
