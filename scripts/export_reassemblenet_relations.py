#!/usr/bin/env python3
"""Export ReassembleNet-style puzzle poses as ROS relative relation JSON.

V1 scope:
- Read a RePAIR/ReassembleNet `data.json` containing fragment target poses.
- Convert absolute pixel target poses into relative piece relations.
- Write a JSON compatible with mirobot_gazebo_demo visual_relative_relation_demo.

This is intentionally an adapter first. The next step is to replace the input
`data.json` with a model-prediction JSON produced by ReassembleNet inference.
"""

import argparse
import json
import math
from pathlib import Path


DEFAULT_INPUT = Path("RePAIR_V2/2D_FINAL/2d_fixed/test/puzzle_000001/data.json")
DEFAULT_OUTPUT = Path("/tmp/reassemblenet_relations.json")


def normalize_deg(angle):
    while angle > 180.0:
        angle -= 360.0
    while angle <= -180.0:
        angle += 360.0
    return angle


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert ReassembleNet absolute fragment poses to ROS relative relation JSON."
    )
    parser.add_argument(
        "--input_json",
        type=Path,
        default=DEFAULT_INPUT,
        help="ReAssemble/RePAIR data.json or prediction JSON with a fragments list.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output relative relation JSON path.",
    )
    parser.add_argument(
        "--reference_piece",
        type=int,
        default=2,
        help="One-based piece id to use as reference. If unavailable, piece 1 is used.",
    )
    parser.add_argument(
        "--pixel_to_meter",
        type=float,
        default=0.00025,
        help="Scale from RePAIR pixel coordinates to robot table meters.",
    )
    parser.add_argument(
        "--keep_image_y",
        action="store_true",
        help="Keep image Y positive downward. Default maps image Y to table Y by negating it.",
    )
    parser.add_argument(
        "--position_tolerance_m",
        type=float,
        default=0.008,
        help="Position tolerance written into the relation JSON.",
    )
    parser.add_argument(
        "--yaw_tolerance_deg",
        type=float,
        default=180.0,
        help="Yaw tolerance written into the relation JSON. Keep loose until yaw is calibrated.",
    )
    return parser.parse_args()


def load_fragments(path):
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    fragments = payload.get("fragments")
    if not isinstance(fragments, list) or len(fragments) < 2:
        raise ValueError(f"{path} must contain at least two fragments")
    return payload, fragments


def fragment_pose(fragment, scale, invert_y):
    pixel_position = fragment.get("pixel_position")
    if not isinstance(pixel_position, list) or len(pixel_position) != 2:
        raise ValueError(f"fragment is missing pixel_position: {fragment}")
    x_px = float(pixel_position[0])
    y_px = float(pixel_position[1])
    y_sign = -1.0 if invert_y else 1.0
    return {
        "x": x_px * scale,
        "y": y_sign * y_px * scale,
        "yaw_deg": float(fragment.get("rotation", 0.0)),
        "filename": fragment.get("filename", ""),
        "pixel_position": pixel_position,
    }


def main():
    args = parse_args()
    input_path = args.input_json
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path

    payload, fragments = load_fragments(input_path)
    piece_count = len(fragments)
    reference_piece = args.reference_piece
    if reference_piece < 1 or reference_piece > piece_count:
        reference_piece = 1

    poses = {}
    invert_y = not args.keep_image_y
    for index, fragment in enumerate(fragments, start=1):
        poses[index] = fragment_pose(fragment, args.pixel_to_meter, invert_y)

    ref = poses[reference_piece]
    relations = []
    for piece_id in range(1, piece_count + 1):
        if piece_id == reference_piece:
            continue
        move = poses[piece_id]
        dx = move["x"] - ref["x"]
        dy = move["y"] - ref["y"]
        dyaw = normalize_deg(move["yaw_deg"] - ref["yaw_deg"])
        relations.append({
            "task_name": f"reassemblenet_piece_{piece_id}_relative_to_piece_{reference_piece}",
            "move_piece": piece_id,
            "reference_piece": reference_piece,
            "relative_dx": dx,
            "relative_dy": dy,
            "relative_dyaw_deg": dyaw,
            "source_fragment": move["filename"],
            "reference_fragment": ref["filename"],
        })

    output = {
        "source": "ReassembleNet adapter v1: converted absolute fragment poses to relative piece relations",
        "mode": "relative_piece_relations",
        "adapter_stage": "offline_pose_json_to_relative_relations",
        "input_json": str(input_path),
        "original_id": payload.get("original_id"),
        "reference_piece": reference_piece,
        "piece_count": piece_count,
        "pixel_to_meter": args.pixel_to_meter,
        "image_y_was_inverted": invert_y,
        "note": (
            "This v1 adapter uses ReassembleNet/RePAIR target poses from data.json. "
            "Replace input_json with model-predicted poses after inference is wired."
        ),
        "units": {
            "relative_dx": "meters in reference/world X axis after pixel_to_meter scaling",
            "relative_dy": "meters in reference/world Y axis after pixel_to_meter scaling",
            "relative_dyaw_deg": "degrees relative to reference piece yaw",
        },
        "tolerances": {
            "position_m": args.position_tolerance_m,
            "yaw_deg": args.yaw_tolerance_deg,
        },
        "piece_poses": poses,
        "relations": relations,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        f.write("\n")

    print(f"wrote {args.output_json}")
    print(f"input: {input_path}")
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
