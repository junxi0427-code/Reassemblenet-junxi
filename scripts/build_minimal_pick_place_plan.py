#!/usr/bin/env python3
"""Build a minimal pick-place plan from camera perception and ReassembleNet relations.

This is execution-backend agnostic: the output can later be consumed by either
Gazebo/MoveIt or a real robot SDK. It does not control the robot.
"""

import argparse
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build minimal robot plan from camera + ReassembleNet outputs.")
    parser.add_argument("--camera_json", type=Path, default=Path("/tmp/camera_piece_keypoints.json"))
    parser.add_argument("--relations_json", type=Path, default=Path("/tmp/camera_reassemblenet_pred_relations_scaled.json"))
    parser.add_argument("--output_json", type=Path, default=Path("/tmp/minimal_pick_place_plan.json"))
    parser.add_argument("--relation_index", type=int, default=0, help="Which predicted relation to convert into one move.")
    parser.add_argument("--pixel_to_meter", type=float, default=0.00025, help="Temporary uncalibrated scale for target pixel preview.")
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pieces_by_id(camera_payload):
    result = {}
    for piece in camera_payload.get("pieces", []):
        result[int(piece["instance_id"])] = piece
    return result


def relation_target_center_pixel(reference_center, relation, pixel_to_meter):
    # This is only a preview until real pixel/table calibration is added.
    dx_px = float(relation["relative_dx"]) / pixel_to_meter
    dy_px = -float(relation["relative_dy"]) / pixel_to_meter
    return [int(round(reference_center[0] + dx_px)), int(round(reference_center[1] + dy_px))]


def main():
    args = parse_args()
    camera_payload = load_json(args.camera_json)
    relations_payload = load_json(args.relations_json)
    relations = relations_payload.get("relations", [])
    if not relations:
        raise ValueError(f"no relations found in {args.relations_json}")
    if args.relation_index < 0 or args.relation_index >= len(relations):
        raise IndexError(f"relation_index={args.relation_index} outside relation count {len(relations)}")

    relation = relations[args.relation_index]
    move_piece_id = int(relation["move_piece"])
    reference_piece_id = int(relation["reference_piece"])
    camera_pieces = pieces_by_id(camera_payload)
    if move_piece_id not in camera_pieces:
        raise ValueError(f"move_piece {move_piece_id} not found in camera pieces {sorted(camera_pieces)}")
    if reference_piece_id not in camera_pieces:
        raise ValueError(f"reference_piece {reference_piece_id} not found in camera pieces {sorted(camera_pieces)}")

    move_piece = camera_pieces[move_piece_id]
    reference_piece = camera_pieces[reference_piece_id]
    target_center_pixel = relation_target_center_pixel(reference_piece["center_pixel"], relation, args.pixel_to_meter)

    plan = {
        "source": "minimal_pick_place_plan_from_camera_and_reassemblenet",
        "status": "candidate_plan_not_robot_calibrated",
        "note": "target_center_pixel uses temporary pixel_to_meter. Real robot execution requires pixel-to-table calibration.",
        "inputs": {
            "camera_json": str(args.camera_json),
            "relations_json": str(args.relations_json),
            "relation_index": args.relation_index,
        },
        "camera": {
            "image_topic": camera_payload.get("image_topic"),
            "image_size": camera_payload.get("image_size"),
        },
        "move": {
            "move_piece": move_piece_id,
            "reference_piece": reference_piece_id,
            "task_name": relation.get("task_name"),
            "grasp_pixel": move_piece.get("grasp_pixel"),
            "current_center_pixel": move_piece.get("center_pixel"),
            "reference_center_pixel": reference_piece.get("center_pixel"),
            "target_center_pixel_preview": target_center_pixel,
            "predicted_relation": {
                "relative_dx": relation.get("relative_dx"),
                "relative_dy": relation.get("relative_dy"),
                "relative_dyaw_deg": relation.get("relative_dyaw_deg"),
            },
            "piece_images": move_piece.get("images", {}),
            "piece_bbox": move_piece.get("bbox"),
        },
        "calibration_required_before_execution": {
            "pixel_to_table_homography": True,
            "robot_pick_height": True,
            "robot_place_height": True,
            "end_effector_grasp_policy": True,
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)
        f.write("\n")

    print(f"wrote {args.output_json}")
    print(f"move_piece={move_piece_id} reference_piece={reference_piece_id}")
    print(f"grasp_pixel={plan['move']['grasp_pixel']}")
    print(f"current_center_pixel={plan['move']['current_center_pixel']}")
    print(f"reference_center_pixel={plan['move']['reference_center_pixel']}")
    print(f"target_center_pixel_preview={plan['move']['target_center_pixel_preview']}")
    print(f"relative_dx={relation.get('relative_dx'):.4f} relative_dy={relation.get('relative_dy'):.4f} dyaw={relation.get('relative_dyaw_deg'):.1f}")


if __name__ == "__main__":
    main()
