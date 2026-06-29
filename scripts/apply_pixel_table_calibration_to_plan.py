#!/usr/bin/env python3
"""Apply a pixel-to-table calibration to a minimal pick-place plan."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Add table XY coordinates to a minimal pick-place plan.")
    parser.add_argument("--plan_json", type=Path, default=Path("/tmp/minimal_pick_place_plan.json"))
    parser.add_argument("--calibration_json", type=Path, default=Path("/tmp/pixel_to_table_calibration.json"))
    parser.add_argument("--output_json", type=Path, default=Path("/tmp/minimal_pick_place_plan_table.json"))
    parser.add_argument(
        "--assembly_anchor_table_xy",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="Optional safe table XY for the reference piece. If set, the move target is anchor + ReassembleNet relative dx/dy.",
    )
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pixel_to_table(h, pixel):
    arr = np.array([[pixel]], dtype=np.float64)
    out = cv2.perspectiveTransform(arr, h).reshape(2)
    return [float(out[0]), float(out[1])]


def main():
    args = parse_args()
    plan = load_json(args.plan_json)
    calibration = load_json(args.calibration_json)
    h = np.array(calibration["homography_pixel_to_table"], dtype=np.float64)
    move = plan.get("move", {})

    for src, dst in [
        ("grasp_pixel", "grasp_table_xy"),
        ("current_center_pixel", "current_center_table_xy"),
        ("reference_center_pixel", "reference_center_table_xy"),
        ("target_center_pixel_preview", "target_center_table_xy_preview"),
    ]:
        if src in move and move[src] is not None:
            move[dst] = pixel_to_table(h, move[src])

    if args.assembly_anchor_table_xy is not None:
        relation = move.get("predicted_relation", {})
        dx = float(relation["relative_dx"])
        dy = float(relation["relative_dy"])
        anchor = [float(args.assembly_anchor_table_xy[0]), float(args.assembly_anchor_table_xy[1])]
        move["assembly_anchor_table_xy"] = anchor
        move["reference_target_table_xy"] = anchor
        move["target_center_table_xy"] = [anchor[0] + dx, anchor[1] + dy]
        move["target_policy"] = "assembly_anchor_plus_reassemblenet_relative_relation"
    else:
        move["target_policy"] = "pixel_preview_homography"

    plan["status"] = "candidate_plan_with_table_coordinates_not_executed"
    plan["calibration"] = {
        "calibration_json": str(args.calibration_json),
        "name": calibration.get("name"),
        "fit_error_m": calibration.get("fit_error_m"),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)
        f.write("\n")
    print(f"wrote {args.output_json}")
    print(f"grasp_table_xy={move.get('grasp_table_xy')}")
    print(f"target_center_table_xy_preview={move.get('target_center_table_xy_preview')}")
    if move.get("target_center_table_xy") is not None:
        print(f"target_center_table_xy={move.get('target_center_table_xy')} policy={move.get('target_policy')}")


if __name__ == "__main__":
    main()
