#!/usr/bin/env python3
"""Inspect a RePAIR/ReassembleNet preprocessed .pt puzzle sample.

The model training/test path consumes these .pt files, not the raw data.json.
This utility prints the sample structure so we can build a camera-to-.pt adapter later.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch


DEFAULT_SAMPLE = Path("RePAIR_dataset/jsons_test/data_0.pt")


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect RePAIR preprocessed .pt sample structure.")
    parser.add_argument("sample", nargs="?", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--max_points", type=int, default=5, help="Number of points to preview for each piece.")
    return parser.parse_args()


def numeric_piece_keys(sample):
    keys = []
    for key in sample.keys():
        text = str(key)
        if text.isdigit() and text != "0":
            keys.append(int(text))
    return sorted(keys)


def point_rows(piece_payload):
    # Original preprocessing stores 100 contour/keypoint rows, then metadata rows.
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


def metadata(piece_payload):
    name = ""
    global_len = 0
    for row in piece_payload:
        if isinstance(row, (list, tuple)) and len(row) >= 2:
            if row[0] == "piece_name":
                name = str(row[1])
            if row[0] == "global" and isinstance(row[1], (list, tuple)):
                global_len = len(row[1])
    return name, global_len


def summarize_piece(piece_id, piece_payload, max_points):
    rows = point_rows(piece_payload)
    name, global_len = metadata(piece_payload)
    points = np.array([row[0] for row in rows], dtype=float) if rows else np.zeros((0, 2))
    geom = np.array([row[1] for row in rows], dtype=float) if rows else np.zeros((0, 2))

    print(f"\npiece_{piece_id}: {name or '(no piece_name)'}")
    print(f"  payload_rows={len(piece_payload)} point_rows={len(rows)} global_texture_dim={global_len}")
    if len(points) == 0:
        print("  no point rows")
        return
    center = points.mean(axis=0)
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    print(f"  pixel_center=({center[0]:.2f}, {center[1]:.2f})")
    print(f"  pixel_bounds=x[{min_xy[0]:.1f}, {max_xy[0]:.1f}] y[{min_xy[1]:.1f}, {max_xy[1]:.1f}]")
    print(
        f"  edge_angle_range=[{geom[:,0].min():.2f}, {geom[:,0].max():.2f}] "
        f"curvature_range=[{geom[:,1].min():.2f}, {geom[:,1].max():.2f}]"
    )
    print("  preview_points:")
    for row in rows[:max_points]:
        local_len = len(row[2]) if len(row) > 2 and isinstance(row[2], (list, tuple)) else 0
        print(f"    point={row[0]} geom={row[1]} local_texture_dim={local_len}")


def main():
    args = parse_args()
    sample_path = args.sample
    if not sample_path.is_absolute():
        sample_path = Path.cwd() / sample_path
    sample = torch.load(sample_path, map_location="cpu", weights_only=False)

    print(f"sample: {sample_path}")
    print(f"type: {type(sample).__name__}")
    print(f"puzzle_name: {sample.get('puzzle_name')}")
    print(f"image_size: {sample.get('0')}")
    piece_keys = numeric_piece_keys(sample)
    print(f"piece_count: {len(piece_keys)}")
    print(f"piece_keys: {piece_keys}")

    for piece_id in piece_keys:
        summarize_piece(piece_id, sample[str(piece_id)], args.max_points)


if __name__ == "__main__":
    main()
