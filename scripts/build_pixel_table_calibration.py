#!/usr/bin/env python3
"""Build a pixel-to-table homography calibration from point correspondences."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Build pixel->table homography calibration JSON.")
    parser.add_argument("--points_json", type=Path, required=True, help="JSON with points: [{pixel:[u,v], table:[x,y]}, ...]")
    parser.add_argument("--output_json", type=Path, default=Path("/tmp/pixel_to_table_calibration.json"))
    parser.add_argument("--name", default="sim_pixel_to_table_calibration")
    return parser.parse_args()


def load_points(path):
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    points = payload.get("points", payload if isinstance(payload, list) else [])
    if len(points) < 4:
        raise ValueError("at least 4 calibration points are required")
    pixels = []
    tables = []
    for item in points:
        pixel = item.get("pixel")
        table = item.get("table")
        if not pixel or not table or len(pixel) != 2 or len(table) != 2:
            raise ValueError(f"invalid calibration point: {item}")
        pixels.append([float(pixel[0]), float(pixel[1])])
        tables.append([float(table[0]), float(table[1])])
    return np.array(pixels, dtype=np.float64), np.array(tables, dtype=np.float64), points


def project_points(h, pixels):
    points = cv2.perspectiveTransform(pixels.reshape(-1, 1, 2).astype(np.float64), h)
    return points.reshape(-1, 2)


def main():
    args = parse_args()
    pixels, tables, raw_points = load_points(args.points_json)
    h, inlier_mask = cv2.findHomography(pixels, tables, method=0)
    if h is None:
        raise RuntimeError("cv2.findHomography failed")
    projected = project_points(h, pixels)
    errors = np.linalg.norm(projected - tables, axis=1)

    output = {
        "source": "pixel_to_table_homography_calibration",
        "name": args.name,
        "input_points_json": str(args.points_json),
        "point_count": int(len(raw_points)),
        "units": {
            "pixel": "camera image pixels [u, v]",
            "table": "table/robot planar coordinates [x, y] in meters",
        },
        "homography_pixel_to_table": h.tolist(),
        "calibration_points": raw_points,
        "fit_error_m": {
            "mean": float(errors.mean()),
            "max": float(errors.max()),
            "per_point": [float(x) for x in errors],
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        f.write("\n")
    print(f"wrote {args.output_json}")
    print(f"points={len(raw_points)} mean_error_m={errors.mean():.6f} max_error_m={errors.max():.6f}")


if __name__ == "__main__":
    main()
