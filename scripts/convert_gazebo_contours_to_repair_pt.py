#!/usr/bin/env python3
"""Convert Gazebo contour JSON into a RePAIR/ReassembleNet-style .pt sample."""

import argparse
import json
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Convert Gazebo contour JSON to ReassembleNet .pt sample.')
    parser.add_argument('--input_json', type=Path, default=Path('/tmp/gazebo_reassemblenet_contours.json'))
    parser.add_argument('--output_pt', type=Path, default=Path('/tmp/gazebo_reassemblenet_input.pt'))
    parser.add_argument('--puzzle_name', default='gazebo_scan_001')
    parser.add_argument('--texture_dim', type=int, default=512)
    parser.add_argument('--points_per_piece', type=int, default=100)
    return parser.parse_args()


def pad_or_trim_points(points, target_count):
    if not points:
        return []
    rows = list(points[:target_count])
    while len(rows) < target_count:
        rows.append(rows[-1])
    return rows


def main():
    args = parse_args()
    with args.input_json.open('r', encoding='utf-8') as f:
        payload = json.load(f)

    image_size = payload.get('image_size', [640, 480])
    result = {
        'puzzle_name': args.puzzle_name,
        '0': [int(image_size[0]), int(image_size[1])],
    }
    zero_texture = [0.0] * args.texture_dim

    pieces = sorted(payload.get('pieces', []), key=lambda item: int(item.get('piece_id', 0)))
    if not pieces:
        raise ValueError(f'no pieces found in {args.input_json}')
    for output_index, piece in enumerate(pieces, start=1):
        rows = []
        for point in pad_or_trim_points(piece.get('points', []), args.points_per_piece):
            rows.append([
                [int(point['pixel'][0]), int(point['pixel'][1])],
                [float(point['geometry'][0]), float(point['geometry'][1])],
                zero_texture,
            ])
        rows.append(['piece_name', piece.get('piece_name', f'puzzle_piece_{output_index}')])
        rows.append(['global', zero_texture])
        result[str(output_index)] = rows

    args.output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output_pt)
    print(f'wrote {args.output_pt}')
    print(f"puzzle_name: {result['puzzle_name']} image_size={result['0']} pieces={len(pieces)}")
    for index in range(1, len(pieces) + 1):
        print(f"piece_{index}: rows={len(result[str(index)])} name={result[str(index)][-2][1]}")


if __name__ == '__main__':
    main()
