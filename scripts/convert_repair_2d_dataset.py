import argparse
import csv
import json
import random
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert raw RePAIR 2D fragments into ReassembleNet data.json format."
    )
    parser.add_argument(
        "--raw_root",
        default="2D_Fragments",
        help="Raw RePAIR 2D root containing 2D_Images and 2D_Ground_Truth.",
    )
    parser.add_argument(
        "--out_root",
        default="RePAIR_V2/2D_FINAL/2d_fixed",
        help="Output root where train/test puzzle folders will be written.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Ratio of valid puzzles assigned to train.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic train/test splitting.",
    )
    parser.add_argument(
        "--min_fragments",
        type=int,
        default=2,
        help="Minimum number of valid fragments required to keep a puzzle.",
    )
    return parser.parse_args()


def resolve_raw_root(raw_root):
    raw_root = Path(raw_root)
    candidates = [
        raw_root,
        raw_root / "2D_Fragments",
        raw_root.parent,
    ]

    for candidate in candidates:
        images_dir = candidate / "2D_Images"
        gt_dir = candidate / "2D_Ground_Truth"
        if images_dir.is_dir() and gt_dir.is_dir():
            return candidate, images_dir, gt_dir

    tried = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Could not find a raw RePAIR 2D root containing both 2D_Images and "
        f"2D_Ground_Truth. Tried:\n{tried}"
    )


def parse_number(value):
    number = float(value)
    return int(number) if number.is_integer() else number


def load_fragments(gt_path, puzzle_dir):
    fragments = []
    with gt_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        expected_columns = {"rpf", "x", "y", "rot"}
        if not reader.fieldnames or not expected_columns.issubset(reader.fieldnames):
            print(f"Warning: malformed ground truth header in {gt_path}; skipped")
            return fragments

        for row in reader:
            filename = row["rpf"].strip()
            if not filename:
                continue

            image_path = puzzle_dir / filename
            if not image_path.is_file():
                print(f"Warning: missing image {image_path}; fragment skipped")
                continue

            try:
                x = parse_number(row["x"])
                y = parse_number(row["y"])
                rot = parse_number(row["rot"])
            except (TypeError, ValueError):
                print(f"Warning: invalid numeric row in {gt_path}: {row}; fragment skipped")
                continue

            fragments.append(
                {
                    "filename": filename,
                    "pixel_position": [x, y],
                    "rotation": rot,
                }
            )

    return fragments


def copy_puzzle(puzzle_name, original_id, source_dir, split_dir, fragments):
    target_dir = split_dir / puzzle_name
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for fragment in fragments:
        filename = fragment["filename"]
        shutil.copy2(source_dir / filename, target_dir / filename)

    data_path = target_dir / "data.json"
    with data_path.open("w", encoding="utf-8") as f:
        json.dump({"original_id": original_id, "fragments": fragments}, f, indent=2)
        f.write("\n")


def main():
    args = parse_args()
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train_ratio must be between 0 and 1")
    if args.min_fragments < 1:
        raise ValueError("--min_fragments must be at least 1")

    raw_root, images_dir, gt_dir = resolve_raw_root(args.raw_root)
    assembled_dir = images_dir / "assembled_objects"
    if not assembled_dir.is_dir():
        raise FileNotFoundError(f"Could not find assembled_objects at {assembled_dir}")

    valid_puzzles = []
    skipped_small = 0
    skipped_missing_gt = 0

    for puzzle_dir in sorted(path for path in assembled_dir.iterdir() if path.is_dir()):
        puzzle_name = puzzle_dir.name
        gt_path = gt_dir / f"{puzzle_name}.txt"
        if not gt_path.is_file():
            print(f"Warning: missing ground truth {gt_path}; puzzle skipped")
            skipped_missing_gt += 1
            continue

        fragments = load_fragments(gt_path, puzzle_dir)
        if len(fragments) < args.min_fragments:
            print(
                f"Warning: puzzle {puzzle_name} has {len(fragments)} valid fragments; "
                f"below min_fragments={args.min_fragments}, puzzle skipped"
            )
            skipped_small += 1
            continue

        valid_puzzles.append((puzzle_name, puzzle_dir, fragments))

    rng = random.Random(args.seed)
    rng.shuffle(valid_puzzles)

    train_count = int(len(valid_puzzles) * args.train_ratio)
    train_puzzles = valid_puzzles[:train_count]
    test_puzzles = valid_puzzles[train_count:]

    out_root = Path(args.out_root)
    train_dir = out_root / "train"
    test_dir = out_root / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    mapping = {"train": {}, "test": {}}

    for idx, (original_id, puzzle_dir, fragments) in enumerate(train_puzzles, start=1):
        puzzle_name = f"puzzle_{idx:06d}"
        copy_puzzle(puzzle_name, original_id, puzzle_dir, train_dir, fragments)
        mapping["train"][puzzle_name] = original_id

    for idx, (original_id, puzzle_dir, fragments) in enumerate(test_puzzles, start=1):
        puzzle_name = f"puzzle_{idx:06d}"
        copy_puzzle(puzzle_name, original_id, puzzle_dir, test_dir, fragments)
        mapping["test"][puzzle_name] = original_id

    mapping_path = out_root / "mapping.json"
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
        f.write("\n")

    train_fragments = sum(len(fragments) for _, _, fragments in train_puzzles)
    test_fragments = sum(len(fragments) for _, _, fragments in test_puzzles)

    print(f"Raw root: {raw_root}")
    print(f"Output root: {out_root}")
    print(f"Train puzzles: {len(train_puzzles)}")
    print(f"Test puzzles: {len(test_puzzles)}")
    print(f"Train fragments: {train_fragments}")
    print(f"Test fragments: {test_fragments}")
    print(f"Total fragments: {train_fragments + test_fragments}")
    print(f"Skipped puzzles without ground truth: {skipped_missing_gt}")
    print(f"Skipped puzzles below min_fragments={args.min_fragments}: {skipped_small}")


if __name__ == "__main__":
    main()
