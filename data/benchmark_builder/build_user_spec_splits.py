from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.benchmark_builder.user_spec_generation import make_user_spec, resources_by_category
from tools.indiana211 import Resource, load_resource_index


DEFAULT_OUTPUT_DIR = Path("data/benchmark")


def main() -> None:
    args = parse_args()
    resources = load_resource_index(args.index_path).resources
    rng = random.Random(args.seed)
    used_resource_ids: set[str] = set()

    split_counts = {
        "train": args.train_count,
        "dev": args.dev_count,
        "test": args.test_count,
    }
    outputs = {
        "train": args.train_out or args.output_dir / f"user_specs_train_{args.train_count}.json",
        "dev": args.dev_out or args.output_dir / f"user_specs_dev_{args.dev_count}.json",
        "test": args.test_out or args.output_dir / f"user_specs_test_{args.test_count}.json",
    }

    for split_name, count in split_counts.items():
        specs = build_disjoint_specs(
            resources,
            count=count,
            rng=rng,
            used_resource_ids=used_resource_ids,
            spec_prefix=f"{split_name}-spec",
            progress_every=args.progress_every,
            split_name=split_name,
        )
        write_json(outputs[split_name], specs)
        print(f"Wrote {len(specs)} {split_name} specs to {outputs[split_name]}")


def build_disjoint_specs(
    resources: list[Resource],
    *,
    count: int,
    rng: random.Random,
    used_resource_ids: set[str],
    spec_prefix: str,
    progress_every: int,
    split_name: str,
) -> list[dict]:
    available = [resource for resource in resources if resource.resource_id not in used_resource_ids]
    by_category = resources_by_category(available)
    categories = sorted(by_category)
    if not categories:
        raise RuntimeError(f"No resources available for {split_name} split.")

    specs: list[dict] = []
    split_resource_ids: set[str] = set()
    category_index = 0
    attempts = 0
    max_attempts = max(count * 200, 5000)

    while len(specs) < count and attempts < max_attempts:
        attempts += 1
        category = categories[category_index % len(categories)]
        category_index += 1
        candidates = [
            resource
            for resource in by_category[category]
            if resource.resource_id not in used_resource_ids and resource.resource_id not in split_resource_ids
        ]
        if not candidates:
            continue
        resource = rng.choice(candidates)
        spec = make_user_spec(resource, category, rng)
        if spec is None:
            continue
        split_resource_ids.add(resource.resource_id)
        specs.append(spec)
        if progress_every and (len(specs) == 1 or len(specs) % progress_every == 0 or len(specs) == count):
            print(f"[{split_name}] selected={len(specs)}/{count} attempts={attempts}")

    if len(specs) < count:
        raise RuntimeError(f"Only generated {len(specs)}/{count} {split_name} specs after {attempts} attempts.")

    for idx, spec in enumerate(specs, start=1):
        spec["user_spec_id"] = f"{spec_prefix}-{idx:03d}"
    used_resource_ids.update(split_resource_ids)
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build disjoint train/dev/test user spec files.")
    parser.add_argument("--index-path", type=Path, default=Path("data/benchmark/filtered_resources_tagged.csv"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-count", type=int, default=100)
    parser.add_argument("--dev-count", type=int, default=50)
    parser.add_argument("--test-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--train-out", type=Path)
    parser.add_argument("--dev-out", type=Path)
    parser.add_argument("--test-out", type=Path)
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
