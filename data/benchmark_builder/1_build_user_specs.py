from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.benchmark_builder.user_spec_generation import build_user_specs
from tools.indiana211 import load_resource_index


CASE_SPECS_PATH = Path("data/benchmark/case_specs.json")


def main() -> None:
    args = parse_args()
    resources = load_resource_index(args.index_path).resources
    specs = build_user_specs(
        resources,
        difficulty_targets={
            "easy": args.easy,
            "medium": args.medium,
            "hard": args.hard,
        },
        seed=args.seed,
        progress_every=args.progress_every,
    )
    write_json(args.case_specs_out, specs)
    print(f"Wrote {args.case_specs_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic benchmark user specs.")
    parser.add_argument("--index-path", type=Path, default=Path("data/benchmark/filtered_resources_tagged.csv"))
    parser.add_argument("--easy", type=int, required=True)
    parser.add_argument("--medium", type=int, required=True)
    parser.add_argument("--hard", type=int, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--case-specs-out", type=Path, default=CASE_SPECS_PATH)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
