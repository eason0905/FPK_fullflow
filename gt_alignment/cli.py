from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluator import AlignmentOptions, evaluate_alignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate canonical multiview graph alignment against ScanResultFormat.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--multiview-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config-json", default="{}")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    options = AlignmentOptions.from_dict(json.loads(args.config_json))
    result = evaluate_alignment(
        dataset_root=args.dataset_root,
        multiview_root=args.multiview_root,
        output_root=args.output_root,
        options=options,
        limit=args.limit,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
