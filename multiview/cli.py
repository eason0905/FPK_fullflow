from __future__ import annotations

import argparse
import json
from pathlib import Path

from .integrator import MultiviewOptions, integrate_graphs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical 2D multiview package graphs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    integrate = subparsers.add_parser("integrate", help="Integrate per-image package graphs by part number.")
    integrate.add_argument("--graph-input", type=Path, required=True)
    integrate.add_argument("--dataset-root", type=Path, default=None)
    integrate.add_argument("--output-root", type=Path, required=True)
    integrate.add_argument("--config-json", default="{}", help="JSON object for multiview options.")
    integrate.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "integrate":
        options = MultiviewOptions.from_dict(json.loads(args.config_json))
        result = integrate_graphs(
            args.graph_input,
            args.output_root,
            dataset_root=args.dataset_root,
            options=options,
            limit=args.limit,
        )
    else:  # pragma: no cover
        raise ValueError(f"Unknown command: {args.command}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
