"""Run project experiments from a single Python entry point."""

from __future__ import annotations

import argparse

from src.experiments import available_experiments, build_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Pix2Pix project experiment.")
    parser.add_argument("experiment", choices=available_experiments().split(", "))
    parser.add_argument("--dry-run", action="store_true", help="Print the command without executing it.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--lambda-l1", type=float, default=None)
    parser.add_argument("--lambda-fm", type=float, default=None)
    parser.add_argument("--ngf", type=int, default=None)
    parser.add_argument("--ndf", type=int, default=None)
    parser.add_argument("--num-scales", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--attention-channels", type=int, default=None)
    parser.add_argument("--amp", dest="amp", action="store_true", default=None)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment = build_experiment(args.experiment, args)
    return experiment.run(dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
