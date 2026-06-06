from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.training_samples import DEFAULT_FEATURE_GROUPS, FEATURE_GROUPS, parse_feature_groups, write_training_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a Parquet track-level fake-vs-truth sample for ghost-suppression neural-net training."
    )
    parser.add_argument("--root", type=Path, default=Path("jan2026_minbias_1p0E34_1000evts.root"), help="input ROOT ntuple")
    parser.add_argument("--output", type=Path, default=Path("training_samples_fakes_vs_truth.parquet"), help="output Parquet file")
    parser.add_argument("--limit", type=int, default=None, help="number of events to read; omit for all events")
    parser.add_argument(
        "--features",
        default=",".join(DEFAULT_FEATURE_GROUPS),
        help=f"comma-separated feature groups to compute; choices: {', '.join(sorted(FEATURE_GROUPS))}",
    )
    parser.add_argument("--balance", action="store_true", help="downsample to the smaller fake/truth class")
    parser.add_argument(
        "--max-tracks-per-class",
        type=int,
        default=None,
        help="optional cap applied independently to fake and truth tracks after reading",
    )
    parser.add_argument("--random-seed", type=int, default=12345, help="random seed for balancing/downsampling")
    parser.add_argument("--compression", default="zstd", help="Parquet compression codec passed to pyarrow")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_groups = parse_feature_groups(args.features)
    output, summary = write_training_sample(
        args.root,
        args.output,
        limit=args.limit,
        feature_groups=feature_groups,
        balance=args.balance,
        max_tracks_per_class=args.max_tracks_per_class,
        random_seed=args.random_seed,
        compression=args.compression,
    )
    print(
        f"wrote={output} events={summary['events']} tracks={summary['tracks']} "
        f"fake_tracks={summary['fake_tracks']} truth_tracks={summary['truth_tracks']} "
        f"columns={summary['columns']} features={','.join(feature_groups)}"
    )


if __name__ == "__main__":
    main()
