from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.chi2_study import (
    _parse_cuts,
    cut_efficiencies,
    load_sample,
    plot_cut_distributions,
    plot_cut_efficiencies,
    plot_cut_variable_response,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Study chi2/ndof cuts on reconstructed tracks.")
    parser.add_argument("--root", type=Path, default=Path("jan2026_minbias_1p0E34_1000evts.root"))
    parser.add_argument("--limit", type=int, default=None, help="number of events to load; omit for all")
    parser.add_argument("--cuts", type=str, default="8,7,6,5,4,3", help="comma/space separated chi2/ndof thresholds")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--efficiency-output", type=Path, default=Path("chi2_cut_efficiency.png"))
    parser.add_argument("--distribution-output", type=Path, default=Path("chi2_cut_distributions.png"))
    parser.add_argument("--response-output", type=Path, default=Path("chi2_cut_variable_response.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cuts = _parse_cuts(args.cuts)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = load_sample(args.root, limit=args.limit)
    metrics = cut_efficiencies(sample, cuts)
    efficiency_output = plot_cut_efficiencies(sample, cuts, output_dir / args.efficiency_output)
    distribution_output = plot_cut_distributions(sample, cuts, output_dir / args.distribution_output)
    response_output = plot_cut_variable_response(sample, cuts, output_dir / args.response_output)

    for cut, rec_eff, truth_eff in zip(metrics["cuts"], metrics["reconstructed_efficiency"], metrics["truth_efficiency"], strict=False):
        print(f"cut<{cut:g} reconstructed_eff={rec_eff:.4f} matched_eff={truth_eff:.4f}")

    print(
        f"events={len(sample.chi2ndof)} cuts={','.join(f'{cut:g}' for cut in cuts)} "
        f"efficiency_plot={efficiency_output} distribution_plot={distribution_output} response_plot={response_output} "
        f"limit={'all' if args.limit is None else args.limit}"
    )


if __name__ == "__main__":
    main()
