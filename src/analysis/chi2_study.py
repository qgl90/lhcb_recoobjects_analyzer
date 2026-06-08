from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

from .composition import TrackCompositionSample, _prepare_matplotlib, _scale_figure, _track_features, load_track_composition_sample


def _parse_cuts(cuts: str | list[float]) -> list[float]:
    if isinstance(cuts, list):
        values = [float(value) for value in cuts]
    else:
        parts = cuts.replace(",", " ").split()
        values = [float(part) for part in parts]
    if not values:
        raise ValueError("at least one chi2/ndof cut must be provided")
    return sorted(values, reverse=True)


def _valid_chi2_mask(sample: TrackCompositionSample) -> np.ndarray:
    return np.isfinite(sample.chi2ndof)


def cut_efficiencies(sample: TrackCompositionSample, cuts: list[float]) -> dict[str, np.ndarray]:
    valid = _valid_chi2_mask(sample)
    reconstructed_total = int(np.sum(valid))
    truth_total = int(np.sum(valid & sample.truth_mask))
    fake_total = int(np.sum(valid & sample.fake_mask))

    reconstructed_efficiency: list[float] = []
    reconstructed_inefficiency: list[float] = []
    truth_efficiency: list[float] = []
    truth_inefficiency: list[float] = []
    fake_efficiency: list[float] = []
    fake_inefficiency: list[float] = []

    for cut in cuts:
        selected = valid & (sample.chi2ndof < cut)
        reconstructed_pass = int(np.sum(selected))
        truth_pass = int(np.sum(selected & sample.truth_mask))
        fake_pass = int(np.sum(selected & sample.fake_mask))

        rec_eff = reconstructed_pass / reconstructed_total if reconstructed_total else float("nan")
        truth_eff = truth_pass / truth_total if truth_total else float("nan")
        fake_eff = fake_pass / fake_total if fake_total else float("nan")

        reconstructed_efficiency.append(rec_eff)
        reconstructed_inefficiency.append(1.0 - rec_eff if np.isfinite(rec_eff) else float("nan"))
        truth_efficiency.append(truth_eff)
        truth_inefficiency.append(1.0 - truth_eff if np.isfinite(truth_eff) else float("nan"))
        fake_efficiency.append(fake_eff)
        fake_inefficiency.append(1.0 - fake_eff if np.isfinite(fake_eff) else float("nan"))

    return {
        "cuts": np.asarray(cuts, dtype=float),
        "reconstructed_efficiency": np.asarray(reconstructed_efficiency, dtype=float),
        "reconstructed_inefficiency": np.asarray(reconstructed_inefficiency, dtype=float),
        "truth_efficiency": np.asarray(truth_efficiency, dtype=float),
        "truth_inefficiency": np.asarray(truth_inefficiency, dtype=float),
        "fake_efficiency": np.asarray(fake_efficiency, dtype=float),
        "fake_inefficiency": np.asarray(fake_inefficiency, dtype=float),
        "reconstructed_total": np.asarray([reconstructed_total], dtype=int),
        "truth_total": np.asarray([truth_total], dtype=int),
        "fake_total": np.asarray([fake_total], dtype=int),
    }


def plot_cut_efficiencies(sample: TrackCompositionSample, cuts: list[float], output_path: str | Path) -> Path:
    plt = _prepare_matplotlib()
    metrics = cut_efficiencies(sample, cuts)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    _scale_figure(fig)

    ax_eff, ax_ineff = axes
    ax_eff.plot(
        metrics["cuts"],
        metrics["reconstructed_efficiency"],
        marker="o",
        color="#2F4B7C",
        label="reconstructed",
    )
    ax_eff.plot(
        metrics["cuts"],
        metrics["truth_efficiency"],
        marker="o",
        color="#4C72B0",
        label="matched truth",
    )
    ax_eff.plot(
        metrics["cuts"],
        metrics["fake_efficiency"],
        marker="o",
        color="#C44E52",
        label="fake",
    )
    ax_eff.set_xlabel("chi2/ndof cut")
    ax_eff.set_ylabel("efficiency")
    ax_eff.set_ylim(0, 1.05)
    ax_eff.grid(True, alpha=0.2)
    ax_eff.set_title("Efficiency vs chi2/ndof cut")
    ax_eff.legend(frameon=False)

    ax_ineff.plot(
        metrics["cuts"],
        metrics["reconstructed_inefficiency"],
        marker="o",
        color="#2F4B7C",
        label="reconstructed",
    )
    ax_ineff.plot(
        metrics["cuts"],
        metrics["truth_inefficiency"],
        marker="o",
        color="#4C72B0",
        label="matched truth",
    )
    ax_ineff.plot(
        metrics["cuts"],
        metrics["fake_inefficiency"],
        marker="o",
        color="#C44E52",
        label="fake",
    )
    ax_ineff.set_xlabel("chi2/ndof cut")
    ax_ineff.set_ylabel("1 - efficiency")
    ax_ineff.set_ylim(0, 1.05)
    ax_ineff.grid(True, alpha=0.2)
    ax_ineff.set_title("Inefficiency vs chi2/ndof cut")
    ax_ineff.legend(frameon=False)

    output_path = Path(output_path)
    fig.suptitle("chi2/ndof cut response")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _distribution_values(sample: TrackCompositionSample, variable: str) -> tuple[np.ndarray, bool]:
    features = _track_features(sample)
    mapping = {
        "p": features["p"],
        "pt": features["pt"],
        "eta": features["eta"],
        "phi": features["phi"],
    }
    if variable not in mapping:
        raise ValueError(f"unsupported variable: {variable}")
    return mapping[variable], variable in {"p", "pt"}


def plot_cut_distributions(sample: TrackCompositionSample, cuts: list[float], output_path: str | Path) -> Path:
    plt = _prepare_matplotlib()
    values_by_var = {
        "p": _distribution_values(sample, "p"),
        "pt": _distribution_values(sample, "pt"),
        "eta": _distribution_values(sample, "eta"),
        "phi": _distribution_values(sample, "phi"),
    }

    fake_mask = sample.fake_mask
    truth_mask = sample.truth_mask
    valid_mask = _valid_chi2_mask(sample)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(cuts)))

    fig, axes = plt.subplots(4, 2, figsize=(13, 16), constrained_layout=True)
    _scale_figure(fig)

    for row, (variable, (values, logx)) in enumerate(values_by_var.items()):
        for col, (mask, title) in enumerate(((fake_mask, "fake"), (truth_mask, "matched truth"))):
            ax = axes[row, col]
            base_values = values[mask & valid_mask]
            finite_base = base_values[np.isfinite(base_values)]
            if logx:
                finite_base = finite_base[finite_base > 0]
            if len(finite_base) == 0:
                ax.set_title(f"{title} {variable}")
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
                continue

            if logx:
                xmin = max(np.min(finite_base), 1e-6)
                xmax = np.max(finite_base)
                bins = np.logspace(np.log10(xmin), np.log10(xmax), 50)
            else:
                bins = np.linspace(np.min(finite_base), np.max(finite_base), 50)

            for color, cut in zip(colors, cuts, strict=False):
                selected = mask & valid_mask & (sample.chi2ndof < cut)
                cut_values = values[selected]
                finite_values = cut_values[np.isfinite(cut_values)]
                if logx:
                    finite_values = finite_values[finite_values > 0]
                if len(finite_values) == 0:
                    continue
                hist, edges = np.histogram(finite_values, bins=bins, density=True)
                centers = 0.5 * (edges[:-1] + edges[1:])
                ax.step(centers, hist, where="mid", color=color, linewidth=1.8, label=f"< {cut:g}")

            ax.set_title(f"{title} {variable}")
            ax.set_ylabel("norm. tracks")
            ax.grid(True, alpha=0.2)
            if logx:
                ax.set_xscale("log")
            if row == len(values_by_var) - 1:
                xlabel = {
                    "p": "p [MeV]",
                    "pt": "pT [MeV]",
                    "eta": "eta",
                    "phi": "phi",
                }[variable]
                ax.set_xlabel(xlabel)
            if row == 0 and col == 0:
                ax.legend(frameon=False, fontsize=8, title="chi2/ndof")

    output_path = Path(output_path)
    fig.suptitle("Track kinematics under chi2/ndof cuts")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _variable_bins(values: np.ndarray, logx: bool, bins: int = 40) -> np.ndarray:
    finite_values = values[np.isfinite(values)]
    if logx:
        finite_values = finite_values[finite_values > 0]
    if len(finite_values) == 0:
        raise ValueError("no finite values available for binning")
    if logx:
        xmin = max(np.min(finite_values), 1e-6)
        xmax = np.max(finite_values)
        return np.logspace(np.log10(xmin), np.log10(xmax), bins)
    return np.linspace(np.min(finite_values), np.max(finite_values), bins)


def plot_cut_variable_response(sample: TrackCompositionSample, cuts: list[float], output_path: str | Path) -> Path:
    plt = _prepare_matplotlib()
    features = _track_features(sample)
    valid_mask = _valid_chi2_mask(sample)
    fake_mask = sample.fake_mask & valid_mask
    truth_mask = sample.truth_mask & valid_mask

    variables = [
        ("p", features["p"], True, "p [MeV]"),
        ("pt", features["pt"], True, "pT [MeV]"),
        ("eta", features["eta"], False, "eta"),
        ("phi", features["phi"], False, "phi"),
    ]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(cuts)))

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    _scale_figure(fig)

    cut_handles = [Line2D([0], [0], color=color, linestyle="-", linewidth=2, label=f"< {cut:g}") for color, cut in zip(colors, cuts, strict=False)]

    for col, (variable_name, values, logx, xlabel) in enumerate(variables):
        truth_values = values[truth_mask]
        fake_values = values[fake_mask]
        finite_truth = truth_values[np.isfinite(truth_values)]
        finite_fake = fake_values[np.isfinite(fake_values)]
        if logx:
            finite_truth = finite_truth[finite_truth > 0]
            finite_fake = finite_fake[finite_fake > 0]
        combined = np.concatenate([finite_truth, finite_fake]) if len(finite_truth) and len(finite_fake) else (
            finite_truth if len(finite_truth) else finite_fake
        )
        bins = _variable_bins(combined, logx=logx)
        centers = 0.5 * (bins[:-1] + bins[1:])
        histograms = {
            "fake": np.histogram(finite_fake, bins=bins)[0],
            "truth": np.histogram(finite_truth, bins=bins)[0],
        }

        for row, (category, mask, title, ylabel) in enumerate(
            (
                ("fake", fake_mask, "fake tracks", "fake(cut) / fake(total)"),
                ("truth", truth_mask, "signal truth tracks", "truth(cut) / truth(total)"),
            )
        ):
            ax = axes[row, col]
            denominator = histograms[category]
            for color, cut in zip(colors, cuts, strict=False):
                selected = valid_mask & (sample.chi2ndof < cut) & mask
                selected_values = values[selected]
                selected_values = selected_values[np.isfinite(selected_values)]
                if logx:
                    selected_values = selected_values[selected_values > 0]
                selected_hist, _ = np.histogram(selected_values, bins=bins)
                pass_fraction = np.divide(
                    selected_hist,
                    denominator,
                    out=np.full_like(selected_hist, np.nan, dtype=float),
                    where=denominator > 0,
                )
                ax.plot(centers, pass_fraction, color=color, linewidth=2, alpha=0.95)

            ax.set_title(f"{title}: {variable_name}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, 1.05)
            if logx:
                ax.set_xscale("log")
            ax.grid(True, alpha=0.2)

    for col in range(len(variables)):
        axes[0, col].set_title(f"fake tracks: {variables[col][0]}")
        axes[1, col].set_title(f"signal truth tracks: {variables[col][0]}")

    fig.legend(
        handles=cut_handles,
        loc="lower center",
        ncol=min(len(cuts), 6),
        frameon=False,
        title="chi2/ndof cut",
        bbox_to_anchor=(0.5, -0.02),
    )

    output_path = Path(output_path)
    fig.suptitle("chi2/ndof pass fraction vs kinematics")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def load_sample(path: str | Path, limit: int | None = None) -> TrackCompositionSample:
    return load_track_composition_sample(path, limit=limit)
