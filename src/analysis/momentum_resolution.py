from __future__ import annotations

from pathlib import Path

import numpy as np

try:  # optional for interactive analysis
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("pandas is required for momentum-resolution analysis; install it with `python3 -m pip install pandas`") from exc

from .composition import TrackCompositionSample, _prepare_matplotlib, _resolution_features, _scale_figure, load_track_composition_sample


def _truth_kinematics_from_arrays(px: np.ndarray, py: np.ndarray, pz: np.ndarray) -> dict[str, np.ndarray]:
    truth_p = np.sqrt(px * px + py * py + pz * pz)
    truth_pt = np.sqrt(px * px + py * py)
    truth_eta = np.arcsinh(np.divide(pz, truth_pt, out=np.full_like(truth_pt, np.nan), where=truth_pt > 0))
    truth_phi = np.arctan2(py, px)
    return {
        "truth_p": truth_p,
        "truth_pt": truth_pt,
        "truth_eta": truth_eta,
        "truth_phi": truth_phi,
    }


def resolution_frame_from_sample(sample: TrackCompositionSample) -> pd.DataFrame:
    features = _resolution_features(sample)
    frame = pd.DataFrame(
        {
            "label": sample.mc_truth.astype(np.int8),
            "chi2ndof": sample.chi2ndof,
            "reco_p": features["p"],
            "reco_pt": features["pt"],
            "reco_eta": features["eta"],
            "reco_phi": features["phi"],
            "truth_p": features["truth_p"],
            "truth_pt": features["truth_pt"],
            "truth_eta": features["truth_eta"],
            "truth_phi": features["truth_phi"],
            "delta_p_over_p": features["p_resolution"],
            "delta_pt_over_pt": features["pt_resolution"],
        }
    )
    return frame


def resolution_frame_from_parquet(path: str | Path, limit: int | None = None) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if limit is not None:
        frame = frame.head(limit).copy()

    if "truth_p" not in frame.columns:
        if {"mc_px", "mc_py", "mc_pz"}.issubset(frame.columns):
            truth = _truth_kinematics_from_arrays(frame["mc_px"].to_numpy(), frame["mc_py"].to_numpy(), frame["mc_pz"].to_numpy())
            for key, values in truth.items():
                frame[key] = values
        else:
            raise ValueError("parquet input is missing truth-momentum columns")

    if "reco_p" not in frame.columns:
        if {"qop", "tx", "ty"}.issubset(frame.columns):
            tx = frame["tx"].to_numpy()
            ty = frame["ty"].to_numpy()
            qop = frame["qop"].to_numpy()
            slope2 = tx * tx + ty * ty
            direction_norm = np.sqrt(1.0 + slope2)
            reco_p = np.divide(1.0, np.abs(qop), out=np.full_like(qop, np.inf, dtype=float), where=qop != 0)
            reco_pt = np.where(np.isfinite(reco_p), reco_p * np.sqrt(slope2) / direction_norm, np.nan)
            reco_eta = np.where(
                np.isfinite(reco_pt) & (reco_pt > 0),
                np.arcsinh(np.divide(reco_p / direction_norm, reco_pt, out=np.full_like(reco_pt, np.nan), where=reco_pt > 0)),
                np.nan,
            )
            reco_phi = np.arctan2(ty, tx)
            frame["reco_p"] = reco_p
            frame["reco_pt"] = reco_pt
            frame["reco_eta"] = reco_eta
            frame["reco_phi"] = reco_phi
        else:
            raise ValueError("parquet input is missing reconstructed momentum columns")

    if "delta_p_over_p" not in frame.columns:
        frame["delta_p_over_p"] = np.divide(
            frame["reco_p"] - frame["truth_p"],
            frame["truth_p"],
            out=np.full(len(frame), np.nan, dtype=float),
            where=frame["truth_p"].to_numpy() != 0,
        )
    if "delta_pt_over_pt" not in frame.columns:
        frame["delta_pt_over_pt"] = np.divide(
            frame["reco_pt"] - frame["truth_pt"],
            frame["truth_pt"],
            out=np.full(len(frame), np.nan, dtype=float),
            where=frame["truth_pt"].to_numpy() != 0,
        )
    if "label" not in frame.columns:
        if "mc_truth" in frame.columns:
            frame["label"] = (frame["mc_truth"] == 1).astype(np.int8)
        else:
            raise ValueError("parquet input is missing track labels")
    return frame


def load_resolution_frame(path: str | Path, limit: int | None = None) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        return resolution_frame_from_parquet(path, limit=limit)
    sample = load_track_composition_sample(path, limit=limit)
    return resolution_frame_from_sample(sample)


def select_truth_frame(frame: pd.DataFrame, chi2_max: float | None = None) -> pd.DataFrame:
    selected = frame[frame["label"] == 1].copy()
    if chi2_max is not None:
        selected = selected[selected["chi2ndof"] < chi2_max].copy()
    return selected


def build_binned_resolution(
    frame: pd.DataFrame,
    *,
    bin_var: str,
    response_var: str = "delta_p_over_p",
    n_bins: int = 20,
    chi2_max: float | None = None,
    fit_mode: str = "moments",
) -> pd.DataFrame:
    if bin_var not in frame.columns:
        raise ValueError(f"unknown bin variable: {bin_var}")
    if response_var not in frame.columns:
        raise ValueError(f"unknown response variable: {response_var}")

    selected = select_truth_frame(frame, chi2_max=chi2_max)
    selected = selected[[bin_var, response_var]].replace([np.inf, -np.inf], np.nan).dropna()
    if selected.empty:
        return pd.DataFrame(columns=["bin_low", "bin_high", "bin_center", "count", "mean", "std", "fit_mu", "fit_sigma"])

    x = selected[bin_var].to_numpy(dtype=float)
    y = selected[response_var].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if bin_var in {"truth_p", "truth_pt", "reco_p", "reco_pt"}:
        x = x[x > 0]
    if len(x) == 0:
        return pd.DataFrame(columns=["bin_low", "bin_high", "bin_center", "count", "mean", "std", "fit_mu", "fit_sigma"])

    if bin_var in {"truth_p", "truth_pt", "reco_p", "reco_pt"}:
        bins = np.logspace(np.log10(max(np.min(x), 1e-6)), np.log10(np.max(x)), n_bins + 1)
    else:
        bins = np.linspace(np.min(x), np.max(x), n_bins + 1)

    rows: list[dict[str, float | int]] = []
    for low, high in zip(bins[:-1], bins[1:], strict=False):
        mask = (x >= low) & (x < high)
        values = y[mask]
        values = values[np.isfinite(values)]
        if len(values) == 0:
            rows.append(
                {
                    "bin_low": float(low),
                    "bin_high": float(high),
                    "bin_center": float(np.sqrt(low * high) if low > 0 and high > 0 else 0.5 * (low + high)),
                    "count": 0,
                    "mean": float("nan"),
                    "std": float("nan"),
                    "fit_mu": float("nan"),
                    "fit_sigma": float("nan"),
                }
            )
            continue
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=0))
        fit_mu = mean
        fit_sigma = std
        if fit_mode == "gaussian":
            try:
                from scipy.stats import norm

                fit_mu, fit_sigma = [float(v) for v in norm.fit(values)]
            except Exception:
                fit_mu, fit_sigma = mean, std
        rows.append(
            {
                "bin_low": float(low),
                "bin_high": float(high),
                "bin_center": float(np.sqrt(low * high) if low > 0 and high > 0 else 0.5 * (low + high)),
                "count": int(len(values)),
                "mean": mean,
                "std": std,
                "fit_mu": float(fit_mu),
                "fit_sigma": float(fit_sigma),
            }
        )
    return pd.DataFrame(rows)


def plot_momentum_resolution_scatter(
    sample: TrackCompositionSample, output_path: str | Path, chi2_max: float | None = None
) -> Path:
    plt = _prepare_matplotlib()
    frame = resolution_frame_from_sample(sample)
    frame = select_truth_frame(frame, chi2_max=chi2_max)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["truth_p", "delta_p_over_p"])
    frame = frame[frame["truth_p"] > 0]
    if frame.empty:
        raise ValueError("no truth-matched tracks available for momentum resolution plot")

    x = frame["truth_p"].to_numpy()
    y = frame["delta_p_over_p"].to_numpy()
    bins = np.logspace(np.log10(max(np.min(x), 1e-6)), np.log10(np.max(x)), 24)
    summary = build_binned_resolution(frame, bin_var="truth_p", response_var="delta_p_over_p", n_bins=23, chi2_max=chi2_max)

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    _scale_figure(fig)
    ax.scatter(x, y, s=4, alpha=0.08, color="#7F7F7F", edgecolors="none")
    centers = summary["bin_center"].to_numpy(dtype=float)
    mean = summary["mean"].to_numpy(dtype=float)
    std = summary["std"].to_numpy(dtype=float)
    valid = summary["count"].to_numpy(dtype=int) > 0
    ax.errorbar(
        centers[valid],
        mean[valid],
        yerr=std[valid],
        fmt="o-",
        color="#2F4B7C",
        ecolor="#2F4B7C",
        capsize=3,
        linewidth=2,
        label="binned mean ± stddev",
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("truth p [MeV]")
    ax.set_ylabel("(p(reco) - p(truth)) / p(truth)")
    ax.set_title("Truth-matched momentum resolution scatter")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)

    output_path = Path(output_path)
    fig.suptitle("Momentum resolution scatter for truth-matched tracks")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
