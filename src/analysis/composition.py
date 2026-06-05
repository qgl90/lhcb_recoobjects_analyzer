from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import awkward as ak
import numpy as np

from .root_loader import _tree_from_file


@dataclass(slots=True)
class TrackCompositionSample:
    mc_truth: np.ndarray
    mc_hasTV: np.ndarray
    mc_hasUP: np.ndarray
    mc_hasMP: np.ndarray
    mc_hasFT: np.ndarray
    tv_hits: np.ndarray
    up_hits: np.ndarray
    ft_hits: np.ndarray
    mp_hits: np.ndarray
    ft_truth_hit_mask: np.ndarray
    mp_truth_hit_mask: np.ndarray
    tx: np.ndarray
    ty: np.ndarray
    qop: np.ndarray
    chi2ndof: np.ndarray

    @property
    def fake_mask(self) -> np.ndarray:
        return self.mc_truth == 0

    @property
    def truth_mask(self) -> np.ndarray:
        return self.mc_truth == 1


@dataclass(slots=True)
class EventCompositionSample:
    event_number: np.ndarray
    run_number: np.ndarray
    bunch_crossing_id: np.ndarray
    pv_count: np.ndarray
    fake_count: np.ndarray
    truth_count: np.ndarray
    fake_rate: np.ndarray


@dataclass(slots=True)
class EventFakePvSample:
    event_number: np.ndarray
    run_number: np.ndarray
    bunch_crossing_id: np.ndarray
    pv_count: np.ndarray
    fake_count: np.ndarray
    truth_count: np.ndarray
    fake_rate: np.ndarray


def _track_level_branch_names() -> list[str]:
    branches = [
        "MC_truth",
        "MC_hasTV",
        "MC_hasUP",
        "MC_hasMP",
        "MC_hasFT",
        "TVHits_n",
        "UPHits_n",
        "FTHits_n",
        "MPHits_n",
        "FTHits_mc_x",
        "MPHits_mc_x",
        "FirstMeasurement_tx",
        "FirstMeasurement_ty",
        "FirstMeasurement_qop",
        "Track_chi2ndof",
    ]
    return branches


def _track_truth_hit_mask(hit_mc: ak.Array, hit_counts: ak.Array) -> np.ndarray:
    masks: list[np.ndarray] = []
    for event_hits_mc, event_counts in zip(hit_mc, hit_counts, strict=False):
        grouped = ak.unflatten(event_hits_mc, event_counts)
        event_mask = ak.to_numpy(ak.any(np.isfinite(grouped), axis=1))
        masks.append(np.asarray(event_mask, dtype=bool))
    return np.concatenate(masks) if masks else np.asarray([], dtype=bool)


def load_track_composition_sample(path: str | Path, limit: int | None = None) -> TrackCompositionSample:
    tree = _tree_from_file(path)
    n_events = int(tree.num_entries if limit is None else min(tree.num_entries, limit))
    arrays = tree.arrays(library="ak", entry_start=0, entry_stop=n_events, filter_name=_track_level_branch_names())

    def flat(name: str) -> np.ndarray:
        return np.asarray(ak.to_numpy(ak.flatten(arrays[name])))

    return TrackCompositionSample(
        mc_truth=flat("MC_truth"),
        mc_hasTV=flat("MC_hasTV"),
        mc_hasUP=flat("MC_hasUP"),
        mc_hasMP=flat("MC_hasMP"),
        mc_hasFT=flat("MC_hasFT"),
        tv_hits=flat("TVHits_n"),
        up_hits=flat("UPHits_n"),
        ft_hits=flat("FTHits_n"),
        mp_hits=flat("MPHits_n"),
        ft_truth_hit_mask=_track_truth_hit_mask(arrays["FTHits_mc_x"], arrays["FTHits_n"]),
        mp_truth_hit_mask=_track_truth_hit_mask(arrays["MPHits_mc_x"], arrays["MPHits_n"]),
        tx=flat("FirstMeasurement_tx"),
        ty=flat("FirstMeasurement_ty"),
        qop=flat("FirstMeasurement_qop"),
        chi2ndof=flat("Track_chi2ndof"),
    )


def load_event_composition_sample(path: str | Path, limit: int | None = None) -> EventCompositionSample:
    tree = _tree_from_file(path)
    n_events = int(tree.num_entries if limit is None else min(tree.num_entries, limit))
    arrays = tree.arrays(
        library="ak",
        entry_start=0,
        entry_stop=n_events,
        filter_name=["EventNumber", "RunNumber", "BunchCrossingID", "MC_truth", "PV_x"],
    )

    event_number = np.asarray(ak.to_numpy(arrays["EventNumber"]))
    run_number = np.asarray(ak.to_numpy(arrays["RunNumber"]))
    bunch_crossing_id = np.asarray(ak.to_numpy(arrays["BunchCrossingID"]))

    fake_count_list: list[int] = []
    truth_count_list: list[int] = []
    pv_count_list: list[int] = []
    fake_rate_list: list[float] = []

    for mc_truth, pv_x in zip(arrays["MC_truth"], arrays["PV_x"], strict=False):
        fake_count = int(ak.sum(mc_truth == 0))
        truth_count = int(ak.sum(mc_truth == 1))
        total_count = fake_count + truth_count
        fake_rate = float(fake_count / total_count) if total_count > 0 else float("nan")
        fake_count_list.append(fake_count)
        truth_count_list.append(truth_count)
        pv_count_list.append(int(len(pv_x)))
        fake_rate_list.append(fake_rate)

    return EventCompositionSample(
        event_number=event_number,
        run_number=run_number,
        bunch_crossing_id=bunch_crossing_id,
        pv_count=np.asarray(pv_count_list, dtype=int),
        fake_count=np.asarray(fake_count_list, dtype=int),
        truth_count=np.asarray(truth_count_list, dtype=int),
        fake_rate=np.asarray(fake_rate_list, dtype=float),
    )


def load_event_fake_pv_sample(path: str | Path, limit: int | None = None) -> EventFakePvSample:
    tree = _tree_from_file(path)
    n_events = int(tree.num_entries if limit is None else min(tree.num_entries, limit))
    arrays = tree.arrays(
        library="ak",
        entry_start=0,
        entry_stop=n_events,
        filter_name=["EventNumber", "RunNumber", "BunchCrossingID", "MC_truth", "PV_x"],
    )

    event_number = np.asarray(ak.to_numpy(arrays["EventNumber"]))
    run_number = np.asarray(ak.to_numpy(arrays["RunNumber"]))
    bunch_crossing_id = np.asarray(ak.to_numpy(arrays["BunchCrossingID"]))

    fake_count_list: list[int] = []
    truth_count_list: list[int] = []
    pv_count_list: list[int] = []
    fake_rate_list: list[float] = []

    for mc_truth, pv_x in zip(arrays["MC_truth"], arrays["PV_x"], strict=False):
        fake_count = int(ak.sum(mc_truth == 0))
        truth_count = int(ak.sum(mc_truth == 1))
        total_count = fake_count + truth_count
        fake_rate = float(fake_count / total_count) if total_count > 0 else float("nan")
        fake_count_list.append(fake_count)
        truth_count_list.append(truth_count)
        pv_count_list.append(int(len(pv_x)))
        fake_rate_list.append(fake_rate)

    return EventFakePvSample(
        event_number=event_number,
        run_number=run_number,
        bunch_crossing_id=bunch_crossing_id,
        pv_count=np.asarray(pv_count_list, dtype=int),
        fake_count=np.asarray(fake_count_list, dtype=int),
        truth_count=np.asarray(truth_count_list, dtype=int),
        fake_rate=np.asarray(fake_rate_list, dtype=float),
    )


def _track_features(sample: TrackCompositionSample) -> dict[str, np.ndarray]:
    slope2 = sample.tx * sample.tx + sample.ty * sample.ty
    direction_norm = np.sqrt(1.0 + slope2)
    p = np.divide(1.0, np.abs(sample.qop), out=np.full_like(sample.qop, np.inf, dtype=float), where=sample.qop != 0)
    pt = np.where(np.isfinite(p), p * np.sqrt(slope2) / direction_norm, np.nan)
    pz = np.where(np.isfinite(p), p / direction_norm, np.nan)
    ratio = np.divide(pz, pt, out=np.full_like(pz, np.nan, dtype=float), where=pt != 0)
    eta = np.where(np.isfinite(pt) & (pt != 0), np.arcsinh(ratio), np.nan)
    phi = np.arctan2(sample.ty, sample.tx)
    return {
        "p": p,
        "pt": pt,
        "pz": pz,
        "eta": eta,
        "phi": phi,
        "chi2ndof": sample.chi2ndof,
    }


def _prepare_matplotlib():
    import os

    mplconfigdir = Path("/private/tmp/mplconfig")
    mplconfigdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mplconfigdir))

    import mplhep as hep
    import matplotlib

    matplotlib.use("Agg", force=True)
    hep.style.use("LHCb2")
    import matplotlib.pyplot as plt

    return plt


def _scale_figure(fig, scale: float = 1.3) -> None:
    width, height = fig.get_size_inches()
    fig.set_size_inches(width * scale, height * scale, forward=True)


def plot_fake_truth_hitcount_distributions(sample: TrackCompositionSample, output_path: str | Path) -> Path:
    plt = _prepare_matplotlib()

    bins_by_name = {
        "TV hits": np.arange(0, int(max(sample.tv_hits.max(), 1)) + 2) - 0.5,
        "UP hits": np.arange(0, int(max(sample.up_hits.max(), 1)) + 2) - 0.5,
        "FT hits": np.arange(0, int(max(sample.ft_hits.max(), 1)) + 2) - 0.5,
        "MP hits": np.arange(0, int(max(sample.mp_hits.max(), 1)) + 2) - 0.5,
    }

    fake = sample.fake_mask
    truth = sample.truth_mask

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    _scale_figure(fig)
    for ax, (label, bins) in zip(axes.flat, bins_by_name.items()):
        values = {
            "TV hits": sample.tv_hits,
            "UP hits": sample.up_hits,
            "FT hits": sample.ft_hits,
            "MP hits": sample.mp_hits,
        }[label]
        fake_values = values[fake]
        truth_values = values[truth]
        ax.hist(
            fake_values,
            bins=bins,
            histtype="step",
            linewidth=2,
            label="fake",
            color="#C44E52",
        )
        ax.hist(
            truth_values,
            bins=bins,
            histtype="step",
            linewidth=2,
            label="truth",
            color="#4C72B0",
        )
        ax.set_title(label)
        ax.set_xlabel("hits per track")
        ax.set_ylabel("tracks")
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False)

    output_path = Path(output_path)
    fig.suptitle("Fake vs truth track hit-count composition")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _hist2d(x: np.ndarray, y: np.ndarray, xbins: np.ndarray, ybins: np.ndarray) -> np.ndarray:
    hist, _, _ = np.histogram2d(x, y, bins=[xbins, ybins])
    return hist


def plot_fake_truth_pt_eta(sample: TrackCompositionSample, output_path: str | Path) -> Path:
    plt = _prepare_matplotlib()
    features = _track_features(sample)

    fake = sample.fake_mask
    truth = sample.truth_mask

    fake_pt = features["pt"][fake]
    fake_eta = features["eta"][fake]
    truth_pt = features["pt"][truth]
    truth_eta = features["eta"][truth]

    fake_mask = np.isfinite(fake_pt) & np.isfinite(fake_eta) & (fake_pt > 0)
    truth_mask = np.isfinite(truth_pt) & np.isfinite(truth_eta) & (truth_pt > 0)
    fake_pt = fake_pt[fake_mask]
    fake_eta = fake_eta[fake_mask]
    truth_pt = truth_pt[truth_mask]
    truth_eta = truth_eta[truth_mask]

    pt_values = np.concatenate([fake_pt, truth_pt])
    eta_values = np.concatenate([fake_eta, truth_eta])
    if len(pt_values) == 0:
        raise ValueError("no finite pT/eta values to plot")

    pt_min = max(np.min(pt_values[pt_values > 0]), 1e-3)
    pt_max = np.max(pt_values)
    eta_min = np.min(eta_values)
    eta_max = np.max(eta_values)
    pt_bins = np.logspace(np.log10(pt_min), np.log10(pt_max), 45)
    eta_bins = np.linspace(eta_min, eta_max, 45)

    fake_hist = _hist2d(fake_pt, fake_eta, pt_bins, eta_bins)
    truth_hist = _hist2d(truth_pt, truth_eta, pt_bins, eta_bins)
    ratio = np.divide(fake_hist, truth_hist, out=np.full_like(fake_hist, np.nan), where=truth_hist > 0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    _scale_figure(fig)
    panels = [
        (axes[0], fake_hist, "fake", "Reds"),
        (axes[1], truth_hist, "truth", "Blues"),
        (axes[2], ratio, "fake / truth", "viridis"),
    ]
    for ax, hist, title, cmap in panels:
        mesh = ax.pcolormesh(pt_bins, eta_bins, hist.T, shading="auto", cmap=cmap)
        ax.set_xscale("log")
        ax.set_xlabel("pT [MeV]")
        ax.set_ylabel("eta")
        ax.set_title(title)
        fig.colorbar(mesh, ax=ax)

    output_path = Path(output_path)
    fig.suptitle("Fake vs truth pT-eta composition")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_fake_truth_kinematics_1d(sample: TrackCompositionSample, output_path: str | Path) -> Path:
    plt = _prepare_matplotlib()
    features = _track_features(sample)

    fake = sample.fake_mask
    truth = sample.truth_mask

    hist_specs = [
        ("p [MeV]", features["p"], True, 50),
        ("pT [MeV]", features["pt"], True, 50),
        ("eta", features["eta"], False, 50),
        ("phi [rad]", features["phi"], False, 50),
        ("tx", sample.tx, False, 50),
        ("ty", sample.ty, False, 50),
    ]

    fig, axes = plt.subplots(len(hist_specs), 3, figsize=(18, 3.2 * len(hist_specs)), constrained_layout=True)
    _scale_figure(fig)
    for row, (label, values, logx, bins) in enumerate(hist_specs):
        fake_values = values[fake]
        truth_values = values[truth]
        finite_fake = fake_values[np.isfinite(fake_values)]
        finite_truth = truth_values[np.isfinite(truth_values)]

        if logx:
            finite_fake = finite_fake[finite_fake > 0]
            finite_truth = finite_truth[finite_truth > 0]
            if len(finite_fake) or len(finite_truth):
                combined = np.concatenate([finite_fake, finite_truth]) if len(finite_fake) and len(finite_truth) else (
                    finite_fake if len(finite_fake) else finite_truth
                )
                xmin = max(np.min(combined), 1e-6)
                xmax = np.max(combined)
                bins_array = np.logspace(np.log10(xmin), np.log10(xmax), bins)
            else:
                bins_array = bins
        else:
            combined = np.concatenate([finite_fake, finite_truth]) if len(finite_fake) and len(finite_truth) else (
                finite_fake if len(finite_fake) else finite_truth
            )
            if len(combined):
                bins_array = np.linspace(np.min(combined), np.max(combined), bins)
            else:
                bins_array = bins

        hist_fake, edges = np.histogram(finite_fake, bins=bins_array)
        hist_truth, _ = np.histogram(finite_truth, bins=edges)
        total = hist_fake + hist_truth
        fake_rate = np.divide(hist_fake, total, out=np.zeros_like(hist_fake, dtype=float), where=total > 0)
        centers = 0.5 * (edges[:-1] + edges[1:])

        ax_fake, ax_truth, ax_rate = axes[row]
        ax_fake.hist(finite_fake, bins=edges, histtype="step", linewidth=2, label="fake", color="#C44E52")
        ax_fake.set_xlabel(label)
        ax_fake.set_ylabel("tracks")
        if logx:
            ax_fake.set_xscale("log")
        ax_fake.grid(True, alpha=0.2)
        ax_fake.legend(frameon=False)
        ax_fake.set_title("fake")

        ax_truth.hist(finite_truth, bins=edges, histtype="step", linewidth=2, label="truth", color="#4C72B0")
        ax_truth.set_xlabel(label)
        ax_truth.set_ylabel("tracks")
        if logx:
            ax_truth.set_xscale("log")
        ax_truth.grid(True, alpha=0.2)
        ax_truth.legend(frameon=False)
        ax_truth.set_title("truth")

        ax_rate.plot(centers, fake_rate, color="#2F4B7C", linewidth=2)
        ax_rate.set_ylim(0, 1)
        ax_rate.set_xlabel(label)
        ax_rate.set_ylabel("fake / total")
        ax_rate.set_title("fake rate")
        if logx:
            ax_rate.set_xscale("log")
        ax_rate.grid(True, alpha=0.2)

    output_path = Path(output_path)
    fig.suptitle("Fake vs truth kinematics")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_fake_truth_kinematics_2d(sample: TrackCompositionSample, output_path: str | Path) -> Path:
    plt = _prepare_matplotlib()
    features = _track_features(sample)

    fake = sample.fake_mask
    truth = sample.truth_mask

    comparisons = [
        ("pT [MeV]", "eta", features["pt"], features["eta"], True, True),
        ("tx", "ty", sample.tx, sample.ty, False, False),
        ("p [MeV]", "pT [MeV]", features["p"], features["pt"], True, True),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(17, 14), constrained_layout=True)
    _scale_figure(fig)
    for row, (xlabel, ylabel, xvals, yvals, logx, logy) in enumerate(comparisons):
        fake_x = xvals[fake]
        fake_y = yvals[fake]
        truth_x = xvals[truth]
        truth_y = yvals[truth]

        fake_mask = np.isfinite(fake_x) & np.isfinite(fake_y)
        truth_mask = np.isfinite(truth_x) & np.isfinite(truth_y)
        fake_x, fake_y = fake_x[fake_mask], fake_y[fake_mask]
        truth_x, truth_y = truth_x[truth_mask], truth_y[truth_mask]

        if len(fake_x) and len(truth_x):
            combined_x = np.concatenate([fake_x, truth_x])
        else:
            combined_x = fake_x if len(fake_x) else truth_x
        if len(fake_y) and len(truth_y):
            combined_y = np.concatenate([fake_y, truth_y])
        else:
            combined_y = fake_y if len(fake_y) else truth_y
        if len(combined_x) == 0 or len(combined_y) == 0:
            continue

        if logx:
            combined_x = combined_x[combined_x > 0]
        if logy:
            combined_y = combined_y[combined_y > 0]

        if len(combined_x) == 0 or len(combined_y) == 0:
            continue

        if logx:
            xmin = max(np.min(np.concatenate([fake_x[fake_x > 0], truth_x[truth_x > 0]]) if len(fake_x) and len(truth_x) else (fake_x[fake_x > 0] if len(fake_x) else truth_x[truth_x > 0])), 1e-6)
            xmax = np.max(np.concatenate([fake_x[fake_x > 0], truth_x[truth_x > 0]]) if len(fake_x) and len(truth_x) else (fake_x[fake_x > 0] if len(fake_x) else truth_x[truth_x > 0]))
            xbins = np.logspace(np.log10(xmin), np.log10(xmax), 40)
        else:
            xmin, xmax = np.min(combined_x), np.max(combined_x)
            xbins = np.linspace(xmin, xmax, 40)

        if logy:
            ymin = max(np.min(np.concatenate([fake_y[fake_y > 0], truth_y[truth_y > 0]]) if len(fake_y) and len(truth_y) else (fake_y[fake_y > 0] if len(fake_y) else truth_y[truth_y > 0])), 1e-6)
            ymax = np.max(np.concatenate([fake_y[fake_y > 0], truth_y[truth_y > 0]]) if len(fake_y) and len(truth_y) else (fake_y[fake_y > 0] if len(fake_y) else truth_y[truth_y > 0]))
            ybins = np.logspace(np.log10(ymin), np.log10(ymax), 40)
        else:
            ymin, ymax = np.min(combined_y), np.max(combined_y)
            ybins = np.linspace(ymin, ymax, 40)

        fake_hist = _hist2d(fake_x, fake_y, xbins, ybins)
        truth_hist = _hist2d(truth_x, truth_y, xbins, ybins)
        total_hist = fake_hist + truth_hist
        fake_rate = np.divide(fake_hist, total_hist, out=np.full_like(fake_hist, np.nan), where=total_hist > 0)

        for col, (hist, title, cmap) in enumerate(
            [
                (fake_hist, "fake", "Reds"),
                (truth_hist, "truth", "Blues"),
                (fake_rate, "fake rate", "viridis"),
            ]
        ):
            ax = axes[row, col]
            mesh = ax.pcolormesh(xbins, ybins, hist.T, shading="auto", cmap=cmap)
            if logx:
                ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            fig.colorbar(mesh, ax=ax)

    output_path = Path(output_path)
    fig.suptitle("Fake vs truth kinematics in 2D")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_ghost_rate(sample: TrackCompositionSample, output_path: str | Path) -> Path:
    plt = _prepare_matplotlib()
    features = _track_features(sample)
    fake = sample.fake_mask
    truth = sample.truth_mask

    variables = [
        ("p [MeV]", features["p"], True),
        ("pT [MeV]", features["pt"], True),
        ("eta", features["eta"], False),
        ("phi [rad]", features["phi"], False),
        ("tx", sample.tx, False),
        ("ty", sample.ty, False),
        ("chi2/ndof", features["chi2ndof"], False),
    ]

    fig, axes = plt.subplots(len(variables), 3, figsize=(16, 3.2 * len(variables)), constrained_layout=True)
    _scale_figure(fig)

    for row, (label, values, logx) in enumerate(variables):
        fake_values = values[fake]
        truth_values = values[truth]
        finite_fake = fake_values[np.isfinite(fake_values)]
        finite_truth = truth_values[np.isfinite(truth_values)]

        if logx:
            finite_fake = finite_fake[finite_fake > 0]
            finite_truth = finite_truth[finite_truth > 0]
            if len(finite_fake) or len(finite_truth):
                combined = np.concatenate([finite_fake, finite_truth]) if len(finite_fake) and len(finite_truth) else (
                    finite_fake if len(finite_fake) else finite_truth
                )
                xmin = max(np.min(combined), 1e-6)
                xmax = np.max(combined)
                bins = np.logspace(np.log10(xmin), np.log10(xmax), 50)
            else:
                bins = 50
        else:
            combined = np.concatenate([finite_fake, finite_truth]) if len(finite_fake) and len(finite_truth) else (
                finite_fake if len(finite_fake) else finite_truth
            )
            if len(combined):
                bins = np.linspace(np.min(combined), np.max(combined), 50)
            else:
                bins = 50

        hist_fake, edges = np.histogram(finite_fake, bins=bins)
        hist_truth, _ = np.histogram(finite_truth, bins=edges)
        total = hist_fake + hist_truth
        ghost_rate = np.divide(hist_fake, total, out=np.zeros_like(hist_fake, dtype=float), where=total > 0)
        centers = 0.5 * (edges[:-1] + edges[1:])

        ax_total, ax_fake_only, ax_rate = axes[row]
        ax_total.hist(
            finite_fake,
            bins=edges,
            histtype="stepfilled",
            alpha=0.35,
            color="#C44E52",
            label="fake",
        )
        ax_total.hist(
            finite_truth,
            bins=edges,
            histtype="stepfilled",
            alpha=0.35,
            color="#4C72B0",
            label="truth",
        )
        ax_total.set_ylabel("tracks")
        ax_total.set_title(f"{label}: total = fake + truth")
        ax_total.grid(True, alpha=0.2)
        if logx:
            ax_total.set_xscale("log")
        ax_total.legend(frameon=False)

        ax_fake_only.hist(finite_fake, bins=edges, histtype="step", linewidth=2, color="#C44E52", label="fake")
        ax_fake_only.hist(finite_truth, bins=edges, histtype="step", linewidth=2, color="#4C72B0", label="truth")
        ax_fake_only.set_ylabel("tracks")
        ax_fake_only.set_title(f"{label}: fake and truth")
        ax_fake_only.grid(True, alpha=0.2)
        if logx:
            ax_fake_only.set_xscale("log")
        ax_fake_only.legend(frameon=False)

        ax_rate.plot(centers, ghost_rate, color="#2F4B7C", linewidth=2)
        ax_rate.set_ylim(0, 1)
        ax_rate.set_ylabel("Fake / Total")
        ax_rate.set_xlabel(label)
        ax_rate.set_title(f"Ghost rate = Fake / (Fake + Truth)")
        ax_rate.grid(True, alpha=0.2)
        if logx:
            ax_rate.set_xscale("log")

    output_path = Path(output_path)
    fig.suptitle("Ghost rate by track property")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_event_fake_rate_vs_pvs(sample: EventCompositionSample | EventFakePvSample, output_path: str | Path) -> Path:
    plt = _prepare_matplotlib()

    pv = sample.pv_count
    fake = sample.fake_count
    fake_rate = sample.fake_rate
    valid_rate_mask = np.isfinite(fake_rate)

    pv_bins = np.arange(0, int(max(pv.max(), 1)) + 2) - 0.5
    rate_bins = np.linspace(0, 1, 41)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), constrained_layout=True)
    _scale_figure(fig)

    axes[0].scatter(pv, fake, alpha=0.7, color="#C44E52", s=12)
    axes[0].set_xlabel("# PVs")
    axes[0].set_ylabel("# fake tracks")
    axes[0].set_title("Fakes vs PVs")
    axes[0].grid(True, alpha=0.2)

    hist = axes[1].hist2d(pv[valid_rate_mask], fake_rate[valid_rate_mask], bins=[pv_bins, rate_bins], cmap="viridis")
    axes[1].set_xlabel("# PVs")
    axes[1].set_ylabel("fake / total")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Event fake rate vs PVs")
    fig.colorbar(hist[3], ax=axes[1], label="events")

    pv_values = np.asarray(pv[valid_rate_mask], dtype=int)
    rate_values = np.asarray(fake_rate[valid_rate_mask], dtype=float)
    unique_pv = np.unique(pv_values)
    profile_x: list[float] = []
    profile_mean: list[float] = []
    profile_std: list[float] = []
    for pv_value in unique_pv:
        rate_subset = rate_values[pv_values == pv_value]
        if len(rate_subset) == 0:
            continue
        profile_x.append(float(pv_value))
        profile_mean.append(float(np.mean(rate_subset)))
        profile_std.append(float(np.std(rate_subset, ddof=0)))

    if profile_x:
        fig2, (ax2, ax2_hist) = plt.subplots(
            2,
            1,
            figsize=(7.5, 8.0),
            constrained_layout=True,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1.2]},
        )
        _scale_figure(fig2)
        profile_x_arr = np.asarray(profile_x, dtype=float)
        profile_mean_arr = np.asarray(profile_mean, dtype=float)
        profile_std_arr = np.asarray(profile_std, dtype=float)
        ax2.errorbar(
            profile_x_arr,
            profile_mean_arr,
            yerr=profile_std_arr,
            fmt="o",
            color="#2F4B7C",
            ecolor="#2F4B7C",
            elinewidth=1.5,
            capsize=3,
            label="mean ± stddev",
        )
        ax2.scatter(
            pv[valid_rate_mask],
            fake_rate[valid_rate_mask],
            alpha=0.18,
            color="#BFBFBF",
            s=10,
            label="events",
        )
        ax2.set_xlabel("# PVs")
        ax2.set_ylabel("fake / total")
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.2)
        ax2.set_title("Profile of event fake rate vs PVs")
        ax2.legend(frameon=False)
        ax2.text(
            0.98,
            0.02,
            "bin stats: mean and stddev",
            transform=ax2.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="0.35",
        )
        pv_hist, _ = np.histogram(pv[valid_rate_mask], bins=pv_bins, density=True)
        pv_centers = 0.5 * (pv_bins[:-1] + pv_bins[1:])
        ax2_hist.bar(
            pv_centers,
            pv_hist,
            width=np.diff(pv_bins),
            align="center",
            color="#4C72B0",
            alpha=0.45,
            edgecolor="#4C72B0",
            linewidth=1.0,
        )
        ax2_hist.set_xlabel("# PVs")
        ax2_hist.set_ylabel("norm. events")
        ax2_hist.set_title("Underlying PV distribution")
        ax2_hist.grid(True, alpha=0.2)
        fig2_path = Path(output_path).with_name(f"{Path(output_path).stem}_profile{Path(output_path).suffix}")
        fig2.savefig(fig2_path, dpi=150)
        plt.close(fig2)

    output_path = Path(output_path)
    fig.suptitle("Per-event ghost rate and PV dependence")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_event_tracks_vs_pvs(sample: EventCompositionSample | EventFakePvSample, output_path: str | Path) -> Path:
    plt = _prepare_matplotlib()

    pv = sample.pv_count
    ntracks = sample.fake_count + sample.truth_count

    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    _scale_figure(fig)
    ax.scatter(pv, ntracks, alpha=0.75, color="#2F4B7C", edgecolors="none", s=12)
    ax.set_xlabel("# PVs")
    ax.set_ylabel("# tracks found")
    ax.set_title("Tracks found per event vs PVs")
    ax.grid(True, alpha=0.2)

    output_path = Path(output_path)
    fig.suptitle("Per-event track multiplicity and PV dependence")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_fake_truth_segment_slopes(sample: TrackCompositionSample, output_path: str | Path) -> Path:
    plt = _prepare_matplotlib()

    fake = sample.fake_mask
    truth = sample.truth_mask
    ft_segment = sample.ft_truth_hit_mask
    mp_segment = sample.mp_truth_hit_mask

    panels = [
        ("FT truth-matched hits: tx", sample.tx, ft_segment),
        ("FT truth-matched hits: ty", sample.ty, ft_segment),
        ("MP truth-matched hits: tx", sample.tx, mp_segment),
        ("MP truth-matched hits: ty", sample.ty, mp_segment),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    _scale_figure(fig)
    for ax, (title, values, segment_mask) in zip(axes.flat, panels):
        fake_values = values[fake & segment_mask]
        truth_values = values[truth & segment_mask]
        finite_fake = fake_values[np.isfinite(fake_values)]
        finite_truth = truth_values[np.isfinite(truth_values)]
        combined = np.concatenate([finite_fake, finite_truth]) if len(finite_fake) and len(finite_truth) else (
            finite_fake if len(finite_fake) else finite_truth
        )
        if len(combined) == 0:
            ax.set_title(title)
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue
        bins = np.linspace(np.min(combined), np.max(combined), 50)
        ax.hist(finite_fake, bins=bins, histtype="step", linewidth=2, label="fake", color="#C44E52")
        ax.hist(finite_truth, bins=bins, histtype="step", linewidth=2, label="truth", color="#4C72B0")
        ax.set_title(title)
        ax.set_xlabel("slope")
        ax.set_ylabel("tracks")
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False)

    output_path = Path(output_path)
    fig.suptitle("Fake vs truth slopes, split by truth-matched FT/MP hits")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
