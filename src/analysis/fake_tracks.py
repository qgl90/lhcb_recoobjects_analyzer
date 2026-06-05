from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

from .models import EventRecord, Track


def track_kinematics(track: Track) -> dict[str, float]:
    tx = track.state.tx
    ty = track.state.ty
    slope2 = tx * tx + ty * ty
    direction_norm = math.sqrt(1.0 + slope2)
    qop = track.state.qop
    p = float("inf") if qop == 0 else abs(1.0 / qop)
    pt = 0.0 if not math.isfinite(p) else p * math.sqrt(slope2) / direction_norm
    pz = 0.0 if not math.isfinite(p) else p / direction_norm
    eta = float("inf") if pt == 0.0 else math.asinh(pz / pt)
    phi = math.atan2(ty, tx)
    return {
        "p": p,
        "pt": pt,
        "pz": pz,
        "eta": eta,
        "phi": phi,
        "tx": tx,
        "ty": ty,
        "qop": qop,
    }


def fake_tracks(event: EventRecord) -> list[Track]:
    return [track for track in event.tracks if track.mc_truth == 0]


def fake_tracks_in_events(events: list[EventRecord]) -> list[Track]:
    tracks: list[Track] = []
    for event in events:
        tracks.extend(fake_tracks(event))
    return tracks


def track_hit_points(track: Track, detector: str | None = None) -> list[tuple[float, float, float, float]]:
    detectors = (detector,) if detector is not None else tuple(track.hits.keys())
    points: list[tuple[float, float, float, float]] = []
    for det in detectors:
        for hit in track.hits.get(det, []):
            points.append((hit.x, hit.y, hit.z, hit.t))
    return points


def _fit_line_1d(z_values: list[float], values: list[float]) -> tuple[float, float]:
    if len(z_values) < 2:
        return float("nan"), float("nan")
    z = np.asarray(z_values, dtype=float)
    v = np.asarray(values, dtype=float)
    mask = np.isfinite(z) & np.isfinite(v)
    if mask.sum() < 2:
        return float("nan"), float("nan")
    z = z[mask]
    v = v[mask]
    slope, intercept = np.polyfit(z, v, deg=1)
    return float(intercept), float(slope)


def fit_track_segment_slopes(track: Track) -> dict[str, dict[str, float]]:
    segments = {
        "velo": ("TV", "UP"),
        "ft": ("FT",),
        "mp": ("MP",),
    }
    fits: dict[str, dict[str, float]] = {}
    for segment_name, detectors in segments.items():
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        hit_count = 0
        for detector in detectors:
            for hit in track.hits.get(detector, []):
                xs.append(hit.x)
                ys.append(hit.y)
                zs.append(hit.z)
                hit_count += 1
        x0, tx = _fit_line_1d(zs, xs)
        y0, ty = _fit_line_1d(zs, ys)
        fits[segment_name] = {
            "x0": x0,
            "tx": tx,
            "y0": y0,
            "ty": ty,
            "n_hits": float(hit_count),
        }
    return fits


def describe_track(track: Track) -> str:
    kin = track_kinematics(track)
    segment_fits = fit_track_segment_slopes(track)
    return (
        f"track={track.index} truth={track.mc_truth} pid={track.mc_pid} "
        f"tx={kin['tx']:.4g} ty={kin['ty']:.4g} qop={kin['qop']:.4g} "
        f"p={kin['p']:.3g}MeV pt={kin['pt']:.3g}MeV eta={kin['eta']:.3g} phi={kin['phi']:.3g} "
        f"velo(x0,tx,y0,ty)=({segment_fits['velo']['x0']:.4g},{segment_fits['velo']['tx']:.4g},"
        f"{segment_fits['velo']['y0']:.4g},{segment_fits['velo']['ty']:.4g}) "
        f"ft(x0,tx,y0,ty)=({segment_fits['ft']['x0']:.4g},{segment_fits['ft']['tx']:.4g},"
        f"{segment_fits['ft']['y0']:.4g},{segment_fits['ft']['ty']:.4g}) "
        f"mp(x0,tx,y0,ty)=({segment_fits['mp']['x0']:.4g},{segment_fits['mp']['tx']:.4g},"
        f"{segment_fits['mp']['y0']:.4g},{segment_fits['mp']['ty']:.4g})"
    )


def summarize_fake_tracks(event: EventRecord, limit: int = 10) -> list[str]:
    return [describe_track(track) for track in fake_tracks(event)[:limit]]


def fake_track_kinematics(event: EventRecord) -> list[dict[str, float]]:
    return [track_kinematics(track) for track in fake_tracks(event)]


def fake_track_kinematics_in_events(events: list[EventRecord]) -> list[dict[str, float]]:
    return [track_kinematics(track) for track in fake_tracks_in_events(events)]


def _prepare_matplotlib():
    try:
        mplconfigdir = Path("/private/tmp/mplconfig")
        mplconfigdir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mplconfigdir))
        import mplhep as hep
        import matplotlib

        matplotlib.use("Agg", force=True)
        hep.style.use("LHCb2")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting") from exc
    return plt


def _scale_figure(fig, scale: float = 1.3) -> None:
    width, height = fig.get_size_inches()
    fig.set_size_inches(width * scale, height * scale, forward=True)


def plot_fake_track_distributions(event_or_events: EventRecord | list[EventRecord], output_path: str | Path) -> Path:
    plt = _prepare_matplotlib()
    if isinstance(event_or_events, list):
        kin = fake_track_kinematics_in_events(event_or_events)
    else:
        kin = fake_track_kinematics(event_or_events)
    if not kin:
        raise ValueError("event has no fake tracks")

    p = [row["p"] for row in kin]
    pt = [row["pt"] for row in kin]
    eta = [row["eta"] for row in kin]
    phi = [row["phi"] for row in kin]
    tx = [row["tx"] for row in kin]
    ty = [row["ty"] for row in kin]

    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    _scale_figure(fig)
    grid = fig.add_gridspec(3, 2)

    hist_specs = [
        (grid[0, 0], p, "p [MeV]", 40, True),
        (grid[0, 1], pt, "pT [MeV]", 40, True),
        (grid[1, 0], eta, "eta", 40, False),
        (grid[1, 1], phi, "phi [rad]", 40, False),
        (grid[2, 0], tx, "tx", 40, False),
        (grid[2, 1], ty, "ty", 40, False),
    ]

    for position, values, label, bins, logx in hist_specs:
        ax = fig.add_subplot(position)
        finite_values = [value for value in values if math.isfinite(value)]
        ax.hist(finite_values, bins=bins, histtype="stepfilled", alpha=0.75, color="#4C72B0")
        ax.set_xlabel(label)
        ax.set_ylabel("tracks")
        if logx and finite_values:
            ax.set_xscale("log")
        ax.grid(True, alpha=0.2)

    output_path = Path(output_path)
    fig.suptitle("Fake-track distributions")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_fake_track_pt_eta(event_or_events: EventRecord | list[EventRecord], output_path: str | Path) -> Path:
    plt = _prepare_matplotlib()
    if isinstance(event_or_events, list):
        kin = fake_track_kinematics_in_events(event_or_events)
    else:
        kin = fake_track_kinematics(event_or_events)
    if not kin:
        raise ValueError("event has no fake tracks")

    pt = [row["pt"] for row in kin if math.isfinite(row["pt"]) and math.isfinite(row["eta"])]
    eta = [row["eta"] for row in kin if math.isfinite(row["pt"]) and math.isfinite(row["eta"])]
    if not pt:
        raise ValueError("no finite fake-track pt/eta values to plot")

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    _scale_figure(fig)
    hist = ax.hist2d(pt, eta, bins=[40, 40], cmap="viridis")
    ax.set_xlabel("pT [MeV]")
    ax.set_ylabel("eta")
    ax.set_xscale("log")
    ax.set_title("Fake-track pT vs eta")
    ax.grid(True, alpha=0.2)
    fig.colorbar(hist[3], ax=ax, label="tracks")

    output_path = Path(output_path)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _segment_fit_arrays(event_or_events: EventRecord | list[EventRecord]) -> dict[str, dict[str, list[float]]]:
    if isinstance(event_or_events, list):
        tracks = [track for event in event_or_events for track in event.tracks if track.mc_truth == 0]
    else:
        tracks = fake_tracks(event_or_events)

    arrays: dict[str, dict[str, list[float]]] = {
        "velo": {"tx": [], "ty": []},
        "ft": {"tx": [], "ty": []},
        "mp": {"tx": [], "ty": []},
    }
    for track in tracks:
        fits = fit_track_segment_slopes(track)
        for segment in arrays:
            tx = fits[segment]["tx"]
            ty = fits[segment]["ty"]
            if math.isfinite(tx):
                arrays[segment]["tx"].append(tx)
            if math.isfinite(ty):
                arrays[segment]["ty"].append(ty)
    return arrays


def _fake_track_segment_records(events: list[EventRecord]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for event in events:
        for track in fake_tracks(event):
            fits = fit_track_segment_slopes(track)
            kin = track_kinematics(track)
            for segment_name, fit in fits.items():
                records.append(
                    {
                        "event_number": event.event_number,
                        "track_index": track.index,
                        "segment": segment_name,
                        "x0": fit["x0"],
                        "y0": fit["y0"],
                        "tx": fit["tx"],
                        "ty": fit["ty"],
                        "n_hits": fit["n_hits"],
                        "qop": track.state.qop,
                        "pt": kin["pt"],
                        "eta": kin["eta"],
                        "phi": kin["phi"],
                    }
                )
    return records


def plot_fake_track_slope_directions(events: list[EventRecord], output_path: str | Path) -> Path:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("plotly is required for the interactive slope plot") from exc

    records = _fake_track_segment_records(events)
    if not records:
        raise ValueError("no fake tracks available for slope plotting")

    segment_style = {
        "velo": {"name": "VELO", "color": "#4C72B0"},
        "ft": {"name": "FT", "color": "#55A868"},
        "mp": {"name": "MP", "color": "#C44E52"},
    }

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Slope direction vectors in (tx, ty)", "tx vs ty"),
        horizontal_spacing=0.12,
    )

    for segment, style in segment_style.items():
        segment_records = [record for record in records if record["segment"] == segment]
        if not segment_records:
            continue

        tx_values = np.asarray([record["tx"] for record in segment_records], dtype=float)
        ty_values = np.asarray([record["ty"] for record in segment_records], dtype=float)
        x0_values = np.asarray([record["x0"] for record in segment_records], dtype=float)
        y0_values = np.asarray([record["y0"] for record in segment_records], dtype=float)
        hover_text = [
            (
                f"event={record['event_number']}<br>"
                f"track={record['track_index']}<br>"
                f"segment={record['segment']}<br>"
                f"x0={record['x0']:.3g}, y0={record['y0']:.3g}<br>"
                f"tx={record['tx']:.3g}, ty={record['ty']:.3g}<br>"
                f"n_hits={int(record['n_hits'])}<br>"
                f"pt={record['pt']:.3g}, eta={record['eta']:.3g}, phi={record['phi']:.3g}"
            )
            for record in segment_records
        ]

        line_x: list[float | None] = []
        line_y: list[float | None] = []
        for record in segment_records:
            line_x.extend([0.0, float(record["tx"]), None])
            line_y.extend([0.0, float(record["ty"]), None])

        fig.add_trace(
            go.Scatter(
                x=tx_values,
                y=ty_values,
                mode="markers",
                name=style["name"],
                marker=dict(color=style["color"], size=8, opacity=0.75),
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                line=dict(color=style["color"], width=1),
                opacity=0.22,
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=x0_values,
                y=y0_values,
                mode="markers",
                name=f"{style['name']} intercepts",
                marker=dict(color=style["color"], size=7, symbol="x", opacity=0.65),
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig.update_xaxes(title_text="tx", row=1, col=1)
    fig.update_yaxes(title_text="ty", row=1, col=1)
    fig.update_xaxes(title_text="x0 [mm]", row=1, col=2)
    fig.update_yaxes(title_text="y0 [mm]", row=1, col=2)
    fig.update_layout(
        title="Interactive fake-track slope directions",
        width=1400,
        height=650,
        legend_title_text="segment",
    )

    output_path = Path(output_path)
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def plot_fake_track_segment_slopes(event_or_events: EventRecord | list[EventRecord], output_path: str | Path) -> Path:
    plt = _prepare_matplotlib()
    slope_arrays = _segment_fit_arrays(event_or_events)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), constrained_layout=True)
    _scale_figure(fig)
    panels = [
        ("velo", "tx", axes[0, 0]),
        ("velo", "ty", axes[0, 1]),
        ("ft", "tx", axes[1, 0]),
        ("ft", "ty", axes[1, 1]),
        ("mp", "tx", axes[2, 0]),
        ("mp", "ty", axes[2, 1]),
    ]
    colors = {"velo": "#4C72B0", "ft": "#55A868", "mp": "#C44E52"}
    titles = {"tx": "slope x(z)", "ty": "slope y(z)"}
    for segment, component, ax in panels:
        values = slope_arrays[segment][component]
        ax.hist(values, bins=40, histtype="stepfilled", alpha=0.75, color=colors[segment])
        ax.set_title(f"{segment.upper()} {titles[component]}")
        ax.set_xlabel(component)
        ax.set_ylabel("tracks")
        ax.grid(True, alpha=0.2)

    output_path = Path(output_path)
    fig.suptitle("Fake-track segment slopes from straight-line fits")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_track_hits_3d(track: Track, output_path: str | Path, detector: str | None = None) -> Path:
    plt = _prepare_matplotlib()

    points = track_hit_points(track, detector=detector)
    if not points:
        raise ValueError("track has no hits to plot")

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    zs = [point[2] for point in points]
    ts = [point[3] for point in points]

    def padded_limits(values: list[float], fraction: float = 0.08, minimum: float = 1.0) -> tuple[float, float]:
        low = min(values)
        high = max(values)
        span = high - low
        pad = max(span * fraction, minimum)
        return low - pad, high + pad

    fig = plt.figure(figsize=(15, 5), constrained_layout=True)
    _scale_figure(fig)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1])

    ax3d = fig.add_subplot(grid[0, 0], projection="3d")
    scatter_3d = ax3d.scatter(xs, ys, zs, c=ts, cmap="viridis", s=10)
    ax3d.set_xlabel("x [mm]")
    ax3d.set_ylabel("y [mm]")
    ax3d.set_zlabel("z [mm]")
    ax3d.set_xlim(*padded_limits(xs))
    ax3d.set_ylim(*padded_limits(ys))
    ax3d.set_zlim(*padded_limits(zs))
    title = f"Track {track.index} hits"
    if detector is not None:
        title += f" ({detector})"
    ax3d.set_title(title)

    ax_xz = fig.add_subplot(grid[0, 1])
    scatter_xz = ax_xz.scatter(zs, xs, c=ts, cmap="viridis", s=10)
    ax_xz.set_xlabel("z [mm]")
    ax_xz.set_ylabel("x [mm]")
    ax_xz.set_title("xz projection")
    ax_xz.grid(True, alpha=0.2)
    ax_xz.set_xlim(*padded_limits(zs))
    ax_xz.set_ylim(*padded_limits(xs))

    ax_yz = fig.add_subplot(grid[0, 2])
    scatter_yz = ax_yz.scatter(zs, ys, c=ts, cmap="viridis", s=10)
    ax_yz.set_xlabel("z [mm]")
    ax_yz.set_ylabel("y [mm]")
    ax_yz.set_title("yz projection")
    ax_yz.grid(True, alpha=0.2)
    ax_yz.set_xlim(*padded_limits(zs))
    ax_yz.set_ylim(*padded_limits(ys))

    fig.colorbar(scatter_3d, ax=[ax3d, ax_xz, ax_yz], label="t [ns]")

    output_path = Path(output_path)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
