from __future__ import annotations

import os
import math
from pathlib import Path

from .ntuple_models import EventRecord, Track


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


def track_hit_points(track: Track, detector: str | None = None) -> list[tuple[float, float, float, float]]:
    detectors = (detector,) if detector is not None else tuple(track.hits.keys())
    points: list[tuple[float, float, float, float]] = []
    for det in detectors:
        for hit in track.hits.get(det, []):
            points.append((hit.x, hit.y, hit.z, hit.t))
    return points


def describe_track(track: Track) -> str:
    kin = track_kinematics(track)
    return (
        f"track={track.index} truth={track.mc_truth} pid={track.mc_pid} "
        f"tx={kin['tx']:.4g} ty={kin['ty']:.4g} qop={kin['qop']:.4g} "
        f"p={kin['p']:.3g}MeV pt={kin['pt']:.3g}MeV eta={kin['eta']:.3g} phi={kin['phi']:.3g}"
    )


def plot_track_hits_3d(track: Track, output_path: str | Path, detector: str | None = None) -> Path:
    try:
        mplconfigdir = Path("/private/tmp/mplconfig")
        mplconfigdir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mplconfigdir))
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for 3D plotting") from exc

    points = track_hit_points(track, detector=detector)
    if not points:
        raise ValueError("track has no hits to plot")

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    zs = [point[2] for point in points]
    ts = [point[3] for point in points]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(xs, ys, zs, c=ts, cmap="viridis", s=18)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    title = f"Track {track.index} hits"
    if detector is not None:
        title += f" ({detector})"
    ax.set_title(title)
    fig.colorbar(scatter, ax=ax, label="t [ns]")
    fig.tight_layout()

    output_path = Path(output_path)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def summarize_fake_tracks(event: EventRecord, limit: int = 10) -> list[str]:
    lines: list[str] = []
    for track in fake_tracks(event)[:limit]:
        lines.append(describe_track(track))
    return lines
