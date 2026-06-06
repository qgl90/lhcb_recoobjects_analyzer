from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

import awkward as ak
import numpy as np

from .root_loader import DETECTORS, _tree_from_file

DEFAULT_FEATURE_GROUPS = ("event", "state", "kinematics", "hits", "hit_truth", "segments", "mc")
FEATURE_GROUPS = frozenset(DEFAULT_FEATURE_GROUPS)

STATE_BRANCHES = (
    "Track_chi2ndof",
    "Track_ndof",
    "FirstMeasurement_x",
    "FirstMeasurement_y",
    "FirstMeasurement_z",
    "FirstMeasurement_tx",
    "FirstMeasurement_ty",
    "FirstMeasurement_qop",
    "FirstMeasurement_cov_0_0",
    "FirstMeasurement_cov_1_1",
    "FirstMeasurement_cov_2_2",
    "FirstMeasurement_cov_3_3",
    "FirstMeasurement_cov_4_4",
)
MC_BRANCHES = (
    "MC_hasTV",
    "MC_hasUP",
    "MC_hasMP",
    "MC_hasFT",
    "MC_fromSignal",
    "MC_pid",
    "MC_key",
    "MC_pv_key",
    "MC_charge",
    "MC_px",
    "MC_py",
    "MC_pz",
    "MC_pe",
)
EVENT_BRANCHES = ("EventNumber", "RunNumber", "BunchCrossingID", "PV_x")


def parse_feature_groups(value: str | Sequence[str] | None) -> tuple[str, ...]:
    """Parse comma-separated feature-group names for the training-sample builder."""
    if value is None:
        return DEFAULT_FEATURE_GROUPS
    if isinstance(value, str):
        requested = tuple(group.strip() for group in value.split(",") if group.strip())
    else:
        requested = tuple(value)
    unknown = sorted(set(requested) - FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"unknown feature group(s): {', '.join(unknown)}; choose from {', '.join(sorted(FEATURE_GROUPS))}")
    return requested


def training_branch_names(feature_groups: Sequence[str]) -> list[str]:
    """Return the minimal ROOT branches needed for selected feature groups."""
    groups = set(feature_groups)
    branches: list[str] = ["MC_truth"]
    if "event" in groups:
        branches.extend(EVENT_BRANCHES)
    if groups & {"state", "kinematics", "segments"}:
        branches.extend(STATE_BRANCHES)
    if "mc" in groups:
        branches.extend(MC_BRANCHES)
    if groups & {"hits", "hit_truth", "segments"}:
        for detector in DETECTORS:
            branches.append(f"{detector}Hits_n")
    if groups & {"hit_truth", "segments"}:
        for detector in DETECTORS:
            branches.extend(
                [
                    f"{detector}Hits_x",
                    f"{detector}Hits_y",
                    f"{detector}Hits_z",
                    f"{detector}Hits_mc_x",
                    f"{detector}Hits_mc_y",
                    f"{detector}Hits_mc_z",
                ]
            )
    # Preserve order while dropping duplicates.
    return list(dict.fromkeys(branches))


def _append_base_columns(columns: dict[str, list[float | int]], arrays: ak.Array, event_index: int, track_index: int) -> None:
    mc_truth = int(arrays["MC_truth"][event_index][track_index])
    columns["label"].append(1 if mc_truth == 1 else 0)
    columns["mc_truth"].append(mc_truth)
    columns["track_index"].append(track_index)


def _append_event_columns(columns: dict[str, list[float | int]], arrays: ak.Array, event_index: int, n_tracks: int) -> None:
    columns["event_number"].append(int(arrays["EventNumber"][event_index]) if "EventNumber" in arrays.fields else event_index)
    columns["run_number"].append(int(arrays["RunNumber"][event_index]) if "RunNumber" in arrays.fields else -1)
    columns["bunch_crossing_id"].append(
        int(arrays["BunchCrossingID"][event_index]) if "BunchCrossingID" in arrays.fields else -1
    )
    columns["n_tracks_event"].append(n_tracks)
    columns["n_pvs"].append(len(arrays["PV_x"][event_index]) if "PV_x" in arrays.fields else -1)


def _append_state_columns(columns: dict[str, list[float | int]], arrays: ak.Array, event_index: int, track_index: int) -> None:
    state_names = {
        "chi2ndof": "Track_chi2ndof",
        "ndof": "Track_ndof",
        "x": "FirstMeasurement_x",
        "y": "FirstMeasurement_y",
        "z": "FirstMeasurement_z",
        "tx": "FirstMeasurement_tx",
        "ty": "FirstMeasurement_ty",
        "qop": "FirstMeasurement_qop",
        "cov_xx": "FirstMeasurement_cov_0_0",
        "cov_yy": "FirstMeasurement_cov_1_1",
        "cov_tx_tx": "FirstMeasurement_cov_2_2",
        "cov_ty_ty": "FirstMeasurement_cov_3_3",
        "cov_qop_qop": "FirstMeasurement_cov_4_4",
    }
    for output_name, branch in state_names.items():
        columns[output_name].append(float(arrays[branch][event_index][track_index]) if branch in arrays.fields else float("nan"))


def _append_kinematic_columns(columns: dict[str, list[float | int]], arrays: ak.Array, event_index: int, track_index: int) -> None:
    tx = float(arrays["FirstMeasurement_tx"][event_index][track_index])
    ty = float(arrays["FirstMeasurement_ty"][event_index][track_index])
    qop = float(arrays["FirstMeasurement_qop"][event_index][track_index])
    slope2 = tx * tx + ty * ty
    direction_norm = math.sqrt(1.0 + slope2)
    p = float("inf") if qop == 0.0 else abs(1.0 / qop)
    pt = float("nan") if not math.isfinite(p) else p * math.sqrt(slope2) / direction_norm
    pz = float("nan") if not math.isfinite(p) else p / direction_norm
    eta = float("nan") if pt == 0.0 or not math.isfinite(pt) else math.asinh(pz / pt)
    phi = math.atan2(ty, tx)
    columns["slope2"].append(slope2)
    columns["p"].append(p)
    columns["pt"].append(pt)
    columns["pz"].append(pz)
    columns["eta"].append(eta)
    columns["phi"].append(phi)


def _append_mc_columns(columns: dict[str, list[float | int]], arrays: ak.Array, event_index: int, track_index: int) -> None:
    for branch in MC_BRANCHES:
        output_name = branch.removeprefix("MC_").lower()
        value = arrays[branch][event_index][track_index] if branch in arrays.fields else -1
        if branch in {"MC_charge", "MC_px", "MC_py", "MC_pz", "MC_pe"}:
            columns[f"mc_{output_name}"].append(float(value))
        else:
            columns[f"mc_{output_name}"].append(int(value))


def _fit_line_1d(z_values: Iterable[float], values: Iterable[float]) -> tuple[float, float]:
    z = np.asarray(list(z_values), dtype=float)
    v = np.asarray(list(values), dtype=float)
    mask = np.isfinite(z) & np.isfinite(v)
    if int(mask.sum()) < 2:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(z[mask], v[mask], deg=1)
    return float(intercept), float(slope)


def _group_hits(arrays: ak.Array, event_index: int, detector: str) -> dict[str, ak.Array]:
    counts = arrays[f"{detector}Hits_n"][event_index]
    return {
        "counts": counts,
        "x": ak.unflatten(arrays[f"{detector}Hits_x"][event_index], counts),
        "y": ak.unflatten(arrays[f"{detector}Hits_y"][event_index], counts),
        "z": ak.unflatten(arrays[f"{detector}Hits_z"][event_index], counts),
        "mc_x": ak.unflatten(arrays[f"{detector}Hits_mc_x"][event_index], counts),
        "mc_y": ak.unflatten(arrays[f"{detector}Hits_mc_y"][event_index], counts),
        "mc_z": ak.unflatten(arrays[f"{detector}Hits_mc_z"][event_index], counts),
    }


def _append_hit_count_columns(
    columns: dict[str, list[float | int]], arrays: ak.Array, event_index: int, track_index: int
) -> None:
    total_hits = 0
    pattern = 0
    for bit, detector in enumerate(DETECTORS):
        branch = f"{detector}Hits_n"
        count = int(arrays[branch][event_index][track_index]) if branch in arrays.fields else 0
        total_hits += count
        if count > 0:
            pattern |= 1 << bit
        prefix = detector.lower()
        columns[f"n_hits_{prefix}"].append(count)
        columns[f"has_{prefix}"].append(int(count > 0))
    columns["n_hits_total"].append(total_hits)
    columns["detector_pattern"].append(pattern)


def _append_hit_truth_columns(
    columns: dict[str, list[float | int]], grouped_hits: dict[str, dict[str, ak.Array]], track_index: int
) -> None:
    total_hits = 0
    total_truth_hits = 0
    for detector in DETECTORS:
        prefix = detector.lower()
        hits = grouped_hits[detector]
        x = np.asarray(ak.to_numpy(hits["x"][track_index]), dtype=float)
        y = np.asarray(ak.to_numpy(hits["y"][track_index]), dtype=float)
        z = np.asarray(ak.to_numpy(hits["z"][track_index]), dtype=float)
        mc_x = np.asarray(ak.to_numpy(hits["mc_x"][track_index]), dtype=float)
        mc_y = np.asarray(ak.to_numpy(hits["mc_y"][track_index]), dtype=float)
        mc_z = np.asarray(ak.to_numpy(hits["mc_z"][track_index]), dtype=float)
        truth_mask = np.isfinite(mc_x) & np.isfinite(mc_y) & np.isfinite(mc_z)
        n_hits = int(x.size)
        n_truth_hits = int(truth_mask.sum())
        total_hits += n_hits
        total_truth_hits += n_truth_hits
        columns[f"n_truth_hits_{prefix}"].append(n_truth_hits)
        columns[f"truth_hit_fraction_{prefix}"].append(n_truth_hits / n_hits if n_hits else float("nan"))
        if n_truth_hits:
            dx = x[truth_mask] - mc_x[truth_mask]
            dy = y[truth_mask] - mc_y[truth_mask]
            dz = z[truth_mask] - mc_z[truth_mask]
            dr = np.sqrt(dx * dx + dy * dy + dz * dz)
            columns[f"hit_residual_r_mean_{prefix}"].append(float(np.mean(dr)))
            columns[f"hit_residual_r_std_{prefix}"].append(float(np.std(dr)))
            columns[f"hit_residual_x_mean_{prefix}"].append(float(np.mean(dx)))
            columns[f"hit_residual_y_mean_{prefix}"].append(float(np.mean(dy)))
        else:
            columns[f"hit_residual_r_mean_{prefix}"].append(float("nan"))
            columns[f"hit_residual_r_std_{prefix}"].append(float("nan"))
            columns[f"hit_residual_x_mean_{prefix}"].append(float("nan"))
            columns[f"hit_residual_y_mean_{prefix}"].append(float("nan"))
    columns["n_truth_hits_total"].append(total_truth_hits)
    columns["truth_hit_fraction_total"].append(total_truth_hits / total_hits if total_hits else float("nan"))


def _append_segment_columns(
    columns: dict[str, list[float | int]], arrays: ak.Array, event_index: int, grouped_hits: dict[str, dict[str, ak.Array]], track_index: int
) -> None:
    segments = {"velo": ("TV", "UP"), "ft": ("FT",), "mp": ("MP",)}
    state_tx = float(arrays["FirstMeasurement_tx"][event_index][track_index])
    state_ty = float(arrays["FirstMeasurement_ty"][event_index][track_index])
    for segment_name, detectors in segments.items():
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        for detector in detectors:
            hits = grouped_hits[detector]
            xs.extend(float(v) for v in ak.to_list(hits["x"][track_index]))
            ys.extend(float(v) for v in ak.to_list(hits["y"][track_index]))
            zs.extend(float(v) for v in ak.to_list(hits["z"][track_index]))
        x0, tx = _fit_line_1d(zs, xs)
        y0, ty = _fit_line_1d(zs, ys)
        columns[f"{segment_name}_x0"].append(x0)
        columns[f"{segment_name}_tx"].append(tx)
        columns[f"{segment_name}_y0"].append(y0)
        columns[f"{segment_name}_ty"].append(ty)
        columns[f"{segment_name}_n_hits"].append(len(zs))
        columns[f"{segment_name}_delta_tx_state"].append(tx - state_tx if math.isfinite(tx) else float("nan"))
        columns[f"{segment_name}_delta_ty_state"].append(ty - state_ty if math.isfinite(ty) else float("nan"))


def _selected_indices(
    labels: np.ndarray,
    balance: bool,
    max_tracks_per_class: int | None,
    random_seed: int,
) -> np.ndarray:
    if not balance and max_tracks_per_class is None:
        return np.arange(labels.size)
    rng = np.random.default_rng(random_seed)
    selected: list[np.ndarray] = []
    for label in (0, 1):
        class_indices = np.flatnonzero(labels == label)
        if max_tracks_per_class is None:
            take = class_indices.size
        else:
            take = min(max_tracks_per_class, class_indices.size)
        if balance:
            other_size = int(np.count_nonzero(labels == (1 - label)))
            take = min(take, other_size)
        if take < class_indices.size:
            class_indices = rng.choice(class_indices, size=take, replace=False)
        selected.append(np.asarray(class_indices, dtype=int))
    if not selected:
        return np.asarray([], dtype=int)
    indices = np.concatenate(selected)
    indices.sort()
    return indices


def write_training_sample(
    input_path: str | Path,
    output_path: str | Path,
    *,
    limit: int | None = None,
    feature_groups: Sequence[str] | None = None,
    balance: bool = False,
    max_tracks_per_class: int | None = None,
    random_seed: int = 12345,
    compression: str = "zstd",
) -> tuple[Path, dict[str, int | str | list[str]]]:
    """Build a fake-vs-truth track table and write it as Parquet."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised only when dependency is missing
        raise RuntimeError("pyarrow is required to write Parquet files; install it with `python3 -m pip install pyarrow`") from exc

    groups = parse_feature_groups(feature_groups)
    tree = _tree_from_file(input_path)
    n_events = int(tree.num_entries if limit is None else min(tree.num_entries, limit))
    arrays = tree.arrays(library="ak", entry_start=0, entry_stop=n_events, filter_name=training_branch_names(groups))
    columns: defaultdict[str, list[float | int]] = defaultdict(list)
    groups_set = set(groups)
    for event_index in range(n_events):
        n_tracks = len(arrays["MC_truth"][event_index])
        grouped_hits = (
            {detector: _group_hits(arrays, event_index, detector) for detector in DETECTORS}
            if groups_set & {"hit_truth", "segments"}
            else {}
        )
        for track_index in range(n_tracks):
            mc_truth = int(arrays["MC_truth"][event_index][track_index])
            if mc_truth not in {0, 1}:
                continue
            _append_base_columns(columns, arrays, event_index, track_index)
            if "event" in groups_set:
                _append_event_columns(columns, arrays, event_index, n_tracks)
            if "state" in groups_set:
                _append_state_columns(columns, arrays, event_index, track_index)
            if "kinematics" in groups_set:
                _append_kinematic_columns(columns, arrays, event_index, track_index)
            if "hits" in groups_set:
                _append_hit_count_columns(columns, arrays, event_index, track_index)
            if "hit_truth" in groups_set:
                _append_hit_truth_columns(columns, grouped_hits, track_index)
            if "segments" in groups_set:
                _append_segment_columns(columns, arrays, event_index, grouped_hits, track_index)
            if "mc" in groups_set:
                _append_mc_columns(columns, arrays, event_index, track_index)

    labels = np.asarray(columns["label"], dtype=np.int8)
    selected = _selected_indices(labels, balance=balance, max_tracks_per_class=max_tracks_per_class, random_seed=random_seed)
    arrays_for_table = {name: np.asarray(values)[selected] for name, values in columns.items()}
    metadata = {
        "input_path": str(input_path),
        "feature_groups": list(groups),
        "label_definition": "label=1 for MC_truth==1 truth tracks, label=0 for MC_truth==0 fake tracks",
        "n_events_read": str(n_events),
    }
    table = pa.table(arrays_for_table)
    table = table.replace_schema_metadata({key: json.dumps(value).encode() for key, value in metadata.items()})
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path, compression=compression)
    label_counts = np.bincount(arrays_for_table["label"].astype(np.int8), minlength=2)
    summary: dict[str, int | str | list[str]] = {
        "output": str(output_path),
        "events": n_events,
        "tracks": int(len(selected)),
        "fake_tracks": int(label_counts[0]),
        "truth_tracks": int(label_counts[1]),
        "columns": int(len(arrays_for_table)),
        "feature_groups": list(groups),
    }
    return output_path, summary
