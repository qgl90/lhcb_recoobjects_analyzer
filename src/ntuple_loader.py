from __future__ import annotations

from pathlib import Path

import awkward as ak
import uproot

from .ntuple_models import EventRecord, Hit, MCParticle, PrimaryVertex, Track, TrackState


DETECTORS = ("TV", "UP", "FT", "MP")


def _tree_from_file(path: str | Path):
    root_file = uproot.open(path)
    if "BestLongTracks" in root_file:
        directory = root_file["BestLongTracks"]
        if "TrackTuple" in directory:
            return directory["TrackTuple"]
    for key in root_file.keys():
        obj = root_file[key]
        if hasattr(obj, "keys") and "TrackTuple" in obj:
            return obj["TrackTuple"]
    raise KeyError("Could not find a TrackTuple tree in the ROOT file")


def _branch_value(array: ak.Array, field: str, event_index: int, track_index: int | None = None):
    value = array[field][event_index]
    if track_index is None:
        return value
    return value[track_index]


def _make_covariance(
    prefix: str,
    event_arrays: ak.Array,
    event_index: int,
    track_index: int | None = None,
    size: int = 5,
):
    cov: dict[tuple[int, int], float] = {}
    for i in range(size):
        for j in range(i + 1):
            name = f"{prefix}_cov_{i}_{j}"
            cov[(i, j)] = float(_branch_value(event_arrays, name, event_index, track_index))
    return cov


def _build_hits_for_track(event_arrays: ak.Array, event_index: int, track_index: int) -> dict[str, list[Hit]]:
    hits: dict[str, list[Hit]] = {}
    for detector in DETECTORS:
        counts = event_arrays[f"{detector}Hits_n"][event_index]
        if int(counts[track_index]) == 0:
            hits[detector] = []
            continue
        reco_x = ak.unflatten(event_arrays[f"{detector}Hits_x"][event_index], counts)[track_index]
        reco_y = ak.unflatten(event_arrays[f"{detector}Hits_y"][event_index], counts)[track_index]
        reco_z = ak.unflatten(event_arrays[f"{detector}Hits_z"][event_index], counts)[track_index]
        reco_t = ak.unflatten(event_arrays[f"{detector}Hits_t"][event_index], counts)[track_index]
        mc_x = ak.unflatten(event_arrays[f"{detector}Hits_mc_x"][event_index], counts)[track_index]
        mc_y = ak.unflatten(event_arrays[f"{detector}Hits_mc_y"][event_index], counts)[track_index]
        mc_z = ak.unflatten(event_arrays[f"{detector}Hits_mc_z"][event_index], counts)[track_index]
        mc_t = ak.unflatten(event_arrays[f"{detector}Hits_mc_t"][event_index], counts)[track_index]
        hits[detector] = [
            Hit(
                x=float(reco_x[i]),
                y=float(reco_y[i]),
                z=float(reco_z[i]),
                t=float(reco_t[i]),
                mc_x=float(mc_x[i]),
                mc_y=float(mc_y[i]),
                mc_z=float(mc_z[i]),
                mc_t=float(mc_t[i]),
            )
            for i in range(len(reco_x))
        ]
    return hits


def _build_mc_particles_for_event(tracks: list[Track]) -> list[MCParticle]:
    particles: list[MCParticle] = []
    for track in tracks:
        if track.mc_truth != 1:
            continue
        particles.append(
            MCParticle(
                key=track.mc_key,
                track_index=track.index,
                pid=track.mc_pid,
                charge=track.mc_charge,
                px=track.mc_px,
                py=track.mc_py,
                pz=track.mc_pz,
                pe=track.mc_pe,
                ovtx_x=track.mc_ovtx_x,
                ovtx_y=track.mc_ovtx_y,
                ovtx_z=track.mc_ovtx_z,
                pv_key=track.mc_pv_key,
                from_signal=track.mc_fromSignal,
                has_tv=track.mc_hasTV,
                has_up=track.mc_hasUP,
                has_mp=track.mc_hasMP,
                has_ft=track.mc_hasFT,
                ancestors_pids=track.ancestors_pids,
                ancestors_keys=track.ancestors_keys,
            )
        )
    return particles


def load_events(path: str | Path, limit: int | None = None) -> list[EventRecord]:
    tree = _tree_from_file(path)
    arrays = tree.arrays(library="ak")
    n_events = int(tree.num_entries if limit is None else min(tree.num_entries, limit))
    events: list[EventRecord] = []

    has_pv = "PV_x" in arrays.fields

    for event_index in range(n_events):
        n_tracks = len(arrays["MC_truth"][event_index])
        tracks: list[Track] = []
        for track_index in range(n_tracks):
            state = TrackState(
                x=float(_branch_value(arrays, "FirstMeasurement_x", event_index, track_index)),
                y=float(_branch_value(arrays, "FirstMeasurement_y", event_index, track_index)),
                z=float(_branch_value(arrays, "FirstMeasurement_z", event_index, track_index)),
                tx=float(_branch_value(arrays, "FirstMeasurement_tx", event_index, track_index)),
                ty=float(_branch_value(arrays, "FirstMeasurement_ty", event_index, track_index)),
                qop=float(_branch_value(arrays, "FirstMeasurement_qop", event_index, track_index)),
                cov=_make_covariance("FirstMeasurement", arrays, event_index, track_index, size=5),
            )
            tracks.append(
                Track(
                    index=track_index,
                    mc_truth=int(_branch_value(arrays, "MC_truth", event_index, track_index)),
                    mc_hasTV=int(_branch_value(arrays, "MC_hasTV", event_index, track_index)),
                    mc_hasUP=int(_branch_value(arrays, "MC_hasUP", event_index, track_index)),
                    mc_hasMP=int(_branch_value(arrays, "MC_hasMP", event_index, track_index)),
                    mc_hasFT=int(_branch_value(arrays, "MC_hasFT", event_index, track_index)),
                    mc_fromSignal=int(_branch_value(arrays, "MC_fromSignal", event_index, track_index)),
                    mc_pid=int(_branch_value(arrays, "MC_pid", event_index, track_index)),
                    mc_key=int(_branch_value(arrays, "MC_key", event_index, track_index)),
                    mc_pv_key=int(_branch_value(arrays, "MC_pv_key", event_index, track_index))
                    if "MC_pv_key" in arrays.fields
                    else -1,
                    mc_charge=float(_branch_value(arrays, "MC_charge", event_index, track_index)),
                    mc_px=float(_branch_value(arrays, "MC_px", event_index, track_index)),
                    mc_py=float(_branch_value(arrays, "MC_py", event_index, track_index)),
                    mc_pz=float(_branch_value(arrays, "MC_pz", event_index, track_index)),
                    mc_pe=float(_branch_value(arrays, "MC_pe", event_index, track_index)),
                    mc_ovtx_x=float(_branch_value(arrays, "MC_ovtx_x", event_index, track_index)),
                    mc_ovtx_y=float(_branch_value(arrays, "MC_ovtx_y", event_index, track_index)),
                    mc_ovtx_z=float(_branch_value(arrays, "MC_ovtx_z", event_index, track_index)),
                    mc_n_ancestors=int(_branch_value(arrays, "MC_n_ancestors", event_index, track_index)),
                    ancestors_pids=[
                        int(x)
                        for x in ak.to_list(
                            ak.unflatten(arrays["MC_ancestor_pids"][event_index], arrays["MC_n_ancestors"][event_index])[
                                track_index
                            ]
                        )
                    ],
                    ancestors_keys=[
                        int(x)
                        for x in ak.to_list(
                            ak.unflatten(arrays["MC_ancestor_keys"][event_index], arrays["MC_n_ancestors"][event_index])[
                                track_index
                            ]
                        )
                    ],
                    chi2ndof=float(_branch_value(arrays, "Track_chi2ndof", event_index, track_index)),
                    ndof=float(_branch_value(arrays, "Track_ndof", event_index, track_index)),
                    state=state,
                    hits=_build_hits_for_track(arrays, event_index, track_index),
                )
            )

        mc_particles = _build_mc_particles_for_event(tracks)

        primary_vertices: list[PrimaryVertex] = []
        if has_pv:
            pv_count = len(arrays["PV_x"][event_index])
            for pv_index in range(pv_count):
                primary_vertices.append(
                    PrimaryVertex(
                        index=pv_index,
                        x=float(arrays["PV_x"][event_index][pv_index]),
                        y=float(arrays["PV_y"][event_index][pv_index]),
                        z=float(arrays["PV_z"][event_index][pv_index]),
                        t=float(arrays["PV_t"][event_index][pv_index]),
                        ndof=float(arrays["PV_ndof"][event_index][pv_index]),
                        chi2ndof=float(arrays["PV_chi2ndof"][event_index][pv_index]),
                        mc_key=int(arrays["PV_mc_key"][event_index][pv_index]) if "PV_mc_key" in arrays.fields else -1,
                        mc_x=float(arrays["PV_mc_x"][event_index][pv_index]) if "PV_mc_x" in arrays.fields else float("nan"),
                        mc_y=float(arrays["PV_mc_y"][event_index][pv_index]) if "PV_mc_y" in arrays.fields else float("nan"),
                        mc_z=float(arrays["PV_mc_z"][event_index][pv_index]) if "PV_mc_z" in arrays.fields else float("nan"),
                        mc_t=float(arrays["PV_mc_t"][event_index][pv_index]) if "PV_mc_t" in arrays.fields else float("nan"),
                        cov=_make_covariance("PV", arrays, event_index, pv_index, size=4),
                    )
                )

        events.append(
            EventRecord(
                event_number=int(arrays["EventNumber"][event_index]),
                run_number=int(arrays["RunNumber"][event_index]),
                bunch_crossing_id=int(arrays["BunchCrossingID"][event_index]),
                tracks=tracks,
                mc_particles=mc_particles,
                primary_vertices=primary_vertices,
            )
        )

    return events

