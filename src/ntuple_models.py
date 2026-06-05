from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Hit:
    x: float
    y: float
    z: float
    t: float
    mc_x: float
    mc_y: float
    mc_z: float
    mc_t: float


@dataclass(slots=True)
class TrackState:
    x: float
    y: float
    z: float
    tx: float
    ty: float
    qop: float
    cov: dict[tuple[int, int], float] = field(default_factory=dict)


@dataclass(slots=True)
class Track:
    index: int
    mc_truth: int
    mc_hasTV: int
    mc_hasUP: int
    mc_hasMP: int
    mc_hasFT: int
    mc_fromSignal: int
    mc_pid: int
    mc_key: int
    mc_pv_key: int
    mc_charge: float
    mc_px: float
    mc_py: float
    mc_pz: float
    mc_pe: float
    mc_ovtx_x: float
    mc_ovtx_y: float
    mc_ovtx_z: float
    mc_n_ancestors: int
    ancestors_pids: list[int]
    ancestors_keys: list[int]
    chi2ndof: float
    ndof: float
    state: TrackState
    hits: dict[str, list[Hit]]


@dataclass(slots=True)
class MCParticle:
    key: int
    track_index: int
    pid: int
    charge: float
    px: float
    py: float
    pz: float
    pe: float
    ovtx_x: float
    ovtx_y: float
    ovtx_z: float
    pv_key: int
    from_signal: int
    has_tv: int
    has_up: int
    has_mp: int
    has_ft: int
    ancestors_pids: list[int]
    ancestors_keys: list[int]


@dataclass(slots=True)
class PrimaryVertex:
    index: int
    x: float
    y: float
    z: float
    t: float
    ndof: float
    chi2ndof: float
    mc_key: int
    mc_x: float
    mc_y: float
    mc_z: float
    mc_t: float
    cov: dict[tuple[int, int], float] = field(default_factory=dict)


@dataclass(slots=True)
class EventRecord:
    event_number: int
    run_number: int
    bunch_crossing_id: int
    tracks: list[Track]
    mc_particles: list[MCParticle]
    primary_vertices: list[PrimaryVertex]

