# TrackTuple ntuple format

Format reference for the reconstructed-track ntuple, written so a future analyst/LLM can read and
interpret it with `uproot` **without any knowledge of how it was produced**.

Reference sample used throughout:
`Moore/build.workspace.tuple_time_tstation/0_test/ntuple.root`
(numbers below come from that file: 1 event, 357 tracks, 27 PVs).

Units are LHCb conventions: **position mm, time ns, momentum/energy MeV** (residuals
`x - mc_x ≈ 0.02 mm` confirm mm).

---

## 1. File structure

```
ntuple.root
└── BestLongTracks/            # a TDirectory; its name = the track collection
    └── TrackTuple             # a TTree
```

- The top directory name is the **track collection** (e.g. `BestLongTracks`, `DownstreamTracks`,
  `LongTracks`, `BestDownstreamTracks`). A file may contain several such directories, each with its
  own `TrackTuple`; they have identical schemas and are analysed the same way.
- **Each TTree entry = one event.** Per-track / per-hit / per-PV quantities live in **variable-length
  vector branches** inside that single entry. There is no flat "one row per track" table — you index
  into the vectors.

### Open it

```python
import uproot, awkward as ak, numpy as np

f = uproot.open(".../ntuple.root")
print(f.keys())                       # ['BestLongTracks;1', 'BestLongTracks/TrackTuple;1', ...]

t = f["BestLongTracks/TrackTuple"]
print(t.num_entries)                  # number of events
print(t.keys())                       # branch names

a = t.arrays()                        # awkward record array; axis 0 = events
```

Every branch is either a jagged `vector<float>`/`vector<int>` (so `a.MC_pid` has layout
`events * var * dtype`) or one of three event scalars. `vector<float>` → `float32`,
`vector<int>` → `int32`.

---

## 2. The five alignment groups — READ THIS FIRST

Within one event, branches only share a length if they are in the **same group**. Do **not** assume
all branches in an entry have equal length.

| Group | One element per | Length (ref. event) | Regroup using |
|-------|-----------------|---------------------|---------------|
| **A — track** | reconstructed track | 357 | already per-track |
| **B — hit** (4 sub-groups, one per detector) | hit on a track | `sum(<Det>Hits_n)`, e.g. TV=3256 | `<Det>Hits_n` (group A) |
| **C — ancestor** | MC ancestor of a track | `sum(MC_n_ancestors)`, e.g. 331 | `MC_n_ancestors` (group A) |
| **D — PV** | reconstructed primary vertex | 27 | already per-PV |
| **S — scalar** | event | 1 | — |

All **group-A** branches are mutually index-aligned: track *i* is the same physical track in
`MC_pid[i]`, `Track_chi2ndof[i]`, `FirstMeasurement_x[i]`, `TVHits_n[i]`, … **Group D (PVs) is
independent of the track count.**

Group-B and group-C branches are *flattened over the tracks of the event*. To get per-track lists:

```python
ev = 0
# per-track TV hit x and the matched-MCHit x, for one event:
tv_x   = ak.unflatten(a.TVHits_x[ev],    a.TVHits_n[ev])   # list-of-lists, one per track
tv_mcx = ak.unflatten(a.TVHits_mc_x[ev], a.TVHits_n[ev])
res_x  = tv_x - tv_mcx                                      # hit residuals per track

# across all events: unflatten each event, or just flatten when per-track grouping isn't needed
tv_x_per_track = [ak.unflatten(a.TVHits_x[i], a.TVHits_n[i]) for i in range(t.num_entries)]
all_res = ak.flatten(a.TVHits_x) - ak.flatten(a.TVHits_mc_x)   # all hits, all events (no grouping)
```

`MC_ancestor_pids` / `MC_ancestor_keys` regroup the same way with `MC_n_ancestors`.

---

## 3. Branch reference

Type column: `f`=`vector<float>`, `i`=`vector<int>`, scalars noted. "default" = value written when the
underlying object is absent — **always filter on these before trusting a value**.

### 3.1 Group A — MC truth of the track (`MC_*`)
The track's **best-match truth particle**. Unmatched track → ints `0`, floats `NaN`. Gate on
`MC_truth == 1`.

| Branch | T | Meaning | default |
|--------|---|---------|---------|
| `MC_truth` | i | 1 = track has a matched MC particle, 0 = ghost/unmatched | 0 |
| `MC_hasTV` | i | matched particle leaves hits in TV (timing VELO) | 0 |
| `MC_hasUP` | i | … in UP (upstream pixel) | 0 |
| `MC_hasMP` | i | … in MP | 0 |
| `MC_hasFT` | i | … in FT (SciFi) | 0 |
| `MC_fromSignal` | i | particle is from the signal decay | 0 |
| `MC_pid` | i | PDG id (e.g. ±211 π, 321 K, 2212 p, ±13 µ) | 0 |
| `MC_key` | i | MC particle key — **join key** within the event | 0 |
| `MC_pv_key` | i | key of the particle's true production PV; join to `PV_mc_key`. *Present only if PV block exists.* | -1 |
| `MC_charge` | f | charge in units of *e* | NaN |
| `MC_px`,`MC_py`,`MC_pz` | f | true momentum components (MeV) | NaN |
| `MC_pe` | f | true energy (MeV) | NaN |
| `MC_ovtx_x/y/z` | f | true origin-vertex position (mm) | NaN |
| `MC_n_ancestors` | i | number of ancestors (mother chain) → counts group C | 0 |

### 3.2 Group C — MC ancestor chain (`MC_ancestor_*`)
Mother chain of each matched particle, immediate mother first walking up to the primary. Flattened
over tracks; regroup with `MC_n_ancestors`.

| Branch | T | Meaning |
|--------|---|---------|
| `MC_ancestor_pids` | i | PDG id of each ancestor |
| `MC_ancestor_keys` | i | MC key of each ancestor |

### 3.3 Group A — track quality (`Track_*`)

| Branch | T | Meaning |
|--------|---|---------|
| `Track_chi2ndof` | f | fit χ²/ndof |
| `Track_ndof` | f | fit degrees of freedom |

### 3.4 Group A — track state(s) (`<State>_*`)
One block per stored state location; the prefix is the location name. **Default: `FirstMeasurement`.**
(If more states were stored you'd also see e.g. `ClosestToBeam_*`.) The state vector is
**(x, y, tx, ty, q/p)**.

| Branch | T | Meaning |
|--------|---|---------|
| `<State>_x`,`_y`,`_z` | f | position (mm) |
| `<State>_tx`,`_ty` | f | slopes dx/dz, dy/dz |
| `<State>_qop` | f | q/p (1/MeV) |
| `<State>_cov_i_j` | f | covariance, lower triangle `i≥j`, **15 branches** |

Covariance index → variable: **0=x, 1=y, 2=tx, 3=ty, 4=q/p**. Branches: `cov_0_0`; `cov_1_0,cov_1_1`;
`cov_2_0..2_2`; `cov_3_0..3_3`; `cov_4_0..4_4`. (e.g. `cov_2_2`=var(tx), diagonal = variances.)

### 3.5 Group B — hits per detector (`<Det>Hits_*`)
`<Det>` ∈ {`TV`, `UP`, `FT`, `MP`}. For each track, the hits assigned to it in that detector. The
`mc_*` fields are the **simulated hit (MCHit)** that produced the reconstructed cluster — i.e. the
truth position/time of that hit. Compare reco vs `mc_*` directly for residuals/resolution.

| Branch | T | Meaning | default |
|--------|---|---------|---------|
| `<Det>Hits_x`,`_y`,`_z` | f | reconstructed cluster position (mm) | — |
| `<Det>Hits_t` | f | reconstructed cluster time (ns) — TV carries real timing | — |
| `<Det>Hits_mc_x`,`_mc_y`,`_mc_z` | f | true MCHit position (mm) | **NaN if no MC match** |
| `<Det>Hits_mc_t` | f | true MCHit time (ns) | NaN if no MC match |
| `<Det>Hits_n` | i | number of `<Det>` hits on the track → counts this group | 0 |

In the reference file `mc_*` is filled for ~100% of TV/UP/MP hits and ~99% of FT hits; the rare NaN
is a reco hit with no associated truth (noise/spillover). A column that is **entirely NaN** means the
truth link was missing for that detector — treat the whole detector's MC info as unavailable.

### 3.6 Group A — RICH PID (`DLL_*`) — *optional*
Present only when RICH PID is included (absent in the reference file). Delta-log-likelihoods relative
to the pion hypothesis; `NaN` if the track has no PID object.

`DLL_Electron`, `DLL_Muon`, `DLL_Pion`, `DLL_Kaon`, `DLL_Proton`, `DLL_Deuteron`,
`DLL_BelowThreshold` (all `f`).

### 3.7 Group S — event scalars
| Branch | type | Meaning |
|--------|------|---------|
| `EventNumber` | `uint64` | event number |
| `RunNumber` | `uint32` | run number |
| `BunchCrossingID` | `uint32` | bunch-crossing id |

*Optional* (present in the reference file). Use these to uniquely identify / join events.

### 3.8 Group D — primary vertices (`PV_*`) — *optional*
One element per **reconstructed PV** (27 in the ref. event) — **not** per track. State vector is
**(x, y, z, t)**.

| Branch | T | Meaning | default |
|--------|---|---------|---------|
| `PV_x`,`PV_y`,`PV_z` | f | PV position (mm) | — |
| `PV_t` | f | PV time (ns) | — |
| `PV_ndof` | f | degrees of freedom | — |
| `PV_chi2ndof` | f | χ²/ndof | — |
| `PV_mc_key` | i | matched true vertex key; join to `MC_pv_key` | -1 |
| `PV_mc_x/_mc_y/_mc_z` | f | matched true vertex position (mm) | NaN |
| `PV_mc_t` | f | matched true vertex time (ns) | NaN |
| `PV_cov_i_j` | f | covariance, lower triangle `i≥j`, **10 branches** | — |

Covariance index → variable: **0=x, 1=y, 2=z, 3=t**. Branches: `cov_0_0`; `cov_1_0,cov_1_1`;
`cov_2_0..2_2`; `cov_3_0..3_3`.

**Track↔PV truth join:** a track's true PV is `MC_pv_key` (group A); each reco PV's true vertex is
`PV_mc_key` (group D). Match on these keys within the same event.

---

## 4. Detector codes

`<Det>` corresponds to the LHCbID detector type = top 4 bits of the 32-bit LHCbID (`id >> 28`):

| Prefix | type (hex) | Detector |
|--------|-----------|----------|
| `TV` | `0xb` | Timing VELO (4D VELO) |
| `UP` | `0xc` | Upstream Pixel |
| `FT` | `0x5` | SciFi / Fibre Tracker |
| `MP` | `0x6` | Mighty/Magnet-station Pixel |

---

## 5. Worked examples (real numbers from the reference file)

```python
import uproot, awkward as ak, numpy as np
t = uproot.open(".../ntuple.root")["BestLongTracks/TrackTuple"]
a = t.arrays()
ev = 0                                          # the file has 1 event

# --- truth-matched tracks and their PDG ids ---
m = a.MC_truth[ev] == 1                          # boolean, per track
print(ak.sum(m), "/", len(m), "tracks matched")  # e.g. ~ most of 357
pid = a.MC_pid[ev][m]                            # e.g. [-211, 321, -211, -211, ...]

# --- true momentum of matched tracks (MeV) ---
p = np.sqrt(a.MC_px[ev]**2 + a.MC_py[ev]**2 + a.MC_pz[ev]**2)[m]

# --- TV hit residuals (reco - truth), per track ---
tv_x   = ak.unflatten(a.TVHits_x[ev],    a.TVHits_n[ev])
tv_mcx = ak.unflatten(a.TVHits_mc_x[ev], a.TVHits_n[ev])
res    = tv_x - tv_mcx          # track0 ~ [0.005, 0.020, 0.017, -0.026, ...] mm
sigma  = ak.std(ak.flatten(res))   # ~ TV x-resolution scale

# --- a track's ancestors ---
anc = ak.unflatten(a.MC_ancestor_pids[ev], a.MC_n_ancestors[ev])  # list per track
```

Sample rows (event 0): `trk0 truth=1 pid=-211 n_anc=1 TVHits_n=16`,
`trk3 truth=0 pid=0 (ghost)`. First TV hits `x,mc_x`: `(1.2118, 1.2064)`, `(1.3610, 1.3407)`, …

---

## 6. Gotchas

- **Unequal branch lengths in one entry are normal** — respect the groups in §2. Mixing group A and
  group B/D lengths is the most common analysis bug.
- **NaN means "no truth / no object", not zero.** Filter (`MC_truth==1`, `~np.isnan(...)`) first.
- **An entirely-NaN `<Det>Hits_mc_*`** ⇒ that detector's hit→truth link was missing in production;
  reco hit positions are still valid, only the MC ones are unavailable.
- **Optional blocks may be absent**: `DLL_*` (no RICH), `PV_*`/`PV_mc_*`/`MC_pv_key` (no PVs),
  `EventNumber`/`RunNumber`/`BunchCrossingID` (event info off). Check `t.keys()` before assuming.
- **`q/p` sign** carries the charge; `MC_charge` is the truth charge in *e*.
- Position from `<Det>Hits_mc_*` is the MCHit mid-point, the natural truth reference for residuals.
