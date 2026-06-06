# Agent guide for `lhcb_recoobjects_analyzer`

This repository analyzes LHCb-style reconstructed-track ROOT ntuples and focuses on fake-track forensics.

## Core layout

- `main_runner.py` — CLI entry point and orchestration
- `src/analysis/root_loader.py` — ROOT loading and event materialization
- `src/analysis/fake_tracks.py` — fake-track selection, kinematics, fits, and visualizations
- `src/analysis/composition.py` — fake-vs-truth, ghost-rate, and PV-dependence studies
- `src/analysis/chi2_study.py` — `chi2/ndof` cut scans and kinematic comparisons
- `src/analysis/models.py` — shared dataclasses used across the analysis code

## Working rules

- Prefer small, targeted changes over broad refactors.
- Keep track/event loaders fast; avoid materializing heavy objects unless the analysis needs them.
- Preserve the fake-only fast path in `main_runner.py` when possible.
- Prefer reusing `TrackCompositionSample` for cut-based studies instead of loading full track objects.
- Use `mplhep.style.use("LHCb2")` for plots.
- Keep plots readable: larger figures, smaller markers, and explicit labels.

## Common commands

```bash
python3 main_runner.py --event-only-fake-rate --limit 10
python3 main_runner.py --composition --limit 100
python3 -m py_compile main_runner.py src/*.py src/analysis/*.py
```

## Data notes

- The sample ROOT file is not meant to be committed.
- Generated figures and notebooks should stay out of version control.
- The analysis often distinguishes:
  - fake tracks via `MC_truth == 0`
  - truth tracks via `MC_truth == 1`
  - PV count via `len(PV_x)` per event

## Useful outputs

- `event_fake_rate_vs_pvs.png`
- `event_fake_rate_vs_pvs_profile.png`
- `event_tracks_vs_pvs.png`
- `fake_truth_kinematics_1d.png`
- `fake_truth_kinematics_2d.png`
- `ghost_rate_tracks.png`
