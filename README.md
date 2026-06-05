# LHCb Reco Objects Analyzer

`lhcb_recoobjects_analyzer` is a small ROOT-analysis toolkit for studying reconstructed tracks, fake tracks, and event-level ghost-rate behavior in LHCb-style ntuples.

It reads a `.root` file with `uproot`, turns the data into lightweight Python objects, and produces track- and event-level plots for quick diagnostics.

## What’s inside

- `main_runner.py` — command-line entry point
- `src/analysis/root_loader.py` — ROOT reading and event assembly
- `src/analysis/composition.py` — fake-vs-truth and event-level studies
- `src/analysis/fake_tracks.py` — fake-track inspection, fits, and plotting
- `src/analysis/models.py` — shared dataclasses for tracks, hits, events, and PVs
- `src/analysis/prompts.md` — prompt notes for analysis workflows

## Install

The project uses a small Python stack:

```bash
python3 -m pip install -r requirements.txt
```

## Run

Quick fake-track pass on a ROOT file:

```bash
python3 main_runner.py --root jan2026_minbias_1p0E34_1000evts.root
```

Event-level fake rate versus PV count only:

```bash
python3 main_runner.py --event-only-fake-rate --limit 10
```

Full fake-vs-truth composition study:

```bash
python3 main_runner.py --composition --limit 100
```

## Outputs

The runner can generate:

- 3D hit clouds for fake tracks
- fake-track kinematic distributions
- fake-vs-truth histograms and 2D maps
- ghost-rate plots
- per-event fake rate versus number of PVs
- per-event track multiplicity versus number of PVs

## Analysis flow

1. Load a `.root` file with `uproot`
2. Materialize fake tracks or event-level summaries
3. Inspect `tx`, `ty`, `qop`, `p`, `pt`, `eta`, `phi`, and `chi2/ndof`
4. Compare fake and truth distributions
5. Study event-level fake rate versus PV multiplicity

## Validation

The repository includes a minimal push-time check in GitHub Actions that compiles the Python sources and runs a small import smoke test.

