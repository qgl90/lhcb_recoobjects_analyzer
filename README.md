# LHCb Reco Objects Analyzer

`lhcb_recoobjects_analyzer` is a small ROOT-analysis toolkit for studying reconstructed tracks, fake tracks, and event-level ghost-rate behavior in LHCb-style ntuples.

It reads a `.root` file with `uproot`, turns the data into lightweight Python objects, and produces track- and event-level plots for quick diagnostics.

## What’s inside

- `main_runner.py` — command-line entry point
- `chi2_cut_study.py` — dedicated `chi2/ndof` cut study
- `src/analysis/root_loader.py` — ROOT reading and event assembly
- `src/analysis/composition.py` — fake-vs-truth and event-level studies
- `src/analysis/fake_tracks.py` — fake-track inspection, fits, and plotting
- `src/analysis/chi2_study.py` — `chi2/ndof` cut efficiency and kinematic impact study
- `src/analysis/models.py` — shared dataclasses for tracks, hits, events, and PVs
- `src/analysis/prompts.md` — prompt notes for analysis workflows

## Install

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

`chi2/ndof` cut study:

```bash
python3 chi2_cut_study.py --cuts 8,7,6,5,4,3 --limit 100
```

That study now also produces variable-response panels showing truth efficiency and fake rejection versus `p`, `pT`, `eta`, and `phi`.

## Outputs

The scripts can generate:

- 3D hit clouds for fake tracks
- fake-track kinematic distributions
- fake-vs-truth histograms and 2D maps
- ghost-rate plots
- event fake rate versus number of PVs
- event track multiplicity versus number of PVs
- `chi2/ndof` efficiency curves and kinematic-shape comparisons
- `chi2/ndof` truth-efficiency and fake-rejection response curves versus `p`, `pT`, `eta`, and `phi`

## Analysis flow

1. Load a `.root` file with `uproot`
2. Materialize fake tracks or event-level summaries
3. Inspect `tx`, `ty`, `qop`, `p`, `pt`, `eta`, `phi`, and `chi2/ndof`
4. Compare fake and truth distributions
5. Study event-level fake rate versus PV multiplicity
6. Scan `chi2/ndof` thresholds and see how fake/truth kinematics respond

## Validation

The repository includes a minimal push-time check in GitHub Actions that compiles the Python sources and runs a small import smoke test.
