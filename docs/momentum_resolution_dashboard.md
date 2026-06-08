# Momentum resolution dashboard

Use `momentum_resolution_dashboard.py` to explore truth-matched momentum residuals interactively.

## Inputs

- ROOT ntuple output from the analysis pipeline, or
- a processed Parquet file from `prepare_training_samples_fakes_vs_truth.py`

## What it shows

- scatter of `(p(reco) - p(truth)) / p(truth)` vs a chosen kinematic variable
- binned mean and spread, or Gaussian-fit bias and width when available
- an overall residual histogram for the selected truth-matched tracks
- a table with per-bin summary statistics

## Controls

- `chi2/ndof` maximum cut
- binning variable: `truth_p`, `truth_pt`, `truth_eta`, `truth_phi`, `reco_p`, `reco_pt`, or `chi2ndof`
- number of bins
- summary mode: moments or Gaussian fit

