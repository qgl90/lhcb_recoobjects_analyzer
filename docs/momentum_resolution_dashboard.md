# Momentum resolution dashboard

Use `momentum_resolution_dashboard.py` to explore truth-matched momentum residuals interactively.

## Inputs

- ROOT ntuple output from the analysis pipeline, or
- a processed Parquet file from `prepare_training_samples_fakes_vs_truth.py`

## What it shows

- scatter of the signed residual `(p(reco) - p(truth)) / p(truth)` vs a chosen kinematic variable
- a combined pad with the scatter profile, `p-bias`, and `p-resolution` in the same view
- the scatter profile uses percent residuals, with mean points and width error bars
- the per-bin fit is a Gaussian on the signed percent residual
- `p-bias` is the Gaussian mean `μ` in percent
- `p-resolution` is the Gaussian width `σ` in percent
- the selected bin readout shows both `p-bias` and `p-resolution` directly
- a selectable per-bin histogram with a Gaussian fit overlay in percent units
- an overall residual histogram for the selected truth-matched tracks
- a table with per-bin summary statistics

## Controls

- `chi2/ndof` maximum cut
- binning variable: `truth_p`, `truth_pt`, `truth_eta`, `truth_phi`, `reco_p`, `reco_pt`, or `chi2ndof`
- explicit bin-range controls, so you can study a slice such as `eta in [2, 5]`
- residual window controls for cutting or zooming the signed `Δp/p` view in percent
- number of bins
- summary mode: moments or Gaussian fit
