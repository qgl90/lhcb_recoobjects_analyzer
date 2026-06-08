# chi2/ndof cut study

This script studies how a `chi2/ndof` cut changes track selection.

## Inputs

- `--cuts` — comma- or space-separated thresholds, for example `8,7,6,5,4,3`
- `--root` — ROOT ntuple to analyze
- `--limit` — optional event cap for faster smoke tests

## What it plots

1. **Efficiency and inefficiency**
   - reconstructed tracks
   - matched truth tracks
   - fake tracks

2. **Kinematic distributions under the same cuts**
   - `p`
   - `pT`
   - `eta`
   - `phi`
   - split into fake and matched-truth tracks

3. **Response curves versus kinematics**
   - fake pass fraction `fake(cut) / fake(total)` vs `p`, `pT`, `eta`, `phi`
   - truth pass fraction `truth(cut) / truth(total)` vs `p`, `pT`, `eta`, `phi`
   - top row: fake tracks, bottom row: signal truth tracks
   - one line per requested chi2/ndof cut

## Notes

- The distributions are normalized so shape changes are easier to compare.
- The study reuses the track-composition loader rather than materializing full track objects.
- This makes it suitable for quick scans of multiple `chi2/ndof` thresholds.
