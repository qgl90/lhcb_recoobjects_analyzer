# Analysis workflow

This project is designed for fast inspection of LHCb-style reconstructed-track ntuples.

## Typical questions

- Where do fake tracks come from?
- How do fake and truth tracks differ in `p`, `pT`, `eta`, `phi`, `tx`, `ty`, and `chi2/ndof`?
- How does the fake rate change with the number of reconstructed PVs?
- What do the hit clouds look like in `xz`, `yz`, and `3D`?

## Recommended steps

1. Start with `--event-only-fake-rate` if you only need event-level PV dependence.
2. Use `--composition` to compare fake and truth track properties.
3. Use the default runner to inspect fake tracks and generate hit-cloud views.
4. Focus on tracks with `MC_truth == 0` when characterizing ghosts.

## Notes on plotting

- The plotting backend uses `mplhep.style.use("LHCb2")`.
- Figures are intentionally sized larger than default for easier inspection.
- Scatter markers are kept small to avoid hiding dense hit patterns.

