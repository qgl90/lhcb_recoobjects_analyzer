# ROOT analysis prompt playbook

Use these prompts when you want an LLM to help with ROOT-based analysis of reconstructed tracks.

## 1) Project architect prompt

> You are a senior HEP software engineer. Inspect this `.root` file with `uproot`, infer the tree/branch layout, and propose a clean Python package structure for analysis. Separate data models, ROOT loading, physics-feature derivation, plotting, and CLI entrypoints. Preserve truth/reco alignment rules, identify optional branches, and avoid assumptions about equal branch lengths. Produce a small, modular design that is easy to extend for fake-track studies.

## 2) ROOT schema audit prompt

> Open the ROOT file with `uproot`, list all trees and branch names, and classify them into track-level, hit-level, vertex-level, and scalar-level groups. Identify any jagged branches, count how they align per event, and note any optional or missing blocks. Summarize the schema in plain language and call out the safest indexing strategy for each branch group.

## 3) Fake-track forensics prompt

> Using the loaded track objects, isolate tracks where `MC_truth == 0`. For those tracks, analyze the first state parameters (`tx`, `ty`, `qop`) and derive `p`, `pt`, `eta`, and `phi`. Then compare fake-track distributions against matched tracks, look for patterns in hit multiplicities and detector occupancy, and suggest concrete hypotheses for why the fake tracks are being built.

## 4) 3D visualization prompt

> For a selected fake track, plot its reconstructed hits in 3D and color them by time. If truth hits are available, overlay them for comparison. Label detector subsets, annotate the first state parameters, and save the figure to disk. Focus on spotting geometric failure modes, discontinuities, and detector-region biases.

## 5) Feature-engineering prompt

> Build a compact feature table for each track using only fields available from the ROOT file. Include truth label, first-state kinematics, track quality, hit counts per detector, ancestor depth, and optional PV associations. Keep the output ready for pandas/CSV/Parquet so I can train a classifier or make diagnostic plots later.

