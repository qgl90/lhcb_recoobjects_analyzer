from __future__ import annotations

from pathlib import Path

import numpy as np

from src.analysis.momentum_resolution import (
    build_binned_resolution,
    load_resolution_frame,
    select_truth_frame,
)


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("streamlit is required for the dashboard; install it with `python3 -m pip install streamlit`") from exc
    import matplotlib.pyplot as plt

    st.set_page_config(page_title="Momentum resolution dashboard", layout="wide")
    st.title("Momentum resolution dashboard")
    st.caption("Explore truth-matched momentum residuals versus kinematic variables.")

    with st.sidebar:
        st.header("Inputs")
        path_text = st.text_input("ROOT or Parquet path", value="training_samples_fakes_vs_truth.parquet")
        limit = st.number_input("Row/event limit", min_value=0, value=0, step=1, help="0 means no limit")
        chi2_max = st.slider("chi2/ndof max cut", min_value=0.0, max_value=20.0, value=8.0, step=0.5)
        bin_var = st.selectbox(
            "Binning variable",
            ["truth_p", "truth_pt", "truth_eta", "truth_phi", "reco_p", "reco_pt", "chi2ndof"],
            index=0,
        )
        response_var = st.selectbox("Resolution variable", ["delta_p_over_p", "delta_pt_over_pt"], index=0)
        n_bins = st.slider("Number of bins", min_value=6, max_value=40, value=16, step=1)
        fit_mode = st.selectbox("Bin summary", ["moments", "gaussian"], index=0)
        show_all_points = st.checkbox("Show all points", value=True)
        selected_limit = None if limit == 0 else int(limit)

    @st.cache_data(show_spinner=False)
    def _load(path_value: str, row_limit: int | None) -> object:
        return load_resolution_frame(path_value, limit=row_limit)

    try:
        frame = _load(path_text, selected_limit)
    except Exception as exc:
        st.error(f"Failed to load input: {exc}")
        return

    truth_frame = select_truth_frame(frame, chi2_max=chi2_max)
    if truth_frame.empty:
        st.warning("No truth-matched tracks passed the current selection.")
        return

    summary = build_binned_resolution(
        frame,
        bin_var=bin_var,
        response_var=response_var,
        n_bins=n_bins,
        chi2_max=chi2_max,
        fit_mode=fit_mode,
    )

    left, right = st.columns([1.3, 1.0])

    with left:
        st.subheader("Scatter + binned profile")
        x = truth_frame[bin_var].to_numpy(dtype=float)
        y = truth_frame[response_var].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        if bin_var in {"truth_p", "truth_pt", "reco_p", "reco_pt"}:
            positive = x > 0
            x = x[positive]
            y = y[positive]
        fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
        if show_all_points:
            ax.scatter(x, y, s=4, alpha=0.08, color="#7F7F7F", edgecolors="none")
        if not summary.empty:
            centers = summary["bin_center"].to_numpy(dtype=float)
            mean = summary["fit_mu"].to_numpy(dtype=float) if fit_mode == "gaussian" else summary["mean"].to_numpy(dtype=float)
            spread = summary["fit_sigma"].to_numpy(dtype=float) if fit_mode == "gaussian" else summary["std"].to_numpy(dtype=float)
            valid = summary["count"].to_numpy(dtype=int) > 0
            ax.errorbar(
                centers[valid],
                mean[valid],
                yerr=spread[valid],
                fmt="o-",
                color="#2F4B7C",
                ecolor="#2F4B7C",
                capsize=3,
                linewidth=2,
                label="profile",
            )
        if bin_var in {"truth_p", "truth_pt", "reco_p", "reco_pt"}:
            ax.set_xscale("log")
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel(bin_var)
        ax.set_ylabel(response_var)
        ax.set_title(f"{response_var} vs {bin_var}")
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False)
        st.pyplot(fig, clear_figure=True)

    with right:
        st.subheader("Bin statistics")
        st.dataframe(summary, use_container_width=True)
        st.download_button(
            "Download summary CSV",
            summary.to_csv(index=False).encode(),
            file_name=f"momentum_resolution_{bin_var}.csv",
            mime="text/csv",
        )

        st.subheader("Overall distribution")
        fig2, ax2 = plt.subplots(figsize=(8, 4), constrained_layout=True)
        ax2.hist(truth_frame[response_var].dropna(), bins=60, histtype="stepfilled", alpha=0.55, color="#4C72B0")
        ax2.axvline(0, color="black", linestyle="--", linewidth=1)
        ax2.set_xlabel(response_var)
        ax2.set_ylabel("tracks")
        ax2.set_title("Truth-matched residual distribution")
        ax2.grid(True, alpha=0.2)
        st.pyplot(fig2, clear_figure=True)


if __name__ == "__main__":
    main()
