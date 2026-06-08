from __future__ import annotations

from pathlib import Path

import numpy as np

from src.analysis.momentum_resolution import (
    load_resolution_frame,
    prepare_binned_resolution,
    selected_bin_histogram,
    select_truth_frame,
)


def _default_range(values: np.ndarray, *, positive_only: bool = False) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if positive_only:
        finite = finite[finite > 0]
    if len(finite) == 0:
        return (0.0, 1.0)
    low, high = np.percentile(finite, [1, 99])
    if low == high:
        low = float(np.min(finite))
        high = float(np.max(finite))
    if positive_only and low <= 0:
        low = max(high * 1e-3, 1e-6)
    return float(low), float(high)


def _render_analysis_tab(
    st,
    plt,
    frame,
    *,
    tab_title: str,
    response_var: str,
    response_label: str,
    profile_stat: str,
    chi2_max: float,
    bin_var: str,
    n_bins: int,
    fit_mode: str,
    show_all_points: bool,
    selected_limit: int | None,
) -> None:
    truth_frame = select_truth_frame(frame, chi2_max=chi2_max)
    if truth_frame.empty:
        st.warning(f"No truth-matched tracks passed the current selection for {tab_title}.")
        return

    positive_only = bin_var in {"truth_p", "truth_pt", "reco_p", "reco_pt"}
    bin_default_min, bin_default_max = _default_range(truth_frame[bin_var].to_numpy(), positive_only=positive_only)
    response_default_min, response_default_max = _default_range(
        truth_frame[response_var].to_numpy(), positive_only=response_var.endswith("_abs")
    )
    if response_var.endswith("_abs"):
        response_default_min = max(0.0, response_default_min)

    with st.expander(f"{tab_title} controls", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            bin_min = st.number_input(f"{tab_title}: bin min", value=float(bin_default_min), format="%.6g", key=f"{tab_title}_bin_min")
        with c2:
            bin_max = st.number_input(f"{tab_title}: bin max", value=float(bin_default_max), format="%.6g", key=f"{tab_title}_bin_max")
        with c3:
            response_max = st.number_input(
                f"{tab_title}: residual max", value=float(response_default_max), format="%.6g", key=f"{tab_title}_res_max"
            )
        response_min_default = float(response_default_min)
        if response_var.endswith("_abs"):
            response_min_default = 0.0
        response_min = st.number_input(
            f"{tab_title}: residual min",
            value=response_min_default,
            format="%.6g",
            key=f"{tab_title}_res_min",
        )
        use_log_x = positive_only
        show_points = st.checkbox(
            f"{tab_title}: show all scatter points", value=show_all_points, key=f"{tab_title}_show_points"
        )

    if positive_only and bin_min <= 0:
        st.warning("Positive bin variables require bin min > 0; adjusting to a small positive value.")
        bin_min = max(bin_max * 1e-3, 1e-6)
    if bin_min >= bin_max:
        st.error("Bin min must be smaller than bin max.")
        return
    if response_min >= response_max:
        st.error("Residual min must be smaller than residual max.")
        return

    selected_frame, summary = prepare_binned_resolution(
        frame,
        bin_var=bin_var,
        response_var=response_var,
        n_bins=n_bins,
        chi2_max=chi2_max,
        bin_min=bin_min,
        bin_max=bin_max,
        response_min=response_min,
        response_max=response_max,
    )
    if summary.empty:
        st.warning("No tracks remain after the current bin and residual selection.")
        return

    selected_bin_options = []
    for index, row in summary.iterrows():
        selected_bin_options.append(
            f"{index}: [{row['bin_low']:.4g}, {row['bin_high']:.4g}) count={int(row['count'])}"
        )
    default_index = int(summary["count"].to_numpy(dtype=int).argmax())
    selected_bin_label = st.selectbox(
        f"{tab_title}: inspect bin", selected_bin_options, index=default_index, key=f"{tab_title}_bin"
    )
    selected_bin_index = selected_bin_options.index(selected_bin_label)
    bin_details = selected_bin_histogram(
        frame,
        bin_var=bin_var,
        response_var=response_var,
        bin_index=selected_bin_index,
        n_bins=n_bins,
        chi2_max=chi2_max,
        bin_min=bin_min,
        bin_max=bin_max,
        response_min=response_min,
        response_max=response_max,
        fit_mode=fit_mode,
    )

    left, right = st.columns([1.4, 1.0])

    with left:
        st.subheader("Scatter + binned profile")
        x = selected_frame[bin_var].to_numpy(dtype=float)
        y = selected_frame[response_var].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        if positive_only:
            positive = x > 0
            x = x[positive]
            y = y[positive]
        fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
        if show_points:
            ax.scatter(x, y, s=4, alpha=0.08, color="#7F7F7F", edgecolors="none")

        centers = summary["bin_center"].to_numpy(dtype=float)
        profile_values = summary[profile_stat].to_numpy(dtype=float)
        valid = summary["count"].to_numpy(dtype=int) > 0
        ax.plot(centers[valid], profile_values[valid], "o-", color="#2F4B7C", linewidth=2, label=profile_stat)

        if positive_only:
            ax.set_xscale("log")
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel(bin_var)
        ax.set_ylabel(response_label)
        ax.set_title(f"{tab_title}: {response_label} vs {bin_var}")
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False)
        st.pyplot(fig, clear_figure=True)

    with right:
        st.subheader("Bin statistics")
        st.dataframe(summary, use_container_width=True)
        st.download_button(
            f"Download {tab_title} summary CSV",
            summary.to_csv(index=False).encode(),
            file_name=f"momentum_resolution_{tab_title}.csv",
            mime="text/csv",
        )

        st.subheader("Selected bin histogram")
        values = np.asarray(bin_details["values"], dtype=float)
        bin_row = bin_details["bin_row"]
        fit_mu = float(bin_details["fit_mu"])
        fit_sigma = float(bin_details["fit_sigma"])

        fig2, ax2 = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        if len(values):
            low = float(bin_row["bin_low"])
            high = float(bin_row["bin_high"])
            ax2.hist(values, bins=50, range=(response_min, response_max), histtype="stepfilled", alpha=0.55, color="#4C72B0")
            if np.isfinite(fit_mu) and np.isfinite(fit_sigma) and fit_sigma > 0:
                xs = np.linspace(response_min, response_max, 400)
                hist_area = len(values) * (response_max - response_min) / 50
                pdf = hist_area * (1.0 / (fit_sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((xs - fit_mu) / fit_sigma) ** 2)
                ax2.plot(xs, pdf, color="#C44E52", linewidth=2, label=f"Gaussian fit μ={fit_mu:.3g}, σ={fit_sigma:.3g}")
            ax2.axvline(0, color="black", linestyle="--", linewidth=1)
            ax2.set_title(f"Bin [{low:.4g}, {high:.4g}) with {len(values)} tracks")
            ax2.legend(frameon=False)
        else:
            ax2.text(0.5, 0.5, "no tracks in selected bin", transform=ax2.transAxes, ha="center", va="center")
        ax2.set_xlabel(response_label)
        ax2.set_ylabel("tracks")
        ax2.grid(True, alpha=0.2)
        st.pyplot(fig2, clear_figure=True)

        st.subheader("Fit summary")
        st.write(
            {
                "bin_range": [float(bin_row["bin_low"]), float(bin_row["bin_high"])],
                "count": int(bin_row["count"]),
                "mean": float(bin_row["mean"]),
                "std": float(bin_row["std"]),
                "fit_mu": fit_mu,
                "fit_sigma": fit_sigma,
            }
        )


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "streamlit is required for the dashboard; install it with `python3 -m pip install streamlit`"
        ) from exc
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
        n_bins = st.slider("Number of bins", min_value=4, max_value=40, value=16, step=1)
        fit_mode = st.selectbox("Bin summary", ["gaussian", "moments"], index=0)
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

    if select_truth_frame(frame, chi2_max=chi2_max).empty:
        st.warning("No truth-matched tracks passed the current selection.")
        return

    tabs = st.tabs(["p-resolution", "p-bias"])
    with tabs[0]:
        _render_analysis_tab(
            st,
            plt,
            frame,
            tab_title="p-resolution",
            response_var="delta_p_over_p_abs",
            response_label="|Δp| / p",
            profile_stat="fit_sigma",
            chi2_max=chi2_max,
            bin_var=bin_var,
            n_bins=n_bins,
            fit_mode=fit_mode,
            show_all_points=show_all_points,
            selected_limit=selected_limit,
        )
    with tabs[1]:
        _render_analysis_tab(
            st,
            plt,
            frame,
            tab_title="p-bias",
            response_var="delta_p_over_p",
            response_label="Δp / p",
            profile_stat="fit_mu",
            chi2_max=chi2_max,
            bin_var=bin_var,
            n_bins=n_bins,
            fit_mode=fit_mode,
            show_all_points=show_all_points,
            selected_limit=selected_limit,
        )


if __name__ == "__main__":
    main()
