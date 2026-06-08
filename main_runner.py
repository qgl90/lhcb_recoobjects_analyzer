from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.composition import (
    load_event_composition_sample,
    load_event_fake_pv_sample,
    load_track_composition_sample,
    plot_event_fake_rate_vs_pvs,
    plot_event_tracks_vs_pvs,
    plot_ghost_rate,
    plot_momentum_resolution_1d,
    plot_momentum_resolution_2d,
    plot_fake_truth_kinematics_1d,
    plot_fake_truth_kinematics_2d,
    plot_fake_truth_hitcount_distributions,
    plot_fake_truth_pt_eta,
    plot_fake_truth_segment_slopes,
)
from src.analysis.momentum_resolution import plot_momentum_resolution_scatter
from src.analysis.fake_tracks import (
    fake_tracks,
    plot_fake_track_distributions,
    plot_fake_track_pt_eta,
    plot_fake_track_slope_directions,
    plot_fake_track_segment_slopes,
    plot_track_hits_3d,
    summarize_fake_tracks,
)
from src.analysis.root_loader import load_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze fake tracks in a ROOT ntuple."
    )
    parser.add_argument(
        "--root", type=Path, default=Path("jan2026_minbias_1p0E34_1000evts.root")
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="number of events to load; omit for all"
    )
    parser.add_argument("--plot-output", type=Path, default=Path("fake_track_3d.png"))
    parser.add_argument(
        "--dist-output", type=Path, default=Path("fake_track_distributions.png")
    )
    parser.add_argument(
        "--pteta-output", type=Path, default=Path("fake_track_pt_eta.png")
    )
    parser.add_argument(
        "--slopes-output", type=Path, default=Path("fake_track_segment_slopes.png")
    )
    parser.add_argument(
        "--interactive-slopes-output",
        type=Path,
        default=Path("fake_track_slope_directions.html"),
    )
    parser.add_argument(
        "--composition",
        action="store_true",
        help="compare fake vs truth track properties",
    )
    parser.add_argument(
        "--comp-hits-output", type=Path, default=Path("fake_truth_hitcounts.png")
    )
    parser.add_argument(
        "--comp-kin1d-output", type=Path, default=Path("fake_truth_kinematics_1d.png")
    )
    parser.add_argument(
        "--comp-kin2d-output", type=Path, default=Path("fake_truth_kinematics_2d.png")
    )
    parser.add_argument(
        "--comp-segments-output",
        type=Path,
        default=Path("fake_truth_segment_slopes.png"),
    )
    parser.add_argument(
        "--ghost-output", type=Path, default=Path("ghost_rate_tracks.png")
    )
    parser.add_argument(
        "--momres-1d-output", type=Path, default=Path("momentum_resolution_1d.png")
    )
    parser.add_argument(
        "--momres-2d-output", type=Path, default=Path("momentum_resolution_2d.png")
    )
    parser.add_argument(
        "--momres-scatter-output",
        type=Path,
        default=Path("momentum_resolution_scatter.png"),
    )
    parser.add_argument(
        "--comp-pteta-output", type=Path, default=Path("fake_truth_pt_eta.png")
    )
    parser.add_argument(
        "--event-rate-output", type=Path, default=Path("event_fake_rate_vs_pvs.png")
    )
    parser.add_argument(
        "--event-tracks-output", type=Path, default=Path("event_tracks_vs_pvs.png")
    )
    parser.add_argument(
        "--event-only-fake-rate",
        action="store_true",
        help="only build the per-event fake-rate vs PV plot",
    )
    parser.add_argument("--no-plot", action="store_true", help="skip 3D plotting")
    parser.add_argument(
        "--fake-limit", type=int, default=8, help="number of fake tracks to print"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.event_only_fake_rate:
        sample = load_event_fake_pv_sample(args.root, limit=args.limit)
        output = plot_event_fake_rate_vs_pvs(sample, args.event_rate_output)
        tracks_output = plot_event_tracks_vs_pvs(sample, args.event_tracks_output)
        print(
            f"events={len(sample.event_number)} fake_rate_plot={output} tracks_plot={tracks_output} "
            f"mode=event-only limit={'all' if args.limit is None else args.limit}"
        )
        return

    show_progress = args.limit is None or args.limit > 1
    events = load_events(
        args.root, limit=args.limit, progress=show_progress, track_selection="fake"
    )
    event = events[0]
    all_fake_tracks = [
        track for event_record in events for track in fake_tracks(event_record)
    ]

    print(
        f"events={len(events)} first_event={event.event_number} run={event.run_number} bx={event.bunch_crossing_id} "
        f"fake_tracks={sum(len(event_record.tracks) for event_record in events)} "
        f"materialized_fake_only=yes"
    )

    print("\nFirst fake tracks:")
    for line in summarize_fake_tracks(events[0], limit=args.fake_limit):
        print("  ", line)

    fake = all_fake_tracks
    if fake:
        first_fake = fake[0]
        print("\nFirst fake track hits:")
        for detector, hits in first_fake.hits.items():
            print(f"  {detector}: {len(hits)} hits")
            for hit_index, hit in enumerate(hits):
                print(
                    f"    hit{hit_index:02d} "
                    f"reco=({hit.x:.3f}, {hit.y:.3f}, {hit.z:.3f}, {hit.t:.3f}) "
                    f"mc=({hit.mc_x:.3f}, {hit.mc_y:.3f}, {hit.mc_z:.3f}, {hit.mc_t:.3f})"
                )

        if not args.no_plot:
            output = plot_track_hits_3d(first_fake, args.plot_output)
            print(f"\nSaved 3D hit plot to {output}")

        dist_output = plot_fake_track_distributions(events, args.dist_output)
        pteta_output = plot_fake_track_pt_eta(events, args.pteta_output)
        slopes_output = plot_fake_track_segment_slopes(events, args.slopes_output)
        interactive_slopes_output = plot_fake_track_slope_directions(
            events, args.interactive_slopes_output
        )
        print(f"Saved fake-track distributions to {dist_output}")
        print(f"Saved fake-track pT-eta map to {pteta_output}")
        print(f"Saved fake-track segment slope plots to {slopes_output}")
        print(f"Saved interactive slope directions to {interactive_slopes_output}")

    if args.composition:
        composition_sample = load_track_composition_sample(args.root, limit=args.limit)
        event_sample = load_event_composition_sample(args.root, limit=args.limit)
        hits_output = plot_fake_truth_hitcount_distributions(
            composition_sample, args.comp_hits_output
        )
        kin1d_output = plot_fake_truth_kinematics_1d(
            composition_sample, args.comp_kin1d_output
        )
        kin2d_output = plot_fake_truth_kinematics_2d(
            composition_sample, args.comp_kin2d_output
        )
        segments_output = plot_fake_truth_segment_slopes(
            composition_sample, args.comp_segments_output
        )
        ghost_output = plot_ghost_rate(composition_sample, args.ghost_output)
        momres_1d_output = plot_momentum_resolution_1d(
            composition_sample, args.momres_1d_output
        )
        momres_2d_output = plot_momentum_resolution_2d(
            composition_sample, args.momres_2d_output
        )
        momres_scatter_output = plot_momentum_resolution_scatter(
            composition_sample, args.momres_scatter_output
        )
        event_rate_output = plot_event_fake_rate_vs_pvs(
            event_sample, args.event_rate_output
        )
        event_tracks_output = plot_event_tracks_vs_pvs(
            event_sample, args.event_tracks_output
        )
        comp_pteta_output = plot_fake_truth_pt_eta(
            composition_sample, args.comp_pteta_output
        )
        print(f"Saved fake/truth hit-count comparison to {hits_output}")
        print(f"Saved fake/truth kinematics comparison to {kin1d_output}")
        print(f"Saved fake/truth 2D kinematics comparison to {kin2d_output}")
        print(f"Saved fake/truth FT/MP segment slope comparison to {segments_output}")
        print(f"Saved ghost-rate plot to {ghost_output}")
        print(f"Saved momentum-resolution summary to {momres_1d_output}")
        print(f"Saved momentum-resolution truth-property checks to {momres_2d_output}")
        print(f"Saved momentum-resolution scatter to {momres_scatter_output}")
        print(f"Saved event fake-rate vs PVs to {event_rate_output}")
        print(f"Saved event tracks vs PVs to {event_tracks_output}")
        print(f"Saved fake/truth pT-eta comparison to {comp_pteta_output}")


if __name__ == "__main__":
    main()
