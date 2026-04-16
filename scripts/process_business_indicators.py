"""
process_business_indicators.py
================================
Stage 2 of the Tennis Analysis Pipeline.

Reads cleaned ball + player trajectories, runs rally segmentation and near-field
hit detection, overlays indicators onto the video, and saves structured results.

Outputs per camera (in --output-dir):
  cam66_annotated.mp4       — annotated video (panel top-right, pose skeleton)
  cam66_rally_stats.csv     — per-rally summary (shots, total player distance)
  cam66_hit_events.csv      — per-hit details (frame, time, speed, rally)
  cam66_summary.json        — overall session summary
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from analysis_module import TennisRallySegmenter, TennisVisualizer

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def _load_json(path: str | None, label: str) -> dict:
    """Load a JSON file; return empty dict on failure."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        logging.warning("[%s] File not found: %s", label, path)
        return {}
    with open(p, 'r') as fh:
        return json.load(fh)


def process_camera(
    label: str,
    video_path: str,
    csv_path: str,
    homography_matrices: dict,
    cal: dict,
    output_dir: Path,
) -> None:
    logging.info("── Processing camera %s ──", label)

    if not Path(csv_path).exists():
        logging.error("CSV not found: %s", csv_path)
        return
    if not Path(video_path).exists():
        logging.error("Video not found: %s", video_path)
        return
    if label not in homography_matrices:
        logging.error("Homography not found for camera '%s'", label)
        return

    H_i2w = homography_matrices[label]['H_image_to_world']
    df     = pd.read_csv(csv_path)

    segmenter  = TennisRallySegmenter()
    visualizer = TennisVisualizer(segmenter, H_i2w, cal=cal)

    output_video = output_dir / f"{label}_annotated.mp4"

    rallies, hit_events, rally_stats = visualizer.process_video(
        video_path=str(video_path),
        trajectory_df=df,
        output_path=str(output_video),
    )

    # ── Rally stats CSV ───────────────────────────────────────────────
    if rallies:
        rally_rows = []
        for r in rallies:
            stats = rally_stats.get(r['rally_id'], {})
            rally_rows.append({
                'rally_id':      r['rally_id'],
                'start_frame':   r['start_frame'],
                'end_frame':     r['end_frame'],
                'rally_start_s': r['rally_start'],
                'rally_end_s':   r['rally_end'],
                'duration_s':    round(r['rally_end'] - r['rally_start'], 3),
                'shots':         stats.get('shots', 0),
                'dist_m':        stats.get('dist_m', 0.0),
            })
        rally_csv = output_dir / f"{label}_rally_stats.csv"
        pd.DataFrame(rally_rows).to_csv(rally_csv, index=False)
        logging.info("Rally stats → %s", rally_csv)

    # ── Hit events CSV ────────────────────────────────────────────────
    if hit_events:
        hits_csv = output_dir / f"{label}_hit_events.csv"
        pd.DataFrame(hit_events).to_csv(hits_csv, index=False)
        logging.info("Hit events  → %s", hits_csv)

    # ── Summary JSON ──────────────────────────────────────────────────
    summary = {
        'camera':             label,
        'total_rallies':      len(rallies),
        'total_hits':         len(hit_events),
        'avg_hits_per_rally': (
            round(len(hit_events) / len(rallies), 2) if rallies else 0
        ),
        'total_dist_m':       round(
            sum(s.get('dist_m', 0) for s in rally_stats.values()), 2
        ),
        'avg_speed_kmh':      (
            round(sum(e['speed_kmh'] for e in hit_events) / len(hit_events), 1)
            if hit_events else 0
        ),
        'y_net_px':           cal.get('y_net'),
    }
    summary_json = output_dir / f"{label}_summary.json"
    with open(summary_json, 'w') as fh:
        json.dump(summary, fh, indent=2)
    logging.info("Summary     → %s", summary_json)
    logging.info("  %s", summary)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process tennis business indicators and annotate video."
    )
    parser.add_argument("--cam66-video",   required=True)
    parser.add_argument("--cam68-video",   required=True)
    parser.add_argument("--cam66-csv",     required=True)
    parser.add_argument("--cam68-csv",     required=True)
    parser.add_argument("--homography-json", required=True)
    parser.add_argument("--cam66-calib",   default=None,
                        help="cal_cam66.json (y_near, y_far, fps, …)")
    parser.add_argument("--cam68-calib",   default=None,
                        help="cal_cam68.json")
    parser.add_argument("--cam66-y-net",   type=float, default=238.0,
                        help="Pixel Y of net in cam66 (default 238)")
    parser.add_argument("--cam68-y-net",   type=float, default=285.0,
                        help="Pixel Y of net in cam68 (default 285)")
    parser.add_argument("--output-dir",    required=True)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    homography_matrices = _load_json(args.homography_json, 'homography')
    if not homography_matrices:
        logging.error("Homography JSON could not be loaded. Aborting.")
        sys.exit(1)

    # Load per-camera calibration and inject net pixel position
    cal66 = _load_json(args.cam66_calib, 'cal_cam66')
    cal66['y_net'] = args.cam66_y_net

    cal68 = _load_json(args.cam68_calib, 'cal_cam68')
    cal68['y_net'] = args.cam68_y_net

    process_camera(
        label='cam66',
        video_path=args.cam66_video,
        csv_path=args.cam66_csv,
        homography_matrices=homography_matrices,
        cal=cal66,
        output_dir=output_dir,
    )

    process_camera(
        label='cam68',
        video_path=args.cam68_video,
        csv_path=args.cam68_csv,
        homography_matrices=homography_matrices,
        cal=cal68,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
