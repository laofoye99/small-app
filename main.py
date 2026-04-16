#!/usr/bin/env python3
"""
Tennis Analysis — entry point.

Runs stages 1-3 (detection → cleaning → sync) in memory and writes only the
final per-camera result files:  output/{label}_result.csv

Usage
-----
python main.py cam1.mp4 cam2.mp4 --output-dir output/
python main.py cam1.mp4 cam2.mp4 --parallel          # concurrent I/O
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import torch

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so package imports resolve correctly
# when the script is invoked from any working directory.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from detectors.wasb_detector import WASBDetector                # noqa: E402
from detectors.yolo_pose import YOLOPoseEstimator               # noqa: E402
from pipeline.processor import VideoProcessor                   # noqa: E402
from postprocess.cleaner_core import clean_df                   # noqa: E402
from scripts.sync_cameras import sync_dfs                       # noqa: E402
from scripts.report_api import (                                # noqa: E402
    report_camera, CAMERA_SERIALS, API_URL,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cam_label(video_path: str) -> str:
    """Extract camera label from video filename, e.g. 'cam66_video.mp4' → 'cam66'."""
    return Path(video_path).stem.partition('_')[0]


def _calib_path(video_path: str) -> str:
    """Convention: calibration JSON lives alongside the video as cal_{label}.json."""
    label = _cam_label(video_path)
    return str(Path(video_path).parent / f"cal_{label}.json")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Tennis Analysis Reasoning Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("videos", nargs="+",
                   help="Input MP4 file(s). Supports two camera perspectives.")
    p.add_argument("--wasb-weights", default="model_weights/wasb_tennis_best.pth.tar",
                   help="WASB tennis ball detector checkpoint (.pth.tar)")
    p.add_argument("--yolo-weights", default="model_weights/yolo26x-pose.pt",
                   help="YOLO pose estimator weights (.pt)")
    p.add_argument("--output-dir",   default="output",
                   help="Directory for result CSVs and annotated MP4s")
    p.add_argument("--ball-conf",    type=float, default=0.5,
                   help="Ball detection confidence threshold [0–1]")
    p.add_argument("--kp-conf",      type=float, default=0.3,
                   help="Keypoint visibility threshold for drawing and CSV [0–1]")
    p.add_argument("--max-players",  type=int,   default=2,
                   help="Maximum number of players to track per frame")
    p.add_argument("--device",       default="auto", choices=["auto", "cpu", "cuda"],
                   help="Inference device ('auto' picks CUDA when available)")
    p.add_argument("--workers",      type=int,   default=4,
                   help="ThreadPoolExecutor worker count for async I/O")
    p.add_argument("--parallel",     action="store_true",
                   help="Process all input videos concurrently via asyncio.gather")
    p.add_argument("--report-api",   action="store_true",
                   help="POST rally results to the API after pipeline completes")
    p.add_argument("--homography",   default="uploads/homography_matrices.json",
                   help="homography_matrices.json path (required for --report-api)")
    p.add_argument("--video-start",  default=None,
                   help="UTC ISO-8601 datetime of video frame 0 (default: now)")
    p.add_argument("--dry-run",      action="store_true",
                   help="With --report-api: build payloads but do not POST")
    return p


async def _run_all(
    args: argparse.Namespace,
    processor: VideoProcessor,
    output_dir: Path,
) -> None:
    executor = ThreadPoolExecutor(max_workers=args.workers)

    # ── Stage 1: Detection ───────────────────────────────────────────────────
    coros = []
    for vpath in args.videos:
        stem = Path(vpath).stem
        coros.append(processor.process_video_async(
            vpath,
            str(output_dir / f"{stem}_annotated.mp4"),
            executor,
        ))

    if args.parallel:
        stage1_results = await asyncio.gather(*coros)
    else:
        stage1_results = []
        for coro in coros:
            stage1_results.append(await coro)

    executor.shutdown(wait=True)

    # stage1_results: list of (frame_count, df_detections) per video

    # ── Stage 2: Cleaning ────────────────────────────────────────────────────
    cleaned: List[Tuple[str, str, object]] = []  # (stem, label, df_clean)
    for vpath, (_, df_det) in zip(args.videos, stage1_results):
        label = _cam_label(vpath)
        calib = _calib_path(vpath)
        logger.info("Cleaning detections for %s (calib: %s)", label, calib)
        df_clean, summary = clean_df(df_det, calib, label=label)
        logger.info(
            "%s: %d raw → %d outliers removed → %d final valid",
            label, summary['raw_detections'], summary['outliers'],
            summary['final_valid'],
        )
        cleaned.append((Path(vpath).stem, label, df_clean))

    # ── Stage 3: Sync + write result CSVs ───────────────────────────────────
    if len(cleaned) == 2:
        (_, label_a, df_a), (_, label_b, df_b) = cleaned
        logger.info("Synchronising %s ↔ %s", label_a, label_b)
        df_a_out, df_b_out, sync_info = sync_dfs(df_a, df_b, label_a, label_b)
        logger.info("Sync: tau=%d frames  SNR=%.2f",
                    sync_info['tau_offset'], sync_info['corr_snr'])

        for label, df_out in ((label_a, df_a_out), (label_b, df_b_out)):
            out_path = output_dir / f"{label}_result.csv"
            df_out.to_csv(out_path, index=False)
            logger.info("Result: %s", out_path)

    elif len(cleaned) == 1:
        # Single camera — no sync, write cleaned result directly
        _, label, df_clean = cleaned[0]
        out_path = output_dir / f"{label}_result.csv"
        df_clean.to_csv(out_path, index=False)
        logger.info("Result: %s", out_path)

    else:
        logger.warning("No videos processed.")
        return

    # ── Stage 4: API reporting (optional) ────────────────────────────────────
    if not args.report_api:
        return

    homog_path = Path(args.homography)
    if not homog_path.exists():
        logger.error("--report-api requires --homography: %s not found", homog_path)
        return

    with open(homog_path, 'r', encoding='utf-8') as fh:
        homography_matrices = json.load(fh)

    video_start: datetime
    if args.video_start:
        video_start = datetime.fromisoformat(
            args.video_start.rstrip('Z')
        ).replace(tzinfo=timezone.utc)
    else:
        video_start = datetime.now(tz=timezone.utc)
        logger.info("video_start not provided — using current UTC time: %s",
                    video_start.strftime('%Y-%m-%dT%H:%M:%SZ'))

    # Y-net pixel values per camera (mirrors run_all_cameras.py defaults)
    Y_NET = {'cam66': 238.0, 'cam68': 285.0}

    for vpath, (_, _df) in zip(args.videos, stage1_results):
        label    = _cam_label(vpath)
        serial   = CAMERA_SERIALS.get(label, f'UNKNOWN_{label}')
        csv_path = output_dir / f"{label}_result.csv"
        cal_path = Path(_calib_path(vpath))

        report_camera(
            label               = label,
            result_csv          = str(csv_path),
            calib_json_path     = str(cal_path),
            y_net               = Y_NET.get(label, 260.0),
            homography_matrices = homography_matrices,
            serial_number       = serial,
            video_start         = video_start,
            api_url             = API_URL,
            dry_run             = args.dry_run,
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _build_parser().parse_args()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    logger.info("Inference device: %s", device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading WASB tennis ball detector …")
    wasb = WASBDetector(args.wasb_weights, device)

    logger.info("Loading YOLO pose estimator …")
    yolo = YOLOPoseEstimator(args.yolo_weights, device)

    processor = VideoProcessor(
        wasb_detector=wasb,
        yolo_estimator=yolo,
        ball_conf_threshold=args.ball_conf,
        kp_conf_threshold=args.kp_conf,
        max_players=args.max_players,
    )

    asyncio.run(_run_all(args, processor, output_dir))


if __name__ == "__main__":
    main()