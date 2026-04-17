"""
live_pipeline.py
================
Continuous tennis analysis pipeline.

Source selection
----------------
At startup each camera is probed independently:
  1. Try to open the configured RTSP URL (5-second timeout).
  2. If unreachable, fall back to the default local video file.

Pass --local to skip the RTSP probe entirely.
Pass --cam66-url / --cam68-url to override the source for a specific camera.

Detection  (every frame, both cameras)
---------------------------------------
  Ball  : WASBDetector – sliding deque of 3 frames, runs every frame
  Pose  : YOLOPoseEstimator – runs every frame
  Two camera workers run in parallel threads; GPU inference is serialised by
  a shared lock so the GPU is never double-submitted.

Processing chain
-----------------
  row buffer → _rows_to_df → clean_df → sync_dfs → report_df (API)

Flush policy
-------------
  Local file  : single final flush after both workers reach EOF
  RTSP        : periodic flush – inactivity (10 s) or force every 5 min

Usage
-----
  python live_pipeline.py                          # auto-detect source
  python live_pipeline.py --local                  # force local files
  python live_pipeline.py --dry-run                # no API POST
  python live_pipeline.py --cam66-url path/to.mp4 --cam68-url path/to.mp4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from settings import (
    CAMERAS_CFG,
    HOMOGRAPHY_JSON,
    WASB_MODEL     as _DEFAULT_WASB,
    YOLO_MODEL     as _DEFAULT_YOLO,
    OUTPUT_DIR     as _DEFAULT_OUTPUT,
    CALIB_CAM66,
    CALIB_CAM68,
    MODEL_FPS,
    MODEL_DEVICE,
)
from detectors.wasb_detector import WASBDetector
from detectors.yolo_pose import YOLOPoseEstimator
from postprocess.cleaner_core import clean_df
from scripts.sync_cameras import sync_dfs
from scripts.report_api import API_URL, report_df as _report_df_api

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(name)s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('live_pipeline')

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_WASB_PATH  = str(_DEFAULT_WASB)
DEFAULT_YOLO_PATH  = str(_DEFAULT_YOLO)
DEFAULT_HOM_PATH   = str(HOMOGRAPHY_JSON)
DEFAULT_DEVICE     = MODEL_DEVICE
DEFAULT_FPS        = MODEL_FPS
DEFAULT_OUTPUT_DIR = str(_DEFAULT_OUTPUT)

_CAM_CFG: Dict[str, dict] = {
    'cam66': {
        'rtsp_url':       CAMERAS_CFG.get('cam66', {}).get(
                              'rtsp_url',
                              'rtsp://admin:motion168@192.168.1.66:554/Streaming/Channels/101'),
        'local_fallback': str(_HERE / 'uploads' / 'cam66_video.mp4'),
        'calib_json':     str(CALIB_CAM66),
        'y_net':          float(CAMERAS_CFG.get('cam66', {}).get('y_net', 238.0)),
        'serial':         CAMERAS_CFG.get('cam66', {}).get('serial', 'FV9942593'),
        'homography_key': CAMERAS_CFG.get('cam66', {}).get('homography_key', 'cam66'),
    },
    'cam68': {
        'rtsp_url':       CAMERAS_CFG.get('cam68', {}).get(
                              'rtsp_url',
                              'rtsp://admin:motion168@192.168.1.68:554/Streaming/Channels/101'),
        'local_fallback': str(_HERE / 'uploads' / 'cam68_video.mp4'),
        'calib_json':     str(CALIB_CAM68),
        'y_net':          float(CAMERAS_CFG.get('cam68', {}).get('y_net', 285.0)),
        'serial':         CAMERAS_CFG.get('cam68', {}).get('serial', 'FV9942588'),
        'homography_key': CAMERAS_CFG.get('cam68', {}).get('homography_key', 'cam68'),
    },
}

# Flush / buffer settings
INACTIVITY_SEC  = 10.0
FORCE_FLUSH_SEC = 300.0
MIN_FLUSH_ROWS  = 200
OVERLAP_SEC     = 30.0
RECONNECT_DELAY = 3.0
RTSP_TIMEOUT_MS = 5_000

# ---------------------------------------------------------------------------
# Joint definitions
# ---------------------------------------------------------------------------

_JOINT_MAP = [
    ('ls', 'left_shoulder'),
    ('rs', 'right_shoulder'),
    ('le', 'left_elbow'),
    ('re', 'right_elbow'),
    ('lw', 'left_wrist'),
    ('rw', 'right_wrist'),
]

_ALL_PLAYER_COLS = (
    [f'p{p}_{j}_{c}' for p in range(2) for j, _ in _JOINT_MAP for c in ('u', 'v')]
    + [f'p{p}_{c}' for p in range(2) for c in ('bx1', 'by1', 'bx2', 'by2')]
)

# ---------------------------------------------------------------------------
# Source probing
# ---------------------------------------------------------------------------

def _probe_rtsp(url: str) -> bool:
    """Return True if the RTSP stream is reachable within RTSP_TIMEOUT_MS."""
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, RTSP_TIMEOUT_MS)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, RTSP_TIMEOUT_MS)
    ok = cap.isOpened()
    cap.release()
    return ok


def _resolve_source(
    cam_name: str,
    override: Optional[str],
    force_local: bool,
) -> tuple[str, bool]:
    """
    Returns (url, is_rtsp).

    Priority:
      1. --cam66-url / --cam68-url override → use as-is (RTSP if starts with rtsp://)
      2. --local flag → skip probe, use local_fallback
      3. Auto: probe RTSP; on failure use local_fallback
    """
    cfg = _CAM_CFG[cam_name]

    if override is not None:
        is_rtsp = override.lower().startswith('rtsp')
        logger.info("[%s] Using override source: %s (rtsp=%s)", cam_name, override, is_rtsp)
        return override, is_rtsp

    if not force_local:
        rtsp = cfg['rtsp_url']
        logger.info("[%s] Probing RTSP %s …", cam_name, rtsp)
        if _probe_rtsp(rtsp):
            logger.info("[%s] RTSP reachable → live stream", cam_name)
            return rtsp, True
        logger.warning("[%s] RTSP unreachable → local fallback", cam_name)

    local = cfg['local_fallback']
    if not Path(local).exists():
        raise FileNotFoundError(
            f"[{cam_name}] Local fallback not found: {local}"
        )
    logger.info("[%s] Using local file: %s", cam_name, local)
    return local, False

# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------

def _make_row(
    cam_name:  str,
    frame_id:  int,
    detected:  bool,
    ball_u:    Optional[float],
    ball_v:    Optional[float],
    ball_conf: float,
    kp_sets:   list,
) -> dict:
    row: dict = {
        'camera_name': cam_name,
        'frame_id':    frame_id,
        'detected':    int(detected),
        'ball_u':      round(ball_u,    2) if ball_u    is not None else None,
        'ball_v':      round(ball_v,    2) if ball_v    is not None else None,
        'ball_conf':   round(ball_conf, 4),
    }
    for p_idx, kp in enumerate(kp_sets[:2]):
        for j_abbr, j_attr in _JOINT_MAP:
            joint = getattr(kp, j_attr, None)
            if joint is not None:
                row[f'p{p_idx}_{j_abbr}_u']    = round(float(joint[0]), 2)
                row[f'p{p_idx}_{j_abbr}_v']    = round(float(joint[1]), 2)
                row[f'p{p_idx}_{j_abbr}_conf'] = round(float(joint[2]), 4)
            else:
                row[f'p{p_idx}_{j_abbr}_u']    = None
                row[f'p{p_idx}_{j_abbr}_v']    = None
                row[f'p{p_idx}_{j_abbr}_conf'] = None
        bbox = getattr(kp, 'bbox', None)
        if bbox is not None:
            row[f'p{p_idx}_bx1'] = round(float(bbox[0]), 1)
            row[f'p{p_idx}_by1'] = round(float(bbox[1]), 1)
            row[f'p{p_idx}_bx2'] = round(float(bbox[2]), 1)
            row[f'p{p_idx}_by2'] = round(float(bbox[3]), 1)
        else:
            for c in ('bx1', 'by1', 'bx2', 'by2'):
                row[f'p{p_idx}_{c}'] = None
    return row

# ---------------------------------------------------------------------------
# DataFrame builder
# ---------------------------------------------------------------------------

def _rows_to_df(rows: List[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=['frame_id', 'detected', 'ball_u', 'ball_v', 'ball_conf']
        )

    df = (
        pd.DataFrame(rows)
        .sort_values('frame_id')
        .drop_duplicates('frame_id', keep='last')
        .reset_index(drop=True)
    )

    ball_cols   = ['frame_id', 'detected', 'ball_u', 'ball_v', 'ball_conf']
    player_cols = [c for c in _ALL_PLAYER_COLS if c in df.columns]

    min_fid = int(df['frame_id'].min())
    max_fid = int(df['frame_id'].max())
    full = pd.DataFrame({'frame_id': pd.RangeIndex(min_fid, max_fid + 1)})
    full = full.merge(df[ball_cols + player_cols], on='frame_id', how='left')
    full['detected']  = full['detected'].fillna(0).astype(int)
    full['ball_conf'] = full['ball_conf'].fillna(0.0)

    for col in player_cols:
        if col in full.columns:
            full[col] = full[col].ffill(limit=5).bfill(limit=5)

    return full.reset_index(drop=True)

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class LivePipeline:
    """
    Two-camera tennis analysis pipeline.

    Each camera runs in its own worker thread that reads frames sequentially
    and performs WASB + YOLO inference.  GPU inference is serialised by a
    shared lock so both threads share one GPU without contention.

    For RTSP sources the flush loop triggers periodic clean → sync → report
    cycles.  For local files a single final flush runs after both workers
    reach EOF.
    """

    def __init__(
        self,
        cam66_url:           Optional[str] = None,
        cam68_url:           Optional[str] = None,
        wasb_weights:        str   = DEFAULT_WASB_PATH,
        yolo_weights:        str   = DEFAULT_YOLO_PATH,
        homography_path:     str   = DEFAULT_HOM_PATH,
        device:              str   = DEFAULT_DEVICE,
        fps:                 float = DEFAULT_FPS,
        dry_run:             bool  = False,
        api_url:             str   = API_URL,
        ball_conf_threshold: float = 0.5,
        kp_conf_threshold:   float = 0.3,
        max_players:         int   = 2,
        output_dir:          str   = DEFAULT_OUTPUT_DIR,
        force_local:         bool  = False,
    ) -> None:
        self.fps        = fps
        self.dry_run    = dry_run
        self.api_url    = api_url
        self.ball_thr   = ball_conf_threshold
        self.kp_thr     = kp_conf_threshold
        self.max_players = max_players
        self.output_dir = output_dir

        # Device
        if device in ('cuda', 'auto'):
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if device == 'cuda' and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available – using CPU")
        else:
            self._device = torch.device('cpu')
        logger.info("Inference device: %s", self._device)

        # Models (shared, one instance each)
        logger.info("Loading WASB detector …")
        self._wasb = WASBDetector(wasb_weights, self._device)

        logger.info("Loading YOLO pose estimator …")
        self._yolo = YOLOPoseEstimator(yolo_weights, self._device)

        # Shared inference lock – serialises GPU access across both workers
        self._infer_lock = threading.Lock()

        # Resolve video sources
        overrides = {'cam66': cam66_url, 'cam68': cam68_url}
        self._sources: Dict[str, tuple[str, bool]] = {}
        for cam_name in _CAM_CFG:
            url, is_rtsp = _resolve_source(cam_name, overrides[cam_name], force_local)
            self._sources[cam_name] = (url, is_rtsp)

        self._is_local = all(not is_rtsp for _, is_rtsp in self._sources.values())
        logger.info(
            "Sources: %s",
            {n: ('rtsp' if r else 'local') for n, (_, r) in self._sources.items()},
        )

        # Homography
        hom_p = Path(homography_path)
        if not hom_p.exists():
            raise FileNotFoundError(f"Homography JSON not found: {hom_p}")
        with open(hom_p, 'r', encoding='utf-8') as fh:
            self._homography = json.load(fh)

        # Per-camera row buffers
        self._buffers:     Dict[str, List[dict]] = {k: [] for k in _CAM_CFG}
        self._last_det_ts: Dict[str, float]      = {k: 0.0 for k in _CAM_CFG}
        self._buf_lock = threading.Lock()

        self._session_start  = datetime.now(tz=timezone.utc)
        self._last_flush_ts  = time.time()
        self._flush_count    = 0
        self._stop           = threading.Event()

    # ------------------------------------------------------------------
    # Camera worker thread
    # ------------------------------------------------------------------

    def _camera_worker(self, cam_name: str) -> None:
        url, is_rtsp = self._sources[cam_name]
        log          = logging.getLogger(f'live_pipeline.{cam_name}')
        ball_buffer  = self._wasb.make_buffer()
        frame_id     = 0

        # Non-overlapping batch accumulators
        # Accumulate exactly frames_in (3) frames, then run WASB once,
        # emit 3 rows, and clear – avoiding per-frame inference overhead.
        batch_frames: list[tuple[int, np.ndarray, int, int]] = []  # (fid, frame, h, w)
        batch_yolo:   list[list]                              = []  # YOLO results per frame

        while not self._stop.is_set():
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if is_rtsp:
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10_000)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10_000)

            if not cap.isOpened():
                cap.release()
                if not is_rtsp:
                    log.error("Cannot open local file: %s", url)
                    break
                log.warning("Cannot open RTSP %s – retrying in %.0fs", url, RECONNECT_DELAY)
                self._stop.wait(RECONNECT_DELAY)
                continue

            log.info("Opened: %s", url)

            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    log.info("%s – end of stream", url)
                    break

                # Mask camera OSD timestamp overlay
                frame[0:41, 0:603] = 0

                h, w = frame.shape[:2]

                # YOLO pose runs every frame (under lock)
                with self._infer_lock:
                    kp_sets = self._yolo.estimate(frame, self.max_players)

                # Push frame and YOLO result into accumulators
                self._wasb.push_frame(frame, ball_buffer)
                batch_frames.append((frame_id, frame, h, w))
                batch_yolo.append(kp_sets)

                # When 3 frames are accumulated, run WASB once for all 3
                if len(batch_frames) == self._wasb.frames_in:
                    _, _, bh, bw = batch_frames[0]
                    with self._infer_lock:
                        ball_results = self._wasb.detect_batch(
                            ball_buffer, bh, bw, self.ball_thr
                        )

                    now_ts = time.time()
                    any_detected = False
                    rows_to_add: list[dict] = []

                    for (fid, _, _, _), (detected, bu, bv, bconf), kps in zip(
                        batch_frames, ball_results, batch_yolo
                    ):
                        row = _make_row(
                            cam_name, fid,
                            detected,
                            bu if detected else None,
                            bv if detected else None,
                            bconf,
                            kps,
                        )
                        rows_to_add.append(row)
                        if detected:
                            any_detected = True

                    with self._buf_lock:
                        self._buffers[cam_name].extend(rows_to_add)
                        if any_detected:
                            self._last_det_ts[cam_name] = now_ts

                    ball_buffer.clear()
                    batch_frames.clear()
                    batch_yolo.clear()

                frame_id += 1
                if frame_id % 500 == 0:
                    log.info("%d frames processed", frame_id)

            cap.release()

            if not is_rtsp:
                break   # local file done

            if not self._stop.is_set():
                log.info("Reconnecting in %.0fs …", RECONNECT_DELAY)
                self._stop.wait(RECONNECT_DELAY)

        # Flush any partial batch at EOF (< 3 frames remaining)
        if batch_frames and ball_buffer:
            _, _, bh, bw = batch_frames[0]
            try:
                with self._infer_lock:
                    ball_results = self._wasb.detect_batch(
                        ball_buffer, bh, bw, self.ball_thr
                    )
                rows_to_add = []
                any_detected = False
                for (fid, _, _, _), (detected, bu, bv, bconf), kps in zip(
                    batch_frames, ball_results, batch_yolo
                ):
                    row = _make_row(
                        cam_name, fid,
                        detected,
                        bu if detected else None,
                        bv if detected else None,
                        bconf,
                        kps,
                    )
                    rows_to_add.append(row)
                    if detected:
                        any_detected = True
                with self._buf_lock:
                    self._buffers[cam_name].extend(rows_to_add)
                    if any_detected:
                        self._last_det_ts[cam_name] = time.time()
            except Exception as exc:
                log.warning("Partial batch flush failed: %s", exc)

        log.info("Worker finished  total_frames=%d", frame_id)

    # ------------------------------------------------------------------
    # Flush loop  (RTSP mode only)
    # ------------------------------------------------------------------

    def _flush_loop(self) -> None:
        while not self._stop.is_set():
            time.sleep(2.0)
            now = time.time()

            with self._buf_lock:
                buf_sizes = {n: len(b) for n, b in self._buffers.items()}
                last_dets = dict(self._last_det_ts)

            if min(buf_sizes.values(), default=0) < MIN_FLUSH_ROWS:
                continue

            active    = [n for n, t in last_dets.items() if t > 0]
            all_quiet = bool(active) and all(
                now - last_dets[n] >= INACTIVITY_SEC for n in active
            )
            force = (now - self._last_flush_ts) >= FORCE_FLUSH_SEC

            if all_quiet or force:
                reason = 'inactivity' if all_quiet else 'force'
                logger.info("Flush triggered (%s)  buf=%s", reason, buf_sizes)
                try:
                    self._do_flush(final=False)
                except Exception as exc:
                    logger.error("Flush error (continuing): %s", exc, exc_info=True)
                self._last_flush_ts = now

    # ------------------------------------------------------------------
    # Core flush: rows → clean → sync → report
    # ------------------------------------------------------------------

    def _do_flush(self, final: bool = False) -> None:
        overlap_rows = 0 if final else int(self.fps * OVERLAP_SEC)

        with self._buf_lock:
            snapshots: Dict[str, List[dict]] = {}
            for name in _CAM_CFG:
                buf = self._buffers[name]
                snapshots[name] = list(buf)
                self._buffers[name] = (
                    [] if final
                    else (buf[-overlap_rows:] if len(buf) > overlap_rows else list(buf))
                )

        self._flush_count += 1
        fid = self._flush_count
        row_counts = {n: len(r) for n, r in snapshots.items()}
        logger.info("=== Flush #%d  final=%s  rows=%s ===", fid, final, row_counts)

        # Stage 1: rows → DataFrames
        dfs: Dict[str, pd.DataFrame] = {}
        for name, rows in snapshots.items():
            if not rows:
                logger.warning("[flush#%d] %s: no rows – skipping", fid, name)
                continue
            dfs[name] = _rows_to_df(rows)
            logger.info(
                "[flush#%d] %s: %d rows  frame %d–%d",
                fid, name, len(dfs[name]),
                int(dfs[name]['frame_id'].min()),
                int(dfs[name]['frame_id'].max()),
            )

        if len(dfs) < 2:
            logger.warning("[flush#%d] Need both cameras – skipping flush", fid)
            return

        # Stage 2: clean each camera
        cleaned: Dict[str, pd.DataFrame] = {}
        for name, df_raw in dfs.items():
            cfg = _CAM_CFG[name]
            try:
                df_clean, summary = clean_df(
                    df_raw,
                    calib_path=cfg['calib_json'],
                    y_net=cfg['y_net'],
                    label=name,
                )
                cleaned[name] = df_clean
                logger.info(
                    "[flush#%d] %s clean: raw=%d  outliers=%d  interp=%d",
                    fid, name,
                    summary['raw_detections'],
                    summary['outliers'],
                    summary['interpolated'],
                )
            except Exception as exc:
                logger.error("[flush#%d] clean_df failed for %s: %s",
                             fid, name, exc, exc_info=True)
                return

        # Stage 3: sync cameras
        try:
            df66, df68, sync_res = sync_dfs(
                cleaned['cam66'], cleaned['cam68'],
                label_a='cam66', label_b='cam68',
            )
            logger.info("[flush#%d] sync: tau=%d fr  SNR=%.2f",
                        fid, sync_res['tau_offset'], sync_res['corr_snr'])
        except Exception as exc:
            logger.error("[flush#%d] sync_dfs failed: %s", fid, exc, exc_info=True)
            return

        # Stage 4: rally detection + API report
        for name, df_synced in (('cam66', df66), ('cam68', df68)):
            cfg = _CAM_CFG[name]
            try:
                sent = _report_df_api(
                    label               = name,
                    df                  = df_synced,
                    calib_json_path     = cfg['calib_json'],
                    y_net               = cfg['y_net'],
                    homography_matrices = self._homography,
                    serial_number       = cfg['serial'],
                    video_start         = self._session_start,
                    fps                 = self.fps,
                    api_url             = self.api_url,
                    dry_run             = self.dry_run,
                )
                logger.info("[flush#%d] %s → %d payloads sent", fid, name, sent)
            except Exception as exc:
                logger.error("[flush#%d] report failed for %s: %s",
                             fid, name, exc, exc_info=True)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        logger.info(
            "Pipeline starting  local=%s  dry_run=%s  device=%s",
            self._is_local, self.dry_run, self._device,
        )

        # One worker thread per camera
        workers = [
            threading.Thread(
                target=self._camera_worker,
                name=f'worker_{name}',
                args=(name,),
                daemon=True,
            )
            for name in _CAM_CFG
        ]
        for t in workers:
            t.start()

        # Flush loop thread for RTSP mode
        if not self._is_local:
            threading.Thread(
                target=self._flush_loop,
                name='flush_loop',
                daemon=True,
            ).start()

        try:
            while any(t.is_alive() for t in workers):
                time.sleep(2.0)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt – shutting down")
            self._stop.set()
            for t in workers:
                t.join(timeout=10)

        # Final flush after all workers finish
        logger.info("All workers finished – running final flush")
        try:
            self._do_flush(final=True)
        except Exception as exc:
            logger.error("Final flush error: %s", exc, exc_info=True)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Tennis analysis pipeline – RTSP live or local file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--dry-run',     action='store_true',
                   help='Print API payloads, do not POST')
    p.add_argument('--local',       action='store_true',
                   help='Skip RTSP probe, use local video files directly')
    p.add_argument('--cam66-url',   default=None, metavar='URL',
                   help='Override cam66 source (RTSP URL or file path)')
    p.add_argument('--cam68-url',   default=None, metavar='URL',
                   help='Override cam68 source (RTSP URL or file path)')
    p.add_argument('--wasb-model',  default=DEFAULT_WASB_PATH, metavar='PATH')
    p.add_argument('--yolo-model',  default=DEFAULT_YOLO_PATH, metavar='PATH')
    p.add_argument('--homography',  default=DEFAULT_HOM_PATH,  metavar='PATH')
    p.add_argument('--device',      default=DEFAULT_DEVICE,
                   choices=['auto', 'cuda', 'cpu'])
    p.add_argument('--fps',         type=float, default=DEFAULT_FPS)
    p.add_argument('--ball-conf',   type=float, default=0.5,
                   help='WASB detection confidence threshold')
    p.add_argument('--kp-conf',     type=float, default=0.3,
                   help='YOLO keypoint visibility threshold')
    p.add_argument('--api-url',     default=API_URL)
    p.add_argument('--output-dir',  default=DEFAULT_OUTPUT_DIR)
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.dry_run:
        logger.info("*** DRY-RUN MODE – payloads printed, not POSTed ***")

    pipeline = LivePipeline(
        cam66_url           = args.cam66_url,
        cam68_url           = args.cam68_url,
        wasb_weights        = args.wasb_model,
        yolo_weights        = args.yolo_model,
        homography_path     = args.homography,
        device              = args.device,
        fps                 = args.fps,
        dry_run             = args.dry_run,
        api_url             = args.api_url,
        ball_conf_threshold = args.ball_conf,
        kp_conf_threshold   = args.kp_conf,
        output_dir          = args.output_dir,
        force_local         = args.local,
    )
    pipeline.run()


if __name__ == '__main__':
    main()
