"""
live_pipeline.py
================
Real-time RTSP → TrackNet → clean → sync → rally → API pipeline.

Two camera subprocesses run continuous TrackNet inference (from the
tennis-3d-tracking library) and push detection rows into per-camera queues.
The main process accumulates rows, then periodically flushes them through:

    clean_df()  →  sync_dfs()  →  detect rallies  →  POST to API

Flush triggers (whichever comes first):
  1. Inactivity: no detections from EITHER camera for INACTIVITY_SEC.
  2. Force flush: every FORCE_FLUSH_SEC even if cameras are still active.

Both triggers respect a minimum buffer size so a 3-second pause right after
startup doesn't cause a premature, near-empty flush.

Usage
-----
    # Dry run — build payloads and print to stdout, no HTTP POST:
    python live_pipeline.py --dry-run

    # Production run:
    python live_pipeline.py

    # Override RTSP streams (e.g. for local replay tests):
    python live_pipeline.py \\
        --cam66-url rtsp://localhost:8554/cam66 \\
        --cam68-url rtsp://localhost:8554/cam68

    # Use CPU inference (no CUDA):
    python live_pipeline.py --device cpu
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Settings (all paths computed from config.yaml / auto-detect)
# ---------------------------------------------------------------------------

# Ensure small-app root is in path so all sibling modules import cleanly.
_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from settings import (
    HOME, CAMERAS_CFG, HOMOGRAPHY_JSON,
    TRACKNET_MODEL as _DEFAULT_MODEL,
    MODEL_DEVICE, MODEL_FPS, MODEL_FRAMES_IN, MODEL_FRAMES_OUT, MODEL_THRESHOLD,
)
from postprocess.cleaner_core import clean_df
from scripts.sync_cameras import sync_dfs
from scripts.report_api import (
    API_URL,
    report_df as _report_df_api,
)
from scripts.analysis_module import TennisRallySegmenter, TennisVisualizer

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
# Default configuration  (values come from settings.py / config.yaml)
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = str(_DEFAULT_MODEL)
DEFAULT_HOM_PATH   = str(HOMOGRAPHY_JSON)
DEFAULT_DEVICE     = MODEL_DEVICE
DEFAULT_FPS        = MODEL_FPS
FRAMES_IN          = MODEL_FRAMES_IN
FRAMES_OUT         = MODEL_FRAMES_OUT
INPUT_SIZE         = (288, 512)   # (H, W)
THRESHOLD          = MODEL_THRESHOLD

# RTSP URLs and per-camera metadata — pulled from settings / config.yaml
_CAM66_RTSP = CAMERAS_CFG.get('cam66', {}).get(
    'rtsp_url', 'rtsp://admin:motion168@192.168.1.66:554/Streaming/Channels/101'
)
_CAM68_RTSP = CAMERAS_CFG.get('cam68', {}).get(
    'rtsp_url', 'rtsp://admin:motion168@192.168.1.68:554/Streaming/Channels/101'
)

# Camera metadata: merge settings config with local calib paths
from settings import CALIB_CAM66, CALIB_CAM68
_CAM_DEFAULTS = {
    'cam66': {
        'homography_key': CAMERAS_CFG.get('cam66', {}).get('homography_key', 'cam66'),
        'calib_json':     str(CALIB_CAM66),
        'y_net':          float(CAMERAS_CFG.get('cam66', {}).get('y_net', 238.0)),
    },
    'cam68': {
        'homography_key': CAMERAS_CFG.get('cam68', {}).get('homography_key', 'cam68'),
        'calib_json':     str(CALIB_CAM68),
        'y_net':          float(CAMERAS_CFG.get('cam68', {}).get('y_net', 285.0)),
    },
}

# Flush policy
INACTIVITY_SEC    = 10.0    # seconds of silence before triggering flush
FORCE_FLUSH_SEC   = 300.0   # force flush every 5 minutes regardless of activity
MIN_FLUSH_ROWS    = 200     # minimum accumulated rows before any flush (~8 s at 25 fps)
OVERLAP_SEC       = 30.0    # seconds of tail kept in buffer after each flush (boundary guard)

# Buffer cap (memory guard: ~10 minutes per camera)
MAX_BUFFER_ROWS = int(DEFAULT_FPS * 600)

# ---------------------------------------------------------------------------
# Camera subprocess entry point
# ---------------------------------------------------------------------------

def _camera_worker(
    name:            str,
    rtsp_url:        str,
    model_path:      str,
    homography_path: str,
    homography_key:  str,
    app_root:        str,   # small-app root dir, added to sys.path in subprocess
    result_queue:    mp.Queue,
    stop_event:      mp.Event,
    status_dict:     dict,
    session_start_ts: float,
    fps:             float = DEFAULT_FPS,
    frames_in:       int   = FRAMES_IN,
    frames_out:      int   = FRAMES_OUT,
    device:          str   = DEFAULT_DEVICE,
    input_size:      tuple = INPUT_SIZE,
    threshold:       float = THRESHOLD,
) -> None:
    """
    Subprocess entry point: RTSP camera → TrackNet inference → detection rows.

    Each detected frame is put into *result_queue* as a dict:
        {
            'camera_name': str,
            'frame_id':    int,   # derived from timestamp at session_start_ts
            'detected':    1,
            'ball_u':      float, # pixel X
            'ball_v':      float, # pixel Y
            'ball_conf':   float,
            'capture_ts':  float,
        }

    Runs until *stop_event* is set.
    """
    # Re-add small-app root to path (Windows spawn starts a fresh interpreter)
    if app_root not in sys.path:
        sys.path.insert(0, app_root)

    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s  [%(name)s]  %(levelname)s  %(message)s',
        datefmt='%H:%M:%S',
    )
    log = logging.getLogger(name)

    try:
        from tracking.camera_stream import CameraStream
        from tracking.inference import create_detector
        from tracking.postprocess import BallTracker
    except ImportError as exc:
        log.error("Import failed (tracking package not found in %s): %s", tracking_dir, exc)
        status_dict['state'] = 'error'
        status_dict['error_msg'] = str(exc)
        return

    log.info("Starting  rtsp=%s  model=%s  device=%s", rtsp_url, model_path, device)
    status_dict['state'] = 'starting'

    stream   = CameraStream(rtsp_url, name)
    stream.start()

    detector = None
    tracker  = None
    try:
        detector = create_detector(
            model_path, input_size, frames_in, frames_out, device,
            detector_type='auto',
        )
        if not getattr(detector, 'returns_blobs', False):
            tracker = BallTracker(original_size=(1920, 1080), threshold=threshold)
        log.info("Inference ready (device=%s)", device)
    except Exception as exc:
        log.warning("Model load failed — inference disabled: %s", exc)

    status_dict['state'] = 'running'

    frame_buffer:    list = []
    ts_buffer:       list = []
    last_frame_id:   int  = -1

    while not stop_event.is_set():
        frame, frame_id, _ts = stream.read()
        if frame is None or frame_id == last_frame_id:
            time.sleep(0.002)
            continue
        last_frame_id  = frame_id
        capture_ts     = time.time()

        frame = frame.copy()
        # Mask camera OSD overlay (same as tennis-3d-tracking)
        frame[0:41, 0:603] = 0

        frame_buffer.append(frame)
        ts_buffer.append(capture_ts)

        if len(frame_buffer) < frames_in:
            continue

        if detector is None:
            frame_buffer.clear()
            ts_buffer.clear()
            continue

        try:
            heatmaps = detector.infer(frame_buffer)
        except Exception as exc:
            log.error("Inference error: %s", exc)
            frame_buffer.clear()
            ts_buffer.clear()
            continue

        if getattr(detector, 'returns_blobs', False):
            # MedianBGDetector path — not used by default (TrackNet is default)
            frame_buffer.clear()
            ts_buffer.clear()
            continue

        # TrackNet / HRNet path
        for i in range(min(frames_out, len(heatmaps))):
            blobs = tracker.process_heatmap_multi(heatmaps[i], max_blobs=1)
            if not blobs:
                continue

            top       = blobs[0]
            px        = top['pixel_x']
            py        = top['pixel_y']
            conf      = top['blob_sum']
            c_ts      = ts_buffer[0] if ts_buffer else capture_ts

            # Derive a monotonic frame_id from elapsed time since session start
            derived_fid = round((c_ts - session_start_ts) * fps)

            row = {
                'camera_name': name,
                'frame_id':    derived_fid,
                'detected':    1,
                'ball_u':      round(float(px),   2),
                'ball_v':      round(float(py),   2),
                'ball_conf':   round(float(conf), 4),
                'capture_ts':  c_ts,
            }
            try:
                result_queue.put_nowait(row)
            except Exception:
                pass  # queue full — drop frame

        frame_buffer.clear()
        ts_buffer.clear()

    stream.stop()
    status_dict['state'] = 'stopped'
    log.info("Worker stopped")


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------

def _rows_to_df(rows: List[dict]) -> pd.DataFrame:
    """
    Convert a list of detection-row dicts to a DataFrame suitable for clean_df().

    Missing frame IDs are filled with detected=0 / NaN ball coords so that
    clean_df's interpolator has a proper frame axis.
    """
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

    # Drop the internal column before building the full index
    extra_cols = [c for c in df.columns
                  if c not in ('frame_id', 'detected', 'ball_u', 'ball_v', 'ball_conf',
                               'capture_ts', 'camera_name')]

    min_fid = int(df['frame_id'].min())
    max_fid = int(df['frame_id'].max())

    full_fids = pd.RangeIndex(min_fid, max_fid + 1)
    full = pd.DataFrame({'frame_id': full_fids})
    full = full.merge(df[['frame_id', 'detected', 'ball_u', 'ball_v', 'ball_conf']],
                      on='frame_id', how='left')
    full['detected']  = full['detected'].fillna(0).astype(int)
    full['ball_conf'] = full['ball_conf'].fillna(0.0)

    return full.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-camera reporting from DataFrame — delegates to report_api.report_df()
# ---------------------------------------------------------------------------

def _report_df(
    label:               str,
    df:                  pd.DataFrame,
    calib_json_path:     str,
    y_net:               float,
    homography_matrices: dict,
    serial_number:       str,
    session_start:       datetime,
    fps:                 float = DEFAULT_FPS,
    api_url:             str   = API_URL,
    dry_run:             bool  = False,
) -> int:
    """Thin wrapper around scripts.report_api.report_df()."""
    return _report_df_api(
        label               = label,
        df                  = df,
        calib_json_path     = calib_json_path,
        y_net               = y_net,
        homography_matrices = homography_matrices,
        serial_number       = serial_number,
        video_start         = session_start,
        fps                 = fps,
        api_url             = api_url,
        dry_run             = dry_run,
    )


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class LivePipeline:
    """
    Orchestrates two camera subprocesses and a periodic flush→report loop.

    Camera workers run in ``mp.Process`` subprocesses.
    Detection rows are consumed by a daemon thread into per-camera deques.
    The flush thread checks inactivity / force-flush conditions and
    dispatches clean → sync → report.
    """

    def __init__(
        self,
        cam66_url:      str   = _CAM66_RTSP,
        cam68_url:      str   = _CAM68_RTSP,
        model_path:     str   = DEFAULT_MODEL_PATH,
        homography_path: str  = DEFAULT_HOM_PATH,
        device:         str   = DEFAULT_DEVICE,
        fps:            float = DEFAULT_FPS,
        dry_run:        bool  = False,
        api_url:        str   = API_URL,
    ):
        self.model_path      = model_path
        self.homography_path = homography_path
        self.device          = device
        self.fps             = fps
        self.dry_run         = dry_run
        self.api_url         = api_url

        # Merge runtime RTSP URLs into camera config (serial from settings/config.yaml)
        self._cam_cfg = {
            'cam66': {**_CAM_DEFAULTS['cam66'], 'rtsp_url': cam66_url,
                      'serial': CAMERAS_CFG.get('cam66', {}).get('serial', 'FV9942593')},
            'cam68': {**_CAM_DEFAULTS['cam68'], 'rtsp_url': cam68_url,
                      'serial': CAMERAS_CFG.get('cam68', {}).get('serial', 'FV9942588')},
        }

        self._session_start_ts: float      = time.time()
        self._session_start:    datetime   = datetime.now(tz=timezone.utc)

        # Per-camera raw detection rows (thread-safe via _buf_lock)
        self._buffers:    Dict[str, List[dict]] = {k: [] for k in self._cam_cfg}
        self._last_det_ts: Dict[str, float]     = {k: 0.0 for k in self._cam_cfg}
        self._buf_lock = threading.Lock()

        self._last_flush_ts = time.time()
        self._flush_count   = 0

        self._stop = mp.Event()

        # mp resources
        self._result_queues:  Dict[str, mp.Queue] = {}
        self._worker_procs:   Dict[str, mp.Process] = {}
        self._status_dicts:   Dict[str, dict] = {}

        # Load homography matrices once
        hom_p = Path(homography_path)
        if not hom_p.exists():
            raise FileNotFoundError(f"Homography JSON not found: {hom_p}")
        with open(hom_p, 'r', encoding='utf-8') as fh:
            self._homography = json.load(fh)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start camera subprocesses and background threads."""
        mgr = mp.Manager()

        for name, cfg in self._cam_cfg.items():
            q      = mp.Queue(maxsize=2000)
            status = mgr.dict()
            status.update({'state': 'starting', 'error_msg': '', 'fps': 0.0,
                           'inference_enabled': True})

            proc = mp.Process(
                target=_camera_worker,
                name=f'camera_{name}',
                kwargs=dict(
                    name             = name,
                    rtsp_url         = cfg['rtsp_url'],
                    model_path       = self.model_path,
                    homography_path  = self.homography_path,
                    homography_key   = cfg['homography_key'],
                    app_root         = str(_HERE),
                    result_queue     = q,
                    stop_event       = self._stop,
                    status_dict      = status,
                    session_start_ts = self._session_start_ts,
                    fps              = self.fps,
                    frames_in        = FRAMES_IN,
                    frames_out       = FRAMES_OUT,
                    device           = self.device,
                    input_size       = INPUT_SIZE,
                    threshold        = THRESHOLD,
                ),
                daemon=True,
            )
            proc.start()
            logger.info("Started subprocess for %s  (pid=%d)", name, proc.pid)

            self._result_queues[name]  = q
            self._worker_procs[name]   = proc
            self._status_dicts[name]   = status

        # Consumer thread: queues → in-memory buffers
        self._consumer_thread = threading.Thread(
            target=self._consume_loop, name='consumer', daemon=True
        )
        self._consumer_thread.start()

        # Flush thread: periodic clean → sync → report
        self._flush_thread = threading.Thread(
            target=self._flush_loop, name='flush', daemon=True
        )
        self._flush_thread.start()

    def stop(self) -> None:
        """Signal all workers and threads to stop."""
        logger.info("Stopping pipeline…")
        self._stop.set()
        for name, proc in self._worker_procs.items():
            proc.terminate()
            proc.join(timeout=8)
            logger.info("[%s] subprocess joined (exitcode=%s)", name, proc.exitcode)

    def run(self) -> None:
        """Start pipeline and block until Ctrl-C."""
        self.start()
        logger.info(
            "Pipeline running.  dry_run=%s  cameras=%s",
            self.dry_run, list(self._cam_cfg)
        )
        try:
            while True:
                time.sleep(15)
                self._log_status()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt — shutting down")
        finally:
            self.stop()

    # ------------------------------------------------------------------
    # Consumer thread
    # ------------------------------------------------------------------

    def _consume_loop(self) -> None:
        """Read detection rows from all camera queues and append to buffers."""
        while not self._stop.is_set():
            any_read = False
            for name, q in self._result_queues.items():
                try:
                    row = q.get_nowait()
                except Exception:
                    continue

                # Skip MedianBG blob_block messages (not used by default)
                if not isinstance(row, dict) or row.get('type') == 'blob_block':
                    continue

                c_ts = float(row.get('capture_ts', time.time()))

                with self._buf_lock:
                    buf = self._buffers[name]
                    buf.append(row)
                    # Cap buffer to avoid unbounded memory growth
                    if len(buf) > MAX_BUFFER_ROWS:
                        del buf[:len(buf) - MAX_BUFFER_ROWS]
                    self._last_det_ts[name] = c_ts

                any_read = True

            if not any_read:
                time.sleep(0.005)

    # ------------------------------------------------------------------
    # Flush loop
    # ------------------------------------------------------------------

    def _flush_loop(self) -> None:
        """Periodically check conditions and trigger a pipeline flush."""
        while not self._stop.is_set():
            time.sleep(2.0)
            now = time.time()

            with self._buf_lock:
                buf_sizes = {n: len(b) for n, b in self._buffers.items()}
                last_dets = dict(self._last_det_ts)

            min_rows = min(buf_sizes.values()) if buf_sizes else 0

            if min_rows < MIN_FLUSH_ROWS:
                continue  # not enough data yet

            # Inactivity condition: all cameras have been quiet
            active_cams = [n for n, t in last_dets.items() if t > 0]
            all_quiet = bool(active_cams) and all(
                now - last_dets[n] >= INACTIVITY_SEC for n in active_cams
            )

            force = (now - self._last_flush_ts) >= FORCE_FLUSH_SEC

            if all_quiet or force:
                reason = 'inactivity' if all_quiet else 'force_flush'
                logger.info("Flush triggered (%s)  buf_sizes=%s", reason, buf_sizes)
                self._do_flush()
                self._last_flush_ts = now

    # ------------------------------------------------------------------
    # Core flush: clean → sync → report
    # ------------------------------------------------------------------

    def _do_flush(self) -> None:
        """
        Extract buffered rows, run the clean→sync→rally→POST pipeline.

        After the flush, the last OVERLAP_SEC seconds of rows are kept so
        that rallies straddling a flush boundary are fully captured next time.
        """
        overlap_rows = int(self.fps * OVERLAP_SEC)

        with self._buf_lock:
            snapshots: Dict[str, List[dict]] = {}
            for name in self._cam_cfg:
                buf = self._buffers[name]
                snapshots[name] = list(buf)           # snapshot
                # Retain tail for next flush
                self._buffers[name] = buf[-overlap_rows:] if len(buf) > overlap_rows else list(buf)

        self._flush_count += 1
        flush_id = self._flush_count
        logger.info("=== Flush #%d  rows=%s ===",
                    flush_id, {n: len(r) for n, r in snapshots.items()})

        # ── Stage 1: rows → DataFrames ────────────────────────────────
        dfs: Dict[str, pd.DataFrame] = {}
        for name, rows in snapshots.items():
            if not rows:
                logger.warning("[flush#%d] No rows for %s — skip", flush_id, name)
                continue
            dfs[name] = _rows_to_df(rows)
            logger.info("[flush#%d] %s: %d raw rows → %d df rows (frame %d–%d)",
                        flush_id, name, len(rows), len(dfs[name]),
                        dfs[name]['frame_id'].min(), dfs[name]['frame_id'].max())

        if len(dfs) < 2:
            logger.warning("[flush#%d] Need both cameras — skipping", flush_id)
            return

        # ── Stage 2: clean each camera ────────────────────────────────
        cleaned: Dict[str, pd.DataFrame] = {}
        for name, df_raw in dfs.items():
            cfg = self._cam_cfg[name]
            try:
                df_clean, summary = clean_df(
                    df_raw,
                    calib_path = cfg['calib_json'],
                    y_net      = cfg['y_net'],
                    label      = name,
                )
                cleaned[name] = df_clean
                logger.info(
                    "[flush#%d] %s clean: %d raw det → %d clean det  outliers=%d",
                    flush_id, name,
                    summary['raw_detections'], summary['clean'],
                    summary['outliers'],
                )
            except Exception as exc:
                logger.error("[flush#%d] clean_df failed for %s: %s", flush_id, name, exc,
                             exc_info=True)
                return

        # ── Stage 3: sync cameras ─────────────────────────────────────
        try:
            df66, df68, sync_res = sync_dfs(
                cleaned['cam66'], cleaned['cam68'],
                label_a='cam66', label_b='cam68',
            )
            logger.info("[flush#%d] sync: tau=%d fr  SNR=%.2f",
                        flush_id, sync_res['tau_offset'], sync_res['corr_snr'])
        except Exception as exc:
            logger.error("[flush#%d] sync_dfs failed: %s", flush_id, exc, exc_info=True)
            return

        # ── Stage 4: detect rallies + POST ────────────────────────────
        for name, df_synced in [('cam66', df66), ('cam68', df68)]:
            cfg = self._cam_cfg[name]
            try:
                sent = _report_df(
                    label               = name,
                    df                  = df_synced,
                    calib_json_path     = cfg['calib_json'],
                    y_net               = cfg['y_net'],
                    homography_matrices = self._homography,
                    serial_number       = cfg['serial'],
                    session_start       = self._session_start,
                    fps                 = self.fps,
                    api_url             = self.api_url,
                    dry_run             = self.dry_run,
                )
                logger.info("[flush#%d] %s → %d payloads sent", flush_id, name, sent)
            except Exception as exc:
                logger.error("[flush#%d] report failed for %s: %s",
                             flush_id, name, exc, exc_info=True)

    # ------------------------------------------------------------------
    # Status logging
    # ------------------------------------------------------------------

    def _log_status(self) -> None:
        with self._buf_lock:
            buf_sizes  = {n: len(b) for n, b in self._buffers.items()}
            last_dets  = dict(self._last_det_ts)
        now = time.time()
        for name, status in self._status_dicts.items():
            idle = now - last_dets.get(name, now)
            logger.info(
                "[%s]  state=%-10s  fps=%4.1f  buf=%5d rows  idle=%.0fs",
                name,
                status.get('state', '?'),
                float(status.get('fps', 0)),
                buf_sizes.get(name, 0),
                idle,
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Real-time RTSP tennis ball tracking → API reporting pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        '--dry-run', action='store_true',
        help='Build and print payloads to stdout; do NOT POST to the API',
    )
    p.add_argument(
        '--cam66-url', default=_CAM66_RTSP,
        help='RTSP stream URL for cam66 (near baseline)',
    )
    p.add_argument(
        '--cam68-url', default=_CAM68_RTSP,
        help='RTSP stream URL for cam68 (far baseline)',
    )
    p.add_argument(
        '--model', default=DEFAULT_MODEL_PATH, metavar='PATH',
        help='TrackNet model weights (.pt)',
    )
    p.add_argument(
        '--homography', default=DEFAULT_HOM_PATH, metavar='PATH',
        help='homography_matrices.json path',
    )
    p.add_argument(
        '--device', default=DEFAULT_DEVICE, choices=['cuda', 'cpu'],
        help='Inference device',
    )
    p.add_argument(
        '--fps', type=float, default=DEFAULT_FPS,
        help='Camera frame rate (used for frame_id derivation and timing)',
    )
    p.add_argument(
        '--api-url', default=API_URL,
        help='POST endpoint URL',
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.dry_run:
        logger.info("*** DRY-RUN MODE — payloads will be printed, not POSTed ***")

    pipeline = LivePipeline(
        cam66_url       = args.cam66_url,
        cam68_url       = args.cam68_url,
        model_path      = args.model,
        homography_path = args.homography,
        device          = args.device,
        fps             = args.fps,
        dry_run         = args.dry_run,
        api_url         = args.api_url,
    )
    pipeline.run()


if __name__ == '__main__':
    # Required on Windows for multiprocessing to work correctly
    mp.freeze_support()
    main()
