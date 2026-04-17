"""
report_api.py
=============
Stage 4 of the Tennis Analysis Pipeline.

Reads the cleaned + synced result CSVs produced by main.py, detects rallies
via net-crossing analysis, builds the API payload, and POSTs one JSON per
rally per camera to the reporting endpoint.

Camera → serial number mapping (edit CAMERA_SERIALS below):
    cam66 → FV9942593
    cam68 → FV9942588

Coordinate system
-----------------
World (metres, from homography calibration):
    X : 0 (left sideline) → COURT_WIDTH_M  (right sideline)
    Y : 0 (near baseline, cam66 side) → COURT_LENGTH_M (far baseline)

API normalised (0 – 1):
    x = x_world / COURT_WIDTH_M          (0 = left, 1 = right)
    y = 1 – y_world / COURT_LENGTH_M     (0 = far baseline, 1 = near baseline)
    result.json reference: y ∈ [0.1, 0.4] far court, y ∈ [0.6, 0.9] near court

Usage
-----
    python scripts/report_api.py                    # use defaults
    python scripts/report_api.py --dry-run          # print payloads, don't POST
    python scripts/report_api.py --result-dir output/ --homography uploads/homography_matrices.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Lazy import so the module loads even without requests installed
try:
    import requests as _requests
except ImportError:
    _requests = None  # type: ignore

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analysis_module import TennisRallySegmenter, TennisVisualizer

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION  — edit these values if the mapping changes
# =============================================================================

API_URL = "https://tennisync.top/api/admin/SpaceParties/reportData"

# Camera label → device serial number
CAMERA_SERIALS: dict[str, str] = {
    'cam66': 'FV9942593',
    'cam68': 'FV9942588',
}

# Per-camera pipeline config (paths relative to project root)
CAMERAS = [
    {
        'label':      'cam66',
        'result_csv': 'output/cam66_result.csv',
        'calib_json': 'uploads/cal_cam66.json',
        'y_net':      238.0,   # pixel V of net in cam66 view
    },
    {
        'label':      'cam68',
        'result_csv': 'output/cam68_result.csv',
        'calib_json': 'uploads/cal_cam68.json',
        'y_net':      285.0,   # pixel V of net in cam68 view
    },
]

HOMOGRAPHY_JSON = 'uploads/homography_matrices.json'

# Standard singles court dimensions (metres)
COURT_LENGTH_M = 23.77
COURT_WIDTH_M  = 8.23

# =============================================================================
# Coordinate helpers
# =============================================================================

def world_to_api(x_world: float, y_world: float) -> Tuple[float, float]:
    """
    Convert world metres → normalised API coordinates, both clamped to [0, 1].

    x_api = x_world / COURT_WIDTH_M
    y_api = 1 – y_world / COURT_LENGTH_M   (flip: API y=0 is far baseline)
    """
    x_api = max(0.0, min(1.0, x_world / COURT_WIDTH_M))
    y_api = max(0.0, min(1.0, 1.0 - y_world / COURT_LENGTH_M))
    return round(x_api, 3), round(y_api, 3)


# =============================================================================
# Per-rally payload builders
# =============================================================================

def _compute_player_stats(
    df_rally: pd.DataFrame,
    visualizer: TennisVisualizer,
    fps: float,
    player_role: str,   # 'near' or 'far'
) -> dict:
    """
    Compute totalDistance / avgMoveSpeed / maxMoveSpeed for near or far player.

    Near = the player with higher pixel-V (closer to camera) each frame.
    Far  = the other player.
    """
    frame_time = 1.0 / fps
    total_dist = 0.0
    max_spd    = 0.0
    prev_w     = None

    for _, row in df_rally.iterrows():
        near_prefix = visualizer._near_player_prefix(row)
        if near_prefix is None:
            prev_w = None   # reset on detection gap
            continue

        prefix = near_prefix if player_role == 'near' else (
            'p1' if near_prefix == 'p0' else 'p0'
        )

        curr_w = visualizer._get_player_world(row, prefix)
        if curr_w and prev_w:
            d = np.hypot(curr_w[0] - prev_w[0], curr_w[1] - prev_w[1])
            if visualizer.min_move_dist < d < visualizer.max_move_dist:
                total_dist += d
                max_spd = max(max_spd, d / frame_time)

        prev_w = curr_w if curr_w else None

    duration = len(df_rally) * frame_time
    avg_spd  = total_dist / duration if duration > 0 and total_dist > 0 else 0.0

    return {
        'totalDistance': round(total_dist, 2),
        'avgMoveSpeed':  round(avg_spd,    3),
        'maxMoveSpeed':  round(max_spd,    3),
    }


def _build_result_matrix(
    df: pd.DataFrame,
    rally_crossings: List[Tuple[int, int]],   # [(frame_id, df_idx), ...]
    visualizer: TennisVisualizer,
    fps: float,
    u_col: str,
    v_col: str,
) -> List[dict]:
    """
    Build resultmatrix — one entry per net-crossing shot within the rally.

    Each entry carries normalised (x, y), type='hit', speed (km/h),
    handType='forehand' (pose-based classification is a future TODO).
    """
    events = []
    for fid, ci in rally_crossings:
        u = df.loc[ci, u_col]
        v = df.loc[ci, v_col]
        if pd.isna(u) or pd.isna(v):
            continue
        w = visualizer.to_world(float(u), float(v))
        if w is None:
            continue
        x_api, y_api = world_to_api(w[0], w[1])
        spd = visualizer._ball_speed_kmh(df, ci, fps, u_col, v_col)
        events.append({
            'x':        x_api,
            'y':        y_api,
            'type':     'hit',
            'speed':    round(spd, 1),
            'handType': 'forehand',   # TODO: classify from pose keypoints
        })
    return events


def _build_track_matrix(
    df_rally: pd.DataFrame,
    visualizer: TennisVisualizer,
    u_col: str,
    v_col: str,
) -> List[dict]:
    """
    Build trackMatrix — per-frame ball + player positions for the rally.

    Only frames with a valid ball position (non-NaN interp_u/v) are included.
    """
    rows = []
    for _, row in df_rally.iterrows():
        u = row.get(u_col)
        v = row.get(v_col)
        if u is None or pd.isna(u):
            continue

        w_ball = visualizer.to_world(float(u), float(v))
        if w_ball is None:
            continue

        x_api, y_api = world_to_api(w_ball[0], w_ball[1])

        near_prefix = visualizer._near_player_prefix(row)
        far_prefix  = ('p1' if near_prefix == 'p0' else 'p0') if near_prefix else None

        near_w = visualizer._get_player_world(row, near_prefix) if near_prefix else None
        far_w  = visualizer._get_player_world(row, far_prefix)  if far_prefix  else None

        track_row: dict = {
            'x':         x_api,
            'y':         y_api,
            'type':      'running',
            'speed':     0,
            'timestamp': int(row['frame_id']),
        }

        if far_w:
            fx, fy = world_to_api(far_w[0], far_w[1])
            track_row['farCountPerson_x'] = fx
            track_row['farCountPerson_y'] = fy
        else:
            track_row['farCountPerson_x'] = None
            track_row['farCountPerson_y'] = None

        if near_w:
            nx, ny = world_to_api(near_w[0], near_w[1])
            track_row['nearCountPerson_x'] = nx
            track_row['nearCountPerson_y'] = ny
        else:
            track_row['nearCountPerson_x'] = None
            track_row['nearCountPerson_y'] = None

        rows.append(track_row)

    return rows


def _build_rally_payload(
    rally: dict,
    df: pd.DataFrame,
    crossing_frames: dict,   # frame_id → df_idx (into full df)
    visualizer: TennisVisualizer,
    fps: float,
    u_col: str,
    v_col: str,
    serial_number: str,
    video_start: datetime,
) -> dict:
    """Build the complete API payload dict for one rally."""
    start_frame = rally['start_frame']
    end_frame   = rally['end_frame']

    # ISO-8601 UTC timestamps
    start_dt = video_start + timedelta(seconds=start_frame / fps)
    end_dt   = video_start + timedelta(seconds=end_frame   / fps)

    # Subset df to this rally's frames
    mask  = (df['frame_id'] >= start_frame) & (df['frame_id'] <= end_frame)
    df_r  = df[mask].copy().reset_index(drop=True)

    if df_r.empty:
        raise ValueError(f"Empty frame range [{start_frame}, {end_frame}]")

    # Crossings within this rally, sorted by frame
    rally_crossings = sorted(
        [(fid, ci) for fid, ci in crossing_frames.items()
         if start_frame <= fid <= end_frame],
        key=lambda x: x[0],
    )

    # Ball speed at each crossing
    crossing_speeds = [
        visualizer._ball_speed_kmh(df, ci, fps, u_col, v_col)
        for _, ci in rally_crossings
    ]

    total_shots    = len(rally_crossings)
    avg_ball_speed = round(sum(crossing_speeds) / len(crossing_speeds), 1) if crossing_speeds else 0.0
    max_ball_speed = round(max(crossing_speeds, default=0.0), 1)

    # Player movement stats
    far_move  = _compute_player_stats(df_r, visualizer, fps, player_role='far')
    near_move = _compute_player_stats(df_r, visualizer, fps, player_role='near')

    # Shared ball stats across both sides
    ball_mete = {
        'totalShots':   total_shots,
        'avgBallSpeed': avg_ball_speed,
        'maxBallSpeed': max_ball_speed,
    }

    # Stats requiring win/point data or serve classification — set to 0
    unknown_stats = {
        'firstServeSuccessRate':  0,
        'returnFirstSuccessRate': 0,
        'baselineShotRate':       0,
        'baselineWinRate':        0,
        'netPointRate':           0,
        'netPointWinRate':        0,
        'aceCount':               0,
        'netApproaches':          0,
    }

    far_count  = {**ball_mete, **far_move,  **unknown_stats}
    near_count = {**ball_mete, **near_move, **unknown_stats}

    return {
        'serial_number': serial_number,
        'startTime': start_dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'endTime':   end_dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'content': {
            'mete': {
                'farCount':  far_count,
                'nearCount': near_count,
            },
            'resultmatrix': _build_result_matrix(
                df, rally_crossings, visualizer, fps, u_col, v_col
            ),
            'trackMatrix': _build_track_matrix(df_r, visualizer, u_col, v_col),
        },
    }


# =============================================================================
# HTTP POST
# =============================================================================

def _post_payload(payload: dict, api_url: str, dry_run: bool = False) -> bool:
    """POST payload to API. Returns True on success."""
    serial = payload.get('serial_number', '?')
    start  = payload.get('startTime', '?')
    end    = payload.get('endTime', '?')

    if dry_run:
        logger.info("[DRY-RUN] serial=%s  %s → %s", serial, start, end)
        logger.info("PAYLOAD:\n%s", json.dumps(payload, indent=2, ensure_ascii=False))
        return True

    if _requests is None:
        raise ImportError("'requests' is not installed. Run: pip install requests")

    try:
        resp = _requests.post(
            api_url,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=30,
        )
        if resp.status_code in (200, 201, 204):
            logger.info("POST OK   serial=%s  %s → %s", serial, start, end)
            return True
        else:
            logger.error(
                "POST FAIL  serial=%s  status=%d  body=%.300s",
                serial, resp.status_code, resp.text,
            )
            return False
    except Exception as exc:
        logger.error("POST ERROR  serial=%s  %s", serial, exc)
        return False


# =============================================================================
# Main per-camera entry point
# =============================================================================

def report_camera(
    label: str,
    result_csv: str,
    calib_json_path: str,
    y_net: float,
    homography_matrices: dict,
    serial_number: str,
    video_start: datetime,
    api_url: str = API_URL,
    dry_run: bool = False,
) -> int:
    """
    Detect rallies for one camera and POST each rally to the API.

    Returns the number of payloads successfully sent (or queued in dry-run).
    """
    SEP = '─' * 60
    logger.info(SEP)
    logger.info("Camera: %s  |  serial: %s", label, serial_number)
    logger.info(SEP)

    # --- Load trajectory CSV ---
    csv_path = Path(result_csv)
    if not csv_path.exists():
        logger.error("Result CSV not found: %s", csv_path)
        return 0

    df = pd.read_csv(csv_path)
    df = df.sort_values('frame_id').reset_index(drop=True)
    logger.info("Loaded %d frames from %s", len(df), csv_path.name)

    # --- Load homography ---
    if label not in homography_matrices:
        logger.error("No homography entry for '%s'", label)
        return 0
    H_i2w = np.array(homography_matrices[label]['H_image_to_world'])

    # --- Load calibration and inject y_net ---
    cal: dict = {}
    cal_path = Path(calib_json_path)
    if cal_path.exists():
        with open(cal_path, 'r', encoding='utf-8') as fh:
            cal = json.load(fh)
        logger.info("Calibration: %s", cal_path.name)
    else:
        logger.warning("Calibration not found: %s — using fallback", cal_path)
    cal['y_net'] = y_net

    fps = float(cal.get('fps', 25.0))

    # --- Build visualizer ---
    segmenter  = TennisRallySegmenter(fps=fps)
    visualizer = TennisVisualizer(segmenter, H_i2w, cal=cal)

    # --- Preprocess DataFrame (sort + time column) ---
    _, df = segmenter.segment_rallies(df)
    u_col     = 'interp_u' if 'interp_u' in df.columns else 'x'
    v_col     = 'interp_v' if 'interp_v' in df.columns else 'y'
    frame_col = 'frame_id' if 'frame_id' in df.columns else 'frame'

    # --- Detect net crossings → rallies ---
    crossings = visualizer._detect_crossings(df)
    if not crossings:
        logger.warning("[%s] No net crossings detected — nothing to report", label)
        return 0

    rallies = visualizer._segment_rallies_by_crossings(
        crossings, df, frame_col, fps
    )
    logger.info("[%s] %d crossing(s) → %d rally(ies)", label, len(crossings), len(rallies))

    # Map frame_id → df row index for speed lookups
    crossing_frames: dict[int, int] = {
        int(df.loc[ci, frame_col]): ci for ci in crossings
    }

    # --- POST each rally ---
    sent = 0
    for rally in rallies:
        try:
            payload = _build_rally_payload(
                rally, df, crossing_frames,
                visualizer, fps, u_col, v_col,
                serial_number, video_start,
            )
            if _post_payload(payload, api_url, dry_run):
                sent += 1
        except Exception as exc:
            logger.error("[%s] Rally %d build/send error: %s",
                         label, rally.get('rally_id', '?'), exc)

    logger.info("[%s] Sent %d / %d rallies", label, sent, len(rallies))
    return sent


# =============================================================================
# CLI
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).parent.parent
    p = argparse.ArgumentParser(
        description='Report tennis rally data to the API endpoint',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--result-dir',  default=str(root / 'output'),
                   help='Directory containing {label}_result.csv files')
    p.add_argument('--homography',  default=str(root / HOMOGRAPHY_JSON),
                   help='homography_matrices.json path')
    p.add_argument('--calib-dir',   default=str(root / 'uploads'),
                   help='Directory containing cal_{label}.json files')
    p.add_argument('--api-url',     default=API_URL,
                   help='POST endpoint URL')
    p.add_argument('--video-start', default=None,
                   help='UTC datetime of video frame 0, ISO-8601 (default: now)')
    p.add_argument('--dry-run',     action='store_true',
                   help='Build payloads but do not POST')
    return p


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)-8s  %(message)s',
        datefmt='%H:%M:%S',
    )

    args = _build_parser().parse_args()

    # Parse video start time
    if args.video_start:
        video_start = datetime.fromisoformat(
            args.video_start.rstrip('Z')
        ).replace(tzinfo=timezone.utc)
    else:
        video_start = datetime.now(tz=timezone.utc)
        logger.info("video_start not provided — using current UTC time: %s",
                    video_start.strftime('%Y-%m-%dT%H:%M:%SZ'))

    # Load homography matrices
    hom_path = Path(args.homography)
    if not hom_path.exists():
        logger.error("Homography JSON not found: %s", hom_path)
        sys.exit(1)
    with open(hom_path, 'r', encoding='utf-8') as fh:
        homography_matrices = json.load(fh)

    result_dir = Path(args.result_dir)
    calib_dir  = Path(args.calib_dir)

    total_sent = 0
    for cam in CAMERAS:
        label    = cam['label']
        serial   = CAMERA_SERIALS.get(label, f'UNKNOWN_{label}')
        csv_path = result_dir / f"{label}_result.csv"
        cal_path = calib_dir  / f"cal_{label}.json"

        sent = report_camera(
            label               = label,
            result_csv          = str(csv_path),
            calib_json_path     = str(cal_path),
            y_net               = cam['y_net'],
            homography_matrices = homography_matrices,
            serial_number       = serial,
            video_start         = video_start,
            api_url             = args.api_url,
            dry_run             = args.dry_run,
        )
        total_sent += sent

    logger.info('=' * 60)
    logger.info('Total payloads sent: %d', total_sent)
    logger.info('=' * 60)


def report_df(
    label: str,
    df: pd.DataFrame,
    calib_json_path: str,
    y_net: float,
    homography_matrices: dict,
    serial_number: str,
    video_start: datetime,
    fps: float = 25.0,
    api_url: str = API_URL,
    dry_run: bool = False,
) -> int:
    """
    Detect rallies in an in-memory DataFrame and POST each rally to the API.

    This is the in-memory equivalent of :func:`report_camera` — it skips the
    CSV-loading step and uses the supplied DataFrame directly.

    Parameters
    ----------
    label : str
        Camera label (e.g. 'cam66').
    df : pd.DataFrame
        Cleaned (and optionally synced) trajectory DataFrame.  Must contain
        at least ``frame_id``, ``interp_u``, ``interp_v`` columns.
    calib_json_path : str
        Path to ``cal_{label}.json``.  Used for fps and y_near/y_far values.
    y_net : float
        Pixel V-coordinate of the net in this camera's view.
    homography_matrices : dict
        Loaded homography JSON (keyed by camera label).
    serial_number : str
        Device serial number sent in the API payload.
    video_start : datetime
        UTC datetime corresponding to frame 0.
    fps : float
        Frame rate (fallback if not in calibration JSON).
    api_url : str
        POST endpoint.
    dry_run : bool
        When True, log payloads but do not POST.

    Returns
    -------
    int
        Number of rally payloads successfully sent (or queued in dry-run).
    """
    SEP = '─' * 60
    logger.info(SEP)
    logger.info("Camera: %s  |  serial: %s  |  frames: %d", label, serial_number, len(df))
    logger.info(SEP)

    if df.empty:
        logger.warning("[%s] Empty DataFrame — nothing to report", label)
        return 0

    # --- Load homography ---
    if label not in homography_matrices:
        logger.error("[%s] No homography entry for '%s'", label, label)
        return 0
    H_i2w = np.array(homography_matrices[label]['H_image_to_world'])

    # --- Load calibration ---
    cal: dict = {}
    cal_path = Path(calib_json_path)
    if cal_path.exists():
        with open(cal_path, 'r', encoding='utf-8') as fh:
            cal = json.load(fh)
        logger.info("[%s] Calibration: %s", label, cal_path.name)
    else:
        logger.warning("[%s] Calibration not found: %s — using fallback", label, cal_path)
    cal['y_net'] = y_net

    fps_cal = float(cal.get('fps', fps))

    # --- Build visualizer ---
    segmenter  = TennisRallySegmenter(fps=fps_cal)
    visualizer = TennisVisualizer(segmenter, H_i2w, cal=cal)

    # --- Preprocess DataFrame ---
    df = df.sort_values('frame_id').reset_index(drop=True)
    _, df = segmenter.segment_rallies(df)

    u_col     = 'interp_u' if 'interp_u' in df.columns else 'ball_u'
    v_col     = 'interp_v' if 'interp_v' in df.columns else 'ball_v'
    frame_col = 'frame_id'

    # --- Detect net crossings → rallies ---
    crossings = visualizer._detect_crossings(df)
    if not crossings:
        logger.warning("[%s] No net crossings detected — nothing to report", label)
        return 0

    rallies = visualizer._segment_rallies_by_crossings(
        crossings, df, frame_col, fps_cal
    )
    logger.info("[%s] %d crossing(s) → %d rally(ies)", label, len(crossings), len(rallies))

    crossing_frames: dict[int, int] = {
        int(df.loc[ci, frame_col]): ci for ci in crossings
    }

    # --- POST each rally ---
    sent = 0
    for rally in rallies:
        try:
            payload = _build_rally_payload(
                rally, df, crossing_frames,
                visualizer, fps_cal, u_col, v_col,
                serial_number, video_start,
            )
            if _post_payload(payload, api_url, dry_run):
                sent += 1
        except Exception as exc:
            logger.error("[%s] Rally %s build/send error: %s",
                         label, rally.get('rally_id', '?'), exc)

    logger.info("[%s] Sent %d / %d rallies", label, sent, len(rallies))
    return sent


if __name__ == '__main__':
    main()
