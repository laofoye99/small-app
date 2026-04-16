"""
cleaner_core.py
===============
All rule logic for ball trajectory cleaning.
Imported by run_all_cameras.py (multi-camera batch).

Rules
-----
  R1  low_conf          ball_conf < CONF_THR
  R2  out_of_bounds     u/v outside valid image area
  R3  snap_back         fast in + fast out + opposite direction
  R4  speed_spike       fast both ways, any direction
  R5  spatial_isolation far from both neighbours
  R6  static_cluster    frozen position for N+ consecutive frames
  R7  short_segment     run too short to compute acceleration
  R8  crossing_time     near<->far in fewer than MIN_CROSSING_FRAMES
  --  low_conf_path     confidence-based path disambiguation
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Global rule parameters  (camera-independent)
# ---------------------------------------------------------------------------
CONF_THR            = 0.50
U_MIN, U_MAX        = 10.0, 1920.0
V_MIN, V_MAX        = 10.0, 1080.0

SNAPBACK_COS        = -0.80
SNAPBACK_MULT       = 1.0       # flag when speed > MULT * v_max_px(v)
ISOLATION_MULT      = 1.5
V_MAX_FLOOR         = 20.0      # px/frame -- guards against negative beta

STATIC_DIST_THR     = 5.0       # px
STATIC_MIN_LEN      = 4         # frames

MIN_SEG_LEN         = 5         # points
SEG_GAP_THR         = 5         # frames

MIN_CROSSING_FRAMES = 8         # physics: ~120 km/h at 25 fps
CROSSING_V_FRAC     = 0.60      # flag if dV > 60% of full y_near-y_far range

DISAMBIG_WINDOW     = 6         # frames to look ahead for path selection
DISAMBIG_V_SPLIT    = 150.0     # px -- two clusters if V differs by this

MAX_INTERP_GAP      = 8         # frames

FALLBACK_SNAPBACK   = 80.0      # px/frame (no calibration)
FALLBACK_ISOLATION  = 150.0     # px/frame (no calibration)


# ---------------------------------------------------------------------------
# Perspective speed model
# ---------------------------------------------------------------------------

class PerspectiveSpeed:
    """
    v_max_px(v) = max(V_MAX_FLOOR, alpha*v + beta)

    snap-back threshold = SNAPBACK_MULT  * v_max_px(v)
    isolation threshold = ISOLATION_MULT * v_max_px(v)
    """
    def __init__(self, alpha, beta, flat_snap, flat_iso):
        self.alpha      = alpha
        self.beta       = beta
        self.flat_snap  = flat_snap
        self.flat_iso   = flat_iso
        self.calibrated = (alpha is not None) and (beta is not None)

    def v_max(self, v):
        if not self.calibrated:
            return None
        return max(V_MAX_FLOOR, self.alpha * v + self.beta)

    def snapback_thr(self, v):
        return self.v_max(v) * SNAPBACK_MULT if self.calibrated else self.flat_snap

    def isolation_thr(self, v):
        return self.v_max(v) * ISOLATION_MULT if self.calibrated else self.flat_iso

    def describe(self, y_far=None, y_near=None):
        if not self.calibrated:
            return (f"flat fallback  snap={self.flat_snap} px/fr  "
                    f"iso={self.flat_iso} px/fr")
        lines = [f"perspective-aware  alpha={self.alpha:.6f}  beta={self.beta:.4f}"]
        for v, label in [(y_far or V_MIN, 'far  baseline'),
                         (y_near or V_MAX, 'near baseline')]:
            vm = self.v_max(v)
            lines.append(
                f"    v={v:6.0f}px ({label})  v_max={vm:7.2f}  "
                f"snap_thr={self.snapback_thr(v):7.2f}  "
                f"iso_thr={self.isolation_thr(v):7.2f}  px/fr"
            )
        return '\n'.join(lines)


def load_calibration(calib_path, alpha_cli=None, beta_cli=None):
    """
    Returns (PerspectiveSpeed, cal_dict).
    Priority: CLI alpha/beta > JSON file > flat fallback.
    """
    if alpha_cli is not None and beta_cli is not None:
        print(f"  [CALIB] CLI: alpha={alpha_cli}  beta={beta_cli}")
        return PerspectiveSpeed(alpha_cli, beta_cli,
                                FALLBACK_SNAPBACK, FALLBACK_ISOLATION), {}

    if calib_path:
        p = Path(calib_path)
        if p.exists():
            cal = json.loads(p.read_text())
            print(f"  [CALIB] {p.name}  "
                  f"y_far={cal['y_far']:.0f}  y_near={cal['y_near']:.0f}  "
                  f"alpha={cal['alpha']:.6f}  beta={cal['beta']:.4f}  "
                  f"fps={cal['fps']}")
            return (PerspectiveSpeed(cal['alpha'], cal['beta'],
                                     FALLBACK_SNAPBACK, FALLBACK_ISOLATION), cal)
        print(f"  [CALIB] '{calib_path}' not found -- flat fallback")

    print(f"  [CALIB] No calibration -- "
          f"snap={FALLBACK_SNAPBACK}  iso={FALLBACK_ISOLATION} px/fr")
    return PerspectiveSpeed(None, None, FALLBACK_SNAPBACK, FALLBACK_ISOLATION), {}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _speed(du, dv, dt):
    return float(np.hypot(du, dv) / max(dt, 1))


def _cos_sim(du1, dv1, du2, dv2):
    v1 = np.array([du1, dv1], dtype=float)
    v2 = np.array([du2, dv2], dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


# ---------------------------------------------------------------------------
# Rules 1-5  (point-level, iterative)
# ---------------------------------------------------------------------------

def flag_point_outliers(df, spd):
    df = df.copy()
    df['clean_u']      = np.where(df['detected'] == 1, df['ball_u'], np.nan)
    df['clean_v']      = np.where(df['detected'] == 1, df['ball_v'], np.nan)
    df['is_outlier']   = False
    df['outlier_rule'] = ''

    det_idx = df.index[df['detected'] == 1].tolist()

    def _set(idx, rule):
        if not df.loc[idx, 'is_outlier']:
            df.loc[idx, 'is_outlier']   = True
            df.loc[idx, 'outlier_rule'] = rule
        df.loc[idx, 'clean_u'] = np.nan
        df.loc[idx, 'clean_v'] = np.nan

    # R1: low confidence
    for idx in det_idx:
        if df.loc[idx, 'ball_conf'] < CONF_THR:
            _set(idx, 'low_conf')

    # R2: out-of-bounds
    for idx in det_idx:
        u, v = df.loc[idx, 'ball_u'], df.loc[idx, 'ball_v']
        if not (U_MIN <= u <= U_MAX and V_MIN <= v <= V_MAX):
            _set(idx, 'out_of_bounds')

    # R3-R5: velocity/spatial (iterate until stable)
    changed, passes = True, 0
    while changed and passes < 10:
        changed = False
        passes += 1
        clean_idx = [i for i in det_idx if not df.loc[i, 'is_outlier']]
        m = len(clean_idx)

        for pos in range(m):
            idx = clean_idx[pos]
            u0, v0, f0 = (df.loc[idx, 'ball_u'],
                          df.loc[idx, 'ball_v'],
                          df.loc[idx, 'frame_id'])
            thr_snap = spd.snapback_thr(v0)
            thr_iso  = spd.isolation_thr(v0)

            if pos > 0:
                p      = clean_idx[pos - 1]
                du_in  = u0 - df.loc[p, 'ball_u']
                dv_in  = v0 - df.loc[p, 'ball_v']
                spd_in = _speed(du_in, dv_in, f0 - df.loc[p, 'frame_id'])
            else:
                du_in = dv_in = spd_in = 0.0

            if pos < m - 1:
                q       = clean_idx[pos + 1]
                du_out  = df.loc[q, 'ball_u'] - u0
                dv_out  = df.loc[q, 'ball_v'] - v0
                spd_out = _speed(du_out, dv_out, df.loc[q, 'frame_id'] - f0)
            else:
                du_out = dv_out = spd_out = 0.0

            # R3: snap-back
            if spd_in > thr_snap and spd_out > thr_snap:
                if _cos_sim(du_in, dv_in, du_out, dv_out) <= SNAPBACK_COS:
                    _set(idx, 'snap_back'); changed = True; continue

            # R4: speed spike
            if spd_in > thr_iso and spd_out > thr_iso:
                _set(idx, 'speed_spike'); changed = True; continue

            # R5: spatial isolation
            if 0 < pos < m - 1:
                p, q   = clean_idx[pos - 1], clean_idx[pos + 1]
                d_prev = np.hypot(u0 - df.loc[p,'ball_u'], v0 - df.loc[p,'ball_v'])
                d_next = np.hypot(u0 - df.loc[q,'ball_u'], v0 - df.loc[q,'ball_v'])
                if d_prev > thr_iso and d_next > thr_iso:
                    _set(idx, 'spatial_isolation'); changed = True; continue

    n_out = int(df['is_outlier'].sum())
    print(f"  [R1-5]  {n_out} point outliers ({passes} passes)")
    for rule, cnt in df[df['is_outlier']]['outlier_rule'].value_counts().items():
        print(f"            {rule:<22}: {cnt}")
    return df


# ---------------------------------------------------------------------------
# Rule 6: static cluster
# ---------------------------------------------------------------------------

def flag_static_clusters(df):
    df = df.copy()
    clean_idx = df.index[(df['detected'] == 1) & (~df['is_outlier'])].tolist()

    flagged = 0
    i = 0
    while i < len(clean_idx):
        run = [clean_idx[i]]
        j   = i + 1
        while j < len(clean_idx):
            dist = np.hypot(
                df.loc[clean_idx[j],   'ball_u'] - df.loc[clean_idx[j-1], 'ball_u'],
                df.loc[clean_idx[j],   'ball_v'] - df.loc[clean_idx[j-1], 'ball_v'],
            )
            if dist < STATIC_DIST_THR:
                run.append(clean_idx[j]); j += 1
            else:
                break
        if len(run) >= STATIC_MIN_LEN:
            for ridx in run:
                if not df.loc[ridx, 'is_outlier']:
                    df.loc[ridx, 'is_outlier']   = True
                    df.loc[ridx, 'outlier_rule']  = 'static_cluster'
                    df.loc[ridx, 'clean_u']       = np.nan
                    df.loc[ridx, 'clean_v']       = np.nan
                    flagged += 1
            i = j
        else:
            i += 1

    print(f"  [R6]    {flagged} static_cluster points "
          f"(dist<{STATIC_DIST_THR}px for >={STATIC_MIN_LEN} frames)")
    return df


# ---------------------------------------------------------------------------
# Rule 7: short segment
# ---------------------------------------------------------------------------

def flag_short_segments(df):
    df = df.copy()
    clean_idx = df.index[(df['detected'] == 1) & (~df['is_outlier'])].tolist()
    if not clean_idx:
        return df

    segments, seg = [], [clean_idx[0]]
    for k in range(1, len(clean_idx)):
        prev, curr = clean_idx[k-1], clean_idx[k]
        if df.loc[curr, 'frame_id'] - df.loc[prev, 'frame_id'] <= SEG_GAP_THR:
            seg.append(curr)
        else:
            segments.append(seg); seg = [curr]
    segments.append(seg)

    flagged = short_n = 0
    for seg in segments:
        if len(seg) < MIN_SEG_LEN:
            short_n += 1
            for idx in seg:
                df.loc[idx, 'is_outlier']   = True
                df.loc[idx, 'outlier_rule']  = 'short_segment'
                df.loc[idx, 'clean_u']       = np.nan
                df.loc[idx, 'clean_v']       = np.nan
                flagged += 1

    print(f"  [R7]    {flagged} points in {short_n} short segments (< {MIN_SEG_LEN} pts)")
    return df


# ---------------------------------------------------------------------------
# Rule 8: crossing-time violation
# ---------------------------------------------------------------------------

def flag_crossing_violations(df, cal):
    if not cal or 'y_near' not in cal:
        print("  [R8]    Skipped (no calibration)")
        return df

    df       = df.copy()
    v_range  = abs(cal['y_near'] - cal['y_far'])
    v_thresh = v_range * CROSSING_V_FRAC

    clean_idx = df.index[(df['detected'] == 1) & (~df['is_outlier'])].tolist()
    flagged = 0
    for k in range(1, len(clean_idx)):
        prev_i, curr_i = clean_idx[k-1], clean_idx[k]
        dt = df.loc[curr_i, 'frame_id'] - df.loc[prev_i, 'frame_id']
        dv = abs(df.loc[curr_i, 'ball_v'] - df.loc[prev_i, 'ball_v'])
        if dt < MIN_CROSSING_FRAMES and dv >= v_thresh:
            if not df.loc[curr_i, 'is_outlier']:
                df.loc[curr_i, 'is_outlier']   = True
                df.loc[curr_i, 'outlier_rule']  = 'crossing_time'
                df.loc[curr_i, 'clean_u']       = np.nan
                df.loc[curr_i, 'clean_v']       = np.nan
                flagged += 1

    print(f"  [R8]    {flagged} crossing_time violations "
          f"(dV>={v_thresh:.0f}px in <{MIN_CROSSING_FRAMES} frames)")
    return df


# ---------------------------------------------------------------------------
# Rule 9: half-court excursion
# ---------------------------------------------------------------------------

HALF_COURT_MIN_RUN = 4      # frames — runs shorter than this on one side are anomalies


def flag_half_court_excursions(df, y_net, min_half_court_frames=HALF_COURT_MIN_RUN):
    """
    R9: Flag brief opposite-half appearances.

    Scans clean detections for contiguous runs on one side of the net.
    Any run shorter than *min_half_court_frames* that is sandwiched between
    detections on the other side is physically implausible — the ball cannot
    teleport across the net for 1–3 frames and return.

    Example anomaly: near-zone ... [1-3 frames far-zone] ... near-zone
    This is distinct from a legitimate crossing (ball stays in the far zone
    for many frames after crossing).
    """
    df = df.copy()
    clean_idx = df.index[(df['detected'] == 1) & (~df['is_outlier'])].tolist()
    if not clean_idx:
        print(f"  [R9]    Skipped (no clean points)")
        return df

    # Group consecutive clean detections into same-side runs
    runs: list[tuple[str, list[int]]] = []   # (side, [idx, ...])
    for cidx in clean_idx:
        side = 'near' if df.loc[cidx, 'ball_v'] > y_net else 'far'
        if runs and runs[-1][0] == side:
            runs[-1][1].append(cidx)
        else:
            runs.append((side, [cidx]))

    flagged = 0
    for k, (side, run_idxs) in enumerate(runs):
        # Only flag runs sandwiched by the opposite side (not first or last run)
        if k == 0 or k == len(runs) - 1:
            continue
        if len(run_idxs) < min_half_court_frames:
            for ridx in run_idxs:
                if not df.loc[ridx, 'is_outlier']:
                    df.loc[ridx, 'is_outlier']   = True
                    df.loc[ridx, 'outlier_rule']  = 'half_court_excursion'
                    df.loc[ridx, 'clean_u']       = np.nan
                    df.loc[ridx, 'clean_v']       = np.nan
                    flagged += 1

    print(f"  [R9]    {flagged} half_court_excursion points "
          f"(y_net={y_net:.0f}px, min_run<{min_half_court_frames}fr)")
    return df


# ---------------------------------------------------------------------------
# Confidence-based path disambiguation
# ---------------------------------------------------------------------------

def disambiguate_by_confidence(df):
    """
    When consecutive clean detections alternate between two spatial clusters
    (V separation > DISAMBIG_V_SPLIT), compare MEAN confidence of each
    cluster over a DISAMBIG_WINDOW-frame look-ahead. Flag the lower-mean
    cluster as 'low_conf_path'.

    Mean (not sum) prevents longer paths from winning by sheer volume.
    """
    df = df.copy()
    clean_idx = df.index[(df['detected'] == 1) & (~df['is_outlier'])].tolist()
    if len(clean_idx) < 4:
        print("  [DISAMB] Skipped (too few clean points)")
        return df

    flagged = 0
    i = 0
    while i < len(clean_idx) - 1:
        curr_i = clean_idx[i]
        next_i = clean_idx[i + 1]
        v_curr = df.loc[curr_i, 'ball_v']
        v_next = df.loc[next_i, 'ball_v']

        if abs(v_curr - v_next) < DISAMBIG_V_SPLIT:
            i += 1
            continue

        window = clean_idx[i: min(i + DISAMBIG_WINDOW, len(clean_idx))]
        cluster_a, cluster_b = [], []
        for widx in window:
            v = df.loc[widx, 'ball_v']
            (cluster_a if abs(v - v_curr) <= abs(v - v_next) else cluster_b).append(widx)

        if not cluster_a or not cluster_b:
            i += 1
            continue

        mean_a = df.loc[cluster_a, 'ball_conf'].mean()
        mean_b = df.loc[cluster_b, 'ball_conf'].mean()
        losers = cluster_b if mean_a >= mean_b else cluster_a

        for lidx in losers:
            if not df.loc[lidx, 'is_outlier']:
                df.loc[lidx, 'is_outlier']   = True
                df.loc[lidx, 'outlier_rule']  = 'low_conf_path'
                df.loc[lidx, 'clean_u']       = np.nan
                df.loc[lidx, 'clean_v']       = np.nan
                flagged += 1

        i += DISAMBIG_WINDOW

    print(f"  [DISAMB] {flagged} points removed by confidence path disambiguation")
    return df


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def interpolate_missing(df):
    df = df.copy()
    df['interp_u']        = df['clean_u'].copy()
    df['interp_v']        = df['clean_v'].copy()
    df['was_interpolated'] = False

    frames = df['frame_id'].values
    u      = df['interp_u'].values.copy()
    v      = df['interp_v'].values.copy()
    is_nan = np.isnan(u)

    filled = 0
    i = 0
    while i < len(u):
        if is_nan[i]:
            j = i
            while j < len(u) and is_nan[j]:
                j += 1
            if i > 0 and j < len(u):
                if int(frames[j] - frames[i-1]) - 1 <= MAX_INTERP_GAP:
                    u0, v0, u1, v1 = u[i-1], v[i-1], u[j], v[j]
                    for k in range(i, j):
                        t    = (frames[k] - frames[i-1]) / (frames[j] - frames[i-1])
                        u[k] = u0 + t * (u1 - u0)
                        v[k] = v0 + t * (v1 - v0)
                        filled += 1
            i = j
        else:
            i += 1

    df['interp_u']        = np.round(u, 2)
    df['interp_v']        = np.round(v, 2)
    df['was_interpolated'] = np.isnan(df['clean_u']) & ~np.isnan(df['interp_u'])
    print(f"  [INTERP] {filled} frames filled (gap<={MAX_INTERP_GAP}fr) | "
          f"{int(np.isnan(u).sum())} still NaN")
    return df


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export(df, path):
    priority = ['frame_id','detected','ball_u','ball_v','ball_conf',
                'is_outlier','outlier_rule','clean_u','clean_v',
                'interp_u','interp_v','was_interpolated']
    rest = [c for c in df.columns if c not in priority]
    df[priority + rest].to_csv(path, index=False)
    print(f"  [SAVE]  -> {path}")


# ---------------------------------------------------------------------------
# Main pipeline function  (called by run_all_cameras.py)
# ---------------------------------------------------------------------------

def clean_df(df, calib_path, alpha_cli=None, beta_cli=None, label='', y_net=None):
    """
    Run the full cleaning pipeline on an already-loaded DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw detection output (columns: frame_id, detected, ball_u, ball_v,
        ball_conf, player keypoints …).
    calib_path : str or None
        Path to the camera calibration JSON.  Pass None to use flat fallback.
    y_net : float, optional
        Pixel V-coordinate of the net.  When provided, R9 (half-court
        excursion filter) is applied.

    Returns
    -------
    (annotated_df, summary_dict)
    """
    spd, cal = load_calibration(calib_path, alpha_cli, beta_cli)
    y_far  = cal.get('y_far')
    y_near = cal.get('y_near')
    fps    = cal.get('fps', 25)

    print(f"  [SPEED] {spd.describe(y_far, y_near)}")
    print(f"          SNAPBACK_MULT={SNAPBACK_MULT}  "
          f"ISOLATION_MULT={ISOLATION_MULT}  "
          f"V_MAX_FLOOR={V_MAX_FLOOR} px/fr")
    print(f"  [R8]    MIN_CROSSING_FRAMES={MIN_CROSSING_FRAMES}  "
          f"(physics: court_diag/v_max*fps ≈ {23.77/55.6*fps:.1f} fr)")
    print()
    print(f"  [LOAD]  {len(df)} frames | {int(df['detected'].sum())} raw detections")

    df = flag_point_outliers(df, spd)
    df = flag_static_clusters(df)
    df = flag_short_segments(df)
    df = flag_crossing_violations(df, cal)
    if y_net is not None:
        df = flag_half_court_excursions(df, y_net)
    df = disambiguate_by_confidence(df)
    df = interpolate_missing(df)

    n_det    = int((df['detected'] == 1).sum())
    n_out    = int(df['is_outlier'].sum())
    n_interp = int(df['was_interpolated'].sum())
    n_final  = int(df['interp_u'].notna().sum())

    summary = {
        'camera':         label,
        'raw_detections': n_det,
        'outliers':       n_out,
        'outlier_pct':    round(100 * n_out / max(n_det, 1), 1),
        'clean':          n_det - n_out,
        'interpolated':   n_interp,
        'final_valid':    n_final,
        'by_rule':        df[df['is_outlier']]['outlier_rule'].value_counts().to_dict(),
    }
    return df, summary


def clean_one_camera(input_csv, output_csv, calib_path,
                     alpha_cli=None, beta_cli=None, label='', y_net=None):
    """
    File-based wrapper: read CSV → clean → write CSV.

    Parameters
    ----------
    y_net : float, optional
        Pixel V-coordinate of the net in this camera's view.  When provided,
        R9 (half-court excursion filter) is applied.

    Returns (annotated_df, summary_dict).
    """
    df = pd.read_csv(input_csv)
    df, summary = clean_df(df, calib_path, alpha_cli, beta_cli, label, y_net)
    export(df, output_csv)
    return df, summary
