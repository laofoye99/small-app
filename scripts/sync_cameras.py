"""
sync_cameras.py
===============
Frame-level synchronisation of two camera trajectories using
cross-correlation of vector acceleration signals.

Pipeline
--------
  1. Load both cleaned trajectory CSVs (output of run_all_cameras.py)
  2. Build a dense vector-acceleration signal per camera
       a(t) = |v(t+1) - v(t)|   where v(t) = [delta_u, delta_v]
     NaN gaps (ball not visible) are zero-filled so scipy.signal.correlate
     can operate on the full frame array.
  3. Z-score normalise both signals to remove perspective-scale differences
     between cameras (px_width_near differs per camera).
  4. Cross-correlate -> tau_offset = lag of global maximum
       positive tau : cam_b lags cam_a  (cam_b started recording later)
       negative tau : cam_b leads cam_a
  5. Detect impact/landing peaks in each aligned signal using
     scipy.signal.find_peaks.
  6. Export:
       - sync_result.json   : tau_offset + peak frames per camera
       - cam_a_synced.csv   : cam_a DataFrame with aligned_frame_id column
       - cam_b_synced.csv   : cam_b DataFrame with aligned_frame_id column

Usage
-----
    # normal: use paths defined in CAMERAS below
    python sync_cameras.py

    # override any path on the CLI
    python sync_cameras.py \\
        --input-a  /path/to/cam66_cleaned.csv \\
        --input-b  /path/to/cam68_cleaned.csv \\
        --output-a /path/to/cam66_synced.csv  \\
        --output-b /path/to/cam68_synced.csv  \\
        --out-json /path/to/sync_result.json

    # inspect the correlation plot without writing files
    python sync_cameras.py --plot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import correlate, find_peaks

# ---------------------------------------------------------------------------
# Resolve paths via settings.py (location-independent)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import OUTPUT_DIR

# =============================================================================
# CONFIGURATION  (defaults — all override-able via CLI)
# =============================================================================

CAMERAS = {
    'cam_a': {
        'label':     'cam66',
        'input_csv': str(OUTPUT_DIR / 'cam66_cleaned.csv'),
        'out_csv':   str(OUTPUT_DIR / 'cam66_synced.csv'),
    },
    'cam_b': {
        'label':     'cam68',
        'input_csv': str(OUTPUT_DIR / 'cam68_cleaned.csv'),
        'out_csv':   str(OUTPUT_DIR / 'cam68_synced.csv'),
    },
}

OUT_JSON = str(OUTPUT_DIR / 'sync_result.json')

# Peak detection parameters
PEAK_HEIGHT_ABS   = 30.0    # minimum |accel| to qualify as a peak (px/frame²)
PEAK_DISTANCE     = 5       # minimum frames between two peaks
PEAK_PROMINENCE   = 10.0    # minimum prominence (suppresses noise shoulders)

# Cross-correlation search window
#   Set to None to search the entire range.
#   Set to e.g. 150 to restrict search to ±150 frames (speeds up + avoids
#   spurious far-off matches when you know cameras are nearly synchronised).
MAX_LAG_FRAMES    = 100     # constrained: cameras recorded same session

# =============================================================================


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------

def build_accel_signal(df: pd.DataFrame, n_frames: int) -> np.ndarray:
    """
    Build a dense vector-acceleration signal of length n_frames.

    For each frame t where t-1, t, t+1 all have valid (non-NaN)
    interp_u / interp_v values:

        v_in (t)  = [u(t)   - u(t-1),  v(t)   - v(t-1)]
        v_out(t)  = [u(t+1) - u(t),    v(t+1) - v(t)  ]
        a(t)      = |v_out - v_in|

    Frames without three consecutive valid neighbours are set to 0.0
    (not NaN) so that scipy.signal.correlate can operate without gaps.

    Why vector and not scalar?
    --------------------------
    Scalar acceleration |v(t+1)| - |v(t)| cancels out at a bounce where
    the ball reverses direction but keeps its speed.  The vector form
    |v_out - v_in| spikes at BOTH speed changes AND direction reversals,
    making it far more sensitive to landing and impact events.
    """
    u = np.full(n_frames, np.nan)
    v = np.full(n_frames, np.nan)

    for _, row in df.iterrows():
        fid = int(row['frame_id'])
        if fid < n_frames and not np.isnan(row['interp_u']):
            u[fid] = row['interp_u']
            v[fid] = row['interp_v']

    accel = np.zeros(n_frames)
    for t in range(1, n_frames - 1):
        if np.isnan(u[t-1]) or np.isnan(u[t]) or np.isnan(u[t+1]):
            continue
        v_in  = np.array([u[t]   - u[t-1], v[t]   - v[t-1]])
        v_out = np.array([u[t+1] - u[t],   v[t+1] - v[t]])
        accel[t] = np.linalg.norm(v_out - v_in)

    return accel


def normalize(sig: np.ndarray) -> np.ndarray:
    """
    Z-score normalise.  Removes amplitude differences caused by
    different px_width_near values between cameras so that the
    cross-correlation is not dominated by the larger-scale camera.
    """
    mu, sd = sig.mean(), sig.std()
    if sd < 1e-9:
        return sig - mu
    return (sig - mu) / sd


# ---------------------------------------------------------------------------
# Cross-correlation alignment
# ---------------------------------------------------------------------------

def compute_offset(sig_a: np.ndarray,
                   sig_b: np.ndarray,
                   max_lag: int | None = None) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Cross-correlate two normalised signals and return the lag at which
    cam_b best matches cam_a.

    Returns
    -------
    tau_offset : int
        Frames by which cam_b should be shifted to align with cam_a.
        Positive  -> cam_b lags cam_a  (cam_b started later)
        Negative  -> cam_b leads cam_a (cam_b started earlier)
    lags : np.ndarray
        Full lag axis (frames).
    corr : np.ndarray
        Full cross-correlation array (same length as lags).
    """
    na = normalize(sig_a)
    nb = normalize(sig_b)

    corr = correlate(na, nb, mode='full')
    N    = len(sig_a)
    lags = np.arange(-(N - 1), N)      # lag axis in frames

    if max_lag is not None:
        mask = np.abs(lags) <= max_lag
        search_corr = np.where(mask, corr, -np.inf)
    else:
        search_corr = corr

    tau_offset = int(lags[np.argmax(search_corr)])
    return tau_offset, lags, corr


# ---------------------------------------------------------------------------
# Peak detection
# ---------------------------------------------------------------------------

def detect_peaks(accel: np.ndarray,
                 height:      float = PEAK_HEIGHT_ABS,
                 distance:    int   = PEAK_DISTANCE,
                 prominence:  float = PEAK_PROMINENCE) -> np.ndarray:
    """
    Detect impact / landing frames as local maxima in the acceleration signal.

    Parameters chosen conservatively:
      height     -- absolute minimum (rejects flat-signal noise)
      distance   -- minimum spacing (rejects double-peaks from one event)
      prominence -- peak must stand out from surroundings (rejects shoulders)
    """
    peak_idx, _ = find_peaks(
        accel,
        height     = height,
        distance   = distance,
        prominence = prominence,
    )
    return peak_idx


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def add_aligned_frame(df: pd.DataFrame, tau: int, role: str) -> pd.DataFrame:
    """
    Add an 'aligned_frame_id' column to a camera DataFrame.

    cam_a (role='a'): aligned_frame_id = frame_id  (cam_a is the reference)
    cam_b (role='b'): aligned_frame_id = frame_id - tau_offset
                      (shift cam_b so its timeline matches cam_a)

    A negative aligned_frame_id means the frame precedes the cam_a
    recording window -- these rows are kept but flagged.
    """
    df = df.copy()
    if role == 'a':
        df['aligned_frame_id'] = df['frame_id']
    else:
        df['aligned_frame_id'] = df['frame_id'] - tau
    return df


def save_json(tau: int,
              peaks_a: np.ndarray,
              peaks_b_aligned: np.ndarray,
              label_a: str,
              label_b: str,
              corr_peak_value: float,
              path: str) -> None:
    result = {
        'tau_offset':           tau,
        'interpretation':       (
            f"{label_b} lags {label_a} by {tau} frames"
            if tau >= 0 else
            f"{label_b} leads {label_a} by {abs(tau)} frames"
        ),
        'corr_peak_value':      round(float(corr_peak_value), 4),
        f'{label_a}_impact_frames': peaks_a.tolist(),
        f'{label_b}_impact_frames_aligned': peaks_b_aligned.tolist(),
    }
    Path(path).write_text(json.dumps(result, indent=2))
    print(f"  [JSON]  -> {path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def sync_dfs(df_a: 'pd.DataFrame',
             df_b: 'pd.DataFrame',
             label_a: str = 'cam66',
             label_b: str = 'cam68',
             out_json: str | None = None) -> tuple:
    """
    Synchronise two already-loaded cleaned DataFrames.

    Parameters
    ----------
    df_a, df_b : pd.DataFrame
        Cleaned trajectory DataFrames (output of ``clean_df``/``clean_one_camera``).
    label_a, label_b : str
        Camera labels used in log output and the optional JSON export.
    out_json : str or None
        If given, write sync metadata to this path.

    Returns
    -------
    df_a_out : pd.DataFrame  — df_a with ``aligned_frame_id`` added
    df_b_out : pd.DataFrame  — df_b with ``aligned_frame_id`` added
    result   : dict          — tau_offset, corr_snr, peak counts
    """
    import pandas as pd  # local import so the module stays importable without pandas

    n_a = int(df_a['frame_id'].max()) + 1
    n_b = int(df_b['frame_id'].max()) + 1
    N   = max(n_a, n_b)
    print(f"  Frame counts: {label_a}={n_a}  {label_b}={n_b}  using N={N}")

    accel_a = build_accel_signal(df_a, N)
    accel_b = build_accel_signal(df_b, N)

    tau, lags, corr = compute_offset(accel_a, accel_b, MAX_LAG_FRAMES)

    corr_peak = float(corr.max())
    corr_mean = float(np.abs(corr).mean())
    snr       = corr_peak / (corr_mean + 1e-9)

    print(f"  tau_offset = {tau} frames  |  Corr SNR = {snr:.2f}")
    if snr < 3.0:
        print(f"  WARNING: low SNR ({snr:.2f}) -- offset may be unreliable.")

    accel_b_aligned = np.zeros(N)
    if tau >= 0:
        accel_b_aligned[tau:] = accel_b[:N - tau]
    else:
        accel_b_aligned[:N + tau] = accel_b[-tau:]

    peaks_a = detect_peaks(accel_a)
    peaks_b = detect_peaks(accel_b_aligned)

    df_a_out = add_aligned_frame(df_a, tau, role='a')
    df_b_out = add_aligned_frame(df_b, tau, role='b')

    priority = ['frame_id', 'aligned_frame_id', 'detected',
                'ball_u', 'ball_v', 'ball_conf',
                'is_outlier', 'outlier_rule',
                'clean_u', 'clean_v', 'interp_u', 'interp_v',
                'was_interpolated']
    rest_a = [c for c in df_a_out.columns if c not in priority]
    rest_b = [c for c in df_b_out.columns if c not in priority]
    df_a_out = df_a_out[priority + rest_a]
    df_b_out = df_b_out[priority + rest_b]

    if out_json:
        save_json(tau, peaks_a, peaks_b, label_a, label_b, corr_peak, out_json)

    result = {
        'tau_offset':    tau,
        'corr_snr':      round(snr, 2),
        'peaks_a':       len(peaks_a),
        'peaks_b':       len(peaks_b),
    }
    return df_a_out, df_b_out, result


def run(input_a:  str,
        input_b:  str,
        output_a: str,
        output_b: str,
        out_json: str,
        plot:     bool = False) -> dict:

    SEP = '=' * 64
    print(SEP)
    print('  CAMERA SYNCHRONISATION  (cross-correlation of vector accel)')
    print(SEP)

    label_a = CAMERAS['cam_a']['label']
    label_b = CAMERAS['cam_b']['label']

    print(f"\n  Loading {label_a}: {input_a}")
    df_a = pd.read_csv(input_a)
    print(f"  Loading {label_b}: {input_b}")
    df_b = pd.read_csv(input_b)

    # ── Build signals (needed for non-zero count log and optional plot) ──────
    print(f"\n  Building vector acceleration signals...")
    n_a = int(df_a['frame_id'].max()) + 1
    n_b = int(df_b['frame_id'].max()) + 1
    N   = max(n_a, n_b)
    accel_a = build_accel_signal(df_a, N)
    accel_b = build_accel_signal(df_b, N)
    print(f"  Non-zero accel frames: {label_a}={(accel_a > 0).sum()}  "
          f"{label_b}={(accel_b > 0).sum()}")

    print(f"\n  Cross-correlating (max_lag={MAX_LAG_FRAMES})...")
    df_a_out, df_b_out, result = sync_dfs(df_a, df_b, label_a, label_b, out_json)

    tau = result['tau_offset']
    snr = result['corr_snr']

    # ── Write output CSVs ────────────────────────────────────────────────────
    print(f"\n  Writing output CSVs...")
    df_a_out.to_csv(output_a, index=False)
    df_b_out.to_csv(output_b, index=False)
    print(f"  [CSV]   -> {output_a}")
    print(f"  [CSV]   -> {output_b}")

    # ── Optional plot ────────────────────────────────────────────────────────
    if plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            # Re-derive signals needed for plotting
            _, lags, corr = compute_offset(accel_a, accel_b, MAX_LAG_FRAMES)
            corr_peak = float(corr.max())
            accel_b_aligned = np.zeros(N)
            if tau >= 0:
                accel_b_aligned[tau:] = accel_b[:N - tau]
            else:
                accel_b_aligned[:N + tau] = accel_b[-tau:]
            peaks_a = detect_peaks(accel_a)
            peaks_b = detect_peaks(accel_b_aligned)

            fig, axes = plt.subplots(3, 1, figsize=(18, 12),
                                     facecolor='#0d1117')
            BG = '#161b22'; GR = '#21262d'

            def style(ax, title):
                ax.set_facecolor(BG)
                ax.grid(color=GR, lw=0.5)
                ax.set_title(title, color='#e6edf3', fontsize=10, fontweight='bold')
                ax.tick_params(colors='#8b949e')
                [s.set_color('#30363d') for s in ax.spines.values()]

            t  = np.arange(N)
            na = normalize(accel_a)
            nb = normalize(accel_b)

            ax = axes[0]
            style(ax, 'Vector Acceleration Signals (normalised)')
            ax.plot(t, na, color='#58a6ff', lw=0.8, alpha=0.8, label=label_a)
            ax.plot(t, nb, color='#3fb950', lw=0.8, alpha=0.8, label=label_b)
            ax.scatter(peaks_a, na[peaks_a], color='#58a6ff', s=40,
                       zorder=5, marker='^')
            ax.scatter(peaks_b, nb[peaks_b] if len(peaks_b) else [],
                       color='#3fb950', s=40, zorder=5, marker='^')
            ax.legend(facecolor=BG, edgecolor='#30363d',
                      labelcolor='#e6edf3', fontsize=8)
            ax.set_xlabel('Frame', color='#8b949e', fontsize=8)

            ax = axes[1]
            style(ax, f'Cross-Correlation  (tau_offset = {tau} fr)')
            ax.plot(lags, corr, color='#ce93d8', lw=0.9)
            ax.axvline(tau, color='#f85149', lw=2, ls='--',
                       label=f'peak lag = {tau}')
            ax.axhline(corr_peak * 0.5, color='#ff9800', lw=0.8,
                       ls=':', label='50% peak')
            ax.legend(facecolor=BG, edgecolor='#30363d',
                      labelcolor='#e6edf3', fontsize=8)
            ax.set_xlabel('Lag (frames)', color='#8b949e', fontsize=8)

            ax = axes[2]
            style(ax, f'Aligned Signals  ({label_b} shifted by -{tau} fr)')
            ax.plot(t, na, color='#58a6ff', lw=0.8, alpha=0.8, label=label_a)
            ax.plot(t, normalize(accel_b_aligned), color='#3fb950', lw=0.8, alpha=0.8,
                    label=f'{label_b} (aligned)')
            ax.legend(facecolor=BG, edgecolor='#30363d',
                      labelcolor='#e6edf3', fontsize=8)
            ax.set_xlabel('Aligned Frame', color='#8b949e', fontsize=8)

            fig.suptitle('Camera Synchronisation via Vector Acceleration '
                         'Cross-Correlation',
                         color='#e6edf3', fontsize=13, fontweight='bold')
            plt.tight_layout()
            plot_path = str(Path(out_json).parent / 'sync_plot.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight',
                        facecolor='#0d1117')
            print(f"  [PLOT]  -> {plot_path}")
        except Exception as e:
            print(f"  [PLOT]  Skipped ({e})")

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print(SEP)
    print('  SUMMARY')
    print(SEP)
    print(f"  tau_offset       : {tau} frames")
    print(f"  Corr SNR         : {snr:.2f}")
    print(f"  {label_a} impact peaks : {result['peaks_a']}")
    print(f"  {label_b} impact peaks : {result['peaks_b']}")
    print(SEP)
    print()
    print('  Column added to both output CSVs:')
    print('    aligned_frame_id  -- common timeline (cam_a is reference)')
    print('                         use this column for cross-camera comparison')
    print(SEP)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    p = argparse.ArgumentParser(
        description='Synchronise two camera trajectories via cross-correlation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--input-a',  default=CAMERAS['cam_a']['input_csv'])
    p.add_argument('--input-b',  default=CAMERAS['cam_b']['input_csv'])
    p.add_argument('--output-a', default=CAMERAS['cam_a']['out_csv'])
    p.add_argument('--output-b', default=CAMERAS['cam_b']['out_csv'])
    p.add_argument('--out-json', default=OUT_JSON)
    p.add_argument('--plot',     action='store_true',
                   help='Save correlation + signal plot to PNG')
    return p


if __name__ == '__main__':
    args = _build_parser().parse_args()
    run(
        input_a  = args.input_a,
        input_b  = args.input_b,
        output_a = args.output_a,
        output_b = args.output_b,
        out_json = args.out_json,
        plot     = args.plot,
    )