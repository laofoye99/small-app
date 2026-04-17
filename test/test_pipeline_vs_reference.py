"""
test_pipeline_vs_reference.py
==============================
Compare the cleaned output produced by live_pipeline's cleaning stage
against pre-saved reference CSVs (output/cam66_cleaned.csv, cam68_cleaned.csv).

Pipeline under test
-------------------
  cam66_video_detections.csv  →  clean_df()  →  compare vs  cam66_cleaned.csv
  cam68_video_detections.csv  →  clean_df()  →  compare vs  cam68_cleaned.csv

Outputs  (all under test/results/)
-------
  comparison_cam66.csv / comparison_cam68.csv   – per-frame error table
  summary.csv                                   – aggregate MAE / RMSE / match rate
  plots/ball_trajectory_<cam>.png               – trajectory overlay
  plots/ball_error_over_time_<cam>.png          – frame-by-frame position error
  plots/player_keypoint_error_<cam>.png         – MAE per joint per player

Usage
-----
    cd <project-root>
    python test/test_pipeline_vs_reference.py

    # override raw-detection CSVs or reference CSVs:
    python test/test_pipeline_vs_reference.py \
        --raw66 output/cam66_video_detections.csv \
        --ref66 output/cam66_cleaned.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path bootstrap – allow running from repo root or from test/
# ---------------------------------------------------------------------------

_HERE    = Path(__file__).parent.resolve()
_ROOT    = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from settings import CALIB_CAM66, CALIB_CAM68, CAMERAS_CFG
from postprocess.cleaner_core import clean_df

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_JOINTS = ['ls', 'rs', 'le', 're', 'lw', 'rw']

_PLAYER_COLS = [
    f'p{p}_{j}_{c}'
    for p in range(2)
    for j in _JOINTS
    for c in ('u', 'v')
]

_BALL_INTERP = ['interp_u', 'interp_v']
_BALL_CLEAN  = ['clean_u',  'clean_v']

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['detected'] = df['detected'].fillna(0).astype(int)
    df['ball_conf'] = df['ball_conf'].fillna(0.0)
    return df


def _run_clean(raw_df: pd.DataFrame, calib_path: str, y_net: float,
               label: str) -> pd.DataFrame:
    cleaned, summary = clean_df(raw_df, calib_path=calib_path, y_net=y_net, label=label)
    print(f"  [{label}] clean_df: raw_det={summary['raw_detections']}  "
          f"clean={summary['clean']}  outliers={summary['outliers']}  "
          f"interp={summary['interpolated']}")
    return cleaned


def _align(actual: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Inner-join on frame_id so both sides are aligned."""
    m = actual.merge(ref, on='frame_id', suffixes=('_act', '_ref'), how='inner')
    return m


def _col_err(merged: pd.DataFrame, col: str) -> pd.Series:
    a = merged.get(f'{col}_act', merged.get(col))
    r = merged.get(f'{col}_ref', merged.get(col))
    if a is None or r is None:
        return pd.Series(dtype=float)
    return (a - r).abs()


def _rmse(series: pd.Series) -> float:
    s = series.dropna()
    return float(np.sqrt((s ** 2).mean())) if len(s) else float('nan')


def _mae(series: pd.Series) -> float:
    s = series.dropna()
    return float(s.mean()) if len(s) else float('nan')


# ---------------------------------------------------------------------------
# Per-frame comparison table
# ---------------------------------------------------------------------------

def build_comparison(actual: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    m = _align(actual, ref)
    rows = pd.DataFrame({'frame_id': m['frame_id']})

    # Ball position error
    for col in _BALL_INTERP + _BALL_CLEAN:
        e = _col_err(m, col)
        if not e.empty:
            rows[f'err_{col}'] = e.values

    # Euclidean ball error (interp)
    eu = _col_err(m, 'interp_u')
    ev = _col_err(m, 'interp_v')
    if not eu.empty and not ev.empty:
        rows['err_ball_interp_px'] = np.sqrt(eu.values**2 + ev.values**2)

    # Player joint errors
    for col in _PLAYER_COLS:
        if f'{col}_act' in m.columns and f'{col}_ref' in m.columns:
            rows[f'err_{col}'] = (m[f'{col}_act'] - m[f'{col}_ref']).abs().values

    # Flag agreement
    if 'is_outlier_act' in m.columns and 'is_outlier_ref' in m.columns:
        rows['outlier_agree'] = (m['is_outlier_act'] == m['is_outlier_ref']).astype(int)

    if 'was_interpolated_act' in m.columns and 'was_interpolated_ref' in m.columns:
        rows['interp_agree'] = (
            m['was_interpolated_act'] == m['was_interpolated_ref']
        ).astype(int)

    # Carry useful columns from actual
    for col in ['detected', 'is_outlier', 'was_interpolated']:
        act_col = f'{col}_act'
        if act_col in m.columns:
            rows[col] = m[act_col].values
        elif col in m.columns:
            rows[col] = m[col].values

    return rows


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def build_summary(comparisons: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records = []
    for cam, df in comparisons.items():
        rec: dict = {'camera': cam}

        # Ball
        if 'err_ball_interp_px' in df.columns:
            rec['ball_interp_mae_px']  = _mae(df['err_ball_interp_px'])
            rec['ball_interp_rmse_px'] = _rmse(df['err_ball_interp_px'])

        for axis in ('u', 'v'):
            col = f'err_interp_{axis}'
            if col in df.columns:
                rec[f'ball_interp_{axis}_mae'] = _mae(df[col])

        # Outlier / interp agreement
        for flag_col in ('outlier_agree', 'interp_agree'):
            if flag_col in df.columns:
                rec[f'{flag_col}_rate'] = df[flag_col].mean()

        # Player joints
        for p in range(2):
            joint_maes = []
            for j in _JOINTS:
                errs = []
                for ax in ('u', 'v'):
                    col = f'err_p{p}_{j}_{ax}'
                    if col in df.columns:
                        errs.append(_mae(df[col]))
                if errs:
                    jmae = float(np.mean([e for e in errs if not np.isnan(e)]))
                    rec[f'p{p}_{j}_mae_px'] = round(jmae, 3)
                    joint_maes.append(jmae)
            if joint_maes:
                valid = [e for e in joint_maes if not np.isnan(e)]
                rec[f'p{p}_avg_joint_mae_px'] = round(float(np.mean(valid)), 3) if valid else float('nan')

        records.append(rec)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _ball_trajectory_plot(actual: pd.DataFrame, ref: pd.DataFrame,
                           cam: str, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(f'{cam} – Ball Trajectory: Actual vs Reference', fontsize=13)

    for ax, (ucol, vcol, title) in zip(axes, [
        ('interp_u', 'interp_v', 'Interpolated trajectory'),
        ('clean_u',  'clean_v',  'Clean (pre-interp) trajectory'),
    ]):
        for df, label, color, alpha in [
            (ref,    'Reference', '#e74c3c', 0.7),
            (actual, 'Actual',    '#2ecc71', 0.7),
        ]:
            mask = df[ucol].notna() & df[vcol].notna()
            ax.scatter(df.loc[mask, ucol], df.loc[mask, vcol],
                       s=2, c=color, alpha=alpha, label=label)
        ax.set_xlabel('u (px)')
        ax.set_ylabel('v (px)')
        ax.set_title(title)
        ax.invert_yaxis()
        ax.legend(markerscale=4)

    plt.tight_layout()
    out = out_dir / f'ball_trajectory_{cam}.png'
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


def _ball_error_over_time_plot(comp: pd.DataFrame, cam: str,
                                out_dir: Path) -> None:
    if 'err_ball_interp_px' not in comp.columns:
        return

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(comp['frame_id'], comp['err_ball_interp_px'],
            lw=0.7, color='#3498db', alpha=0.8)
    ax.set_xlabel('Frame ID')
    ax.set_ylabel('Ball position error (px)')
    ax.set_title(f'{cam} – Ball Trajectory Error per Frame')
    mae_val = _mae(comp['err_ball_interp_px'])
    ax.axhline(mae_val, color='red', lw=1.2, ls='--',
               label=f'MAE = {mae_val:.2f} px')
    ax.legend()

    plt.tight_layout()
    out = out_dir / f'ball_error_over_time_{cam}.png'
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


def _player_keypoint_error_plot(comp: pd.DataFrame, cam: str,
                                 out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle(f'{cam} – Player Keypoint MAE (px)', fontsize=13)

    for ax, pid in zip(axes, [0, 1]):
        maes, labels = [], []
        for j in _JOINTS:
            vals = []
            for ax_c in ('u', 'v'):
                col = f'err_p{pid}_{j}_{ax_c}'
                if col in comp.columns:
                    vals.append(_mae(comp[col]))
            if vals:
                valid = [v for v in vals if not np.isnan(v)]
                maes.append(float(np.mean(valid)) if valid else float('nan'))
                labels.append(j)

        colors = ['#2ecc71' if m < 5 else '#e67e22' if m < 15 else '#e74c3c'
                  for m in maes]
        bars = ax.barh(labels, maes, color=colors)
        ax.set_xlabel('MAE (px)')
        ax.set_title(f'Player {pid}')
        for bar, v in zip(bars, maes):
            if not np.isnan(v):
                ax.text(v + 0.3, bar.get_y() + bar.get_height() / 2,
                        f'{v:.1f}', va='center', fontsize=8)

    plt.tight_layout()
    out = out_dir / f'player_keypoint_error_{cam}.png'
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Compare live_pipeline cleaning output vs reference CSVs'
    )
    parser.add_argument('--raw66',  default=str(_ROOT / 'output' / 'cam66_video_detections.csv'))
    parser.add_argument('--raw68',  default=str(_ROOT / 'output' / 'cam68_video_detections.csv'))
    parser.add_argument('--ref66',  default=str(_ROOT / 'output' / 'cam66_cleaned.csv'))
    parser.add_argument('--ref68',  default=str(_ROOT / 'output' / 'cam68_cleaned.csv'))
    parser.add_argument('--out',    default=str(_HERE / 'results'),
                        help='Output directory for CSVs and plots')
    args = parser.parse_args()

    out_dir   = Path(args.out)
    plots_dir = out_dir / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    cam_configs = {
        'cam66': {
            'raw':   Path(args.raw66),
            'ref':   Path(args.ref66),
            'calib': str(CALIB_CAM66),
            'y_net': float(CAMERAS_CFG.get('cam66', {}).get('y_net', 238.0)),
        },
        'cam68': {
            'raw':   Path(args.raw68),
            'ref':   Path(args.ref68),
            'calib': str(CALIB_CAM68),
            'y_net': float(CAMERAS_CFG.get('cam68', {}).get('y_net', 285.0)),
        },
    }

    comparisons: dict[str, pd.DataFrame] = {}

    for cam, cfg in cam_configs.items():
        print(f"\n{'='*60}")
        print(f"Camera: {cam}")
        print(f"  Raw detections : {cfg['raw']}")
        print(f"  Reference      : {cfg['ref']}")

        if not cfg['raw'].exists():
            print(f"  [SKIP] Raw CSV not found: {cfg['raw']}")
            continue
        if not cfg['ref'].exists():
            print(f"  [SKIP] Reference CSV not found: {cfg['ref']}")
            continue

        raw_df = _load_raw(cfg['raw'])
        print(f"  Loaded raw: {len(raw_df)} rows  (frames {raw_df['frame_id'].min()}–{raw_df['frame_id'].max()})")

        actual = _run_clean(raw_df, cfg['calib'], cfg['y_net'], label=cam)
        ref    = pd.read_csv(cfg['ref'])

        print(f"  Actual cleaned: {len(actual)} rows")
        print(f"  Reference     : {len(ref)} rows")
        print(f"  Frame overlap : {len(set(actual['frame_id']) & set(ref['frame_id']))} frames")

        comp = build_comparison(actual, ref)
        comparisons[cam] = comp

        comp_path = out_dir / f'comparison_{cam}.csv'
        comp.to_csv(comp_path, index=False)
        print(f"  Saved per-frame comparison: {comp_path}")

        # Plots
        _ball_trajectory_plot(actual, ref, cam, plots_dir)
        _ball_error_over_time_plot(comp, cam, plots_dir)
        _player_keypoint_error_plot(comp, cam, plots_dir)

    if not comparisons:
        print("\nNo cameras processed. Check that raw/reference CSVs exist.")
        return

    print(f"\n{'='*60}")
    print("Summary statistics")
    summary = build_summary(comparisons)
    summary_path = out_dir / 'summary.csv'
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved: {summary_path}")
    print(f"All outputs written to: {out_dir}")


if __name__ == '__main__':
    main()
