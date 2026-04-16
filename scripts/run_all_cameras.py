"""
run_all_cameras.py
==================
Multi-camera batch entry point.  Runs the full cleaning pipeline for
every camera defined in CAMERAS, producing one cleaned CSV per camera,
then prints a combined summary table.

Configuration
-------------
Edit the CAMERAS list below to add or remove cameras.  Each entry is a
dict with:
    label       : short name used in log output and summary table
    input_csv   : raw model-output CSV for this camera
    output_csv  : where to write the cleaned CSV
    calib_json  : calibration JSON produced by calibrate_court.py

Usage
-----
    python run_all_cameras.py               # process all cameras
    python run_all_cameras.py --cam cam66   # process one camera by label
    python run_all_cameras.py --dry-run     # validate paths, do not process
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path so all packages resolve correctly
sys.path.insert(0, str(Path(__file__).parent.parent))

from postprocess.cleaner_core import clean_one_camera
from settings import OUTPUT_DIR, UPLOADS_DIR, CAMERAS_CFG

# =============================================================================
# CAMERA REGISTRY  (derived from settings.py / config.yaml — no hardcoded paths)
#
# Edit 'input_csv' to match the filename main.py produced.
# Pattern: output/{video_stem}_detections.csv
# =============================================================================
CAMERAS = [
    {
        'label':      'cam66',
        'input_csv':  str(OUTPUT_DIR / 'cam66_video_detections.csv'),
        'output_csv': str(OUTPUT_DIR / 'cam66_cleaned.csv'),
        'calib_json': str(UPLOADS_DIR / 'cal_cam66.json'),
        'y_net':      float(CAMERAS_CFG.get('cam66', {}).get('y_net', 238)),
    },
    {
        'label':      'cam68',
        'input_csv':  str(OUTPUT_DIR / 'cam68_video_detections.csv'),
        'output_csv': str(OUTPUT_DIR / 'cam68_cleaned.csv'),
        'calib_json': str(UPLOADS_DIR / 'cal_cam68.json'),
        'y_net':      float(CAMERAS_CFG.get('cam68', {}).get('y_net', 285)),
    },
]
# =============================================================================


def _build_parser():
    p = argparse.ArgumentParser(
        description='Multi-camera trajectory cleaning batch runner',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--cam',     default=None,
                   help='Process only this camera label (default: all)')
    p.add_argument('--dry-run', action='store_true',
                   help='Validate paths and config, do not process')
    return p


def _validate(cameras):
    """Check all input files and calibration JSONs exist. Return list of issues."""
    issues = []
    for cam in cameras:
        if not Path(cam['input_csv']).exists():
            issues.append(f"  MISSING input : {cam['input_csv']}  [{cam['label']}]")
        if not Path(cam['calib_json']).exists():
            issues.append(f"  MISSING calib : {cam['calib_json']}  [{cam['label']}]")
        Path(cam['output_csv']).parent.mkdir(parents=True, exist_ok=True)
    return issues


def _print_summary_table(summaries):
    """Print a compact side-by-side summary for all processed cameras."""
    SEP = '=' * 72
    print()
    print(SEP)
    print('  COMBINED SUMMARY')
    print(SEP)
    header = f"  {'Camera':<8}  {'Raw':>6}  {'Outliers':>9}  {'%':>5}  {'Clean':>6}  {'Interp':>7}  {'Final':>6}"
    print(header)
    print('  ' + '-' * 68)
    for s in summaries:
        print(f"  {s['camera']:<8}  {s['raw_detections']:>6}  "
              f"{s['outliers']:>9}  {s['outlier_pct']:>4.1f}%  "
              f"{s['clean']:>6}  {s['interpolated']:>7}  {s['final_valid']:>6}")
    print()

    # Per-rule breakdown
    all_rules = sorted({r for s in summaries for r in s['by_rule']})
    print(f"  {'Rule':<24}" + ''.join(f"  {s['camera']:>8}" for s in summaries))
    print('  ' + '-' * (24 + 10 * len(summaries)))
    for rule in all_rules:
        row = f"  {rule:<24}"
        for s in summaries:
            row += f"  {s['by_rule'].get(rule, 0):>8}"
        print(row)
    print(SEP)


def main():
    args    = _build_parser().parse_args()
    cameras = CAMERAS

    if args.cam:
        cameras = [c for c in cameras if c['label'] == args.cam]
        if not cameras:
            sys.exit(f"ERROR: no camera with label '{args.cam}' in registry")

    BIG_SEP = '=' * 64
    print(BIG_SEP)
    print(f"  MULTI-CAMERA TRAJECTORY CLEANER  ({len(cameras)} camera(s))")
    print(BIG_SEP)

    # Validate
    issues = _validate(cameras)
    missing_input = [i for i in issues if 'input' in i]
    missing_calib = [i for i in issues if 'calib' in i]

    if missing_calib:
        print('\n  WARNING: missing calibration files (flat fallback will be used):')
        for m in missing_calib: print(m)

    if missing_input:
        print('\n  ERROR: missing input detection files:')
        for m in missing_input: print(m)
        skip_labels = {i.split('[')[1].rstrip(']') for i in missing_input}
        cameras = [c for c in cameras if c['label'] not in skip_labels]
        if not cameras:
            sys.exit('  No cameras left to process. Exiting.')
        print(f'\n  Continuing with {len(cameras)} camera(s)...')

    if args.dry_run:
        print('\n  [DRY-RUN] Paths validated. No processing performed.')
        for c in cameras:
            status_i = 'OK' if Path(c['input_csv']).exists()  else 'MISSING'
            status_c = 'OK' if Path(c['calib_json']).exists() else 'MISSING'
            print(f"    {c['label']:<8}  input={status_i}  calib={status_c}"
                  f"  -> {c['output_csv']}")
        return

    # Process each camera
    summaries  = []
    t_start    = time.time()

    for cam in cameras:
        print()
        print(f"  {'─'*60}")
        print(f"  Camera: {cam['label']}")
        print(f"  {'─'*60}")

        _, summary = clean_one_camera(
            input_csv  = cam['input_csv'],
            output_csv = cam['output_csv'],
            calib_path = cam['calib_json'],
            label      = cam['label'],
            y_net      = cam.get('y_net'),
        )
        summaries.append(summary)

    elapsed = time.time() - t_start
    _print_summary_table(summaries)
    print(f"  Total time: {elapsed:.1f}s")
    print()
    print('  Output files:')
    for cam in cameras:
        print(f"    {cam['label']:<8} -> {cam['output_csv']}")
    print(BIG_SEP)


if __name__ == '__main__':
    main()