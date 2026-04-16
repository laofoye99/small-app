#!/usr/bin/env python3
"""
Court homography annotation tool — 12 keypoints.

Lets you click 12 standard court landmarks on a video frame and computes a
perspective homography H that maps image pixels to normalised court coordinates
(x ∈ [0,1] left→right, y ∈ [0,1] far→near baseline).

Standard singles court layout used as world reference
------------------------------------------------------
  Court length : 23.77 m   (baseline to baseline)
  Court width  :  8.23 m   (singles sideline to sideline)
  Service line :  6.40 m   from each baseline
  Net          : 11.885 m  from each baseline  (y = 0.5 in normalised space)

  (0,0)─────────────────(1,0)   ← far  baseline
    │   (0, SL)─T─(1,SL)  │    ← far  service line   SL = 6.40/23.77 ≈ 0.269
    │      │   │   │       │
  (0,.5)───────────────(1,.5)   ← net
    │      │   │   │       │
    │(0,1-SL)─T─(1,1-SL)  │    ← near service line
  (0,1)─────────────────(1,1)   ← near baseline

Click order
-----------
  1  Far  baseline – LEFT  singles corner
  2  Far  baseline – RIGHT singles corner
  3  Far  service line – LEFT  sideline
  4  Far  service line – RIGHT sideline
  5  Far  T-point (service × centre service line)
  6  Net  – LEFT  sideline
  7  Net  – RIGHT sideline
  8  Near T-point (service × centre service line)
  9  Near service line – LEFT  sideline
  10 Near service line – RIGHT sideline
  11 Near baseline – LEFT  singles corner
  12 Near baseline – RIGHT singles corner

Usage
-----
python mark_court.py <video.mp4> [--frame 0] [--out court_H.json]

Controls
--------
  d / →   next frame      a / ←   previous frame
  f       jump to frame N
  r       undo last click
  s       compute H and save     q   quit without saving
  scroll  zoom in / out          middle-drag  pan
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Court geometry
# ---------------------------------------------------------------------------

_SL = 6.40 / 23.77   # service line normalised distance from each baseline

# 12 world points in normalised court coordinates, matching the click order above
WORLD_PTS: List[Tuple[float, float]] = [
    (0.0, 0.0),        #  1  far-left  baseline
    (1.0, 0.0),        #  2  far-right baseline
    (0.0, _SL),        #  3  far-left  service
    (1.0, _SL),        #  4  far-right service
    (0.5, _SL),        #  5  far T-point
    (0.0, 0.5),        #  6  net-left
    (1.0, 0.5),        #  7  net-right
    (0.5, 1.0 - _SL),  #  8  near T-point
    (0.0, 1.0 - _SL),  #  9  near-left  service
    (1.0, 1.0 - _SL),  # 10  near-right service
    (0.0, 1.0),        # 11  near-left  baseline
    (1.0, 1.0),        # 12  near-right baseline
]

PROMPTS = [
    " 1/12 — Far  baseline      : click LEFT  corner",
    " 2/12 — Far  baseline      : click RIGHT corner",
    " 3/12 — Far  service line  : click LEFT  sideline",
    " 4/12 — Far  service line  : click RIGHT sideline",
    " 5/12 — Far  T-point       : click centre service line",
    " 6/12 — Net                : click LEFT  sideline",
    " 7/12 — Net                : click RIGHT sideline",
    " 8/12 — Near T-point       : click centre service line",
    " 9/12 — Near service line  : click LEFT  sideline",
    "10/12 — Near service line  : click RIGHT sideline",
    "11/12 — Near baseline      : click LEFT  corner",
    "12/12 — Near baseline      : click RIGHT corner",
]

# Colour per click group
_COLOURS = [
    (0, 200, 255),   # far baseline (1-2)
    (0, 200, 255),
    (0, 160, 200),   # far service (3-5)
    (0, 160, 200),
    (0, 160, 200),
    (200, 200, 0),   # net (6-7)
    (200, 200, 0),
    (80, 200, 80),   # near service (8-10)
    (80, 200, 80),
    (80, 200, 80),
    (0, 255, 100),   # near baseline (11-12)
    (0, 255, 100),
]

_LABELS = [
    "FAR-L-BASE", "FAR-R-BASE",
    "FAR-L-SVC",  "FAR-R-SVC",  "FAR-T",
    "NET-L",      "NET-R",
    "NEAR-T",     "NEAR-L-SVC", "NEAR-R-SVC",
    "NEAR-L-BASE","NEAR-R-BASE",
]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class _State:
    def __init__(self, frame: np.ndarray) -> None:
        self.orig:        np.ndarray              = frame.copy()
        self.clicks:      List[Tuple[int, int]]   = []
        self.zoom:        float                   = 1.0
        self.pan:         List[float]             = [0.0, 0.0]
        self._drag_start: Optional[Tuple[int,int]] = None
        self._pan_start:  Optional[List[float]]   = None

    def win_to_img(self, wx: int, wy: int) -> Tuple[int, int]:
        ix = int(wx / self.zoom + self.pan[0])
        iy = int(wy / self.zoom + self.pan[1])
        h, w = self.orig.shape[:2]
        return int(np.clip(ix, 0, w - 1)), int(np.clip(iy, 0, h - 1))

    def render(self) -> np.ndarray:
        h, w = self.orig.shape[:2]
        vis  = self.orig.copy()

        for i, (px, py) in enumerate(self.clicks):
            col = _COLOURS[i]
            cv2.circle(vis, (px, py), 7, col, -1)
            cv2.circle(vis, (px, py), 7, (255, 255, 255), 1)
            cv2.putText(vis, _LABELS[i], (px + 9, py - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

        # Draw court lines between matched groups
        def _safe_line(a, b, col):
            if a < len(self.clicks) and b < len(self.clicks):
                cv2.line(vis, self.clicks[a], self.clicks[b], col, 1)

        _safe_line(0, 1, (0, 200, 255))   # far baseline
        _safe_line(2, 3, (0, 160, 200))   # far service line
        _safe_line(5, 6, (200, 200, 0))   # net
        _safe_line(8, 9, (80, 200, 80))   # near service line
        _safe_line(10, 11, (0, 255, 100)) # near baseline
        # sidelines
        _safe_line(0, 10, (180, 180, 180))
        _safe_line(1, 11, (180, 180, 180))

        # Zoom / pan
        x0, y0 = int(self.pan[0]), int(self.pan[1])
        x1 = int(x0 + w / self.zoom);  y1 = int(y0 + h / self.zoom)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        zoomed = cv2.resize(vis[y0:y1, x0:x1], (w, h),
                            interpolation=cv2.INTER_LINEAR)

        # HUD
        n = len(self.clicks)
        prompt = (PROMPTS[n] if n < 12
                  else "All done — press S to save | R to undo | Q to quit")
        cv2.rectangle(zoomed, (0, 0), (w, 36), (30, 30, 30), -1)
        cv2.putText(zoomed, prompt, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(zoomed,
                    f"zoom {self.zoom:.1f}x  clicks {n}/12",
                    (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)
        return zoomed


def _make_callback(state: _State):
    def cb(event, wx, wy, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            if state._drag_start is None and len(state.clicks) < 12:
                ix, iy = state.win_to_img(wx, wy)
                state.clicks.append((ix, iy))

        elif event == cv2.EVENT_MBUTTONDOWN:
            state._drag_start = (wx, wy)
            state._pan_start  = state.pan[:]
        elif event == cv2.EVENT_MBUTTONUP:
            state._drag_start = None
        elif event == cv2.EVENT_MOUSEMOVE:
            if state._drag_start:
                dx = (state._drag_start[0] - wx) / state.zoom
                dy = (state._drag_start[1] - wy) / state.zoom
                h, w = state.orig.shape[:2]
                state.pan[0] = float(np.clip(
                    state._pan_start[0] + dx, 0, w * (1 - 1/state.zoom)))
                state.pan[1] = float(np.clip(
                    state._pan_start[1] + dy, 0, h * (1 - 1/state.zoom)))

        elif event == cv2.EVENT_MOUSEWHEEL:
            h, w = state.orig.shape[:2]
            if flags > 0:
                state.zoom = min(state.zoom * 1.15, 8.0)
            else:
                state.zoom = max(state.zoom / 1.15, 1.0)
            ix, iy = state.win_to_img(wx, wy)
            state.pan[0] = float(np.clip(
                ix - wx / state.zoom, 0, w * (1 - 1/state.zoom)))
            state.pan[1] = float(np.clip(
                iy - wy / state.zoom, 0, h * (1 - 1/state.zoom)))
    return cb


# ---------------------------------------------------------------------------
# Homography computation
# ---------------------------------------------------------------------------

def compute_homography(
    clicks: List[Tuple[int, int]],
) -> Tuple[np.ndarray, float]:
    """
    Compute H mapping image pixels → normalised court coordinates.

    Returns (H, rms_reprojection_error_px).
    """
    src = np.array([[float(u), float(v)] for u, v in clicks],  dtype=np.float64)
    dst = np.array([[float(x), float(y)] for x, y in WORLD_PTS[:len(clicks)]],
                   dtype=np.float64)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=0.01)
    if H is None:
        raise RuntimeError("findHomography failed — check that clicks are spread "
                           "across the full court")

    # Compute RMS reprojection error in world-space units (then ×court size for px)
    src_h   = np.hstack([src, np.ones((len(src), 1))])
    proj    = (H @ src_h.T).T
    proj    /= proj[:, 2:3]
    err     = np.linalg.norm(proj[:, :2] - dst, axis=1)
    rms_err = float(np.sqrt(np.mean(err ** 2)))

    return H, rms_err


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Interactive 12-point court homography annotation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("video", help="Input video file")
    p.add_argument("--frame", type=int, default=0,
                   help="Frame index to annotate on")
    p.add_argument("--out", default="court_homography.json",
                   help="Output JSON path")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    src  = Path(args.video)

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        sys.exit(f"ERROR: cannot open {src}")

    total_f   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = max(0, min(args.frame, total_f - 1))

    def read_frame(idx: int) -> np.ndarray:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if not ok:
            sys.exit(f"ERROR: could not read frame {idx}")
        return f

    frame = read_frame(frame_idx)
    state = _State(frame)

    win = "Mark Court — LMB: click point | scroll: zoom | middle-drag: pan | S: save | Q: quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, frame.shape[1], frame.shape[0])
    cv2.setMouseCallback(win, _make_callback(state))

    print(__doc__)
    print(f"Video : {src}  ({total_f} frames)")
    print(f"Frame : {frame_idx}\n")
    print("Keys: d/→ next  a/← prev  f jump  r undo  s save  q quit\n")

    while True:
        cv2.imshow(win, state.render())
        key = cv2.waitKey(20) & 0xFF

        if key in (ord('d'), 83):
            frame_idx = min(frame_idx + 1, total_f - 1)
            state.orig = read_frame(frame_idx)
            state.clicks.clear()
        elif key in (ord('a'), 81):
            frame_idx = max(frame_idx - 1, 0)
            state.orig = read_frame(frame_idx)
            state.clicks.clear()
        elif key == ord('f'):
            try:
                n = int(input(f"Jump to frame [0-{total_f-1}]: ").strip())
                frame_idx = max(0, min(n, total_f - 1))
                state.orig = read_frame(frame_idx)
                state.clicks.clear()
            except (ValueError, EOFError):
                pass
        elif key == ord('r'):
            if state.clicks:
                state.clicks.pop()
                print(f"Undo — {len(state.clicks)}/12 clicks")
        elif key == ord('s'):
            if len(state.clicks) < 12:
                print(f"Need 12 clicks ({len(state.clicks)}/12 so far)")
            else:
                try:
                    H, rms = compute_homography(state.clicks)
                except RuntimeError as e:
                    print(f"ERROR: {e}")
                    continue

                result = {
                    "H_image_to_world": H.tolist(),
                    "court_points_px":  [[float(u), float(v)]
                                         for u, v in state.clicks],
                    "court_points_world": [[float(x), float(y)]
                                           for x, y in WORLD_PTS],
                    "frame_index": frame_idx,
                    "rms_world_error": round(rms, 6),
                    "note": (
                        "x in [0,1] left→right singles sideline, "
                        "y in [0,1] far→near baseline"
                    ),
                }
                out_path = Path(args.out)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(result, indent=2))
                print(f"\nSaved → {out_path}")
                print(f"RMS reprojection error (world units): {rms:.6f}")
                print("  (< 0.01 is excellent;  < 0.02 is acceptable)")
                break

        elif key == ord('q') or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            print("Quit without saving.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
