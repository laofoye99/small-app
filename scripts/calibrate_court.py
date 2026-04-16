#!/usr/bin/env python3
"""
Court calibration tool.

Lets you click the left and right endpoints of the near and far baselines on
a video frame, then computes y_near, px_width_near, y_far, px_width_far and
writes them (plus the derived alpha/beta) to a JSON file.

Usage
-----
python calibrate_court.py <video.mp4> [--frame 100] [--out calibration.json]

Controls
--------
  d / →    next frame          a / ←    previous frame
  f        jump to frame N     r        redo last baseline
  s        save and quit       q        quit without saving
  scroll   zoom in / out       drag     pan when zoomed

Click order
-----------
  Step 1: click the LEFT  endpoint of the FAR  baseline (top of court)
  Step 2: click the RIGHT endpoint of the FAR  baseline
  Step 3: click the LEFT  endpoint of the NEAR baseline (bottom of court)
  Step 4: click the RIGHT endpoint of the NEAR baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# State shared with mouse callback
# ---------------------------------------------------------------------------

class _State:
    """Mutable annotation state passed into the OpenCV mouse callback."""

    # Click labels shown in sequence
    PROMPTS = [
        "Step 1/4 — FAR  baseline : click LEFT  endpoint",
        "Step 2/4 — FAR  baseline : click RIGHT endpoint",
        "Step 3/4 — NEAR baseline : click LEFT  endpoint",
        "Step 4/4 — NEAR baseline : click RIGHT endpoint",
    ]

    def __init__(self, frame: np.ndarray) -> None:
        self.orig: np.ndarray        = frame.copy()
        self.clicks: List[Tuple[int, int]] = []   # image-space coords
        self.zoom: float             = 1.0
        self.pan:  List[float]       = [0.0, 0.0]  # tx, ty in image pixels
        self._drag_start: Optional[Tuple[int, int]] = None
        self._pan_start:  Optional[List[float]]     = None

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def win_to_img(self, wx: int, wy: int) -> Tuple[int, int]:
        """Convert window pixel → image pixel (accounting for zoom/pan)."""
        ix = int(wx / self.zoom + self.pan[0])
        iy = int(wy / self.zoom + self.pan[1])
        h, w = self.orig.shape[:2]
        return int(np.clip(ix, 0, w - 1)), int(np.clip(iy, 0, h - 1))

    def img_to_win(self, ix: float, iy: float) -> Tuple[int, int]:
        """Convert image pixel → window pixel."""
        return int((ix - self.pan[0]) * self.zoom), int((iy - self.pan[1]) * self.zoom)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> np.ndarray:
        """Return the current annotated view (zoomed + panned)."""
        h, w = self.orig.shape[:2]
        vis  = self.orig.copy()

        # Draw finished baselines
        colors = [(0, 200, 255), (0, 200, 255), (0, 255, 100), (0, 255, 100)]
        labels = ["FAR-L", "FAR-R", "NEAR-L", "NEAR-R"]
        for i, (px, py) in enumerate(self.clicks):
            cv2.circle(vis, (px, py), 7, colors[i], -1)
            cv2.putText(vis, labels[i], (px + 8, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2, cv2.LINE_AA)

        if len(self.clicks) >= 2:   # FAR baseline complete
            cv2.line(vis, self.clicks[0], self.clicks[1], (0, 200, 255), 2)
        if len(self.clicks) == 4:   # NEAR baseline complete
            cv2.line(vis, self.clicks[2], self.clicks[3], (0, 255, 100), 2)

        # Apply zoom + pan
        x0 = int(self.pan[0]);  y0 = int(self.pan[1])
        x1 = int(x0 + w / self.zoom); y1 = int(y0 + h / self.zoom)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        crop    = vis[y0:y1, x0:x1]
        zoomed  = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

        # HUD overlay
        step   = len(self.clicks)
        prompt = self.PROMPTS[step] if step < 4 else "All done — press S to save, R to redo, Q to quit"
        cv2.rectangle(zoomed, (0, 0), (w, 36), (30, 30, 30), -1)
        cv2.putText(zoomed, prompt, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        zoom_str = f"zoom {self.zoom:.1f}x  |  frame coords: pan=({int(self.pan[0])},{int(self.pan[1])})"
        cv2.putText(zoomed, zoom_str, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
        return zoomed


def _make_mouse_callback(state: _State):
    def callback(event, wx, wy, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if state._drag_start is None and len(state.clicks) < 4:
                ix, iy = state.win_to_img(wx, wy)
                state.clicks.append((ix, iy))

        elif event == cv2.EVENT_MBUTTONDOWN:
            state._drag_start = (wx, wy)
            state._pan_start  = state.pan[:]

        elif event == cv2.EVENT_MBUTTONUP:
            state._drag_start = None

        elif event == cv2.EVENT_MOUSEMOVE:
            if state._drag_start is not None:
                dx = (state._drag_start[0] - wx) / state.zoom
                dy = (state._drag_start[1] - wy) / state.zoom
                h, w = state.orig.shape[:2]
                state.pan[0] = float(np.clip(state._pan_start[0] + dx, 0, w * (1 - 1/state.zoom)))
                state.pan[1] = float(np.clip(state._pan_start[1] + dy, 0, h * (1 - 1/state.zoom)))

        elif event == cv2.EVENT_MOUSEWHEEL:
            h, w = state.orig.shape[:2]
            old_z = state.zoom
            if flags > 0:
                state.zoom = min(state.zoom * 1.15, 8.0)
            else:
                state.zoom = max(state.zoom / 1.15, 1.0)
            # Keep the image point under the cursor fixed
            ix, iy = state.win_to_img(wx, wy)
            state.pan[0] = float(np.clip(ix - wx / state.zoom, 0, w * (1 - 1/state.zoom)))
            state.pan[1] = float(np.clip(iy - wy / state.zoom, 0, h * (1 - 1/state.zoom)))

    return callback


# ---------------------------------------------------------------------------
# Result computation
# ---------------------------------------------------------------------------

def _compute_calibration(
    clicks: List[Tuple[int, int]],
    fps: float,
    v_max_ms: float,
    court_width_m: float,
) -> dict:
    """Derive calibration parameters from four click coordinates."""
    far_l, far_r, near_l, near_r = clicks

    y_far       = float((far_l[1]  + far_r[1])  / 2)
    y_near      = float((near_l[1] + near_r[1]) / 2)
    px_width_far  = float(abs(far_r[0]  - far_l[0]))
    px_width_near = float(abs(near_r[0] - near_l[0]))

    K     = v_max_ms / (fps * court_width_m)
    slope = (px_width_near - px_width_far) / (y_near - y_far)
    alpha = K * slope
    beta  = K * px_width_far - alpha * y_far

    return {
        "y_far":          y_far,
        "px_width_far":   px_width_far,
        "y_near":         y_near,
        "px_width_near":  px_width_near,
        "alpha":          alpha,
        "beta":           beta,
        "fps":            fps,
        "v_max_ms":       v_max_ms,
        "court_width_m":  court_width_m,
        # Raw pixel click coordinates -- required by spatial_mapping.py
        # to compute an accurate homography H for Y (along-court) mapping.
        # Order: [far_left, far_right, near_left, near_right], each [u, v].
        "clicks_px": [
            [float(far_l[0]),  float(far_l[1])],
            [float(far_r[0]),  float(far_r[1])],
            [float(near_l[0]), float(near_l[1])],
            [float(near_r[0]), float(near_r[1])],
        ],
    }


def _print_result(cal: dict) -> None:
    print("\n── Calibration result ─────────────────────────────────────────")
    print(f"  y_far         = {cal['y_far']:.1f} px")
    print(f"  px_width_far  = {cal['px_width_far']:.1f} px")
    print(f"  y_near        = {cal['y_near']:.1f} px")
    print(f"  px_width_near = {cal['px_width_near']:.1f} px")
    print(f"  α (alpha)     = {cal['alpha']:.6f}")
    print(f"  β (beta)      = {cal['beta']:.4f}")
    vpx_far  = cal['alpha'] * cal['y_far']  + cal['beta']
    vpx_near = cal['alpha'] * cal['y_near'] + cal['beta']
    print(f"  Vpx_max(y_far)  = {vpx_far:.1f} px/frame")
    print(f"  Vpx_max(y_near) = {vpx_near:.1f} px/frame")
    print("────────────────────────────────────────────────────────────────")
    print("\nUsage in clean_trajectory.py:")
    print(f"  --alpha {cal['alpha']:.6f} --beta {cal['beta']:.4f}")
    print("  — or —")
    print(f"  --y-near {cal['y_near']:.0f} --px-near {cal['px_width_near']:.0f} "
          f"--y-far {cal['y_far']:.0f} --px-far {cal['px_width_far']:.0f}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Interactive court calibration tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("video", help="Video file to annotate")
    p.add_argument("--frame",         type=int,   default=0,
                   help="Frame index to start on")
    p.add_argument("--out",           default="calibration.json",
                   help="Output JSON file path")
    p.add_argument("--fps",           type=float, default=None,
                   help="Override FPS (default: read from video)")
    p.add_argument("--v-max-ms",      type=float, default=55.6,
                   help="Physical ball speed ceiling in m/s")
    p.add_argument("--court-width-m", type=float, default=10.97,
                   help="Real court width in metres")
    return p


def main() -> None:
    args    = _build_parser().parse_args()
    src     = Path(args.video)
    out_path = Path(args.out)

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        sys.exit(f"ERROR: cannot open {src}")

    fps       = args.fps or (cap.get(cv2.CAP_PROP_FPS) or 30.0)
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

    win = "Court Calibration — scroll=zoom  middle-drag=pan  LMB=click  S=save  Q=quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, frame.shape[1], frame.shape[0])
    cv2.setMouseCallback(win, _make_mouse_callback(state))

    result: Optional[dict] = None

    print(__doc__)
    print(f"Video : {src}  ({total_f} frames, {fps:.1f} fps)")
    print(f"Frame : {frame_idx}\n")
    print("Keyboard:  d/→ next frame | a/← prev frame | f jump | r redo | s save | q quit")

    while True:
        cv2.imshow(win, state.render())
        key = cv2.waitKey(20) & 0xFF

        # ---- frame navigation ----
        if key in (ord('d'), 83):   # d or →
            frame_idx = min(frame_idx + 1, total_f - 1)
            state.orig = read_frame(frame_idx)
            print(f"Frame {frame_idx}")
        elif key in (ord('a'), 81):  # a or ←
            frame_idx = max(frame_idx - 1, 0)
            state.orig = read_frame(frame_idx)
            print(f"Frame {frame_idx}")
        elif key == ord('f'):
            try:
                n = int(input(f"Jump to frame [0–{total_f-1}]: ").strip())
                frame_idx = max(0, min(n, total_f - 1))
                state.orig = read_frame(frame_idx)
                state.clicks.clear()
                print(f"Jumped to frame {frame_idx}")
            except (ValueError, EOFError):
                pass

        # ---- annotation ----
        elif key == ord('r'):       # redo last click
            if state.clicks:
                state.clicks.pop()
                print(f"Removed last click  ({len(state.clicks)}/4)")

        # ---- save / quit ----
        elif key == ord('s'):
            if len(state.clicks) < 4:
                print(f"Need 4 clicks first ({len(state.clicks)}/4 so far).")
            else:
                result = _compute_calibration(
                    state.clicks, fps, args.v_max_ms, args.court_width_m
                )
                _print_result(result)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(result, indent=2))
                print(f"Saved → {out_path}")
                break

        elif key == ord('q') or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            print("Quit without saving.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()