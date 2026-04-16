"""Visualisation utilities — draw ball, skeleton, and HUD onto video frames."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from config.model_config import (
    COLOR_BALL,
    COLOR_BBOX,
    COLOR_KP,
    COLOR_SKEL_BAR,
    COLOR_SKEL_L,
    COLOR_SKEL_R,
)
from pipeline.data_types import FrameResult


def _draw_kp(
    frame: np.ndarray,
    kp: Optional[Tuple[float, float, float]],
    color: tuple,
    radius: int = 5,
    conf_thr: float = 0.3,
) -> None:
    """Draw a filled circle at the keypoint location if confidence is sufficient."""
    if kp is None:
        return
    u, v, c = kp
    if c >= conf_thr and u > 0 and v > 0:
        cv2.circle(frame, (int(u), int(v)), radius, color, -1)


def _draw_bone(
    frame: np.ndarray,
    kp1: Optional[Tuple[float, float, float]],
    kp2: Optional[Tuple[float, float, float]],
    color: tuple,
    conf_thr: float = 0.3,
) -> None:
    """Draw a line between two keypoints if both are sufficiently confident."""
    if kp1 is None or kp2 is None:
        return
    u1, v1, c1 = kp1
    u2, v2, c2 = kp2
    if c1 >= conf_thr and c2 >= conf_thr:
        cv2.line(frame, (int(u1), int(v1)), (int(u2), int(v2)), color, 2)


def visualize(
    frame: np.ndarray,
    result: FrameResult,
    kp_conf_thr: float = 0.3,
) -> np.ndarray:
    """Return an annotated copy of *frame* with ball circle and arm skeleton.

    Parameters
    ----------
    frame:
        Original BGR frame (not modified).
    result:
        Detection outputs for this frame.
    kp_conf_thr:
        Minimum keypoint confidence required to draw a joint or bone.
    """
    vis = frame.copy()

    # ---- Ball ---------------------------------------------------------------
    if result.ball.detected and result.ball.u is not None:
        cx, cy = int(result.ball.u), int(result.ball.v)
        r = max(8, int(result.ball.conf * 20))
        cv2.circle(vis, (cx, cy), r, COLOR_BALL, 2)
        cv2.circle(vis, (cx, cy), 3, COLOR_BALL, -1)
        cv2.putText(vis, f"{result.ball.conf:.2f}", (cx + 6, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_BALL, 1, cv2.LINE_AA)

    # ---- Players ------------------------------------------------------------
    for player in result.players:
        kp = player.keypoints
        if kp.bbox is not None:
            x1, y1, x2, y2 = (int(v) for v in kp.bbox)
            cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_BBOX, 1)
            cv2.putText(vis, f"P{player.player_id}", (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BBOX, 1, cv2.LINE_AA)

        _draw_bone(vis, kp.left_shoulder,  kp.left_elbow,    COLOR_SKEL_L,    kp_conf_thr)
        _draw_bone(vis, kp.left_elbow,     kp.left_wrist,    COLOR_SKEL_L,    kp_conf_thr)
        _draw_bone(vis, kp.right_shoulder, kp.right_elbow,   COLOR_SKEL_R,    kp_conf_thr)
        _draw_bone(vis, kp.right_elbow,    kp.right_wrist,   COLOR_SKEL_R,    kp_conf_thr)
        _draw_bone(vis, kp.left_shoulder,  kp.right_shoulder, COLOR_SKEL_BAR, kp_conf_thr)

        for joint in (kp.left_shoulder, kp.right_shoulder,
                      kp.left_elbow,    kp.right_elbow,
                      kp.left_wrist,    kp.right_wrist):
            _draw_kp(vis, joint, COLOR_KP, conf_thr=kp_conf_thr)

    # ---- HUD ----------------------------------------------------------------
    label = "BALL" if result.ball.detected else "NO BALL"
    cv2.putText(vis, f"#{result.frame_id:06d}  {label}", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return vis
