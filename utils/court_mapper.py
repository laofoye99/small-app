"""Map pixel coordinates to normalised court coordinates via homography H.

Produced by scripts/mark_court.py.

Normalised court space
----------------------
  x ∈ [0, 1]   left singles sideline → right singles sideline
  y ∈ [0, 1]   far baseline           → near baseline

Physical dimensions used for speed conversion
---------------------------------------------
  Length (baseline-to-baseline) : 23.77 m
  Width  (singles sideline)     :  8.23 m
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from pipeline.data_types import FrameResult

_COURT_LENGTH_M = 23.77
_COURT_WIDTH_M  = 8.23


class CourtMapper:
    """Convert image pixel coordinates to normalised court coordinates.

    Parameters
    ----------
    H:
        3×3 homography matrix produced by :mod:`scripts.mark_court`.
        Maps image homogeneous coords ``[u, v, 1]`` to normalised court
        homogeneous coords ``[x·w, y·w, w]``.
    """

    def __init__(self, H: np.ndarray) -> None:
        self.H = np.array(H, dtype=float)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: str | Path, camera: str | None = None) -> "CourtMapper":
        """Load H from a JSON file saved by ``mark_court.py``.

        Parameters
        ----------
        path:
            Path to the JSON file.
        camera:
            If the file wraps multiple cameras as ``{"cam66": {...}, ...}``,
            pass the camera label to select the right entry.  Leave ``None``
            for single-camera files (``{"H_image_to_world": [...]}``) and for
            the format written by ``mark_court.py``.
        """
        data = json.loads(Path(path).read_text())
        if camera is not None:
            H = data[camera]["H_image_to_world"]
        else:
            H = data.get("H_image_to_world", data)
        return cls(np.array(H, dtype=float))

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def pixel_to_court(
        self, u: float, v: float
    ) -> Optional[Tuple[float, float]]:
        """Map a single pixel to normalised court coordinates.

        Returns ``None`` if the homogeneous weight is near zero (degenerate).
        """
        res = self.H @ np.array([u, v, 1.0])
        if abs(res[2]) < 1e-10:
            return None
        return float(res[0] / res[2]), float(res[1] / res[2])

    def court_to_metres(
        self, x: float, y: float
    ) -> Tuple[float, float]:
        """Convert normalised court (x, y) to physical metres from far-left corner."""
        return x * _COURT_WIDTH_M, y * _COURT_LENGTH_M

    # ------------------------------------------------------------------
    # Batch application
    # ------------------------------------------------------------------

    def apply_to_results(
        self,
        frames: List[FrameResult],
        fps: float,
    ) -> None:
        """Fill ``ball.x``, ``ball.y``, ``ball.speed`` and ``player.x``,
        ``player.y`` for every :class:`~pipeline.data_types.FrameResult`
        in *frames* in-place.

        Ball speed is computed from the displacement between consecutive
        detected frames (km/h).  Frames where the ball is not detected
        break the speed chain — the next detected frame will have
        ``speed = None``.

        Player position is estimated from the midpoint of the left and right
        shoulder keypoints (minimum confidence 0.1 on each shoulder).
        """
        prev_bx: Optional[float] = None
        prev_by: Optional[float] = None

        for fr in frames:
            # ── Ball ──────────────────────────────────────────────────
            if fr.ball.detected and fr.ball.u is not None:
                ct = self.pixel_to_court(fr.ball.u, fr.ball.v)
                if ct is not None:
                    fr.ball.x, fr.ball.y = ct

                    if prev_bx is not None:
                        dx_m = (fr.ball.x - prev_bx) * _COURT_WIDTH_M
                        dy_m = (fr.ball.y - prev_by) * _COURT_LENGTH_M
                        fr.ball.speed = round(
                            math.hypot(dx_m, dy_m) * fps * 3.6, 1
                        )  # km/h

                    prev_bx, prev_by = fr.ball.x, fr.ball.y
                else:
                    prev_bx = prev_by = None
            else:
                prev_bx = prev_by = None

            # ── Players ───────────────────────────────────────────────
            for player in fr.players:
                kp = player.keypoints
                ls = kp.left_shoulder
                rs = kp.right_shoulder
                if (ls is not None and rs is not None
                        and ls[2] >= 0.1 and rs[2] >= 0.1
                        and ls[0] > 5 and rs[0] > 5):
                    u_mid = (ls[0] + rs[0]) / 2
                    v_mid = (ls[1] + rs[1]) / 2
                    ct = self.pixel_to_court(u_mid, v_mid)
                    if ct is not None:
                        player.x, player.y = ct
