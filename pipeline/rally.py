"""Rally-level data aggregation and result.json generation.

A Rally accumulates FrameResult objects for one rally and exposes methods
that compute the statistics required by the small-app POST API:

    startTime / endTime          — derived from start/end frame index + fps
    farCount / nearCount stats   — ball speed, player movement
    resultmatrix                 — hit / bounce events
    trackMatrix                  — per-frame ball + player positions

Court-normalised (x, y) coordinates and ball speed must be filled in on each
FrameResult.ball and FrameResult.players[*] before calling the stat methods;
they are None until a court-mapping step populates them.

Rally detection
---------------
Use :func:`detect_rallies` to segment a flat list of FrameResult objects
into Rally instances.  The algorithm mirrors the net-crossing logic in
``scripts/analysis_module.py``:

  1. Find every near→far net crossing  (ball pixel-V drops below y_net_px).
  2. Group crossings whose inter-crossing gap ≤ *inter_rally_gap_s* into the
     same rally.
  3. Expand each rally window by *pre_crossing_s* / *post_crossing_s*.
  4. Clip overlapping windows between adjacent rallies at their midpoint.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np

from pipeline.data_types import FrameResult

logger = logging.getLogger(__name__)

# Physical singles-court dimensions used to convert normalised → metres
_COURT_LENGTH_M = 23.77
_COURT_WIDTH_M  = 10.97


@dataclass
class Rally:
    """One tennis rally: a contiguous sequence of FrameResult objects.

    Parameters
    ----------
    start_frame:
        Index of the first frame of the rally (inclusive).
    end_frame:
        Index of the last frame of the rally (inclusive).
    fps:
        Video frame rate, used to convert frame indices to timestamps.
    frames:
        Ordered list of FrameResult objects for every frame in the rally.
    """

    start_frame: int
    end_frame:   int
    fps:         float
    frames:      List[FrameResult] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Time helpers
    # ------------------------------------------------------------------

    def _offset(self, frame_id: int) -> timedelta:
        return timedelta(seconds=frame_id / self.fps)

    def start_time_iso(self, video_start: datetime) -> str:
        """ISO-8601 UTC string for the rally's first frame."""
        return (video_start + self._offset(self.start_frame)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

    def end_time_iso(self, video_start: datetime) -> str:
        """ISO-8601 UTC string for the rally's last frame."""
        return (video_start + self._offset(self.end_frame)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

    # ------------------------------------------------------------------
    # Ball stats  (totalShots / avgBallSpeed / maxBallSpeed)
    # ------------------------------------------------------------------

    def ball_stats(self) -> dict:
        """Aggregate ball statistics across the rally.

        Returns a dict with keys: totalShots, avgBallSpeed, maxBallSpeed.
        Ball speed is in km/h; only frames with a detected ball and a
        non-None speed value contribute to the speed averages.
        """
        total_shots = sum(1 for f in self.frames if f.ball.detected)
        speeds = [
            f.ball.speed
            for f in self.frames
            if f.ball.detected and f.ball.speed is not None
        ]
        avg_speed = (sum(speeds) / len(speeds)) if speeds else 0.0
        max_speed = max(speeds, default=0.0)
        return {
            "totalShots":   total_shots,
            "avgBallSpeed": round(avg_speed, 1),
            "maxBallSpeed": round(max_speed, 1),
        }

    # ------------------------------------------------------------------
    # Player movement stats  (totalDistance / avgMoveSpeed / maxMoveSpeed)
    # ------------------------------------------------------------------

    def player_stats(self, player_id: int) -> dict:
        """Aggregate movement statistics for one player across the rally.

        Parameters
        ----------
        player_id:
            0 for the far-end player, 1 for the near-end player.

        Returns a dict with keys: totalDistance (m), avgMoveSpeed (m/s),
        maxMoveSpeed (m/s).  Requires Player.x / Player.y to be populated
        by a court-mapping step; returns zeros if fewer than two positions
        are available.
        """
        # Collect (frame_id, x, y) tuples for this player
        coords = []
        for f in self.frames:
            p = next((p for p in f.players if p.player_id == player_id), None)
            if p is not None and p.x is not None and p.y is not None:
                coords.append((f.frame_id, p.x, p.y))

        if len(coords) < 2:
            return {"totalDistance": 0.0, "avgMoveSpeed": 0.0, "maxMoveSpeed": 0.0}

        frame_time  = 1.0 / self.fps
        total_dist  = 0.0
        max_speed   = 0.0

        for (f0, x0, y0), (f1, x1, y1) in zip(coords, coords[1:]):
            dx   = (x1 - x0) * _COURT_WIDTH_M
            dy   = (y1 - y0) * _COURT_LENGTH_M
            dist = math.hypot(dx, dy)
            dt   = (f1 - f0) * frame_time
            total_dist += dist
            if dt > 0:
                max_speed = max(max_speed, dist / dt)

        duration  = (coords[-1][0] - coords[0][0]) * frame_time
        avg_speed = (total_dist / duration) if duration > 0 else 0.0

        return {
            "totalDistance": round(total_dist, 2),
            "avgMoveSpeed":  round(avg_speed, 2),
            "maxMoveSpeed":  round(max_speed, 2),
        }

    # ------------------------------------------------------------------
    # result.json matrices
    # ------------------------------------------------------------------

    def result_matrix(self) -> List[dict]:
        """Hit / bounce events with normalised court coordinates.

        Only frames whose ball has normalised (x, y) populated are included.
        The ``type`` and ``handType`` fields are placeholders until a
        hit-classification step is wired in.
        """
        events = []
        for f in self.frames:
            if f.ball.detected and f.ball.x is not None:
                events.append({
                    "x":        round(f.ball.x, 3),
                    "y":        round(f.ball.y, 3),
                    "type":     "hit",        # TODO: distinguish hit vs bounce
                    "speed":    round(f.ball.speed, 1) if f.ball.speed else 0,
                    "handType": "forehand",   # TODO: derive from pose analysis
                })
        return events

    def track_matrix(self) -> List[dict]:
        """Per-frame ball position + both players' court positions.

        Only frames with a detected ball and populated (x, y) are included.
        """
        rows = []
        for f in self.frames:
            if not (f.ball.detected and f.ball.x is not None):
                continue

            row: dict = {
                "x":         round(f.ball.x, 3),
                "y":         round(f.ball.y, 3),
                "type":      "hit",   # TODO: event classification
                "speed":     round(f.ball.speed, 1) if f.ball.speed else 0,
                "timestamp": f.frame_id,
            }
            for p in f.players:
                prefix = "farCount" if p.player_id == 0 else "nearCount"
                row[f"{prefix}Person_x"] = round(p.x, 3) if p.x is not None else None
                row[f"{prefix}Person_y"] = round(p.y, 3) if p.y is not None else None
            rows.append(row)
        return rows

    # ------------------------------------------------------------------
    # Top-level serialiser
    # ------------------------------------------------------------------

    def to_result_json(
        self,
        serial_number: str,
        video_start: datetime,
    ) -> dict:
        """Build the full result.json payload for this rally.

        Parameters
        ----------
        serial_number:
            Device serial number written to the ``serial_number`` field.
        video_start:
            UTC datetime of the first frame of the source video, used to
            compute absolute ISO-8601 timestamps.
        """
        ball  = self.ball_stats()
        far   = {**ball, **self.player_stats(player_id=0)}
        near  = {**ball, **self.player_stats(player_id=1)}

        return {
            "serial_number": serial_number,
            "startTime": self.start_time_iso(video_start),
            "endTime":   self.end_time_iso(video_start),
            "content": {
                "mete": {
                    "farCount":  far,
                    "nearCount": near,
                },
                "resultmatrix": self.result_matrix(),
                "trackMatrix":  self.track_matrix(),
            },
        }


# ---------------------------------------------------------------------------
# Rally detection  (mirrors analysis_module.py net-crossing logic)
# ---------------------------------------------------------------------------

def detect_rallies(
    frames: List[FrameResult],
    fps: float,
    y_net_px: float,
    inter_rally_gap_s: float = 8.0,
    pre_crossing_s:    float = 3.0,
    post_crossing_s:   float = 3.0,
    min_far_count:     int   = 2,
    debounce_frames:   int   = 20,
    max_gap:           int   = 25,
) -> List[Rally]:
    """Segment a list of FrameResult objects into Rally instances.

    The algorithm is the net-crossing method used by
    ``scripts/analysis_module.TennisVisualizer``:

    1. Scan for every near→far crossing: ball pixel-V drops from above
       *y_net_px* to below *y_net_px*.  Gap-bridging crossings (ball
       disappears in near zone, reappears in far zone within *max_gap*
       frames) are also detected.
    2. Group crossings separated by ≤ *inter_rally_gap_s* into the same
       rally.
    3. Expand each rally window *pre_crossing_s* before the first crossing
       and *post_crossing_s* after the last crossing.
    4. Clip overlapping windows at their midpoint.

    Parameters
    ----------
    frames:
        Ordered list of FrameResult objects for the full video.
    fps:
        Video frame rate.
    y_net_px:
        Pixel V-coordinate of the net in this camera view.  Ball detections
        above this line (smaller V) are in the far half-court; below (larger
        V) are in the near half-court.
    inter_rally_gap_s:
        Maximum gap between two crossings that still belong to the same rally.
    pre_crossing_s / post_crossing_s:
        Rally window padding around the crossing cluster.
    min_far_count:
        Number of consecutive far-zone frames required after a crossing to
        confirm it (noise filter).
    debounce_frames:
        Minimum frame gap between two accepted crossings.
    max_gap:
        Maximum missing-frame gap to bridge when looking for gap crossings.

    Returns
    -------
    List[Rally]
        One Rally per detected rally, ordered chronologically.
        Returns an empty list if no crossings are found.
    """
    if not frames:
        return []

    # Build arrays for fast NumPy operations
    n          = len(frames)
    frame_ids  = np.array([f.frame_id for f in frames], dtype=int)
    ball_v     = np.full(n, np.nan, dtype=float)

    for i, f in enumerate(frames):
        if f.ball.detected and f.ball.v is not None:
            ball_v[i] = f.ball.v

    valid_idx = [i for i in range(n) if not np.isnan(ball_v[i])]

    # ── Step 1: find near→far crossings ──────────────────────────────
    crossings: List[int] = []   # indices into `frames`
    last_frame = -debounce_frames

    for k in range(1, len(valid_idx)):
        prev_i = valid_idx[k - 1]
        curr_i = valid_idx[k]

        if curr_i - prev_i > max_gap:
            continue

        # Must transition from near (V > y_net) to far (V < y_net)
        if not (ball_v[prev_i] > y_net_px and ball_v[curr_i] < y_net_px):
            continue

        # Require ball to dwell in far zone for min_far_count frames
        far_count = 0
        for j in valid_idx[k:]:
            if j >= curr_i + debounce_frames:
                break
            if ball_v[j] < y_net_px:
                far_count += 1
                if far_count >= min_far_count:
                    break
            else:
                break
        if far_count < min_far_count:
            continue

        # Debounce
        fi = int(frame_ids[curr_i])
        if fi - last_frame < debounce_frames:
            continue

        crossings.append(curr_i)
        last_frame = fi

    logger.info(
        "Net crossings detected: %d  (y_net=%.0f px, debounce=%d fr)",
        len(crossings), y_net_px, debounce_frames,
    )

    if not crossings:
        return []

    # ── Step 2: group crossings by time gap ──────────────────────────
    gap_frames  = int(inter_rally_gap_s * fps)
    pre_frames  = int(pre_crossing_s    * fps)
    post_frames = int(post_crossing_s   * fps)

    crossing_frame_ids = [int(frame_ids[ci]) for ci in crossings]
    groups: List[List[int]] = [[crossing_frame_ids[0]]]
    for cf in crossing_frame_ids[1:]:
        if cf - groups[-1][-1] <= gap_frames:
            groups[-1].append(cf)
        else:
            groups.append([cf])

    # ── Step 3: build Rally objects with padded windows ──────────────
    f_min = int(frame_ids[0])
    f_max = int(frame_ids[-1])

    # Map frame_id → list index for window boundary lookup
    fid_to_idx = {int(f.frame_id): i for i, f in enumerate(frames)}

    rally_defs: List[dict] = []
    for group in groups:
        start_fid = max(group[0]  - pre_frames,  f_min)
        end_fid   = min(group[-1] + post_frames, f_max)

        # Snap to nearest available frame
        start_fid = next(
            (int(frame_ids[i]) for i in range(n) if frame_ids[i] >= start_fid),
            f_min,
        )
        end_fid = next(
            (int(frame_ids[i]) for i in range(n - 1, -1, -1)
             if frame_ids[i] <= end_fid),
            f_max,
        )
        rally_defs.append({"start": start_fid, "end": end_fid})

    # ── Step 4: clip overlapping windows ─────────────────────────────
    for i in range(len(rally_defs) - 1):
        if rally_defs[i]["end"] >= rally_defs[i + 1]["start"]:
            mid = (rally_defs[i]["end"] + rally_defs[i + 1]["start"]) // 2
            rally_defs[i]["end"]          = mid
            rally_defs[i + 1]["start"]    = mid + 1

    # ── Build Rally objects ───────────────────────────────────────────
    rallies: List[Rally] = []
    for rd in rally_defs:
        s_idx = fid_to_idx.get(rd["start"])
        e_idx = fid_to_idx.get(rd["end"])
        if s_idx is None or e_idx is None:
            continue
        rally_frames = frames[s_idx: e_idx + 1]
        rallies.append(Rally(
            start_frame = rd["start"],
            end_frame   = rd["end"],
            fps         = fps,
            frames      = rally_frames,
        ))

    logger.info(
        "detect_rallies: %d crossings → %d rallies",
        len(crossings), len(rallies),
    )
    return rallies
