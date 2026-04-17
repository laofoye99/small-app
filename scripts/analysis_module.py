"""
analysis_module.py
==================
Tennis Rally Segmentation and Business Indicator Visualization

Rally Detection Logic
---------------------
  Net-crossing groups: crossings separated by ≤ inter_rally_gap_s (default 8 s)
  belong to the same rally.  The display window is expanded pre/post_crossing_s
  (default 3 s) around the crossing cluster.

  TennisRallySegmenter.segment_rallies() is still called for DataFrame
  preprocessing (sort, time column, pixel-speed column) but its rally
  boundary output is discarded.

Shot / Serve Detection  (Net-Crossing method)
---------------------------------------------
Any ball transition from the near side (V > y_net) to the far side (V < y_net)
is a near-field player shot — this holds for groundstrokes, volleys, and serves.

Two crossing modes are detected:
  A. Consecutive-frame crossing: V[i-1] > y_net  and  V[i] < y_net
  B. Gap-bridging crossing: ball last seen in near zone, then (after up to
     max_gap missing frames) first seen in far zone.

Speed at the crossing frame is computed from world-coordinate displacement
and divided by NET_SPEED_CORRECTION (default 2.0) to compensate for
single-camera perspective overestimation at net height.

Rally Merging
-------------
After shot detection, any rally with 0 net crossings is merged with the
next rally (consecutive 0-shot rallies are all merged with the first
following non-empty rally).  Trailing 0-shot rallies are absorbed into the
preceding non-empty rally.

Display (top-left panel, persists between rallies)
--------------------------------------------------
  - Rally ID: keeps showing last known rally number (no "---" reset)
  - Ball speed: updated only at shot moments; shows last shot speed otherwise
  - Distance: all valid player (p0 + p1) displacement summed per rally
  - Near-field player pose skeleton drawn on every frame
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# ──────────────────────────────────────────────────────────────────────────────
# Rally Segmenter
# ──────────────────────────────────────────────────────────────────────────────

class TennisRallySegmenter:
    """
    Segment rallies from ball trajectory data.

    A rally ends when:
      - no valid ball position for > gap_threshold consecutive frames, OR
      - ball pixel speed < min_speed_threshold for > slow_duration_threshold frames
    """

    def __init__(
        self,
        fps: float = 30,
        gap_threshold: int = 25,
        min_speed_threshold: float = 5.0,
        slow_duration_threshold: int = 30,
        min_rally_frames: int = 150,
        interp_limit: int = 5,
    ):
        self.fps = fps
        self.gap_threshold = gap_threshold
        self.min_speed_threshold = min_speed_threshold
        self.slow_duration_threshold = slow_duration_threshold
        self.min_rally_frames = min_rally_frames  # default 150 ≈ 6s at 25fps
        self.interp_limit = interp_limit

    def _find_valid_motion(self, df: pd.DataFrame, x_col: str, y_col: str):
        df = df.copy()
        df[[x_col, y_col]] = df[[x_col, y_col]].interpolate(
            method='linear', limit=self.interp_limit
        )
        dx = df[x_col].diff().fillna(0)
        dy = df[y_col].diff().fillna(0)
        df['_v_px'] = np.sqrt(dx ** 2 + dy ** 2)

        is_missing = df[x_col].isna()
        gap_grp  = (is_missing != is_missing.shift()).cumsum()
        gap_sz   = is_missing.groupby(gap_grp).transform('count')
        silence_trigger = is_missing & (gap_sz > self.gap_threshold)

        is_slow  = df['_v_px'] < self.min_speed_threshold
        slow_grp = (is_slow != is_slow.shift()).cumsum()
        slow_sz  = is_slow.groupby(slow_grp).transform('count')
        slow_trigger = is_slow & (slow_sz > self.slow_duration_threshold)

        is_valid = ~(silence_trigger | slow_trigger | is_missing)
        return df, is_valid

    def segment_rallies(self, trajectory_df: pd.DataFrame):
        """
        Returns
        -------
        rallies : list[dict]  {rally_id, start_frame, end_frame,
                               rally_start (sec), rally_end (sec)}
        df      : processed DataFrame with reset integer index 0..N-1
        """
        df = trajectory_df.copy()

        frame_col = 'frame_id' if 'frame_id' in df.columns else 'frame'
        x_col     = 'interp_u' if 'interp_u' in df.columns else 'x'
        y_col     = 'interp_v' if 'interp_v' in df.columns else 'y'

        if frame_col not in df.columns:
            raise KeyError("Input DataFrame must have 'frame_id' or 'frame' column.")

        df = df.sort_values(frame_col).reset_index(drop=True)
        if 'time' not in df.columns:
            df['time'] = df[frame_col] / self.fps

        df, is_valid = self._find_valid_motion(df, x_col, y_col)

        rallies: list[dict] = []
        rally_id   = 1
        in_rally   = False
        start_idx: int | None = None

        for i in range(len(df)):
            valid = bool(is_valid.iloc[i])
            if valid and not in_rally:
                in_rally  = True
                start_idx = i
            elif not valid and in_rally:
                end_idx = i - 1
                if end_idx - start_idx + 1 >= self.min_rally_frames:
                    rallies.append({
                        'rally_id':    rally_id,
                        'start_frame': int(df.loc[start_idx, frame_col]),
                        'end_frame':   int(df.loc[end_idx,   frame_col]),
                        'rally_start': round(float(df.loc[start_idx, 'time']), 3),
                        'rally_end':   round(float(df.loc[end_idx,   'time']), 3),
                    })
                    rally_id += 1
                in_rally = False

        if in_rally and start_idx is not None:
            end_idx = len(df) - 1
            if end_idx - start_idx + 1 >= self.min_rally_frames:
                rallies.append({
                    'rally_id':    rally_id,
                    'start_frame': int(df.loc[start_idx, frame_col]),
                    'end_frame':   int(df.loc[end_idx,   frame_col]),
                    'rally_start': round(float(df.loc[start_idx, 'time']), 3),
                    'rally_end':   round(float(df.loc[end_idx,   'time']), 3),
                })

        logging.info(
            "Segmented %d rallies  "
            "(silence>%d fr  OR  speed<%.1f px/fr for >%d fr,  min=%d fr)",
            len(rallies), self.gap_threshold,
            self.min_speed_threshold, self.slow_duration_threshold,
            self.min_rally_frames,
        )
        return rallies, df


# ──────────────────────────────────────────────────────────────────────────────
# Visualizer
# ──────────────────────────────────────────────────────────────────────────────

class TennisVisualizer:
    """
    Overlay business-logic indicators onto the tennis video.

    Shot detection uses net-crossing events (ball V > y_net → V < y_net).
    """

    # Skeleton connections for pose drawing
    _SKELETON = [('ls', 'rs'), ('ls', 'le'), ('le', 'lw'),
                 ('rs', 're'), ('re', 'rw')]
    _KP_SHOULDER_COLOR = (0,  80, 255)
    _KP_JOINT_COLOR    = (0, 200, 255)
    _SKEL_COLOR        = (0, 230,   0)

    # Speed correction for single-camera perspective at net height
    NET_SPEED_CORRECTION = 2.0

    def __init__(
        self,
        segmenter: TennisRallySegmenter,
        H_i2w,
        cal: dict | None = None,
    ):
        self.segmenter = segmenter
        self.H         = np.array(H_i2w, dtype=float)
        self.font      = cv2.FONT_HERSHEY_SIMPLEX
        self.cal       = cal or {}
        self.y_near    = self.cal.get('y_near')
        self.y_far     = self.cal.get('y_far')
        self.y_net     = self.cal.get('y_net')

        self.min_move_dist = 0.05   # metres — filter sub-noise jitter
        self.max_move_dist = 2.0    # metres — filter ID-swap jumps

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def to_world(self, u, v):
        if u is None or v is None: return None
        try:
            u, v = float(u), float(v)
        except (TypeError, ValueError):
            return None
        if np.isnan(u) or np.isnan(v): return None
        res = self.H @ np.array([u, v, 1.0])
        if abs(res[2]) < 1e-10: return None
        return (res[0] / res[2], res[1] / res[2])

    # ------------------------------------------------------------------
    # Player helpers
    # ------------------------------------------------------------------

    def _near_player_prefix(self, row: pd.Series) -> str | None:
        """
        Return 'p0' or 'p1' for the nearest-camera player (highest pixel-V
        shoulder).  Players where BOTH shoulders have U ≤ 20 are excluded —
        YOLO encodes invisible/out-of-frame keypoints as (0, 0), so partial
        detections (legs only, spectators outside court) are filtered out.
        """
        def is_valid_player(prefix: str) -> bool:
            ls_u = row.get(f'{prefix}_ls_u')
            rs_u = row.get(f'{prefix}_rs_u')
            return (
                ls_u is not None and rs_u is not None
                and not pd.isna(ls_u) and not pd.isna(rs_u)
                and float(ls_u) > 20 and float(rs_u) > 20
            )

        def avg_shoulder_v(prefix: str) -> float:
            vals = [row.get(f'{prefix}_ls_v'), row.get(f'{prefix}_rs_v')]
            vals = [float(x) for x in vals if x is not None and not pd.isna(x)]
            return float(np.mean(vals)) if vals else np.nan

        p0_ok = is_valid_player('p0')
        p1_ok = is_valid_player('p1')

        if not p0_ok and not p1_ok: return None
        if not p0_ok: return 'p1'
        if not p1_ok: return 'p0'

        v0 = avg_shoulder_v('p0')
        v1 = avg_shoulder_v('p1')
        if np.isnan(v0) and np.isnan(v1): return None
        if np.isnan(v0): return 'p1'
        if np.isnan(v1): return 'p0'
        return 'p0' if v0 > v1 else 'p1'

    def _get_player_world(self, row: pd.Series, prefix: str):
        vals = {k: row.get(f'{prefix}_{k}')
                for k in ('ls_u', 'ls_v', 'rs_u', 'rs_v')}
        if any(v is None or pd.isna(v) for v in vals.values()):
            return None
        # YOLO zero-marker: u = 0 means "keypoint not visible"
        if float(vals['ls_u']) < 5 or float(vals['rs_u']) < 5:
            return None
        u = (float(vals['ls_u']) + float(vals['rs_u'])) / 2
        v = (float(vals['ls_v']) + float(vals['rs_v'])) / 2
        return self.to_world(u, v)

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_player_keypoints(
        self, frame: np.ndarray, row: pd.Series, prefix: str | None
    ) -> None:
        if prefix is None:
            return
        kp_defs = {
            'ls': (f'{prefix}_ls_u', f'{prefix}_ls_v'),
            'rs': (f'{prefix}_rs_u', f'{prefix}_rs_v'),
            'le': (f'{prefix}_le_u', f'{prefix}_le_v'),
            're': (f'{prefix}_re_u', f'{prefix}_re_v'),
            'lw': (f'{prefix}_lw_u', f'{prefix}_lw_v'),
            'rw': (f'{prefix}_rw_u', f'{prefix}_rw_v'),
        }
        kps: dict[str, tuple[int, int]] = {}
        for name, (u_key, v_key) in kp_defs.items():
            u = row.get(u_key)
            v = row.get(v_key)
            if u is None or v is None or pd.isna(u) or pd.isna(v):
                continue
            ui, vi = int(float(u)), int(float(v))
            if ui > 5 and vi > 5:
                kps[name] = (ui, vi)
        for a, b in self._SKELETON:
            if a in kps and b in kps:
                cv2.line(frame, kps[a], kps[b], self._SKEL_COLOR, 2)
        for name, pt in kps.items():
            color = (self._KP_SHOULDER_COLOR
                     if name in ('ls', 'rs') else self._KP_JOINT_COLOR)
            cv2.circle(frame, pt, 5, color, -1)
            cv2.circle(frame, pt, 5, (255, 255, 255), 1)

    # ------------------------------------------------------------------
    # Shot / speed helpers
    # ------------------------------------------------------------------

    def _ball_speed_kmh(
        self,
        df: pd.DataFrame,
        idx: int,
        fps: float,
        u_col: str,
        v_col: str,
        correction: float = NET_SPEED_CORRECTION,
        max_lookback: int = 30,
    ) -> float:
        """
        Compute world-coordinate ball speed at frame idx (km/h).

        Scans backward up to max_lookback rows to find the last valid ball
        position before idx — this handles gap-bridging crossings where the
        immediately preceding row may be NaN.
        """
        if idx < 1:
            return 0.0
        w1 = self.to_world(df.loc[idx, u_col], df.loc[idx, v_col])
        if w1 is None:
            return 0.0

        # Find last valid row before idx
        prev_idx = idx - 1
        while prev_idx >= max(0, idx - max_lookback):
            w0 = self.to_world(df.loc[prev_idx, u_col], df.loc[prev_idx, v_col])
            if w0 is not None:
                frame_gap = max(idx - prev_idx, 1)
                dist_m = np.hypot(w1[0] - w0[0], w1[1] - w0[1])
                spd    = dist_m * fps * 3.6 / frame_gap / correction
                if spd > 60:
                    spd = 60 + (spd - 60) * 0.2
                spd = min(spd, 110)
                if spd >= 10:
                    return spd
                return 0.0
            prev_idx -= 1
        return 0.0

    def _detect_crossings(
        self,
        df: pd.DataFrame,
        min_far_count: int = 2,
        debounce_frames: int = 20,
        max_gap: int = 25,
    ) -> list[int]:
        """
        Detect every near→far net crossing in the ball trajectory.

        Two modes:
          A. Consecutive: V[i-1] > y_net  and  V[i] < y_net
          B. Gap-bridging: last valid near-zone frame, then gap ≤ max_gap,
             then first valid far-zone frame — records crossing at the first
             far-zone frame after the gap.

        Both require the ball to stay in the far zone for min_far_count
        additional frames (noise filter) and observe debounce_frames between
        consecutive crossings.

        Returns list of DataFrame row indices where crossings occur.
        """
        if self.y_net is None:
            logging.warning("y_net not configured — net-crossing detection skipped")
            return []

        v_col = 'interp_v' if 'interp_v' in df.columns else 'y'
        f_col = ('frame_id' if 'frame_id' in df.columns
                 else ('frame' if 'frame' in df.columns else None))

        v      = df[v_col].values.astype(float)
        frames = (df[f_col].values.astype(int)
                  if f_col else np.arange(len(df), dtype=int))
        n      = len(v)
        y_net  = float(self.y_net)

        crossings: list[int] = []
        last_frame = -debounce_frames

        # Find all valid (non-NaN) indices
        valid_idx = [i for i in range(n) if not np.isnan(v[i])]

        for k in range(1, len(valid_idx)):
            prev_i = valid_idx[k - 1]
            curr_i = valid_idx[k]

            # Skip if gap is too large (likely scene cut or long occlusion)
            gap = curr_i - prev_i
            if gap > max_gap:
                continue

            # Near → far crossing?
            if not (v[prev_i] > y_net and v[curr_i] < y_net):
                continue

            # Far-zone dwell: ball stays in far zone for min_far_count frames
            far_count = 0
            for j in valid_idx[k:]:
                if j >= curr_i + debounce_frames:
                    break
                if v[j] < y_net:
                    far_count += 1
                    if far_count >= min_far_count:
                        break
                else:
                    break   # returned to near zone

            if far_count < min_far_count:
                continue

            # Debounce
            fi = int(frames[curr_i])
            if fi - last_frame < debounce_frames:
                continue

            crossings.append(curr_i)
            last_frame = fi

        logging.info(
            "Net crossings (shots+serves): %d  "
            "(y_net=%.0f, min_far=%d fr, debounce=%d fr, max_gap=%d fr)",
            len(crossings), y_net, min_far_count, debounce_frames, max_gap,
        )
        return crossings

    def _segment_rallies_by_crossings(
        self,
        crossings: list[int],
        df: pd.DataFrame,
        frame_col: str,
        fps: float,
        inter_rally_gap_s: float = 8.0,
        pre_crossing_s: float = 3.0,
        post_crossing_s: float = 3.0,
    ) -> list[dict]:
        """
        Group net crossings into rallies based on inter-crossing gap.

        A new rally starts whenever two consecutive crossings are more than
        *inter_rally_gap_s* apart — a long pause means a point ended and a
        new serve is pending.

        Each rally's display window is expanded *pre_crossing_s* seconds
        before the first crossing (to show the serve toss) and
        *post_crossing_s* seconds after the last crossing (to show the
        ball landing / player reaction).

        Returns a list of rally dicts with the same schema as
        segment_rallies():
            {rally_id, start_frame, end_frame, rally_start, rally_end}

        Every rally is guaranteed to contain ≥ 1 crossing, so no
        empty-rally merging is needed.
        """
        if not crossings:
            logging.warning("No crossings detected — no rallies to segment")
            return []

        all_frame_ids   = df[frame_col].values.astype(int)
        crossing_frames = [int(df.loc[ci, frame_col]) for ci in crossings]

        gap_frames  = int(inter_rally_gap_s * fps)
        pre_frames  = int(pre_crossing_s  * fps)
        post_frames = int(post_crossing_s * fps)

        f_min = int(all_frame_ids[0])
        f_max = int(all_frame_ids[-1])

        # ── Group crossings by time proximity ──────────────────────────
        groups: list[list[int]] = [[crossing_frames[0]]]
        for cf in crossing_frames[1:]:
            if cf - groups[-1][-1] <= gap_frames:
                groups[-1].append(cf)
            else:
                groups.append([cf])

        # ── Convert each group to a rally dict ─────────────────────────
        rallies: list[dict] = []
        for rally_id, group in enumerate(groups, 1):
            start_target = max(group[0]  - pre_frames,  f_min)
            end_target   = min(group[-1] + post_frames, f_max)

            mask = (all_frame_ids >= start_target) & (all_frame_ids <= end_target)
            if not mask.any():
                continue

            start_frame = int(all_frame_ids[mask][0])
            end_frame   = int(all_frame_ids[mask][-1])

            rallies.append({
                'rally_id':    rally_id,
                'start_frame': start_frame,
                'end_frame':   end_frame,
                'rally_start': round(start_frame / fps, 3),
                'rally_end':   round(end_frame   / fps, 3),
            })

        # ── Clip overlapping windows between adjacent rallies ──────────
        for i in range(len(rallies) - 1):
            if rallies[i]['end_frame'] >= rallies[i + 1]['start_frame']:
                mid = (rallies[i]['end_frame'] + rallies[i + 1]['start_frame']) // 2
                rallies[i]['end_frame']        = mid
                rallies[i]['rally_end']        = round(mid / fps, 3)
                rallies[i + 1]['start_frame']  = mid + 1
                rallies[i + 1]['rally_start']  = round((mid + 1) / fps, 3)

        logging.info(
            "Crossing-based segmentation: %d crossings → %d rallies  "
            "(inter_gap=%.0fs  pre=%.0fs  post=%.0fs)",
            len(crossings), len(rallies),
            inter_rally_gap_s, pre_crossing_s, post_crossing_s,
        )
        return rallies

    def _merge_empty_rallies(
        self,
        rallies: list[dict],
        crossings: list[int],
        df: pd.DataFrame,
        frame_col: str,
    ) -> list[dict]:
        """
        Group rallies: any rally with 0 net crossings is merged forward into
        the next non-empty rally.  Consecutive 0-shot rallies are all merged
        with the first following non-empty rally.  Trailing 0-shot rallies
        (no subsequent non-empty rally) are merged backward into the last
        non-empty rally.

        Returns re-numbered merged rally list.
        """
        if not rallies:
            return []

        # ── Count crossings per rally ──────────────────────────────────
        frame_to_rally: dict[int, int] = {}
        for r in rallies:
            for fid in range(r['start_frame'], r['end_frame'] + 1):
                frame_to_rally[fid] = r['rally_id']

        shot_counts: dict[int, int] = {r['rally_id']: 0 for r in rallies}
        for ci in crossings:
            fid = int(df.loc[ci, frame_col]) if frame_col in df.columns else ci
            rid = frame_to_rally.get(fid)
            if rid is not None:
                shot_counts[rid] += 1

        # ── Forward pass: group 0-shot rallies with next non-empty rally ──
        merged: list[dict] = []
        pending: list[dict] = []          # accumulates consecutive 0-shot rallies

        for r in rallies:
            shots = shot_counts.get(r['rally_id'], 0)
            if shots == 0:
                pending.append(r)
            else:
                group = pending + [r]
                pending = []
                merged.append({
                    'rally_id':    0,   # assigned below
                    'start_frame': group[0]['start_frame'],
                    'end_frame':   group[-1]['end_frame'],
                    'rally_start': group[0]['rally_start'],
                    'rally_end':   group[-1]['rally_end'],
                })

        # ── Trailing 0-shot rallies: absorb into last non-empty ────────
        if pending:
            if merged:
                merged[-1]['end_frame'] = pending[-1]['end_frame']
                merged[-1]['rally_end'] = pending[-1]['rally_end']
            else:
                # Every rally had 0 shots — keep one combined rally
                merged.append({
                    'rally_id':    0,
                    'start_frame': pending[0]['start_frame'],
                    'end_frame':   pending[-1]['end_frame'],
                    'rally_start': pending[0]['rally_start'],
                    'rally_end':   pending[-1]['rally_end'],
                })

        # ── Re-number ──────────────────────────────────────────────────
        for new_id, r in enumerate(merged, 1):
            r['rally_id'] = new_id

        n_orig   = len(rallies)
        n_merged = len(merged)
        if n_orig != n_merged:
            logging.info(
                "Rally merge: %d → %d  (%d empty rallies absorbed)",
                n_orig, n_merged, n_orig - n_merged,
            )
        return merged

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def process_video(
        self,
        video_path: str,
        trajectory_df: pd.DataFrame,
        output_path: str,
    ) -> tuple[list, list, dict]:
        """
        Read video, compute indicators, write annotated output.

        Rally segmentation strategy
        ---------------------------
        Rallies are defined by net-crossing groups rather than ball-trajectory
        continuity.  Two crossings belong to the same rally when they are no
        more than *inter_rally_gap_s* seconds apart.  The rally display window
        is padded by *pre/post_crossing_s* seconds around the crossing cluster.

        This avoids false rally splits caused by detection gaps during bounces
        and ensures every rally contains at least one shot.

        Returns
        -------
        rallies    : list of rally dicts
        hit_events : list of per-crossing dicts
        rally_stats: dict  rally_id → {shots, dist_m}
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error("Cannot open video: %s", video_path)
            return [], [], {}

        fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.segmenter.fps = fps

        # ── 1. Preprocess DataFrame (sort, add time & speed columns) ───
        # segment_rallies() output is discarded; we use crossings instead.
        _, df = self.segmenter.segment_rallies(trajectory_df)

        # ── Column names ───────────────────────────────────────────────
        frame_col = 'frame_id' if 'frame_id' in df.columns else 'frame'
        u_col     = 'interp_u' if 'interp_u' in df.columns else 'x'
        v_col     = 'interp_v' if 'interp_v' in df.columns else 'y'

        # ── 2. Detect all net crossings (shots + serves) ───────────────
        crossings = self._detect_crossings(df)

        # ── 3. Build rallies from crossing groups ──────────────────────
        # Every rally has ≥1 crossing; no empty-rally merging needed.
        rallies = self._segment_rallies_by_crossings(
            crossings, df, frame_col, fps)

        # ── 4. Build frame → rally lookup ──────────────────────────────
        frame_to_rally: dict[int, int] = {}
        for r in rallies:
            for fid in range(r['start_frame'], r['end_frame'] + 1):
                frame_to_rally[fid] = r['rally_id']

        # ── 5. Pre-compute crossing speeds ─────────────────────────────
        crossing_set: set[int] = set(crossings)
        crossing_speeds: dict[int, float] = {
            i: self._ball_speed_kmh(df, i, fps, u_col, v_col)
            for i in crossings
        }

        # ── 6. Video writer ────────────────────────────────────────────
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # ── Per-rally state ────────────────────────────────────────────
        current_rally_id: int | None = None
        shots_in_rally   = 0
        dist_total       = 0.0
        prev_p0_w        = None
        prev_p1_w        = None

        # ── Display state (persists between rallies) ───────────────────
        display_rally_id: int | None = None
        display_shots    = 0
        display_dist     = 0.0
        display_speed    = 0.0

        hit_events:  list[dict]      = []
        rally_stats: dict[int, dict] = {}

        n_frames = min(len(df),
                       int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or len(df)))

        for frame_idx in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break

            row      = df.loc[frame_idx]
            frame_id = int(row[frame_col])

            # ── Rally state transition ─────────────────────────────────
            new_rally_id = frame_to_rally.get(frame_id)

            if new_rally_id != current_rally_id:
                if current_rally_id is not None:
                    rally_stats[current_rally_id] = {
                        'shots':  shots_in_rally,
                        'dist_m': round(dist_total, 2),
                    }
                    display_shots = shots_in_rally
                    display_dist  = dist_total

                current_rally_id = new_rally_id
                shots_in_rally   = 0
                dist_total       = 0.0
                prev_p0_w        = None
                prev_p1_w        = None

                if new_rally_id is not None:
                    display_rally_id = new_rally_id

            if current_rally_id is not None:
                display_shots = shots_in_rally
                display_dist  = dist_total

            # ── Near-field player ──────────────────────────────────────
            near_prefix = self._near_player_prefix(row)

            # ── Net-crossing shot event ────────────────────────────────
            if frame_idx in crossing_set:
                spd           = crossing_speeds.get(frame_idx, 0.0)
                display_speed = spd

                if current_rally_id is not None:
                    shots_in_rally += 1
                    hit_events.append({
                        'frame':       frame_id,
                        'time_s':      round(frame_id / fps, 3),
                        'rally_id':    current_rally_id,
                        'speed_kmh':   round(spd, 1),
                        'near_player': near_prefix or 'unknown',
                    })

            # ── Player distance (both players, merged) ─────────────────
            if current_rally_id is not None:
                for prefix in ('p0', 'p1'):
                    curr_w = self._get_player_world(row, prefix)
                    prev_w = prev_p0_w if prefix == 'p0' else prev_p1_w
                    if curr_w and prev_w:
                        d = np.hypot(curr_w[0] - prev_w[0],
                                     curr_w[1] - prev_w[1])
                        if self.min_move_dist < d < self.max_move_dist:
                            dist_total += d
                    if curr_w:
                        if prefix == 'p0':
                            prev_p0_w = curr_w
                        else:
                            prev_p1_w = curr_w

            # ── Draw nearest-player pose skeleton ──────────────────────
            self._draw_player_keypoints(frame, row, near_prefix)

            # ── Draw ball ──────────────────────────────────────────────
            bu = row.get(u_col)
            bv = row.get(v_col)
            if (bu is not None and bv is not None
                    and not pd.isna(bu) and not pd.isna(bv)):
                cv2.circle(frame,
                           (int(float(bu)), int(float(bv))),
                           8, (0, 255, 0), 2)

            # ── Panel (top-left, semi-transparent) ────────────────────
            panel_w = 320
            panel_h = 140
            px1     = 10
            py1     = 10
            px2     = px1 + panel_w
            py2     = py1 + panel_h

            overlay = frame.copy()
            cv2.rectangle(overlay, (px1, py1), (px2, py2), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            tx  = px1 + 15
            ty1 = py1 + 32
            ty2 = py1 + 64
            ty3 = py1 + 96
            ty4 = py1 + 125

            rally_lbl = (f"Rally: #{display_rally_id}"
                         if display_rally_id else "Rally: ---")
            speed_lbl = (f"Speed: {display_speed:.1f} km/h"
                         if display_speed > 0 else "Speed: ---")

            cv2.putText(frame, rally_lbl,
                        (tx, ty1), self.font, 0.7, (255, 255,   0), 2)
            cv2.putText(frame, f"Shots: {display_shots}",
                        (tx, ty2), self.font, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, speed_lbl,
                        (tx, ty3), self.font, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Dist: {display_dist:.1f} m",
                        (tx, ty4), self.font, 0.5, (0,   255, 255), 1)

            out.write(frame)

        # ── Archive last rally ─────────────────────────────────────────
        if current_rally_id is not None:
            rally_stats[current_rally_id] = {
                'shots':  shots_in_rally,
                'dist_m': round(dist_total, 2),
            }

        cap.release()
        out.release()
        logging.info(
            "Video done → %s  |  rallies=%d  shots=%d",
            output_path, len(rallies), len(hit_events),
        )
        return rallies, hit_events, rally_stats
