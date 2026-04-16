"""CSV schema and serialisation for per-frame detection results."""

from __future__ import annotations

from typing import List, Optional

from pipeline.data_types import FrameResult

# ---------------------------------------------------------------------------
# Column layout
# ---------------------------------------------------------------------------
# Joint abbreviations: ls=left_shoulder, rs=right_shoulder,
#                      le=left_elbow,    re=right_elbow,
#                      lw=left_wrist,    rw=right_wrist
_JOINT_ABBREVS = ["ls", "rs", "le", "re", "lw", "rw"]
_JOINT_ATTRS   = [
    "left_shoulder", "right_shoulder",
    "left_elbow",    "right_elbow",
    "left_wrist",    "right_wrist",
]

CSV_HEADER: List[str] = (
    ["frame_id", "detected", "ball_u", "ball_v", "ball_conf"]
    + [
        f"p{p}_{j}_{c}"
        for p in range(2)
        for j in _JOINT_ABBREVS
        for c in ("u", "v", "conf")
    ]
)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _fmt(v: Optional[float], decimals: int = 2) -> str:
    return "" if v is None else f"{v:.{decimals}f}"


def result_to_row(r: FrameResult) -> List:
    """Serialise a :class:`~pipeline.data_types.FrameResult` to a CSV row list."""
    row: List = [
        r.frame_id,
        int(r.ball.detected),
        _fmt(r.ball.u),
        _fmt(r.ball.v),
        _fmt(r.ball.conf, 4),
    ]
    for p_idx in range(2):
        player = r.players[p_idx] if p_idx < len(r.players) else None
        kp = player.keypoints if player is not None else None
        for attr in _JOINT_ATTRS:
            joint = getattr(kp, attr) if kp is not None else None
            if joint is not None:
                row += [_fmt(joint[0]), _fmt(joint[1]), _fmt(joint[2], 4)]
            else:
                row += ["", "", ""]
    return row
