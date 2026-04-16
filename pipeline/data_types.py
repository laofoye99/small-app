"""Shared data structures passed between detectors, visualiser, and CSV writer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class KeypointSet:
    """Wrist / elbow / shoulder keypoints for one player.

    Each field is either None (joint not detected) or a (u, v, conf) tuple
    where u is the horizontal pixel coordinate, v is the vertical pixel
    coordinate, and conf is the model's keypoint visibility confidence.
    """
    left_shoulder:  Optional[Tuple[float, float, float]] = None
    right_shoulder: Optional[Tuple[float, float, float]] = None
    left_elbow:     Optional[Tuple[float, float, float]] = None
    right_elbow:    Optional[Tuple[float, float, float]] = None
    left_wrist:     Optional[Tuple[float, float, float]] = None
    right_wrist:    Optional[Tuple[float, float, float]] = None
    bbox:           Optional[Tuple[float, float, float, float]] = None  # x1,y1,x2,y2


@dataclass
class TennisBall:
    """Ball detection result for a single frame.

    Pixel coordinates (u, v) come directly from the WASB detector.
    Normalised court coordinates (x, y) and speed are filled in later
    by a court-mapping / trajectory-analysis step; they are None until then.
    """
    detected: bool
    u:        Optional[float]         # pixel x in original frame
    v:        Optional[float]         # pixel y in original frame
    conf:     float                   # sigmoid confidence
    x:        Optional[float] = None  # normalised court coord [0, 1]
    y:        Optional[float] = None  # normalised court coord [0, 1]
    speed:    Optional[float] = None  # km/h, filled after trajectory analysis


@dataclass
class Player:
    """Per-frame player detection result.

    player_id 0 = far-end player, 1 = near-end player (sorted by bbox area).
    Normalised court coordinates (x, y) are filled in by a court-mapping step.
    """
    player_id: int
    keypoints: KeypointSet
    x:         Optional[float] = None  # normalised court position [0, 1]
    y:         Optional[float] = None  # normalised court position [0, 1]


@dataclass
class FrameResult:
    """All detection outputs for a single video frame."""
    frame_id: int
    ball:     TennisBall
    players:  List[Player] = field(default_factory=list)
