"""YOLO-based human pose estimator with Ultralytics fuse-bug workaround."""

from __future__ import annotations

import threading
from typing import List

import numpy as np
import torch

from config.model_config import (
    KP_LEFT_ELBOW,
    KP_LEFT_SHOULDER,
    KP_LEFT_WRIST,
    KP_RIGHT_ELBOW,
    KP_RIGHT_SHOULDER,
    KP_RIGHT_WRIST,
)
from pipeline.data_types import KeypointSet


# ---------------------------------------------------------------------------
# Ultralytics fuse-bug workaround
# ---------------------------------------------------------------------------

def patch_repvggdw_fuse() -> None:
    """Fix the Ultralytics 8.x ``AttributeError: 'Conv' object has no attribute 'bn'``
    crash that occurs on models containing ``RepVGGDW`` blocks.

    Root cause
    ----------
    ``BaseModel.fuse()`` in ``ultralytics/nn/tasks.py`` iterates modules
    depth-first, so the inner ``Conv`` children of each ``RepVGGDW`` block
    have their ``BatchNorm`` absorbed and ``bn`` deleted *before*
    ``RepVGGDW.fuse()`` is called.  ``RepVGGDW.fuse()`` then crashes at
    ``self.conv.bn`` because the attribute no longer exists.

    Fix
    ---
    Replace ``RepVGGDW.fuse`` with a version that detects the pre-fused state
    (``hasattr(self.conv, "bn") is False``) and manually merges the two
    depthwise branches by padding the 3×3 kernel to 7×7 and adding the
    weight/bias tensors element-wise — producing the same result the original
    implementation would have computed.
    """
    try:
        import torch as _torch
        import torch.nn as _nn
        import torch.nn.functional as _F
        from ultralytics.nn.modules.block import RepVGGDW
    except ImportError:
        return  # ultralytics not installed; skip silently

    _orig_fuse = RepVGGDW.fuse

    @_torch.no_grad()
    def _safe_fuse(self: RepVGGDW) -> None:  # type: ignore[override]
        if not hasattr(self, "conv1"):
            return  # already single-branch

        if hasattr(self.conv, "bn"):
            # Normal path: inner Conv layers still have their bn — delegate.
            _orig_fuse(self)
            return

        # Pre-fused path: outer BaseModel.fuse loop has already absorbed bn
        # into self.conv.conv and self.conv1.conv (plain nn.Conv2d with bias).
        # Merge the two depthwise branches by padding conv1 (3×3 → 7×7) and
        # adding weights + biases element-wise.
        conv_w  = self.conv.conv.weight.data   # [C, 1, 7, 7]
        conv_b  = self.conv.conv.bias.data      # [C]
        conv1_w = self.conv1.conv.weight.data   # [C, 1, 3, 3]
        conv1_b = self.conv1.conv.bias.data     # [C]

        conv1_w_padded = _F.pad(conv1_w, [2, 2, 2, 2])   # [C, 1, 7, 7]

        merged = _nn.Conv2d(
            in_channels=self.conv.conv.in_channels,
            out_channels=self.conv.conv.out_channels,
            kernel_size=7,
            stride=self.conv.conv.stride,
            padding=self.conv.conv.padding,
            groups=self.conv.conv.groups,
            bias=True,
        )
        merged.weight.data.copy_(conv_w + conv1_w_padded)
        merged.bias.data.copy_(conv_b + conv1_b)

        self.conv = merged   # replace Conv wrapper with plain Conv2d
        del self.conv1       # drop second branch (outer loop then sets forward_fuse)

    RepVGGDW.fuse = _safe_fuse


# ---------------------------------------------------------------------------
# YOLOPoseEstimator
# ---------------------------------------------------------------------------

class YOLOPoseEstimator:
    """Human pose estimator wrapping an Ultralytics YOLO pose model.

    Returns up to ``max_players`` :class:`~pipeline.data_types.KeypointSet`
    objects per frame, ordered by bounding-box area (largest first).

    Parameters
    ----------
    weights_path:
        Path to the ``.pt`` YOLO pose model file.
    device:
        PyTorch device for inference.
    """

    _JOINTS = {
        "left_shoulder":  KP_LEFT_SHOULDER,
        "right_shoulder": KP_RIGHT_SHOULDER,
        "left_elbow":     KP_LEFT_ELBOW,
        "right_elbow":    KP_RIGHT_ELBOW,
        "left_wrist":     KP_LEFT_WRIST,
        "right_wrist":    KP_RIGHT_WRIST,
    }

    def __init__(self, weights_path: str, device: torch.device) -> None:
        patch_repvggdw_fuse()

        from ultralytics import YOLO  # deferred — large package
        self._model      = YOLO(weights_path)
        self._device_str = "cuda" if device.type == "cuda" else "cpu"
        self._lock       = threading.Lock()

        # Pre-warm: triggers setup_model → fuse exactly once, before any
        # thread pool is active.  Without this, parallel video threads each
        # hit the first YOLO call simultaneously, both call setup_model/fuse,
        # and the second thread sees `bn` already deleted by the first → crash.
        _dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        self._model(_dummy, device=self._device_str, verbose=False)

    def estimate(self, bgr: np.ndarray, max_players: int = 2) -> List[KeypointSet]:
        """Run pose estimation on *bgr* and return up to *max_players* results.

        The lock ensures only one thread executes YOLO inference at a time,
        preventing CUDA stream races when processing multiple videos in parallel.
        """
        with self._lock:
            results = self._model(bgr, device=self._device_str, verbose=False)

        if not results or results[0].keypoints is None:
            return []

        kp_xy   = results[0].keypoints.xy.cpu().numpy()    # [N, 17, 2]
        kp_conf = results[0].keypoints.conf.cpu().numpy()  # [N, 17]
        boxes   = results[0].boxes.xyxy.cpu().numpy()       # [N, 4]

        if len(boxes) == 0:
            return []

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        order = np.argsort(-areas)

        players: List[KeypointSet] = []
        for idx in order[:max_players]:
            kp = KeypointSet(bbox=tuple(boxes[idx].tolist()))
            for attr, ki in self._JOINTS.items():
                x, y = kp_xy[idx, ki]
                c    = kp_conf[idx, ki]
                setattr(kp, attr, (float(x), float(y), float(c)))
            players.append(kp)

        return players
