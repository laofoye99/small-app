"""WASB tennis ball detector wrapping the HRNet backbone."""

from __future__ import annotations

from collections import deque
from typing import Tuple

import cv2
import numpy as np
import torch

from config.model_config import INP_H, INP_W, IMG_MEAN, IMG_STD, WASB_CFG
from model_definitions.wasb import HRNet


class WASBDetector:
    """
    Tennis ball detector wrapping the HRNet-based WASB model.

    The detector is deliberately stateless with respect to the frame buffer:
    callers supply an explicit *buffer* (a deque of preprocessed tensors) so
    that multiple videos can run concurrently without sharing mutable state.
    Create one buffer per video with :meth:`make_buffer`.

    Parameters
    ----------
    weights_path:
        Path to the ``.pth.tar`` checkpoint file (key: ``model_state_dict``).
    device:
        PyTorch device for inference.
    """

    def __init__(self, weights_path: str, device: torch.device) -> None:
        self.device    = device
        self.frames_in = WASB_CFG["frames_in"]

        model = HRNet(WASB_CFG)
        ckpt  = torch.load(weights_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        model.load_state_dict(state, strict=True)
        self.model = model.to(device).eval()

    # ------------------------------------------------------------------
    # Per-video buffer factory
    # ------------------------------------------------------------------

    def make_buffer(self) -> deque:
        """Return a fresh, empty sliding-window buffer for one video stream."""
        return deque(maxlen=self.frames_in)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess(bgr: np.ndarray) -> torch.Tensor:
        """BGR HWC uint8 → resized (INP_H × INP_W) normalised float32 CHW tensor."""
        resized = cv2.resize(bgr, (INP_W, INP_H), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - IMG_MEAN) / IMG_STD
        return torch.from_numpy(rgb.transpose(2, 0, 1))  # [3, INP_H, INP_W]

    def push_frame(self, bgr: np.ndarray, buffer: deque) -> None:
        """Append a preprocessed frame tensor to *buffer*.

        On the very first call the buffer is pre-filled with copies of the
        first frame so that :meth:`detect` can be called immediately from
        frame 0 without special-casing.
        """
        t = self._preprocess(bgr)
        if not buffer:
            for _ in range(self.frames_in):
                buffer.append(t)
        else:
            buffer.append(t)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def detect(
        self,
        buffer: deque,
        orig_h: int,
        orig_w: int,
        conf_threshold: float = 0.5,
    ) -> Tuple[bool, float, float, float]:
        """Run ball detection on the current *buffer* contents.

        Parameters
        ----------
        buffer:
            Sliding window of preprocessed tensors for the current video.
        orig_h, orig_w:
            Original frame dimensions used to scale detected coordinates back.
        conf_threshold:
            Minimum sigmoid activation to count as a detection.

        Returns
        -------
        (detected, u, v, confidence)
            ``u``, ``v`` are pixel coordinates in the *original* frame.
        """
        # Stack buffer → [1, frames_in*3, INP_H, INP_W]
        x = torch.cat(list(buffer), dim=0).unsqueeze(0).to(self.device)

        # out: {0: [1, frames_out, INP_H, INP_W]}  (STRIDES=[1,1] → no stem downscale)
        out     = self.model(x)
        heatmap = out[0][0, -1]  # last channel = current frame heatmap

        conf_map = torch.sigmoid(heatmap)
        flat_idx = conf_map.argmax().item()
        v_model  = flat_idx // INP_W
        u_model  = flat_idx  % INP_W
        conf     = conf_map[v_model, u_model].item()

        # Scale from model input space back to original frame space
        u = float(u_model) * (orig_w / INP_W)
        v = float(v_model) * (orig_h / INP_H)

        return conf >= conf_threshold, u, v, float(conf)

    @torch.no_grad()
    def detect_batch(
        self,
        buffer: deque,
        orig_h: int,
        orig_w: int,
        conf_threshold: float = 0.5,
    ) -> list[tuple[bool, float, float, float]]:
        """Run inference once and return results for all frames_in frames.

        Used in non-overlapping batch mode: caller accumulates exactly
        frames_in frames, calls this method, then clears the buffer.

        Returns
        -------
        List of (detected, u, v, confidence) – one entry per input frame,
        in the same order as the frames were pushed into *buffer*.
        """
        x = torch.cat(list(buffer), dim=0).unsqueeze(0).to(self.device)
        out = self.model(x)
        heatmaps = out[0][0]  # [frames_in, INP_H, INP_W]

        results: list[tuple[bool, float, float, float]] = []
        for i in range(heatmaps.shape[0]):
            conf_map = torch.sigmoid(heatmaps[i])
            flat_idx = int(conf_map.argmax().item())
            v_model  = flat_idx // INP_W
            u_model  = flat_idx  % INP_W
            conf     = float(conf_map[v_model, u_model].item())
            u        = float(u_model) * (orig_w / INP_W)
            v        = float(v_model) * (orig_h / INP_H)
            results.append((conf >= conf_threshold, u, v, conf))
        return results
