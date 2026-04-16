"""Async-ready video processing pipeline."""

from __future__ import annotations

import asyncio
import csv
import io
import logging
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from detectors.wasb_detector import WASBDetector
from detectors.yolo_pose import YOLOPoseEstimator
from pipeline.data_types import FrameResult, Player, TennisBall
from utils.csv_io import CSV_HEADER, result_to_row
from utils.visualizer import visualize

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Async-ready pipeline that runs WASB + YOLO on each frame of a video.

    Design notes
    ------------
    * **I/O concurrency** — frame reads and writes are dispatched to a
      ``ThreadPoolExecutor`` so the ``asyncio`` event loop is never blocked.
    * **Inference serialisation** — ``_infer_lock`` ensures only one thread
      executes GPU inference at a time, making it safe to call
      :meth:`process_video_async` concurrently on multiple videos via
      ``asyncio.gather`` (``--parallel`` flag).
    * **Per-video WASB buffers** — :meth:`process_video_async` creates its own
      :func:`~detectors.wasb_detector.WASBDetector.make_buffer` deque so
      parallel video streams never mix frame history.

    Usage
    -----
    Single video (synchronous convenience)::

        processor.process_video(src, csv_out, mp4_out)

    Multiple videos with overlapping I/O::

        await asyncio.gather(*[
            processor.process_video_async(v, c, m, executor)
            for v, c, m in jobs
        ])
    """

    def __init__(
        self,
        wasb_detector:       WASBDetector,
        yolo_estimator:      YOLOPoseEstimator,
        ball_conf_threshold: float = 0.5,
        kp_conf_threshold:   float = 0.3,
        max_players:         int   = 2,
    ) -> None:
        self.ball_det    = wasb_detector
        self.yolo        = yolo_estimator
        self.ball_thr    = ball_conf_threshold
        self.kp_thr      = kp_conf_threshold
        self.max_players = max_players
        self._infer_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Sync helpers (executed inside ThreadPoolExecutor)
    # ------------------------------------------------------------------

    @staticmethod
    def _read_frame_sync(cap: cv2.VideoCapture) -> Optional[np.ndarray]:
        ok, frame = cap.read()
        return frame if ok else None

    def _infer_sync(self, frame: np.ndarray, ball_buffer: deque) -> FrameResult:
        """Run WASB + YOLO inference, serialised by ``_infer_lock``."""
        h, w = frame.shape[:2]
        with self._infer_lock:
            self.ball_det.push_frame(frame, ball_buffer)
            detected, u, v, conf = self.ball_det.detect(
                ball_buffer, h, w, self.ball_thr
            )
            kp_sets = self.yolo.estimate(frame, self.max_players)

        ball = TennisBall(
            detected=detected,
            u=u if detected else None,
            v=v if detected else None,
            conf=conf,
        )
        players = [Player(player_id=i, keypoints=kp) for i, kp in enumerate(kp_sets)]
        return FrameResult(frame_id=0, ball=ball, players=players)  # frame_id set by caller

    # ------------------------------------------------------------------
    # Async coroutines
    # ------------------------------------------------------------------

    async def _read_frame_async(
        self, cap: cv2.VideoCapture, executor: ThreadPoolExecutor
    ) -> Optional[np.ndarray]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, self._read_frame_sync, cap)

    async def _write_frame_async(
        self,
        writer: cv2.VideoWriter,
        frame: np.ndarray,
        executor: ThreadPoolExecutor,
    ) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, writer.write, frame)

    async def _infer_async(
        self,
        frame: np.ndarray,
        ball_buffer: deque,
        executor: ThreadPoolExecutor,
    ) -> FrameResult:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            executor, self._infer_sync, frame, ball_buffer
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_video_async(
        self,
        video_path: str,
        output_video_path: str,
        executor: ThreadPoolExecutor,
    ) -> Tuple[int, pd.DataFrame]:
        """Process *video_path* asynchronously.

        Returns
        -------
        frame_count : int
        df : pd.DataFrame
            Per-frame detection results with columns matching ``CSV_HEADER``.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        ball_buffer = self.ball_det.make_buffer()  # isolated per-video state
        frame_count = 0
        rows: List = []

        while True:
            frame = await self._read_frame_async(cap, executor)
            if frame is None:
                break

            result          = await self._infer_async(frame, ball_buffer, executor)
            result.frame_id = frame_count
            rows.append(result_to_row(result))

            vis = visualize(frame, result, self.kp_thr)
            await self._write_frame_async(writer, vis, executor)

            frame_count += 1
            if frame_count % 200 == 0:
                logger.info("%s: %d frames", Path(video_path).name, frame_count)

        cap.release()
        writer.release()

        # Build DataFrame via CSV round-trip so pandas handles type inference
        # (empty strings → NaN, numeric strings → float) identically to
        # reading the file from disk.
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(CSV_HEADER)
        w.writerows(rows)
        buf.seek(0)
        df = pd.read_csv(buf)

        logger.info("Done %s → %s (%d frames)", video_path, output_video_path, frame_count)
        return frame_count, df

    def process_video(
        self,
        video_path: str,
        output_video_path: str,
        workers: int = 2,
    ) -> Tuple[int, pd.DataFrame]:
        """Synchronous convenience wrapper around :meth:`process_video_async`."""
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return asyncio.run(
                self.process_video_async(video_path, output_video_path, executor)
            )
