"""
Model configuration and shared constants.

All values here were derived by introspecting model_weights/wasb_tennis_best.pth.tar:
  conv1.weight          → [64,  9, 3,3]  → frames_in  = 3  (9 = 3×3 RGB channels)
  final_layers.0.weight → [3,  16, 1,1]  → frames_out = 3, in_ch = 16 (W16 variant)
  Stage branch channels : [16], [16,32], [16,32,64], [16,32,64,128]
  No deconv layers      : NUM_DECONVS = 0
"""

import os

import numpy as np

# ---------------------------------------------------------------------------
# Suppress OMP duplicate-DLL warning common on Windows before torch is loaded
# ---------------------------------------------------------------------------
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ---------------------------------------------------------------------------
# Fixed model input size
# Frames are always resized to this before WASB inference.
# Chosen so every stage's spatial dimension is divisible by 2^(num_branches-1):
#   288 / 8 = 36,  512 / 8 = 64  — all integer.
#
# CRITICAL — STEM STRIDES must be [1, 1]:
#   With STRIDES=[2,2] and a 1080p input the stem reduces H=1080 → 270.
#   Stage-2 branch-1 downsamples 270 → 135, stage-3 branch-2 → 68.
#   HRNet fuse-layer upsamples branch-2 by ×4 = 68×4 = 272 ≠ 270 → crash.
#   STRIDES=[1,1] keeps the stem at full resolution so branches stay at
#   clean multiples of the fixed input size.
# ---------------------------------------------------------------------------
INP_H: int = 288
INP_W: int = 512

# ---------------------------------------------------------------------------
# WASB / HRNet model architecture configuration
# ---------------------------------------------------------------------------
WASB_CFG: dict = {
    "frames_in":  3,
    "frames_out": 3,
    "out_scales": [0],
    "MODEL": {
        "EXTRA": {
            "STEM": {"STRIDES": [1, 1], "INPLANES": 64},
            "STAGE1": {
                "NUM_MODULES": 1, "NUM_BRANCHES": 1,
                "NUM_BLOCKS": [1], "NUM_CHANNELS": [32],
                "BLOCK": "BOTTLENECK", "FUSE_METHOD": "SUM",
            },
            "STAGE2": {
                "NUM_MODULES": 1, "NUM_BRANCHES": 2,
                "NUM_BLOCKS": [2, 2], "NUM_CHANNELS": [16, 32],
                "BLOCK": "BASIC", "FUSE_METHOD": "SUM",
            },
            "STAGE3": {
                "NUM_MODULES": 1, "NUM_BRANCHES": 3,
                "NUM_BLOCKS": [2, 2, 2], "NUM_CHANNELS": [16, 32, 64],
                "BLOCK": "BASIC", "FUSE_METHOD": "SUM",
            },
            "STAGE4": {
                "NUM_MODULES": 1, "NUM_BRANCHES": 4,
                "NUM_BLOCKS": [2, 2, 2, 2], "NUM_CHANNELS": [16, 32, 64, 128],
                "BLOCK": "BASIC", "FUSE_METHOD": "SUM",
            },
            "DECONV": {"NUM_DECONVS": 0, "KERNEL_SIZE": [], "NUM_CHANNELS": []},
            "FINAL_CONV_KERNEL": 1,
            "PRETRAINED_LAYERS": ["*"],
        }
    },
}

# ---------------------------------------------------------------------------
# Preprocessing normalization (ImageNet statistics)
# ---------------------------------------------------------------------------
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ---------------------------------------------------------------------------
# COCO keypoint indices used for tennis arm analysis
# ---------------------------------------------------------------------------
KP_LEFT_SHOULDER  = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW     = 7
KP_RIGHT_ELBOW    = 8
KP_LEFT_WRIST     = 9
KP_RIGHT_WRIST    = 10

# ---------------------------------------------------------------------------
# Drawing colors (BGR)
# ---------------------------------------------------------------------------
COLOR_BALL     = (0,   255, 255)   # yellow  – ball
COLOR_SKEL_L   = (255, 128,   0)   # blue    – left arm
COLOR_SKEL_R   = (0,   128, 255)   # orange  – right arm
COLOR_SKEL_BAR = (200, 200, 200)   # grey    – shoulder bar
COLOR_KP       = (0,   255,   0)   # green   – joint dot
COLOR_BBOX     = (255,   0, 255)   # magenta – player bbox
