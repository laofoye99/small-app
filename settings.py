"""
settings.py
===========
Central path resolver for the Tennis Analysis pipeline.

All scripts import this module to get computed absolute paths.
No script should contain hardcoded absolute paths.

    from settings import HOME, PATHS, CAMERAS_CFG

How home_dir is resolved (in priority order):
  1. home_dir in config.yaml  — set this when you move the repo
  2. Auto-detect              — directory containing this settings.py file

So on any new machine you only need to set home_dir in config.yaml
(or leave it blank if the project is in the same relative layout).
"""

from __future__ import annotations

from pathlib import Path

try:
    import yaml
    _YAML_OK = True
except ImportError:
    _YAML_OK = False

# ---------------------------------------------------------------------------
# Locate config.yaml (always next to this file)
# ---------------------------------------------------------------------------

_SELF_DIR   = Path(__file__).parent.resolve()
_CONFIG_FILE = _SELF_DIR / 'config.yaml'


def _load_config() -> dict:
    if not _CONFIG_FILE.exists() or not _YAML_OK:
        return {}
    try:
        return yaml.safe_load(_CONFIG_FILE.read_text(encoding='utf-8')) or {}
    except Exception:
        return {}


_cfg = _load_config()

# ---------------------------------------------------------------------------
# Resolve HOME
# ---------------------------------------------------------------------------

def _resolve_home() -> Path:
    raw = str(_cfg.get('home_dir', '') or '').strip()
    if raw:
        p = Path(raw)
        if p.is_absolute():
            return p.resolve()
        # relative to config.yaml location
        return (_SELF_DIR / p).resolve()
    return _SELF_DIR


HOME: Path = _resolve_home()

# ---------------------------------------------------------------------------
# Sub-directory helpers
# ---------------------------------------------------------------------------

def _rel(section: str, key: str, fallback: str) -> Path:
    """Return HOME / value-from-config, with fallback."""
    val = _cfg.get(section, {}).get(key, fallback)
    return HOME / val


# Core directories
UPLOADS_DIR = _rel('paths', 'uploads', 'uploads')
OUTPUT_DIR  = _rel('paths', 'output',  'output')
WEIGHTS_DIR = _rel('paths', 'weights', 'model_weights')

# Calibration / geometry files
HOMOGRAPHY_JSON = _rel('paths', 'homography', 'uploads/homography_matrices.json')
CALIB_CAM66     = _rel('paths', 'calib_cam66', 'uploads/cal_cam66.json')
CALIB_CAM68     = _rel('paths', 'calib_cam68', 'uploads/cal_cam68.json')

# Model weights
TRACKNET_MODEL = _rel('model', 'path',      'model_weights/TrackNet_finetuned.pt')
WASB_MODEL     = _rel('model', 'wasb_path', 'model_weights/wasb_tennis_best.pth.tar')
YOLO_MODEL     = _rel('model', 'yolo_path', 'model_weights/yolo26x-pose.pt')

# Inference settings
_model_cfg      = _cfg.get('model', {})
MODEL_DEVICE    = _model_cfg.get('device',     'cuda')
MODEL_FPS       = float(_model_cfg.get('fps',       25.0))
MODEL_FRAMES_IN = int(_model_cfg.get('frames_in',   8))
MODEL_FRAMES_OUT= int(_model_cfg.get('frames_out',  8))
MODEL_THRESHOLD = float(_model_cfg.get('threshold', 0.3))

# ---------------------------------------------------------------------------
# Camera config
# ---------------------------------------------------------------------------

_DEFAULT_CAMERAS = {
    'cam66': {
        'rtsp_url':       'rtsp://admin:motion168@192.168.1.66:554/Streaming/Channels/101',
        'homography_key': 'cam66',
        'y_net':          238.0,
        'serial':         'FV9942593',
    },
    'cam68': {
        'rtsp_url':       'rtsp://admin:motion168@192.168.1.68:554/Streaming/Channels/101',
        'homography_key': 'cam68',
        'y_net':          285.0,
        'serial':         'FV9942588',
    },
}

CAMERAS_CFG: dict = _cfg.get('cameras', _DEFAULT_CAMERAS)

# Ensure every camera entry has a 'serial' field (back-compat)
for _name, _cam in CAMERAS_CFG.items():
    if 'serial' not in _cam:
        _cam['serial'] = _DEFAULT_CAMERAS.get(_name, {}).get('serial', f'UNKNOWN_{_name}')

# ---------------------------------------------------------------------------
# Convenience: expose raw config for any caller that needs deeper values
# ---------------------------------------------------------------------------

raw_cfg: dict = _cfg


# ---------------------------------------------------------------------------
# Quick self-test / debug
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print(f"HOME            : {HOME}")
    print(f"UPLOADS_DIR     : {UPLOADS_DIR}")
    print(f"OUTPUT_DIR      : {OUTPUT_DIR}")
    print(f"WEIGHTS_DIR     : {WEIGHTS_DIR}")
    print(f"HOMOGRAPHY_JSON : {HOMOGRAPHY_JSON}")
    print(f"TRACKNET_MODEL  : {TRACKNET_MODEL}")
    print(f"WASB_MODEL      : {WASB_MODEL}")
    print(f"YOLO_MODEL      : {YOLO_MODEL}")
    print(f"MODEL_DEVICE    : {MODEL_DEVICE}  fps={MODEL_FPS}")
    print()
    for cam, info in CAMERAS_CFG.items():
        print(f"Camera {cam}  serial={info.get('serial')}  y_net={info.get('y_net')}")
        print(f"  rtsp: {info.get('rtsp_url')}")
