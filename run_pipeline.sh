#!/usr/bin/env bash
# ============================================================
#  Tennis Analysis Pipeline — Linux / macOS / Git-Bash Runner
#  run_pipeline.sh
#
#  MODES
#  -----
#  bash run_pipeline.sh                        batch pipeline (default videos)
#  bash run_pipeline.sh cam66.mp4 cam68.mp4    batch pipeline (custom videos)
#  bash run_pipeline.sh --live                 real-time RTSP pipeline (production)
#  bash run_pipeline.sh --live --dry-run       real-time RTSP pipeline (dry run)
#  bash run_pipeline.sh --setup                one-time camera calibration
#  bash run_pipeline.sh --stage N              resume batch from stage N (0-2)
#
#  FIRST-TIME SETUP
#  ----------------
#  1. Copy your TrackNet model to:  model_weights/TrackNet_finetuned.pt
#  2. Copy WASB model to:           model_weights/wasb_tennis_best.pth.tar
#  3. Copy YOLO model to:           model_weights/yolo26x-pose.pt
#  4. Edit home_dir in config.yaml  (only if auto-detect fails)
#  5. Run:  bash run_pipeline.sh --setup
# ============================================================
set -euo pipefail

# ── Project root (always the directory containing this script) ─────────────
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$ROOT/scripts"
UPLOADS="$ROOT/uploads"
OUTPUT="$ROOT/output"
WEIGHTS="$ROOT/model_weights"

# ── Default video filenames ────────────────────────────────────────────────
CAM66_VIDEO="$UPLOADS/cam66_video.mp4"
CAM68_VIDEO="$UPLOADS/cam68_video.mp4"

# ── Python executable ──────────────────────────────────────────────────────
PYTHON="${PYTHON:-python}"

# ── Colours ───────────────────────────────────────────────────────────────
GREEN="\033[92m"; YELLOW="\033[93m"; RED="\033[91m"
CYAN="\033[96m";  RESET="\033[0m"

info()  { echo -e "${CYAN}$*${RESET}"; }
ok()    { echo -e "${GREEN}[OK]${RESET} $*"; }
die()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; exit 1; }
stage() { echo -e "\n${YELLOW}[Stage $1]${RESET} $2"; }

# ── Argument parsing ───────────────────────────────────────────────────────
MODE="batch"
DRY_RUN=""
START_STAGE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --live)       MODE="live";  shift ;;
        --setup)      MODE="setup"; shift ;;
        --dry-run)    DRY_RUN="--dry-run"; shift ;;
        --stage)      START_STAGE="$2"; shift 2 ;;
        --python)     PYTHON="$2"; shift 2 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,2\}//'
            exit 0 ;;
        -*)           die "Unknown option: $1" ;;
        *)
            if [[ -z "${CAM66_SET:-}" ]]; then CAM66_VIDEO="$1"; CAM66_SET=1
            else CAM68_VIDEO="$1"; fi
            shift ;;
    esac
done

# ============================================================
#  ONE-TIME SETUP
# ============================================================
if [[ "$MODE" == "setup" ]]; then
    info "============================================================"
    info "  One-Time Camera Setup"
    info "============================================================"
    echo "This opens interactive windows for each camera."
    echo "Follow the on-screen instructions to click court lines."
    echo "Press Enter to continue..."
    read -r

    stage "S1/4" "Speed calibration — cam66"
    "$PYTHON" "$SCRIPTS/calibrate_court.py" "$CAM66_VIDEO" \
        --out "$UPLOADS/cal_cam66.json"

    stage "S2/4" "Speed calibration — cam68"
    "$PYTHON" "$SCRIPTS/calibrate_court.py" "$CAM68_VIDEO" \
        --out "$UPLOADS/cal_cam68.json"

    stage "S3/4" "Homography annotation — cam66"
    "$PYTHON" "$SCRIPTS/annotate_homography.py" "$CAM66_VIDEO" \
        --calib "$UPLOADS/cal_cam66.json"

    stage "S4/4" "Homography annotation — cam68"
    "$PYTHON" "$SCRIPTS/annotate_homography.py" "$CAM68_VIDEO" \
        --calib "$UPLOADS/cal_cam68.json"

    echo ""
    ok "Setup complete.  Now run:  bash run_pipeline.sh --live"
    exit 0
fi

# ============================================================
#  LIVE PIPELINE  (real-time RTSP → API)
# ============================================================
if [[ "$MODE" == "live" ]]; then
    info "============================================================"
    info "  Tennis Live Pipeline  (RTSP → TrackNet → API)"
    info "============================================================"
    if [[ -n "$DRY_RUN" ]]; then
        echo "  Mode : DRY RUN (payloads printed, not POSTed)"
    else
        echo "  Mode : PRODUCTION (POSTing to API)"
    fi
    echo "  Root : $ROOT"
    echo ""

    mkdir -p "$OUTPUT"

    "$PYTHON" "$ROOT/live_pipeline.py" $DRY_RUN
    ok "Live pipeline exited."
    exit 0
fi

# ============================================================
#  BATCH PIPELINE  (offline video files → API)
# ============================================================
info "============================================================"
info "  Tennis Batch Pipeline"
info "============================================================"
echo "  Root   : $ROOT"
echo "  Cam66  : $CAM66_VIDEO"
echo "  Cam68  : $CAM68_VIDEO"
echo "  Output : $OUTPUT"
echo "  Resume from stage: $START_STAGE"

mkdir -p "$OUTPUT"

# ── Stage 0: Detection ───────────────────────────────────────────────────
if [[ "$START_STAGE" -le 0 ]]; then
    stage 0 "Ball + player detection (both cameras)..."
    "$PYTHON" "$ROOT/main.py" \
        "$CAM66_VIDEO" "$CAM68_VIDEO" \
        --wasb-weights "$WEIGHTS/wasb_tennis_best.pth.tar" \
        --yolo-weights "$WEIGHTS/yolo26x-pose.pt" \
        --output-dir   "$OUTPUT" \
        --device auto \
        --parallel
    ok "Stage 0 complete."
fi

# ── Stage 1: Cleaning ────────────────────────────────────────────────────
if [[ "$START_STAGE" -le 1 ]]; then
    stage 1 "Cleaning raw detections..."
    "$PYTHON" "$SCRIPTS/run_all_cameras.py"
    ok "Stage 1 complete."
fi

# ── Stage 2: Business Logic & API report ────────────────────────────────
if [[ "$START_STAGE" -le 2 ]]; then
    stage 2 "Processing business indicators and reporting..."
    "$PYTHON" "$SCRIPTS/process_business_indicators.py" \
        --cam66-video    "$CAM66_VIDEO" \
        --cam68-video    "$CAM68_VIDEO" \
        --cam66-csv      "$OUTPUT/cam66_cleaned.csv" \
        --cam68-csv      "$OUTPUT/cam68_cleaned.csv" \
        --homography-json "$UPLOADS/homography_matrices.json" \
        --cam66-calib    "$UPLOADS/cal_cam66.json" \
        --cam68-calib    "$UPLOADS/cal_cam68.json" \
        --cam66-y-net    238 \
        --cam68-y-net    285 \
        --output-dir     "$OUTPUT"
    ok "Stage 2 complete."
fi

echo ""
ok "Batch pipeline complete.  Output is in: $OUTPUT"
exit 0
