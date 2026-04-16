@echo off
REM ============================================================
REM  Tennis Analysis Pipeline — Windows Batch Runner
REM  run_pipeline.bat
REM
REM  MODES
REM  -----
REM  run_pipeline.bat                         batch pipeline (default videos)
REM  run_pipeline.bat cam66.mp4 cam68.mp4     batch pipeline (custom videos)
REM  run_pipeline.bat --live                  real-time RTSP pipeline (production)
REM  run_pipeline.bat --live --dry-run        real-time RTSP pipeline (dry run)
REM  run_pipeline.bat --setup                 one-time camera calibration
REM
REM  FIRST-TIME SETUP
REM  ----------------
REM  1. Copy your TrackNet model to:  model_weights\TrackNet_finetuned.pt
REM  2. Copy WASB model to:           model_weights\wasb_tennis_best.pth.tar
REM  3. Copy YOLO model to:           model_weights\yolo26x-pose.pt
REM  4. Edit home_dir in config.yaml  (only if auto-detect fails)
REM  5. Run:  run_pipeline.bat --setup
REM ============================================================

setlocal EnableDelayedExpansion

REM ── Project root = folder containing this .bat file ───────────────────────
set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"

set "SCRIPTS=%ROOT%\scripts"
set "UPLOADS=%ROOT%\uploads"
set "OUTPUT=%ROOT%\output"
set "WEIGHTS=%ROOT%\model_weights"

REM ── Default video filenames ───────────────────────────────────────────────
set "CAM66_VIDEO=%UPLOADS%\cam66_video.mp4"
set "CAM68_VIDEO=%UPLOADS%\cam68_video.mp4"

REM ── Python executable ─────────────────────────────────────────────────────
set "PYTHON=python"
set "KMP_DUPLICATE_LIB_OK=TRUE"

REM ── Colours (Windows 10+ Terminal) ────────────────────────────────────────
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "CYAN=[96m"
set "RESET=[0m"

REM ============================================================
REM  Argument parsing
REM ============================================================
set "MODE=batch"
set "DRY_RUN="

:PARSE_ARGS
if "%1"==""         goto :DISPATCH
if "%1"=="--live"   ( set "MODE=live"     & shift & goto :PARSE_ARGS )
if "%1"=="--setup"  ( set "MODE=setup"    & shift & goto :PARSE_ARGS )
if "%1"=="--dry-run"( set "DRY_RUN=--dry-run" & shift & goto :PARSE_ARGS )
if "%1"=="--help"   goto :USAGE
if "%1"=="-h"       goto :USAGE
REM positional: first two non-flag args are video paths
if not "%1"=="" (
    if not defined _CAM66_SET ( set "CAM66_VIDEO=%1" & set "_CAM66_SET=1" )
    else ( set "CAM68_VIDEO=%1" )
    shift & goto :PARSE_ARGS
)
goto :DISPATCH

:USAGE
echo.
echo Usage:
echo   run_pipeline.bat                        ^(batch: use default videos^)
echo   run_pipeline.bat cam66.mp4 cam68.mp4   ^(batch: custom video paths^)
echo   run_pipeline.bat --live                 ^(live RTSP, POST to API^)
echo   run_pipeline.bat --live --dry-run       ^(live RTSP, print only^)
echo   run_pipeline.bat --setup                ^(one-time calibration^)
goto :END

:DISPATCH
if "%MODE%"=="live"  goto :LIVE
if "%MODE%"=="setup" goto :SETUP
goto :BATCH

REM ============================================================
REM  LIVE PIPELINE  (real-time RTSP → API)
REM ============================================================
:LIVE
echo.
echo %CYAN%============================================================%RESET%
echo %CYAN%  Tennis Live Pipeline  (RTSP → TrackNet → API)%RESET%
echo %CYAN%============================================================%RESET%
if defined DRY_RUN (
    echo   Mode : DRY RUN ^(payloads printed, not POSTed^)
) else (
    echo   Mode : PRODUCTION ^(POSTing to API^)
)
echo   Root : %ROOT%
echo.

if not exist "%OUTPUT%" mkdir "%OUTPUT%"

%PYTHON% "%ROOT%\live_pipeline.py" %DRY_RUN%

if errorlevel 1 (
    echo %RED%[ERROR]%RESET% Live pipeline exited with an error.
    goto :FAIL
)
goto :END

REM ============================================================
REM  BATCH PIPELINE  (offline video files → API)
REM ============================================================
:BATCH
echo.
echo %CYAN%============================================================%RESET%
echo %CYAN%  Tennis Batch Pipeline%RESET%
echo %CYAN%============================================================%RESET%
echo   Root   : %ROOT%
echo   Cam66  : %CAM66_VIDEO%
echo   Cam68  : %CAM68_VIDEO%
echo   Output : %OUTPUT%
echo.

if not exist "%OUTPUT%" mkdir "%OUTPUT%"

REM ── Stage 0: Detection ───────────────────────────────────────────────────
echo %YELLOW%[Stage 0]%RESET% Ball + player detection (both cameras)...
%PYTHON% "%ROOT%\main.py" ^
    "%CAM66_VIDEO%" "%CAM68_VIDEO%" ^
    --wasb-weights "%WEIGHTS%\wasb_tennis_best.pth.tar" ^
    --yolo-weights "%WEIGHTS%\yolo26x-pose.pt" ^
    --output-dir   "%OUTPUT%" ^
    --device auto ^
    --parallel

if errorlevel 1 ( echo %RED%[ERROR]%RESET% Stage 0 failed. & goto :FAIL )
echo %GREEN%[OK]%RESET% Stage 0 complete.
echo.

REM ── Stage 1: Cleaning ────────────────────────────────────────────────────
echo %YELLOW%[Stage 1]%RESET% Cleaning raw detections...
%PYTHON% "%SCRIPTS%\run_all_cameras.py"

if errorlevel 1 ( echo %RED%[ERROR]%RESET% Stage 1 failed. & goto :FAIL )
echo %GREEN%[OK]%RESET% Stage 1 complete.
echo.

REM ── Stage 2: Business Logic & API report ────────────────────────────────
echo %YELLOW%[Stage 2]%RESET% Processing business indicators and reporting...
%PYTHON% "%SCRIPTS%\process_business_indicators.py" ^
    --cam66-video     "%CAM66_VIDEO%" ^
    --cam68-video     "%CAM68_VIDEO%" ^
    --cam66-csv       "%OUTPUT%\cam66_cleaned.csv" ^
    --cam68-csv       "%OUTPUT%\cam68_cleaned.csv" ^
    --homography-json "%UPLOADS%\homography_matrices.json" ^
    --cam66-calib     "%UPLOADS%\cal_cam66.json" ^
    --cam68-calib     "%UPLOADS%\cal_cam68.json" ^
    --cam66-y-net     238 ^
    --cam68-y-net     285 ^
    --output-dir      "%OUTPUT%"

if errorlevel 1 ( echo %RED%[ERROR]%RESET% Stage 2 failed. & goto :FAIL )
echo %GREEN%[OK]%RESET% Stage 2 complete.
echo.

echo %GREEN%Batch pipeline complete!%RESET%
echo Output is in: %OUTPUT%
goto :END

REM ============================================================
REM  ONE-TIME SETUP
REM ============================================================
:SETUP
echo.
echo %CYAN%============================================================%RESET%
echo %CYAN%  One-Time Camera Setup%RESET%
echo %CYAN%============================================================%RESET%
echo This opens interactive windows for each camera.
echo Follow the on-screen instructions to click court lines.
echo.
pause

echo %YELLOW%[Setup 1/4]%RESET% Speed calibration — cam66
%PYTHON% "%SCRIPTS%\calibrate_court.py" "%CAM66_VIDEO%" --out "%UPLOADS%\cal_cam66.json"
if errorlevel 1 goto :FAIL

echo %YELLOW%[Setup 2/4]%RESET% Speed calibration — cam68
%PYTHON% "%SCRIPTS%\calibrate_court.py" "%CAM68_VIDEO%" --out "%UPLOADS%\cal_cam68.json"
if errorlevel 1 goto :FAIL

echo %YELLOW%[Setup 3/4]%RESET% Homography annotation — cam66
%PYTHON% "%SCRIPTS%\annotate_homography.py" "%CAM66_VIDEO%" ^
    --calib "%UPLOADS%\cal_cam66.json"
if errorlevel 1 goto :FAIL

echo %YELLOW%[Setup 4/4]%RESET% Homography annotation — cam68
%PYTHON% "%SCRIPTS%\annotate_homography.py" "%CAM68_VIDEO%" ^
    --calib "%UPLOADS%\cal_cam68.json"
if errorlevel 1 goto :FAIL

echo.
echo %GREEN%Setup complete!%RESET%
echo Now run:  run_pipeline.bat --live
goto :END

REM ============================================================
REM  Helpers
REM ============================================================
:FAIL
echo.
echo %RED%Pipeline stopped due to an error.%RESET%
exit /b 1

:END
echo.
echo %GREEN%Done.%RESET%
endlocal
