"""
serve_viewer.py
===============
Local HTTP server: video frame (with ball overlay) + live 3D trajectory.

Changes vs previous version
----------------------------
  Fix 1 - Overlay simplified:
      Only ONE circle is drawn -- the green detected pixel (u_a, v_a).
      The cyan "Z-corrected" circle is removed because Z comes from mock
      cam68 data and is meaningless until real dual-camera data is ready.
      The red "shadow projection" is also removed -- it duplicates the
      green circle when the homography is correct, adding visual noise.

  Fix 2 - 3D plot shows current segment only, with trail up to current frame:
      Instead of rendering all segments at all times, the plot shows:
        (a) the CURRENT SEGMENT -- the continuous arc the ball is on now
        (b) points UP TO the current frame (trail mode, not future points)
        (c) the CURRENT POINT highlighted as a large white dot
      Other segments are shown faintly in the background as context.

Usage
-----
  python serve_viewer.py
  python serve_viewer.py --video cam66_video.mp4 --camera cam66
  python serve_viewer.py --no-browser   # start server without opening browser
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Resolve paths via settings.py (location-independent)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import UPLOADS_DIR, OUTPUT_DIR, HOMOGRAPHY_JSON as _HOM_JSON

# =============================================================================
# CONFIGURATION  (defaults — all override-able via CLI)
# =============================================================================
VIDEO_A      = str(UPLOADS_DIR / 'cam66_video.mp4')
VIDEO_B      = str(UPLOADS_DIR / 'cam68_video.mp4')
TRAJ_CSV     = str(OUTPUT_DIR  / 'trajectory_3d.csv')
HOMOG_JSON   = str(_HOM_JSON)
PORT         = 8765
JPEG_QUALITY = 80
CACHE_FRAMES = True
# =============================================================================

_video_cap    = None
_total_frames = 0
_fps          = 25.0
_frame_cache  = {}
_traj_by_frame = {}
_traj_json    = '{}'
_H_w2i        = None
_cam_label    = 'cam66'
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Overlay drawing  (Fix 1: one circle only)
# ---------------------------------------------------------------------------

def render_frame(frame_id: int) -> bytes:
    if CACHE_FRAMES and frame_id in _frame_cache:
        return _frame_cache[frame_id]
    with _lock:
        _video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, img = _video_cap.read()
    if not ok or img is None:
        img = np.full((540, 960, 3), 40, dtype=np.uint8)
        cv2.putText(img, f'Frame {frame_id} unavailable',
                    (20, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180,180,180), 2)
    else:
        img = _draw_overlay(img, frame_id)
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    data = buf.tobytes()
    if CACHE_FRAMES:
        _frame_cache[frame_id] = data
    return data


def _draw_overlay(img: np.ndarray, frame_id: int) -> np.ndarray:
    row = _traj_by_frame.get(frame_id)
    h, w = img.shape[:2]

    # Dark top bar for text
    bar = img.copy()
    cv2.rectangle(bar, (0, 0), (w, 60), (15, 15, 15), -1)
    img = cv2.addWeighted(bar, 0.7, img, 0.3, 0)
    cv2.putText(img, f'Frame {frame_id}  |  {_cam_label}',
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

    if row is None or int(row.get('n_cameras', 0)) == 0:
        cv2.putText(img, 'No detection this frame',
                    (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)
        return img

    u_a   = row.get('u_a',   float('nan'))
    v_a   = row.get('v_a',   float('nan'))
    X     = row.get('X',     float('nan'))
    Y     = row.get('Y',     float('nan'))
    Z     = row.get('Z',     float('nan'))
    ca    = row.get('conf_a', 0.0)
    cb    = row.get('conf_b', 0.0)
    ncam  = int(row.get('n_cameras', 0))

    # Info line
    parts = []
    if not np.isnan(X):
        parts.append(f'X={X:.2f}m  Y={Y:.2f}m  Z={Z:.2f}m')
    parts.append(f'conf={ca:.2f}/{cb:.2f}  cam={ncam}')
    cv2.putText(img, '  '.join(parts), (10, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 210, 255), 1, cv2.LINE_AA)

    # Green circle at detected pixel -- the ONLY overlay circle
    if not (np.isnan(u_a) or np.isnan(v_a)):
        pu, pv = int(round(u_a)), int(round(v_a))
        if 0 <= pu < w and 0 <= pv < h:
            # Outer ring
            cv2.circle(img, (pu, pv), 16, (0, 230, 60), 2, cv2.LINE_AA)
            # Inner dot
            cv2.circle(img, (pu, pv),  4, (0, 230, 60), -1, cv2.LINE_AA)
            # Small crosshair
            cv2.line(img, (pu-22, pv), (pu-16, pv), (0, 230, 60), 1, cv2.LINE_AA)
            cv2.line(img, (pu+16, pv), (pu+22, pv), (0, 230, 60), 1, cv2.LINE_AA)
            cv2.line(img, (pu, pv-22), (pu, pv-16), (0, 230, 60), 1, cv2.LINE_AA)
            cv2.line(img, (pu, pv+16), (pu, pv+22), (0, 230, 60), 1, cv2.LINE_AA)

    return img


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Tennis Ball Trajectory Viewer</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
* { box-sizing:border-box; margin:0; padding:0; }
body { background:#0d1117; color:#e6edf3; font-family:'Segoe UI',sans-serif;
       display:flex; flex-direction:column; height:100vh; overflow:hidden; }
#topbar { background:#161b22; border-bottom:1px solid #30363d; padding:7px 14px;
          display:flex; align-items:center; gap:10px; flex-wrap:wrap; flex-shrink:0; }
h1 { font-size:14px; color:#58a6ff; white-space:nowrap; }
.stat { background:#21262d; padding:3px 9px; border-radius:5px;
        font-size:11px; color:#8b949e; white-space:nowrap; }
.stat b { color:#e6edf3; }
#det-info { font-size:11px; color:#8b949e; margin-left:auto; }
#main { display:flex; flex:1; overflow:hidden; min-height:0; }
#left { display:flex; flex-direction:column; width:56%; border-right:1px solid #30363d; min-width:0; }
#right { flex:1; min-width:0; }
#video-wrap { flex:1; background:#000; display:flex; align-items:center;
              justify-content:center; position:relative; overflow:hidden; min-height:0; }
#video-img { max-width:100%; max-height:100%; object-fit:contain; }
#loading { position:absolute; color:#58a6ff; font-size:13px;
           background:rgba(0,0,0,0.6); padding:6px 12px; border-radius:6px; }
#controls { background:#161b22; border-top:1px solid #30363d;
            padding:7px 12px; display:flex; flex-direction:column;
            gap:5px; flex-shrink:0; }
#slider-row { display:flex; align-items:center; gap:8px; }
#frame-slider { flex:1; accent-color:#58a6ff; cursor:pointer; height:4px; }
#fdisplay { font-size:11px; color:#8b949e; min-width:90px; text-align:right; }
#btn-row { display:flex; gap:5px; align-items:center; flex-wrap:wrap; }
button { background:#21262d; color:#e6edf3; border:1px solid #30363d;
         padding:4px 11px; border-radius:5px; cursor:pointer; font-size:11px; }
button:hover { background:#30363d; }
button.active { background:#1f6feb; border-color:#388bfd; }
#jump-input { background:#0d1117; color:#e6edf3; border:1px solid #30363d;
              border-radius:5px; padding:3px 7px; font-size:11px; width:76px; }
#trail-row { display:flex; align-items:center; gap:8px; font-size:11px; color:#8b949e; }
#trail-slider { width:120px; accent-color:#58a6ff; }
#plot { width:100%; height:100%; }
</style>
</head>
<body>
<div id="topbar">
  <h1>&#127934; Trajectory Viewer</h1>
  <div class="stat">Frames: <b id="st-total">--</b></div>
  <div class="stat">Detections: <b id="st-det">--</b></div>
  <div class="stat">Segments: <b id="st-seg">--</b></div>
  <div class="stat">Camera: <b id="st-cam">--</b></div>
  <div id="det-info">--</div>
</div>
<div id="main">
  <div id="left">
    <div id="video-wrap">
      <img id="video-img" src="" alt="frame">
      <div id="loading">Loading...</div>
    </div>
    <div id="controls">
      <div id="slider-row">
        <span style="font-size:11px;color:#8b949e">Frame</span>
        <input type="range" id="frame-slider" min="0" value="0">
        <span id="fdisplay">0 / 0</span>
      </div>
      <div id="btn-row">
        <button id="btn-pd" title="Prev detection (,)">&#9664;&#9664;Det</button>
        <button id="btn-p"  title="Prev frame (Left)">&#9664;</button>
        <button id="btn-play" class="active" title="Play/Pause (Space)">&#9654; Play</button>
        <button id="btn-n"  title="Next frame (Right)">&#9654;</button>
        <button id="btn-nd" title="Next detection (.)">Det&#9654;&#9654;</button>
        <span style="font-size:11px;color:#8b949e;margin-left:4px">Speed:</span>
        <button onclick="setSpeed(0.25)">x0.25</button>
        <button onclick="setSpeed(0.5)">x0.5</button>
        <button onclick="setSpeed(1)" id="sp1" class="active">x1</button>
        <button onclick="setSpeed(2)">x2</button>
        <button onclick="setSpeed(4)">x4</button>
        <input id="jump-input" type="number" placeholder="Jump to..." min="0">
        <button onclick="jumpFrame()">Go</button>
      </div>
      <div id="trail-row">
        Trail length:
        <input type="range" id="trail-slider" min="5" max="200" value="60">
        <span id="trail-val">60 frames</span>
        &nbsp;&nbsp;
        <label><input type="checkbox" id="show-all" onchange="updatePlot()"> Show all segments</label>
      </div>
    </div>
  </div>
  <div id="right"><div id="plot"></div></div>
</div>

<script>
// ── State ──────────────────────────────────────────────────────────────────
let DATA = null, totalFrames = 0, fps = 25, detFrames = [];
let curFrame = 0, playing = false, playTimer = null, playSpeed = 1;
let imgPending = false, nextPending = null;
const imgEl    = document.getElementById('video-img');
const loadEl   = document.getElementById('loading');
const slider   = document.getElementById('frame-slider');
const trailSl  = document.getElementById('trail-slider');
const trailVal = document.getElementById('trail-val');
let trailLen   = 60;
let plotReady  = false;

// ── Boot ───────────────────────────────────────────────────────────────────
fetch('/data').then(r=>r.json()).then(d=>{
  DATA = d;
  totalFrames = d.total_frames;
  fps         = d.fps;
  detFrames   = d.detection_frames;
  slider.max  = totalFrames - 1;
  document.getElementById('st-total').textContent = totalFrames;
  document.getElementById('st-det').textContent   = detFrames.length;
  document.getElementById('st-seg').textContent   = d.segments.length;
  document.getElementById('st-cam').textContent   = d.camera_label;
  buildPlot(d);
  gotoFrame(0);
});

// ── Frame navigation ───────────────────────────────────────────────────────
function gotoFrame(fid, fromPlot) {
  fid = Math.max(0, Math.min(fid, totalFrames-1));
  curFrame = fid;
  slider.value = fid;
  document.getElementById('fdisplay').textContent = `${fid} / ${totalFrames-1}`;

  const row = DATA?.frames[fid];
  const info = document.getElementById('det-info');
  if (row && row.n > 0) {
    const zStr = (row.Z != null) ? `Z=${row.Z.toFixed(2)}m` : 'Z=--';
    info.innerHTML =
      `<b style="color:#3fb950">f${fid}</b>&nbsp; `+
      `X=${row.X!=null?row.X.toFixed(2):'--'}m &nbsp;`+
      `Y=${row.Y!=null?row.Y.toFixed(2):'--'}m &nbsp;`+
      `${zStr} &nbsp;conf=${row.ca.toFixed(2)}/${row.cb.toFixed(2)} &nbsp;cams=${row.n}`;
  } else {
    info.textContent = `Frame ${fid} -- no detection`;
  }

  if (!fromPlot) updatePlot();
  fetchFrame(fid);
}

function fetchFrame(fid) {
  if (imgPending) { nextPending = fid; return; }
  imgPending = true;
  loadEl.style.display = 'block';
  const tmp = new Image();
  tmp.onload = () => {
    imgEl.src = tmp.src;
    loadEl.style.display = 'none';
    imgPending = false;
    if (nextPending !== null && nextPending !== fid) {
      const nf = nextPending; nextPending = null; fetchFrame(nf);
    }
  };
  tmp.onerror = () => { imgPending=false; loadEl.style.display='none'; };
  tmp.src = `/frame?id=${fid}&t=${Date.now()}`;
}

// ── Playback ───────────────────────────────────────────────────────────────
function setPlaying(p) {
  playing = p;
  const btn = document.getElementById('btn-play');
  btn.innerHTML = p ? '&#9646;&#9646; Pause' : '&#9654; Play';
  btn.className = p ? 'active' : '';
  if (playTimer) clearInterval(playTimer);
  if (p) {
    const ms = Math.round(1000 / (fps * playSpeed));
    playTimer = setInterval(()=>{
      if (curFrame >= totalFrames-1) { setPlaying(false); return; }
      gotoFrame(curFrame+1);
    }, ms);
  }
}

function setSpeed(s) {
  playSpeed = s;
  ['sp025','sp05','sp1','sp2','sp4'].forEach(id=>{
    const el=document.getElementById(id); if(el) el.className='';
  });
  if (playing) { setPlaying(false); setPlaying(true); }
}

function prevDet() {
  const p = detFrames.filter(f=>f<curFrame).slice(-1)[0];
  if (p!==undefined) gotoFrame(p);
}
function nextDet() {
  const n = detFrames.find(f=>f>curFrame);
  if (n!==undefined) gotoFrame(n);
}
function jumpFrame() {
  const v = parseInt(document.getElementById('jump-input').value);
  if (!isNaN(v)) gotoFrame(v);
}

document.getElementById('btn-play').onclick = ()=>setPlaying(!playing);
document.getElementById('btn-p').onclick    = ()=>{ setPlaying(false); gotoFrame(curFrame-1); };
document.getElementById('btn-n').onclick    = ()=>{ setPlaying(false); gotoFrame(curFrame+1); };
document.getElementById('btn-pd').onclick   = ()=>{ setPlaying(false); prevDet(); };
document.getElementById('btn-nd').onclick   = ()=>{ setPlaying(false); nextDet(); };
slider.oninput = ()=>{ setPlaying(false); gotoFrame(parseInt(slider.value)); };
document.getElementById('jump-input').onkeydown = e=>{ if(e.key==='Enter') jumpFrame(); };

trailSl.oninput = ()=>{
  trailLen = parseInt(trailSl.value);
  trailVal.textContent = `${trailLen} frames`;
  updatePlot();
};

document.addEventListener('keydown', e=>{
  if (e.target.tagName==='INPUT') return;
  if (e.key===' '||e.key==='k') { e.preventDefault(); setPlaying(!playing); }
  else if (e.key==='ArrowLeft')  { setPlaying(false); gotoFrame(curFrame-1); }
  else if (e.key==='ArrowRight') { setPlaying(false); gotoFrame(curFrame+1); }
  else if (e.key===',')  { setPlaying(false); prevDet(); }
  else if (e.key==='.') { setPlaying(false); nextDet(); }
});

// ── 3D Plot (Fix 2: trail mode + segment highlighting) ────────────────────
const PALETTE = [
  '#4fc3f7','#81c784','#ffb74d','#f06292','#ce93d8',
  '#4dd0e1','#aed581','#ff8a65','#90caf9','#80cbc4',
];
const DIM_COLOR = '#2a3040';   // faded colour for background segments

function line3(xs,ys,zs,col,w=1) {
  return {type:'scatter3d',mode:'lines',x:xs,y:ys,z:zs,
          line:{color:col,width:w},hoverinfo:'skip',showlegend:false};
}

function buildPlot(d) {
  const cw=d.court_width, cl=d.court_length, ny=d.net_y;

  const court = [
    {type:'mesh3d',x:[0,cw,cw,0],y:[0,0,cl,cl],z:[0,0,0,0],
     i:[0,0],j:[1,2],k:[2,3],color:'#1a3a16',opacity:0.25,
     hoverinfo:'skip',showlegend:false},
    line3([0,cw],[0,0],[0,0],'#4a8c4a',2),
    line3([0,cw],[cl,cl],[0,0],'#4a8c4a',2),
    line3([0,0],[0,cl],[0,0],'#4a8c4a',2),
    line3([cw,cw],[0,cl],[0,0],'#4a8c4a',2),
    line3([0,cw],[ny,ny],[1.07,1.07],'#4a8cff',3),
    line3([0,0],[ny,ny],[0,1.07],'#4a8cff',2),
    line3([cw,cw],[ny,ny],[0,1.07],'#4a8cff',2),
  ];
  window._nCourt = court.length;

  // One trace per segment (background, dimmed) -- static, never updated
  const bgTraces = d.segments.map((seg,si)=>({
    type:'scatter3d', mode:'lines',
    x:seg.x, y:seg.y, z:seg.z,
    line:{color:DIM_COLOR, width:1},
    opacity:0.4,
    hoverinfo:'skip', showlegend:false,
  }));
  window._nBg = bgTraces.length;

  // Active trail trace (updated every frame)
  const trailTrace = {
    type:'scatter3d', mode:'lines+markers', name:'trail',
    x:[], y:[], z:[],
    line:{color:'#58a6ff', width:3},
    marker:{color:[], colorscale:'Plasma', size:4, opacity:0.9,
            cmin:0, cmax:5,
            colorbar:{title:'Z (m)',thickness:12,len:0.45,x:1.01,
                      tickfont:{color:'#8b949e'},titlefont:{color:'#8b949e'}},
            showscale:true},
    hoverinfo:'skip', showlegend:false,
  };

  // Current point marker (large white dot)
  const curTrace = {
    type:'scatter3d', mode:'markers', name:'current',
    x:[null], y:[null], z:[null],
    marker:{size:12, color:'#ffffff', symbol:'circle',
            line:{color:'#f85149', width:3}},
    hoverinfo:'skip', showlegend:false,
  };

  window._trailIdx = court.length + bgTraces.length;
  window._curIdx   = court.length + bgTraces.length + 1;

  const allTraces = [...court, ...bgTraces, trailTrace, curTrace];

  const layout = {
    paper_bgcolor:'#0d1117', plot_bgcolor:'#0d1117',
    margin:{l:0,r:55,t:0,b:0},
    scene:{
      bgcolor:'#0d1117',
      xaxis:{title:'X (m)',range:[-1,cw+1],gridcolor:'#1c2128',
             color:'#8b949e',zerolinecolor:'#30363d'},
      yaxis:{title:'Y (m)',range:[-1,cl+1],gridcolor:'#1c2128',
             color:'#8b949e',zerolinecolor:'#30363d'},
      zaxis:{title:'Z (m)',range:[0,7],gridcolor:'#1c2128',
             color:'#8b949e',zerolinecolor:'#30363d'},
      camera:{eye:{x:-1.3,y:-2.0,z:0.9},up:{x:0,y:0,z:1}},
      aspectmode:'manual',
      aspectratio:{x:0.55,y:1.2,z:0.38},
    },
    showlegend:false,
  };

  Plotly.newPlot('plot', allTraces, layout, {responsive:true, displayModeBar:true});
  plotReady = true;

  // Click a point in a background segment -> jump to nearest detection frame
  document.getElementById('plot').on('plotly_click', evt=>{
    const pt = evt.points?.[0];
    if (!pt || !DATA) return;
    // Find nearest detection frame by world position
    const px=pt.x, py=pt.y;
    let best=null, bestD=Infinity;
    for (const f of detFrames) {
      const r=DATA.frames[f];
      if (!r||r.X==null) continue;
      const d=Math.hypot(r.X-px, r.Y-py);
      if (d<bestD) { bestD=d; best=f; }
    }
    if (best!==null) gotoFrame(best, true);
  });
}

function updatePlot() {
  if (!plotReady || !DATA) return;

  const showAll = document.getElementById('show-all').checked;
  const row = DATA.frames[curFrame];
  const curX = row?.X ?? null;
  const curY = row?.Y ?? null;
  const curZ = row?.Z ?? null;

  // Find which segment the current frame belongs to
  let curSeg = -1;
  for (let si=0; si<DATA.segments.length; si++) {
    const frames = DATA.segments[si].frames;
    if (frames[0] <= curFrame && curFrame <= frames[frames.length-1]) {
      curSeg = si; break;
    }
  }

  // Background segments: show all dimmed, or hide non-current
  const bgUpdates = DATA.segments.map((seg,si)=>({
    visible: showAll || si === curSeg ? true : false,
  }));
  const bgIndices = DATA.segments.map((_,si)=>window._nCourt+si);
  if (bgIndices.length>0) {
    Plotly.restyle('plot',
      { visible: bgUpdates.map(u=>u.visible) },
      bgIndices
    );
  }

  // Trail: points in curSeg from (curFrame - trailLen) to curFrame
  let trailX=[], trailY=[], trailZ=[];
  if (curSeg >= 0) {
    const seg = DATA.segments[curSeg];
    const frames = seg.frames;
    const minF = curFrame - trailLen;
    for (let i=0; i<frames.length; i++) {
      if (frames[i] >= minF && frames[i] <= curFrame) {
        trailX.push(seg.x[i]);
        trailY.push(seg.y[i]);
        trailZ.push(seg.z[i]);
      }
    }
  }

  // Update trail and current marker in one restyle call
  Plotly.restyle('plot', {
    x: [trailX, curX!=null?[curX]:[null]],
    y: [trailY, curY!=null?[curY]:[null]],
    z: [trailZ, curZ!=null?[curZ]:[null]],
    'marker.color': [trailZ],
  }, [window._trailIdx, window._curIdx]);
}
</script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        p  = urlparse(self.path)
        qs = parse_qs(p.query)

        if p.path == '/':
            self._send(200, 'text/html; charset=utf-8',
                       HTML_PAGE.encode('utf-8'))

        elif p.path == '/data':
            self._send(200, 'application/json',
                       _traj_json.encode('utf-8'))

        elif p.path == '/frame':
            fid = int(qs.get('id', ['0'])[0])
            fid = max(0, min(fid, _total_frames - 1))
            try:
                self._send(200, 'image/jpeg', render_frame(fid))
            except Exception as e:
                self._send(500, 'text/plain', str(e).encode())

        else:
            self._send(404, 'text/plain', b'Not found')

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _build_traj_json(df: pd.DataFrame, cal: dict,
                     total_frames: int, fps: float,
                     cam_label: str) -> str:
    cd = cal['court_dimensions']

    # Per-frame lookup
    frames_dict = {}
    for _, row in df.iterrows():
        fid = int(row['aligned_frame_id'])
        def safe(v):
            try:
                f = float(v)
                return None if np.isnan(f) else round(f, 3)
            except Exception:
                return None
        frames_dict[fid] = {
            'X':  safe(row.get('X')),
            'Y':  safe(row.get('Y')),
            'Z':  safe(row.get('Z')),
            'n':  int(row.get('n_cameras', 0)),
            'ca': round(float(row.get('conf_a', 0.0) or 0.0), 3),
            'cb': round(float(row.get('conf_b', 0.0) or 0.0), 3),
        }

    # Segments (split by gaps > 10 frames)
    valid = df[df['X'].notna()].copy().sort_values('aligned_frame_id')
    valid['_gap'] = valid['aligned_frame_id'].diff().fillna(1)
    valid['_seg'] = (valid['_gap'] > 10).cumsum()

    segments = []
    for _, grp in valid.groupby('_seg'):
        segments.append({
            'x':      grp['X'].round(3).tolist(),
            'y':      grp['Y'].round(3).tolist(),
            'z':      grp['Z'].round(3).tolist(),
            'frames': grp['aligned_frame_id'].tolist(),
        })

    det_frames = sorted(
        df[df['n_cameras'].fillna(0).gt(0)]['aligned_frame_id'].tolist()
    )

    payload = {
        'total_frames':     total_frames,
        'fps':              fps,
        'camera_label':     cam_label,
        'court_width':      cd['width_m'],
        'court_length':     cd['length_m'],
        'net_y':            cd['net_y_m'],
        'segments':         segments,
        'frames':           frames_dict,
        'detection_frames': det_frames,
    }
    return json.dumps(payload, allow_nan=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _video_cap, _total_frames, _fps, _traj_by_frame
    global _traj_json, _H_w2i, _cam_label

    ap = argparse.ArgumentParser(
        description='Video + 3D trajectory viewer (local server)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--video',      default=VIDEO_A)
    ap.add_argument('--traj',       default=TRAJ_CSV)
    ap.add_argument('--homography', default=HOMOG_JSON)
    ap.add_argument('--camera',     default='cam66', choices=['cam66','cam68'])
    ap.add_argument('--port',       default=PORT, type=int)
    ap.add_argument('--no-browser', action='store_true')
    args = ap.parse_args()

    _cam_label = args.camera

    print(f"Loading homography : {args.homography}")
    cal    = json.loads(Path(args.homography).read_text(encoding='utf-8'))
    _H_w2i = np.array(cal[_cam_label]['H_world_to_image'])

    print(f"Loading trajectory : {args.traj}")
    df = pd.read_csv(args.traj)
    for _, row in df.iterrows():
        _traj_by_frame[int(row['aligned_frame_id'])] = row.to_dict()

    print(f"Opening video      : {args.video}")
    _video_cap = cv2.VideoCapture(args.video)
    if not _video_cap.isOpened():
        sys.exit(f"ERROR: cannot open {args.video}")

    _total_frames = int(_video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    _fps          = float(_video_cap.get(cv2.CAP_PROP_FPS)) or 25.0
    W = int(_video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(_video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  {_total_frames} frames  {W}x{H}  {_fps:.1f} fps")

    _traj_json = _build_traj_json(df, cal, _total_frames, _fps, _cam_label)

    url    = f'http://localhost:{args.port}'
    server = HTTPServer(('localhost', args.port), _Handler)
    t      = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    print(f"\nServer: {url}  |  Press Ctrl+C to stop")
    print("Keys: Space=play  Left/Right=frame  ,/.=prev/next detection  Click 3D=jump")

    if not args.no_browser:
        time.sleep(0.4)
        webbrowser.open(url)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped.")
        server.shutdown()
        _video_cap.release()


if __name__ == '__main__':
    main()