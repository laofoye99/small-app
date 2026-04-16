"""
triangulate_3d.py
=================
True 3D ball trajectory reconstruction from two-camera PnP homographies.

Method
------
Given PnP-calibrated homography matrices for two cameras (sharing a common
world coordinate frame), this script triangulates the ball's 3D position
(X, Y, Z) by finding the analytical closest point between two rays.

Each camera projects a ray from its 3D position (Cx, Cy, Cz) through the
observed ground-plane shadow point (X_shadow, Y_shadow, 0). The true ball
position lies somewhere along this ray.

Triangulation formula
---------------------
Ray A: Pa(s) = CamA + s * (GroundA - CamA)
Ray B: Pb(t) = CamB + t * (GroundB - CamB)

We find s, t in [0, 1] that minimize the distance between Pa(s) and Pb(t).
The 3D position is the midpoint (or confidence-weighted mean) of the
resulting points on each ray.

Camera height (Cz) is the only parameter not in the JSON. Default 8.0m
is typical for broadcast-mounted cameras. Calibrate it by finding a frame
where the ball passes over the net (Z=1.07m) and tuning Cz until the
reconstructed Z matches.

World coordinate system (from homography_matrices.json)
--------------------------------------------------------
  X : cross-court, 0 (left sideline) -> 8.23m (right sideline)
  Y : along-court, 0 (near baseline, cam66 side) -> 23.77m (far baseline)
  Z : height above court surface, metres

Usage
-----
    python triangulate_3d.py                          # use defaults below
    python triangulate_3d.py --cz-a 9.5 --cz-b 8.0  # tune camera heights
    python triangulate_3d.py --plot                   # generate HTML viewer
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Resolve paths via settings.py (location-independent)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import OUTPUT_DIR, HOMOGRAPHY_JSON as _HOM_JSON

# =============================================================================
# CONFIGURATION  (defaults — all override-able via CLI)
# =============================================================================

HOMOGRAPHY_JSON = str(_HOM_JSON)

INPUT_A    = str(OUTPUT_DIR / 'cam66_synced.csv')
INPUT_B    = str(OUTPUT_DIR / 'cam68_synced.csv')
OUTPUT_3D  = str(OUTPUT_DIR / 'trajectory_3d.csv')
OUTPUT_HTML = str(OUTPUT_DIR / 'trajectory_3d_viewer.html')

# Camera heights above court surface (metres).
# Default 8.0m is typical for broadcast-mounted cameras.
# Tune by finding a net-crossing frame (Z=1.07m) and adjusting until
# the reconstructed Z equals 1.07m at that frame.
CAM_A_HEIGHT_M  = 8.0
CAM_B_HEIGHT_M  = 8.0

# Maximum plausible ball height (m) -- flag anything above as suspect
Z_MAX_PLAUSIBLE = 12.0

# Segment gap: frames further apart than this start a new trajectory segment
SEG_GAP_FRAMES  = 10

# =============================================================================


# ---------------------------------------------------------------------------
# Core geometry
# ---------------------------------------------------------------------------

def cam_ground_pos(H_world_to_image: np.ndarray) -> tuple[float, float]:
    """
    Estimate camera's ground-plane position (Cx, Cy) in world coords.
    The camera centre satisfies H_w2i @ [Cx, Cy, 1]^T = 0,
    i.e. it is the null vector of H_w2i.
    """
    _, _, Vt = np.linalg.svd(H_world_to_image)
    null = Vt[-1]
    null /= null[2]
    return float(null[0]), float(null[1])


def px_to_world_shadow(H_i2w: np.ndarray,
                        u: float, v: float) -> tuple[float, float]:
    """
    Map pixel (u, v) -> world (X, Y) assuming Z=0 (ground shadow).
    Returns (nan, nan) if the mapping is degenerate.
    """
    ph = H_i2w @ np.array([u, v, 1.0])
    if abs(ph[2]) < 1e-12:
        return float('nan'), float('nan')
    ph /= ph[2]
    return float(ph[0]), float(ph[1])


def triangulate_frame(Xa: float, Ya: float,
                       Xb: float, Yb: float,
                       Cx_a: float, Cy_a: float, Cz_a: float,
                       Cx_b: float, Cy_b: float, Cz_b: float,
                       conf_a: float, conf_b: float
                       ) -> tuple[float, float, float, float, str]:
    """
    Triangulate true 3D position (X, Y, Z) from two ground-shadow points
    using the analytical closest-point-between-rays method.

    Parameters
    ----------
    Xa, Ya  : ground shadow from camera A (metres, Z=0 assumption)
    Xb, Yb  : ground shadow from camera B
    Cx_a, Cy_a, Cz_a : camera A position (metres)
    Cx_b, Cy_b, Cz_b : camera B position
    conf_a, conf_b   : detection confidence weights

    Returns
    -------
    X, Y, Z : 3D position in metres
    delta_D  : 3D distance between the two closest points on the rays (quality)
    Z_source : method used ('ray_intersect', 'parallel')
    """
    cam1 = np.array([Cx_a, Cy_a, Cz_a], dtype=np.float64)
    cam2 = np.array([Cx_b, Cy_b, Cz_b], dtype=np.float64)
    ground1 = np.array([Xa, Ya, 0.0], dtype=np.float64)
    ground2 = np.array([Xb, Yb, 0.0], dtype=np.float64)

    d1 = ground1 - cam1  # ray 1 direction
    d2 = ground2 - cam2  # ray 2 direction

    # Analytical closest-point between two rays
    w = cam1 - cam2
    a = float(np.dot(d1, d1))
    b = float(np.dot(d1, d2))
    c = float(np.dot(d2, d2))
    d_val = float(np.dot(d1, w))
    e = float(np.dot(d2, w))

    denom = a * c - b * b
    if abs(denom) < 1e-10:
        # Rays nearly parallel — fall back to midpoint of ground projections
        w_a = conf_a + 1e-9
        w_b = conf_b + 1e-9
        X = (Xa * w_a + Xb * w_b) / (w_a + w_b)
        Y = (Ya * w_a + Yb * w_b) / (w_a + w_b)
        return float(X), float(Y), 0.0, float(np.hypot(Xa-Xb, Ya-Yb)), 'parallel'

    s = (b * e - c * d_val) / denom
    t = (a * e - b * d_val) / denom

    # Clamp to [0, 1] — ball must be between camera and ground projection
    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)

    # Re-solve for t given fixed s
    p1_fixed = cam1 + s * d1
    t = float(np.dot(p1_fixed - cam2, d2)) / c if c > 1e-10 else t
    t = np.clip(t, 0.0, 1.0)

    # Re-solve for s given fixed t
    p2_fixed = cam2 + t * d2
    s = float(np.dot(p2_fixed - cam1, d1)) / a if a > 1e-10 else s
    s = np.clip(s, 0.0, 1.0)

    p1 = cam1 + s * d1
    p2 = cam2 + t * d2

    # Confidence-weighted final position
    w_a = conf_a + 1e-9
    w_b = conf_b + 1e-9
    X = (p1[0] * w_a + p2[0] * w_b) / (w_a + w_b)
    Y = (p1[1] * w_a + p2[1] * w_b) / (w_a + w_b)
    Z = (p1[2] * w_a + p2[2] * w_b) / (w_a + w_b)

    if Z < 0:
        Z = 0.0

    delta_D = float(np.linalg.norm(p1 - p2))

    return float(X), float(Y), float(Z), delta_D, 'ray_intersect'


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_homography(path: str) -> dict:
    cal = json.loads(Path(path).read_text())
    return cal


def build_trajectory(df_a: pd.DataFrame,
                     df_b: pd.DataFrame,
                     H_a: np.ndarray, H_b: np.ndarray,
                     Cx_a: float, Cy_a: float, Cz_a: float,
                     Cx_b: float, Cy_b: float, Cz_b: float) -> pd.DataFrame:
    """
    Merge both cameras on aligned_frame_id and compute 3D position per frame.
    Also handles single-camera frames (one camera doesn't see the ball).
    """
    cols = ['aligned_frame_id', 'interp_u', 'interp_v', 'ball_conf',
            'is_outlier', 'was_interpolated']
    # Keep only columns that exist
    cols_a = [c for c in cols if c in df_a.columns]
    cols_b = [c for c in cols if c in df_b.columns]

    ma = df_a[cols_a].copy().rename(columns={
        'interp_u':'u_a','interp_v':'v_a','ball_conf':'conf_a',
        'is_outlier':'outlier_a','was_interpolated':'interp_a'})
    mb = df_b[cols_b].copy().rename(columns={
        'interp_u':'u_b','interp_v':'v_b','ball_conf':'conf_b',
        'is_outlier':'outlier_b','was_interpolated':'interp_b'})

    merged = pd.merge(ma, mb, on='aligned_frame_id', how='outer')
    merged = merged.sort_values('aligned_frame_id').reset_index(drop=True)

    records = []
    for _, row in merged.iterrows():
        fid     = int(row['aligned_frame_id'])
        u_a     = row.get('u_a', float('nan'))
        v_a     = row.get('v_a', float('nan'))
        u_b     = row.get('u_b', float('nan'))
        v_b     = row.get('v_b', float('nan'))
        conf_a  = float(row['conf_a']) if not pd.isna(row.get('conf_a')) else 0.0
        conf_b  = float(row['conf_b']) if not pd.isna(row.get('conf_b')) else 0.0

        has_a = not (pd.isna(u_a) or pd.isna(v_a))
        has_b = not (pd.isna(u_b) or pd.isna(v_b))

        Xa = Ya = Xb = Yb = float('nan')
        if has_a:
            Xa, Ya = px_to_world_shadow(H_a, u_a, v_a)
        if has_b:
            Xb, Yb = px_to_world_shadow(H_b, u_b, v_b)

        rec = {
            'aligned_frame_id': fid,
            'u_a': round(u_a, 2) if has_a else float('nan'),
            'v_a': round(v_a, 2) if has_a else float('nan'),
            'u_b': round(u_b, 2) if has_b else float('nan'),
            'v_b': round(v_b, 2) if has_b else float('nan'),
            'Xa_shadow': round(Xa, 4) if has_a else float('nan'),
            'Ya_shadow': round(Ya, 4) if has_a else float('nan'),
            'Xb_shadow': round(Xb, 4) if has_b else float('nan'),
            'Yb_shadow': round(Yb, 4) if has_b else float('nan'),
            'conf_a': round(conf_a, 4),
            'conf_b': round(conf_b, 4),
        }

        if has_a and has_b and not any(np.isnan([Xa, Ya, Xb, Yb])):
            # Full triangulation
            X, Y, Z, dD, z_src = triangulate_frame(
                Xa, Ya, Xb, Yb,
                Cx_a, Cy_a, Cz_a,
                Cx_b, Cy_b, Cz_b,
                conf_a, conf_b)
            rec.update({
                'X': round(X, 4), 'Y': round(Y, 4), 'Z': round(Z, 4),
                'delta_D': round(dD, 4),
                'z_source': z_src,
                'n_cameras': 2,
                'z_suspect': Z > Z_MAX_PLAUSIBLE or Z < 0,
            })
        elif has_a and not np.isnan(Xa):
            # Single camera A -- Z=0 estimate only
            rec.update({
                'X': round(Xa, 4), 'Y': round(Ya, 4), 'Z': 0.0,
                'delta_D': float('nan'),
                'z_source': 'single_cam_A',
                'n_cameras': 1,
                'z_suspect': False,
            })
        elif has_b and not np.isnan(Xb):
            # Single camera B -- Z=0 estimate only
            rec.update({
                'X': round(Xb, 4), 'Y': round(Yb, 4), 'Z': 0.0,
                'delta_D': float('nan'),
                'z_source': 'single_cam_B',
                'n_cameras': 1,
                'z_suspect': False,
            })
        else:
            rec.update({
                'X': float('nan'), 'Y': float('nan'), 'Z': float('nan'),
                'delta_D': float('nan'),
                'z_source': 'no_detection',
                'n_cameras': 0,
                'z_suspect': False,
            })

        records.append(rec)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Interactive HTML viewer
# ---------------------------------------------------------------------------

def build_html_viewer(df3d: pd.DataFrame, cal: dict) -> str:
    """Generate a self-contained Plotly HTML viewer for the 3D trajectory."""
    cd    = cal['court_dimensions']
    cw    = cd['width_m']
    cl    = cd['length_m']
    net_y = cd['net_y_m']
    svc_n = cd['service_near_y_m']
    svc_f = cd['service_far_y_m']

    # Only frames with valid 3D positions
    valid = df3d[df3d['X'].notna() & df3d['n_cameras'] == 2].copy()
    valid = valid.sort_values('aligned_frame_id').reset_index(drop=True)

    # Segment by frame gaps
    valid['gap'] = valid['aligned_frame_id'].diff().fillna(1)
    valid['seg'] = (valid['gap'] > SEG_GAP_FRAMES).cumsum()

    PALETTE = [
        '#4fc3f7','#81c784','#ffb74d','#f06292','#ce93d8','#4dd0e1',
        '#aed581','#ff8a65','#90caf9','#80cbc4','#fff176','#ef9a9a',
        '#80deea','#a5d6a7','#ffe082','#b39ddb','#f48fb1','#bcaaa4',
    ]

    segs_json = []
    for sid, grp in valid.groupby('seg'):
        segs_json.append({
            'x':      grp['X'].round(3).tolist(),
            'y':      grp['Y'].round(3).tolist(),
            'z':      grp['Z'].round(3).tolist(),
            'frame':  grp['aligned_frame_id'].tolist(),
            'conf_a': grp['conf_a'].round(3).tolist(),
            'conf_b': grp['conf_b'].round(3).tolist(),
            'dD':     grp['delta_D'].round(3).tolist(),
            'color':  PALETTE[int(sid) % len(PALETTE)],
            'seg_id': int(sid),
        })

    # Detect bounces: Z near 0 after being elevated
    z_vals = valid['Z'].values
    bounce_rows = []
    for i in range(2, len(z_vals)-1):
        if z_vals[i] < 0.3 and max(z_vals[max(0,i-4):i]) > 0.8:
            bounce_rows.append(valid.iloc[i])
    bounces_json = [{'X': float(r['X']), 'Y': float(r['Y']),
                     'Z': float(r['Z']), 'frame': int(r['aligned_frame_id'])}
                    for r in bounce_rows]

    payload = json.dumps({
        'segments':     segs_json,
        'bounces':      bounces_json,
        'court_width':  cw,
        'court_length': cl,
        'net_y':        net_y,
        'svc_near':     svc_n,
        'svc_far':      svc_f,
        'total_points': int((df3d['n_cameras']==2).sum()),
        'n_segments':   int(valid['seg'].nunique()),
        'n_bounces':    len(bounces_json),
        'cam_a_pos':    {'Cx': round(cam_a_Cx, 2), 'Cy': round(cam_a_Cy, 2)},
        'cam_b_pos':    {'Cx': round(cam_b_Cx, 2), 'Cy': round(cam_b_Cy, 2)},
    })

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>3D Ball Trajectory — Dual Camera Triangulation</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0d1117;font-family:'Segoe UI',sans-serif;color:#e6edf3;overflow:hidden}}
  #header{{padding:10px 20px;background:#161b22;border-bottom:1px solid #30363d;
           display:flex;align-items:center;gap:16px;flex-wrap:wrap}}
  h1{{font-size:16px;color:#58a6ff;white-space:nowrap}}
  .stat{{background:#21262d;padding:5px 12px;border-radius:6px;font-size:12px;
         color:#8b949e;white-space:nowrap}}
  .stat b{{color:#e6edf3}}
  #controls{{padding:8px 20px;background:#161b22;border-bottom:1px solid #30363d;
             display:flex;gap:8px;flex-wrap:wrap;align-items:center}}
  button{{background:#21262d;color:#e6edf3;border:1px solid #30363d;
          padding:5px 14px;border-radius:6px;cursor:pointer;font-size:12px}}
  button:hover{{background:#30363d}}
  button.on{{background:#1f6feb;border-color:#388bfd}}
  #plot{{width:100%;height:calc(100vh - 90px)}}
  #tooltip{{position:fixed;bottom:16px;right:16px;background:#161b22;
            border:1px solid #30363d;border-radius:8px;padding:10px 14px;
            font-size:11px;color:#8b949e;max-width:200px;line-height:1.6}}
</style>
</head>
<body>
<div id="header">
  <h1>3D Trajectory — Dual-Camera Triangulation</h1>
  <div class="stat">Dual-cam pts: <b id="npts">—</b></div>
  <div class="stat">Segments: <b id="nsegs">—</b></div>
  <div class="stat">Bounces: <b id="nbounces">—</b></div>
  <div class="stat">Z max: <b id="zmax">—</b>m</div>
  <div class="stat">delta_D median: <b id="ddmed">—</b>m</div>
</div>
<div id="controls">
  <button class="on" onclick="setView('3d',this)">3D</button>
  <button onclick="setView('top',this)">Top (X-Y)</button>
  <button onclick="setView('side',this)">Side (Y-Z)</button>
  <button onclick="setView('front',this)">Front (X-Z)</button>
  <button class="on" id="btnBounce" onclick="toggle('bounce',this)">Bounces &#10003;</button>
  <button class="on" id="btnCourt" onclick="toggle('court',this)">Court &#10003;</button>
  <button class="on" id="btnCam" onclick="toggle('cam',this)">Cameras &#10003;</button>
  <button onclick="resetCam()">Reset View</button>
</div>
<div id="plot"></div>
<div id="tooltip">
  <b style="color:#58a6ff">Controls</b><br>
  Rotate: left-drag<br>Pan: right-drag<br>Zoom: scroll<br>
  Hover: frame &amp; position<br><br>
  <b style="color:#58a6ff">Legend</b><br>
  Colour = trajectory segment<br>
  Colour intensity = height Z<br>
  ✕ = detected bounce<br>
  ▲ = camera position
</div>

<script>
const D = {payload};

// Populate stats
document.getElementById('npts').textContent     = D.total_points;
document.getElementById('nsegs').textContent    = D.n_segments;
document.getElementById('nbounces').textContent = D.n_bounces;
const allZ  = D.segments.flatMap(s=>s.z);
const allDD = D.segments.flatMap(s=>s.dD).filter(v=>v>0);
document.getElementById('zmax').textContent  = Math.max(...allZ).toFixed(2);
const sorted = [...allDD].sort((a,b)=>a-b);
document.getElementById('ddmed').textContent = sorted[Math.floor(sorted.length/2)]?.toFixed(3) ?? '—';

const CW=D.court_width, CL=D.court_length, NY=D.net_y, SN=D.svc_near, SF=D.svc_far;

function line3(xs,ys,zs,col,name,w=1){{
  return {{type:'scatter3d',mode:'lines',name,x:xs,y:ys,z:zs,
           line:{{color:col,width:w}},hoverinfo:'skip',showlegend:false}};
}}

// Court surface
const courtTraces = [
  // Ground plane
  {{type:'mesh3d',name:'Court surface',
    x:[0,CW,CW,0],y:[0,0,CL,CL],z:[0,0,0,0],i:[0,0],j:[1,2],k:[2,3],
    color:'#1a3a16',opacity:0.35,hoverinfo:'skip',showlegend:false}},
  // Baselines & sidelines
  line3([0,CW],[0,0],[0,0],'#6aff6a','Near baseline',2),
  line3([0,CW],[CL,CL],[0,0],'#6aff6a','Far baseline',2),
  line3([0,0],[0,CL],[0,0],'#6aff6a','Left sideline',2),
  line3([CW,CW],[0,CL],[0,0],'#6aff6a','Right sideline',2),
  // Net (height 1.07m at posts, 0.914m at centre -- use 1.0m approx)
  line3([0,CW],[NY,NY],[1.07,1.07],'#58a6ff','Net',3),
  line3([0,0],[NY,NY],[0,1.07],'#58a6ff','Net post L',2),
  line3([CW,CW],[NY,NY],[0,1.07],'#58a6ff','Net post R',2),
  // Service lines
  line3([0,CW],[SN,SN],[0,0],'#447744','Service near',1),
  line3([0,CW],[SF,SF],[0,0],'#447744','Service far',1),
  // Centre service line
  line3([CW/2,CW/2],[NY,SN],[0,0],'#447744','Centre near',1),
  line3([CW/2,CW/2],[NY,SF],[0,0],'#447744','Centre far',1),
];

// Camera positions
const camTraces = [
  {{type:'scatter3d',mode:'markers+text',name:'cam66',
    x:[D.cam_a_pos.Cx],y:[D.cam_a_pos.Cy],z:[0],
    marker:{{symbol:'diamond',size:6,color:'#4fc3f7'}},
    text:['cam66'],textposition:'top center',
    textfont:{{color:'#4fc3f7',size:11}},hoverinfo:'name'}},
  {{type:'scatter3d',mode:'markers+text',name:'cam68',
    x:[D.cam_b_pos.Cx],y:[D.cam_b_pos.Cy],z:[0],
    marker:{{symbol:'diamond',size:6,color:'#ffb74d'}},
    text:['cam68'],textposition:'top center',
    textfont:{{color:'#ffb74d',size:11}},hoverinfo:'name'}},
];

// Trajectory segments
const trajTraces = D.segments.map(seg=>{{
  const zMin=Math.min(...seg.z), zMax=Math.max(...seg.z)+0.001;
  return {{
    type:'scatter3d', mode:'lines+markers', name:`Seg ${{seg.seg_id}}`,
    x:seg.x, y:seg.y, z:seg.z,
    line:{{color:seg.color,width:3}},
    marker:{{
      color:seg.z, colorscale:'Plasma', size:3, opacity:0.85,
      cmin:0, cmax:Math.max(...allZ),
      colorbar:seg.seg_id===0?{{title:'Z height (m)',thickness:14,len:0.5,x:1.01,
                                tickfont:{{color:'#8b949e'}},
                                titlefont:{{color:'#8b949e'}}}}:undefined,
      showscale: seg.seg_id===0,
    }},
    text:seg.frame.map((f,i)=>
      `<b>Frame ${{f}}</b><br>`+
      `X: ${{seg.x[i].toFixed(3)}}m<br>`+
      `Y: ${{seg.y[i].toFixed(3)}}m<br>`+
      `Z: ${{seg.z[i].toFixed(3)}}m<br>`+
      `delta_D: ${{seg.dD[i]?.toFixed(3)??'—'}}m<br>`+
      `Conf A: ${{seg.conf_a[i].toFixed(3)}}<br>`+
      `Conf B: ${{seg.conf_b[i].toFixed(3)}}`
    ),
    hovertemplate:'%{{text}}<extra>Seg ${{seg.seg_id}}</extra>',
  }};
}});

// Bounce markers
const bounceTrace = {{
  type:'scatter3d', mode:'markers', name:'Bounces',
  x:D.bounces.map(b=>b.X), y:D.bounces.map(b=>b.Y), z:D.bounces.map(b=>b.Z),
  marker:{{symbol:'x',size:9,color:'#f85149',line:{{color:'white',width:2}}}},
  text:D.bounces.map(b=>`<b>Bounce</b><br>Frame ${{b.frame}}<br>X:${{b.X.toFixed(2)}}m Y:${{b.Y.toFixed(2)}}m`),
  hovertemplate:'%{{text}}<extra>Bounce</extra>',
}};

const allTraces = [...courtTraces, ...camTraces, ...trajTraces, bounceTrace];
const nCourt = courtTraces.length;
const nCam   = camTraces.length;
const nTraj  = trajTraces.length;
// Indices: court=0..nCourt-1, cam=nCourt..nCourt+nCam-1,
//          traj=nCourt+nCam..., bounce=last

const layout = {{
  paper_bgcolor:'#0d1117', plot_bgcolor:'#0d1117',
  margin:{{l:0,r:60,t:0,b:0}},
  scene:{{
    bgcolor:'#0d1117',
    xaxis:{{title:'X — Cross-court (m)',range:[-35,43],
            gridcolor:'#1c2128',color:'#8b949e',zerolinecolor:'#30363d'}},
    yaxis:{{title:'Y — Along-court (m)',range:[-30,55],
            gridcolor:'#1c2128',color:'#8b949e',zerolinecolor:'#30363d'}},
    zaxis:{{title:'Z — Height (m)',range:[0,14],
            gridcolor:'#1c2128',color:'#8b949e',zerolinecolor:'#30363d'}},
    camera:{{eye:{{x:-1.2,y:-2.0,z:0.9}},up:{{x:0,y:0,z:1}}}},
    aspectmode:'manual',
    aspectratio:{{x:0.8,y:1.5,z:0.4}},
  }},
  showlegend:false,
}};

Plotly.newPlot('plot', allTraces, layout, {{responsive:true, displayModeBar:true}});

// View presets
function setView(mode, btn) {{
  document.querySelectorAll('#controls button').forEach(b=>{{
    if(['3d','top','side','front'].some(v=>b.textContent.startsWith(v.slice(0,2))||b.textContent===v.toUpperCase()||b.textContent.startsWith(v[0].toUpperCase())))
      b.classList.remove('on');
  }});
  btn.classList.add('on');
  const cameras = {{
    '3d':   {{eye:{{x:-1.2,y:-2.0,z:0.9}},up:{{x:0,y:0,z:1}}}},
    'top':  {{eye:{{x:0,y:0,z:4}},   up:{{x:0,y:1,z:0}}}},
    'side': {{eye:{{x:0,y:-4,z:0.5}},up:{{x:0,y:0,z:1}}}},
    'front':{{eye:{{x:-4,y:0,z:0.5}},up:{{x:0,y:0,z:1}}}},
  }};
  Plotly.relayout('plot', {{'scene.camera': cameras[mode]}});
}}

// Layer toggles
const visible = {{bounce:true, court:true, cam:true}};
function toggle(layer, btn) {{
  visible[layer] = !visible[layer];
  btn.classList.toggle('on', visible[layer]);
  btn.textContent = (layer==='bounce'?'Bounces':layer==='court'?'Court':'Cameras') +
                    (visible[layer]?' &#10003;':'');
  let idxs;
  if (layer==='bounce') idxs = [allTraces.length-1];
  else if (layer==='court') idxs = [...Array(nCourt).keys()];
  else if (layer==='cam') idxs = [...Array(nCam).keys()].map(i=>i+nCourt);
  Plotly.restyle('plot', {{visible: visible[layer] ? true : false}}, idxs);
}}

function resetCam() {{
  Plotly.relayout('plot', {{'scene.camera':{{eye:{{x:-1.2,y:-2.0,z:0.9}},up:{{x:0,y:0,z:1}}}}}});
}}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Global camera positions (computed from H, used in HTML builder)
cam_a_Cx = cam_a_Cy = cam_b_Cx = cam_b_Cy = 0.0


def run(input_a:  str = INPUT_A,
        input_b:  str = INPUT_B,
        out_3d:   str = OUTPUT_3D,
        out_html: str = OUTPUT_HTML,
        cz_a:     float = CAM_A_HEIGHT_M,
        cz_b:     float = CAM_B_HEIGHT_M,
        plot:     bool  = True) -> pd.DataFrame:

    global cam_a_Cx, cam_a_Cy, cam_b_Cx, cam_b_Cy

    SEP = '=' * 64
    print(SEP)
    print('  3D TRIANGULATION  (dual-camera PnP homography)')
    print(SEP)

    # ── Load homographies ────────────────────────────────────────────────────
    cal = load_homography(HOMOGRAPHY_JSON)
    H_a    = np.array(cal['cam66']['H_image_to_world'])
    H_b    = np.array(cal['cam68']['H_image_to_world'])
    Hw2i_a = np.array(cal['cam66']['H_world_to_image'])
    Hw2i_b = np.array(cal['cam68']['H_world_to_image'])

    cam_a_Cx, cam_a_Cy = cam_ground_pos(Hw2i_a)
    cam_b_Cx, cam_b_Cy = cam_ground_pos(Hw2i_b)

    err_a = cal['cam66']['reprojection_error_m']
    err_b = cal['cam68']['reprojection_error_m']
    cd    = cal['court_dimensions']

    print(f"\n  Court: {cd['width_m']}m x {cd['length_m']}m")
    print(f"  cam66: ground pos ({cam_a_Cx:.2f}, {cam_a_Cy:.2f})  "
          f"height={cz_a}m  reproj_err={err_a}m")
    print(f"  cam68: ground pos ({cam_b_Cx:.2f}, {cam_b_Cy:.2f})  "
          f"height={cz_b}m  reproj_err={err_b}m")
    print(f"\n  Y-baseline: {abs(cam_a_Cy - cam_b_Cy):.1f}m  "
          f"X-baseline: {abs(cam_a_Cx - cam_b_Cx):.1f}m")

    # ── Load trajectories ────────────────────────────────────────────────────
    print(f"\n  Loading trajectories...")
    df_a = pd.read_csv(input_a)
    df_b = pd.read_csv(input_b)
    print(f"  cam66: {len(df_a)} rows  |  cam68: {len(df_b)} rows")

    # ── Triangulate ──────────────────────────────────────────────────────────
    print(f"\n  Triangulating...")
    df3d = build_trajectory(df_a, df_b,
                             H_a, H_b,
                             cam_a_Cx, cam_a_Cy, cz_a,
                             cam_b_Cx, cam_b_Cy, cz_b)

    n2    = int((df3d['n_cameras'] == 2).sum())
    n1    = int((df3d['n_cameras'] == 1).sum())
    n0    = int((df3d['n_cameras'] == 0).sum())
    nsus  = int(df3d.get('z_suspect', pd.Series(False)).sum())
    valid = df3d[df3d['n_cameras'] == 2]
    zmax  = float(valid['Z'].max()) if len(valid) else 0.0
    ddmed = float(valid['delta_D'].median()) if len(valid) else 0.0

    print(f"  Dual-camera frames : {n2}")
    print(f"  Single-camera      : {n1}")
    print(f"  No detection       : {n0}")
    print(f"  Suspect Z > {Z_MAX_PLAUSIBLE}m   : {nsus}")
    print(f"  Max Z (height)     : {zmax:.3f} m")
    print(f"  Median delta_D     : {ddmed:.4f} m  (cross-cam position error)")

    # ── Save CSV ─────────────────────────────────────────────────────────────
    df3d.to_csv(out_3d, index=False)
    print(f"\n  [CSV]  -> {out_3d}")

    # ── Build HTML viewer ────────────────────────────────────────────────────
    if plot:
        html = build_html_viewer(df3d, cal)
        Path(out_html).write_text(html, encoding='utf-8')
        size_kb = Path(out_html).stat().st_size // 1024
        print(f"  [HTML] -> {out_html}  ({size_kb} KB)")

    print()
    print(SEP)
    print('  SUMMARY')
    print(SEP)
    print(f"  Dual-camera 3D points : {n2}")
    print(f"  Max ball height Z     : {zmax:.3f} m")
    print(f"  Median position error : {ddmed:.4f} m")
    print()
    print('  Output CSV columns:')
    print('    X, Y, Z          — 3D world position (metres)')
    print('    Xa/Ya_shadow     — ground shadow from cam66 (Z=0 projection)')
    print('    Xb/Yb_shadow     — ground shadow from cam68')
    print('    delta_D          — 3D distance between rays (m, quality metric)')
    print('    z_source         — method used (ray_intersect or parallel)')
    print('    n_cameras        — 2=triangulated, 1=single-cam, 0=no detection')
    print('    z_suspect        — True if Z > plausible threshold')
    print(SEP)
    return df3d


def _build_parser():
    p = argparse.ArgumentParser(
        description='3D triangulation from dual PnP homography matrices',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--input-a',  default=INPUT_A)
    p.add_argument('--input-b',  default=INPUT_B)
    p.add_argument('--out-3d',   default=OUTPUT_3D)
    p.add_argument('--out-html', default=OUTPUT_HTML)
    p.add_argument('--cz-a',     type=float, default=CAM_A_HEIGHT_M,
                   help='cam66 height above court (m)')
    p.add_argument('--cz-b',     type=float, default=CAM_B_HEIGHT_M,
                   help='cam68 height above court (m)')
    p.add_argument('--no-plot',  action='store_true',
                   help='Skip HTML viewer generation')
    return p


if __name__ == '__main__':
    args = _build_parser().parse_args()
    run(input_a  = args.input_a,
        input_b  = args.input_b,
        out_3d   = args.out_3d,
        out_html = args.out_html,
        cz_a     = args.cz_a,
        cz_b     = args.cz_b,
        plot     = not args.no_plot)