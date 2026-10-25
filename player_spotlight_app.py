import json
import os
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Player Spotlight Tracker",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==============================================================
# الثوابت والإعدادات
# ==============================================================
FPS = 25
PITCH_L, PITCH_W = 105.0, 68.0
STEP = 2

HOME_FILE = "Sample_Game_1_RawTrackingData_Home_Team.csv"
AWAY_FILE = "Sample_Game_1_RawTrackingData_Away_Team.csv"
SPOTLIGHT_FILE = "players_spotlight_profiles.json"

st.markdown(
    """
<style>
    .stat-card { background:#1a202c; border:1px solid #2d3748; border-radius:8px; padding:10px; text-align:center; margin-bottom:8px; }
    .stat-val { font-size:20px; font-weight:bold; color:#ecc94b; }
    .stat-lbl { font-size:11px; color:#a0aec0; margin-top:2px; }
    .play-card { background:#232c3b; border-radius:8px; padding:10px 12px; margin-bottom:8px; border-inline-start:4px solid #3182ce; }
</style>
""",
    unsafe_allow_html=True,
)


# ==============================================================
# 1. قراءة البيانات
# ==============================================================
def load_metrica_tracking(csv_path, team_prefix):
    with open(csv_path, "r", encoding="utf-8") as f:
        _ = f.readline()
        row1 = f.readline().strip().split(",")
        _ = f.readline()

    col_names = ["Period", "Frame", "Time [s]"]
    i = 3
    while i < len(row1):
        if i >= len(row1) - 2:
            col_names.extend(["Ball_x", "Ball_y"])
            break
        num = str(row1[i]).strip()
        if not num:
            num = str(row1[i - 1]).strip()
        col_names.append(f"{team_prefix}_Player{num}_x")
        col_names.append(f"{team_prefix}_Player{num}_y")
        i += 2

    df = pd.read_csv(csv_path, skiprows=3, header=None, names=col_names, low_memory=False)
    for c in df.columns:
        if c not in ["Period", "Frame"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    x_cols = [c for c in df.columns if c.endswith("_x")]
    y_cols = [c for c in df.columns if c.endswith("_y")]
    if x_cols and np.nanmax(df[x_cols].to_numpy(dtype=float)) <= 1.5:
        for c in x_cols:
            df[c] = df[c] * PITCH_L
        for c in y_cols:
            df[c] = (1.0 - df[c]) * PITCH_W
    return df


@st.cache_data(show_spinner="جاري تحميل البيانات...")
def load_all_data():
    with open(SPOTLIGHT_FILE, "r", encoding="utf-8") as f:
        spotlight_db = json.load(f)
    df_home = load_metrica_tracking(HOME_FILE, "Home").set_index("Frame", drop=False)
    df_away = load_metrica_tracking(AWAY_FILE, "Away").set_index("Frame", drop=False)
    return spotlight_db, df_home, df_away


_missing = [f for f in (SPOTLIGHT_FILE, HOME_FILE, AWAY_FILE) if not os.path.exists(f)]
if _missing:
    st.error("ملفات ناقصة في الفولدر: " + " , ".join(_missing))
    st.stop()

spotlight_db, df_home, df_away = load_all_data()


# ==============================================================
# 2. بناء تسلسل الحركة (الكل مرسوم من منظور فريق اللاعب: الهجوم لفوق)
# ==============================================================
def attacks_positive_x(df_team, prefix, period):
    cols = [c for c in df_team.columns if c.startswith(f"{prefix}_Player") and c.endswith("_x")]
    sub = df_team[df_team["Period"] == period].head(50)
    if sub.empty or not cols:
        return period == 1 and prefix == "Home"
    return bool(np.nanmean(sub[cols].to_numpy(dtype=float)) < PITCH_L / 2)


def _r(v):
    return round(float(v), 1)


@st.cache_data(show_spinner=False)
def build_spotlight_sequence(frame_id, s_frame, e_frame, period, team, spotlight_pid):
    prefix = team if team in ("Home", "Away") else "Home"
    positive = attacks_positive_x(df_home if prefix == "Home" else df_away, prefix, period)

    def tr(x, y):
        return (PITCH_W - y, x) if positive else (y, PITCH_L - x)

    h = df_home.loc[s_frame:e_frame].iloc[::STEP]
    if h.empty:
        return None
    frames = h["Frame"].to_numpy()
    a = df_away.reindex(frames)
    s0 = int(frames[0])

    def team_block(df_i, pfx):
        xs = [c for c in df_i.columns if c.startswith(f"{pfx}_Player") and c.endswith("_x")]
        ids = [c[len(f"{pfx}_Player"):-2] for c in xs]
        X = df_i[xs].to_numpy(dtype=float)
        Y = df_i[[c[:-2] + "_y" for c in xs]].to_numpy(dtype=float)
        sx, sy = tr(X, Y)
        out = []
        for i in range(len(df_i)):
            out.append([None if (np.isnan(sx[i, k]) or np.isnan(sy[i, k])) else [_r(sx[i, k]), _r(sy[i, k])]
                        for k in range(len(xs))])
        return ids, out

    h_ids, h_pos = team_block(h, "Home")
    a_ids, a_pos = team_block(a, "Away")

    bx = pd.Series(h["Ball_x"].to_numpy(dtype=float)).ffill().bfill().fillna(PITCH_L / 2).to_numpy()
    by = pd.Series(h["Ball_y"].to_numpy(dtype=float)).ffill().bfill().fillna(PITCH_W / 2).to_numpy()
    sbx, sby = tr(bx, by)
    ball = [[_r(sbx[i]), _r(sby[i])] for i in range(len(h))]

    target_ids = h_ids if prefix == "Home" else a_ids
    target_pos = h_pos if prefix == "Home" else a_pos
    pid_str = str(spotlight_pid)

    found = pid_str in target_ids
    if found:
        idx = target_ids.index(pid_str)
        player_track = []
        last_pt = None
        for i in range(len(h)):
            pos = target_pos[i][idx]
            if pos:
                last_pt = pos
            player_track.append(pos or last_pt or ball[i])
    else:
        player_track = ball  # اللاعب مش في الفريمات دي: الكاميرا تتابع الكرة

    n = len(h)
    fps_data = FPS / STEP
    time_start = float(h["Time [s]"].iloc[0]) if "Time [s]" in h.columns else s0 / FPS

    return {
        "fps": fps_data, "n": n, "T": round((n - 1) / fps_data, 2),
        "homeIds": h_ids, "awayIds": a_ids, "home": h_pos, "away": a_pos,
        "ball": ball, "playerTrack": player_track,
        "attack": prefix, "clock0": round(time_start, 2),
        "spotlightId": pid_str, "spotlightTeam": prefix, "found": found,
        "label": f"Player {pid_str} ({prefix})",
    }


# ==============================================================
# 3. Canvas: كاميرا بتتحرك مع اللاعب (من غير zoom) + خريطة صغيرة
# ==============================================================
VIEWER_HTML = r"""
<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  *{box-sizing:border-box}
  body{margin:0;background:#0c131a;color:#e2e8f0;font-family:"Segoe UI",Tahoma,Arial,sans-serif}
  .wrap{display:flex;flex-direction:column;align-items:center;padding:4px}
  #cw{width:100%;max-width:760px}
  canvas{display:block;border-radius:10px;border:1px solid #2d3748;background:#0f2a1a;width:100%}
  .controls{display:flex;gap:8px;align-items:center;margin-top:8px;flex-wrap:wrap;width:100%}
  button,select{background:#1f2a37;color:#e2e8f0;border:1px solid #3a4a5e;border-radius:8px;padding:6px 12px;font-size:13px;cursor:pointer}
  button:hover{background:#2b3a4d}
  #sl{flex:1 1 140px;accent-color:#ecc94b}
  #tm{font-size:12px;color:#a0aec0;min-width:80px;text-align:center}
</style></head>
<body>
<div class="wrap">
  <div id="cw">
    <canvas id="c"></canvas>
    <div class="controls">
      <button id="bp">▶ Play</button>
      <input id="sl" type="range" min="0" max="1000" value="0">
      <span id="tm">0.0s</span>
      <select id="sp"><option value="0.5">0.5x</option><option value="1" selected>1.0x</option><option value="1.5">1.5x</option></select>
    </div>
  </div>
</div>
<script>
const D = __DATA__;
let playing = true, t = 0, speed = 1, last = null;

// الكادر: عرض الملعب كله + ~62 متر من طوله، والكاميرا بتنزلق مع اللاعب (بدون تكبير)
const VIEW_X0 = -3, VIEW_X1 = 71, VIEW_H = 62, PAD = 4, LEAD = 4;
const cv = document.getElementById('c'), ctx = cv.getContext('2d');
const pitchCv = document.createElement('canvas');
let W = 0, H = 0, S = 1, dpr = window.devicePixelRatio || 1, camY = 52.5;
const PX = x => (x - VIEW_X0) * S;
const PY = y => (camY + VIEW_H / 2 - y) * S;

// مسار الكاميرا: متوسط متحرك لمكان اللاعب (±1.2 ثانية) = حركة ناعمة + مساحة زيادة قدام اللاعب
(function buildCam(){
  const w = Math.max(1, Math.round(1.2 * D.fps)), n = D.n;
  const ys = D.playerTrack.map(p => p ? p[1] : 52.5);
  D.cam = [];
  for (let i = 0; i < n; i++) {
    let s = 0, c = 0;
    for (let j = Math.max(0, i - w); j <= Math.min(n - 1, i + w); j++) { s += ys[j]; c++; }
    let y = s / c + LEAD;
    y = Math.max(-PAD + VIEW_H / 2, Math.min(105 + PAD - VIEW_H / 2, y));
    D.cam.push(y);
  }
})();

function lerpPt(a, b, f){ if (a && b) return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f]; return a || b; }

function drawPitchStatic(){
  pitchCv.width = W; pitchCv.height = Math.round((105 + 2 * PAD) * S);
  const g = pitchCv.getContext('2d');
  const px = x => (x - VIEW_X0) * S, py = y => (105 + PAD - y) * S;
  g.fillStyle = "#0b1f14"; g.fillRect(0, 0, pitchCv.width, pitchCv.height);
  for (let y = 0, k = 0; y < 105; y += 7, k++) {
    const y2 = Math.min(y + 7, 105);
    g.fillStyle = k % 2 ? "#1a5130" : "#164428";
    g.fillRect(px(0), py(y2), px(68) - px(0), py(y) - py(y2));
  }
  g.strokeStyle = "rgba(255,255,255,.85)"; g.lineWidth = Math.max(1.5, S * 0.14); g.lineJoin = "round";
  const line = pts => { g.beginPath(); pts.forEach((p, i) => i ? g.lineTo(px(p[0]), py(p[1])) : g.moveTo(px(p[0]), py(p[1]))); g.stroke(); };
  line([[0,0],[0,105],[68,105],[68,0],[0,0]]);
  line([[0,52.5],[68,52.5]]);
  g.beginPath(); g.arc(px(34), py(52.5), 9.15 * S, 0, Math.PI * 2); g.stroke();
  g.fillStyle = "#fff"; g.beginPath(); g.arc(px(34), py(52.5), Math.max(2, S * 0.25), 0, Math.PI * 2); g.fill();
  for (const [yg, dir] of [[0, 1], [105, -1]]) {
    line([[13.84,yg],[13.84,yg+dir*16.5],[54.16,yg+dir*16.5],[54.16,yg]]);
    line([[24.84,yg],[24.84,yg+dir*5.5],[43.16,yg+dir*5.5],[43.16,yg]]);
    g.fillStyle = "#fff"; g.beginPath(); g.arc(px(34), py(yg + dir * 11), Math.max(2, S * 0.25), 0, Math.PI * 2); g.fill();
    let seg = [], arcs = [];
    for (let a = 0; a <= Math.PI * 2 + 0.05; a += 0.05) {
      const x = 34 + 9.15 * Math.cos(a), y = yg + dir * 11 + 9.15 * Math.sin(a);
      if ((y - yg) * dir > 16.5) seg.push([x, y]); else if (seg.length) { arcs.push(seg); seg = []; }
    }
    if (seg.length) arcs.push(seg);
    arcs.forEach(s => { if (s.length > 1) line(s); });
    const y1 = yg - dir * 2.5;
    g.fillStyle = "rgba(255,255,255,.18)";
    g.fillRect(px(30.34), py(Math.max(yg, y1)), px(37.66) - px(30.34), Math.abs(py(yg) - py(y1)));
    const sv = g.strokeStyle, lw = g.lineWidth;
    g.strokeStyle = "#ffd700"; g.lineWidth = Math.max(2, S * 0.3);
    line([[30.34, yg], [30.34, y1], [37.66, y1], [37.66, yg]]);
    g.strokeStyle = sv; g.lineWidth = lw;
  }
}

function resize(){
  const w = document.getElementById('cw').clientWidth;
  const h = w * VIEW_H / (VIEW_X1 - VIEW_X0);
  cv.style.width = w + 'px'; cv.style.height = h + 'px';
  cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  W = cv.width; H = cv.height; S = W / (VIEW_X1 - VIEW_X0);
  drawPitchStatic(); draw();
}

function drawPlayers(list, ids, color, i, i2, f){
  const r = 1.8 * S;
  ctx.font = "bold " + Math.max(9, r * 0.9) + "px Arial"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  for (let k = 0; k < ids.length; k++) {
    const p = lerpPt(list[i][k], list[i2][k], f); if (!p) continue;
    const cx = PX(p[0]), cy = PY(p[1]);
    if (cy < -30 || cy > H + 30) continue;
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = color; ctx.fill(); ctx.lineWidth = Math.max(1.2, r * 0.12); ctx.strokeStyle = "#fff"; ctx.stroke();
    ctx.fillStyle = "#fff"; ctx.fillText(ids[k], cx, cy + r * 0.05);
  }
}

function drawSpot(i, i2, f){
  const isHome = D.spotlightTeam === "Home";
  const list = isHome ? D.home : D.away, ids = isHome ? D.homeIds : D.awayIds;
  const k = ids.indexOf(String(D.spotlightId)); if (k < 0) return;
  const p = lerpPt(list[i][k], list[i2][k], f); if (!p) return;
  const cx = PX(p[0]), cy = PY(p[1]), r = 2.3 * S, pulse = 1 + 0.18 * Math.sin(t * 7);
  ctx.save();
  ctx.beginPath(); ctx.arc(cx, cy, (r + S * 1.1) * pulse, 0, Math.PI * 2);
  ctx.strokeStyle = "#ffd700"; ctx.lineWidth = Math.max(3, S * 0.4); ctx.shadowColor = "#ffeb3b"; ctx.shadowBlur = 16 * dpr; ctx.stroke();
  ctx.shadowBlur = 0;
  ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.fillStyle = isHome ? "#1f6feb" : "#da3633"; ctx.fill(); ctx.lineWidth = Math.max(2, r * 0.14); ctx.strokeStyle = "#fff"; ctx.stroke();
  ctx.fillStyle = "#fff"; ctx.font = "bold " + (r * 1.0) + "px Arial"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText(String(D.spotlightId), cx, cy + r * 0.05);
  ctx.fillStyle = "#ffd700"; ctx.beginPath();
  ctx.moveTo(cx, cy - (r + S * 1.6)); ctx.lineTo(cx - S * 0.9, cy - (r + S * 2.8)); ctx.lineTo(cx + S * 0.9, cy - (r + S * 2.8)); ctx.closePath(); ctx.fill();
  ctx.restore();
}

function drawMini(i, i2, f, bp){
  const mw = S * 7, mh = mw * 105 / 68, ml = W - mw - S * 1.2, mt = S * 1.2;
  const mx = x => ml + x / 68 * mw, my = y => mt + (105 - y) / 105 * mh;
  ctx.save();
  ctx.fillStyle = "rgba(10,16,24,.75)"; ctx.fillRect(ml - S * 0.4, mt - S * 0.4, mw + S * 0.8, mh + S * 0.8);
  ctx.fillStyle = "#1a5130"; ctx.fillRect(ml, mt, mw, mh);
  ctx.strokeStyle = "rgba(255,255,255,.7)"; ctx.lineWidth = 1; ctx.strokeRect(ml, mt, mw, mh);
  ctx.beginPath(); ctx.moveTo(ml, my(52.5)); ctx.lineTo(ml + mw, my(52.5)); ctx.stroke();
  const top = Math.min(105, camY + VIEW_H / 2), bot = Math.max(0, camY - VIEW_H / 2);
  ctx.fillStyle = "rgba(255,215,0,.18)"; ctx.fillRect(ml, my(top), mw, my(bot) - my(top));
  ctx.strokeStyle = "#ffd700"; ctx.lineWidth = Math.max(1, dpr); ctx.strokeRect(ml, my(top), mw, my(bot) - my(top));
  const dot = (p, c, rad) => { if (!p) return; ctx.beginPath(); ctx.arc(mx(Math.min(68, Math.max(0, p[0]))), my(Math.min(105, Math.max(0, p[1]))), rad, 0, Math.PI * 2); ctx.fillStyle = c; ctx.fill(); };
  D.home[i].forEach((_, k) => dot(lerpPt(D.home[i][k], D.home[i2][k], f), "#1f6feb", S * 0.28));
  D.away[i].forEach((_, k) => dot(lerpPt(D.away[i][k], D.away[i2][k], f), "#da3633", S * 0.28));
  dot(lerpPt(D.playerTrack[i], D.playerTrack[i2], f), "#ffd700", S * 0.55);
  dot(bp, "#fff", S * 0.4);
  ctx.restore();
}

function draw(){
  if (!W) return;
  const fi = Math.min(Math.max(t * D.fps, 0), D.n - 1), i = Math.floor(fi), i2 = Math.min(i + 1, D.n - 1), f = fi - i;
  camY = D.cam[i] + (D.cam[i2] - D.cam[i]) * f;

  const srcY = Math.max(0, Math.min(pitchCv.height - H, (105 + PAD - (camY + VIEW_H / 2)) * S));
  ctx.clearRect(0, 0, W, H);
  ctx.drawImage(pitchCv, 0, srcY, W, H, 0, 0, W, H);

  const bp = lerpPt(D.ball[i], D.ball[i2], f);
  const j0 = Math.max(0, i - 12), tr = [];
  for (let j = j0; j <= i; j++) tr.push(D.ball[j]); tr.push(bp);
  for (let j = 1; j < tr.length; j++) {
    const a = j / tr.length;
    ctx.beginPath(); ctx.moveTo(PX(tr[j-1][0]), PY(tr[j-1][1])); ctx.lineTo(PX(tr[j][0]), PY(tr[j][1]));
    ctx.strokeStyle = "rgba(255,255,255," + (0.5 * a) + ")"; ctx.lineWidth = Math.max(2, S * 0.35 * a); ctx.lineCap = "round"; ctx.stroke();
  }

  drawPlayers(D.away, D.awayIds, "#da3633", i, i2, f);
  drawPlayers(D.home, D.homeIds, "#1f6feb", i, i2, f);
  drawSpot(i, i2, f);

  // الكرة (أو سهم لو برا الكادر)
  const bx = PX(bp[0]), by = PY(bp[1]);
  if (by < 0 || by > H) {
    const up = by < 0, ey = up ? S * 4.5 : H - S * 2.5, ex = Math.min(W - S * 2, Math.max(S * 2, bx));
    ctx.save(); ctx.fillStyle = "rgba(255,255,255,.92)"; ctx.beginPath();
    if (up) { ctx.moveTo(ex, ey - S * 2); ctx.lineTo(ex - S * 1.3, ey); ctx.lineTo(ex + S * 1.3, ey); }
    else { ctx.moveTo(ex, ey + S * 2); ctx.lineTo(ex - S * 1.3, ey); ctx.lineTo(ex + S * 1.3, ey); }
    ctx.closePath(); ctx.fill(); ctx.font = (S * 1.5) + "px Arial"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText("⚽", ex + S * 2.6, up ? ey - S * 0.6 : ey + S * 0.6); ctx.restore();
  } else {
    ctx.save(); ctx.shadowColor = "#ffffff"; ctx.shadowBlur = 12 * dpr;
    ctx.beginPath(); ctx.arc(bx, by, Math.max(3.5, 0.9 * S), 0, Math.PI * 2);
    ctx.fillStyle = "#ffffff"; ctx.fill(); ctx.lineWidth = Math.max(1.2, S * 0.15); ctx.strokeStyle = "#111"; ctx.stroke(); ctx.restore();
  }

  drawMini(i, i2, f, bp);

  const clk = D.clock0 + t, mm = Math.floor(clk / 60), ss = Math.floor(clk % 60);
  const txt = "🎥 " + D.label + "  " + (mm < 10 ? "0" : "") + mm + ":" + (ss < 10 ? "0" : "") + ss;
  ctx.font = "bold " + (S * 1.5) + "px Arial"; const tw = ctx.measureText(txt).width + S * 2.4;
  ctx.fillStyle = "rgba(10,16,24,.85)"; ctx.beginPath(); ctx.roundRect(S, H - S * 4.6, tw, S * 3.4, 6 * dpr); ctx.fill();
  ctx.fillStyle = "#ecc94b"; ctx.textAlign = "left"; ctx.textBaseline = "middle"; ctx.fillText(txt, S * 2.2, H - S * 2.9);
}

function updateUI(){
  document.getElementById('bp').textContent = playing ? "⏸ Pause" : (t >= D.T - 0.05 ? "↺ Replay" : "▶ Play");
  document.getElementById('sl').value = Math.round(t / D.T * 1000);
  document.getElementById('tm').textContent = t.toFixed(1) + "s / " + D.T.toFixed(1) + "s";
}

function loop(ts){
  if (last === null) last = ts;
  const dt = (ts - last) / 1000; last = ts;
  if (playing) { t += dt * speed; if (t >= D.T) { t = D.T; playing = false; } }
  draw(); updateUI(); requestAnimationFrame(loop);
}

document.getElementById('bp').onclick = () => { if (!playing && t >= D.T - 0.05) t = 0; playing = !playing; };
document.getElementById('sl').oninput = e => { t = e.target.value / 1000 * D.T; };
document.getElementById('sp').onchange = e => { speed = parseFloat(e.target.value); };
window.addEventListener('resize', resize);
if (!CanvasRenderingContext2D.prototype.roundRect) {
  CanvasRenderingContext2D.prototype.roundRect = function(x,y,w,h,r){this.moveTo(x+r,y);this.arcTo(x+w,y,x+w,y+h,r);this.arcTo(x+w,y+h,x,y+h,r);this.arcTo(x,y+h,x,y,r);this.arcTo(x,y,x+w,y,r);this.closePath();};
}
resize(); requestAnimationFrame(loop);
</script></body></html>
"""


# ==============================================================
# 4. واجهة Streamlit
# ==============================================================
def stat_card(val, lbl):
    st.markdown(
        f'<div class="stat-card"><div class="stat-val">{val}</div><div class="stat-lbl">{lbl}</div></div>',
        unsafe_allow_html=True,
    )


col_title, col_lang = st.columns([8.5, 1.5])
with col_lang:
    lang = st.selectbox("🌐 Language / اللغة", ["العربية", "English"], index=0)
is_ar = lang == "العربية"

with col_title:
    st.markdown(f"### {'🌟 رادار النجوم: كاميرا تتبع اللاعب المفضل' if is_ar else '🌟 Player Spotlight: Dynamic Follow Cam'}")

# ---------- الفلتر: فريق (Home / Away) ← وتحته لاعبين الفريق ده بس ----------
teams_available = [t for t in ("Home", "Away") if any(v.get("team") == t for v in spotlight_db.values())]
team_names = {
    "Home": "🔵 الفريق الأزرق (Home)" if is_ar else "🔵 Home Team",
    "Away": "🔴 الفريق الأحمر (Away)" if is_ar else "🔴 Away Team",
}

f1, f2 = st.columns([4, 6])
with f1:
    team_choice = st.radio(
        "⚽ اختر الفريق" if is_ar else "⚽ Select Team",
        teams_available,
        format_func=lambda t: team_names[t],
        horizontal=True,
    )
with f2:
    team_keys = [k for k, v in spotlight_db.items() if v.get("team") == team_choice]
    selected_player_key = st.selectbox(
        "⭐ اختر اللاعب" if is_ar else "⭐ Select Player",
        team_keys,
        format_func=lambda k: f"Player {spotlight_db[k]['player_id']}",
        key=f"player_sel_{team_choice}",
    )

# لو اللاعب اتغير: امسح اللقطة القديمة (عشان متتعرضش لقطة لاعب تاني)
if st.session_state.get("spotlight_owner") != selected_player_key:
    st.session_state["spotlight_owner"] = selected_player_key
    st.session_state["current_spotlight_play"] = None

st.markdown("---")

p_info = spotlight_db[selected_player_key]
stats = p_info.get("profile_stats", {})
categories = p_info.get("moments_categories", {})

col_pitch, col_details = st.columns([5.5, 4.5])

with col_details:
    team_color = "#1f6feb" if p_info["team"] == "Home" else "#da3633"
    team_label = "الفريق الأزرق (Home)" if p_info["team"] == "Home" else "الفريق الأحمر (Away)"
    st.markdown(
        f"""
    <div style="border-inline-start: 6px solid {team_color}; background: #1a202c; padding: 14px 18px; border-radius: 10px; margin-bottom: 12px;">
        <h2 style="margin: 0; color: white; font-size: 24px;">Player {p_info['player_id']}
            <span style="font-size: 14px; color: #a0aec0; font-weight: normal;">({team_label})</span>
        </h2>
    </div>
    """,
        unsafe_allow_html=True,
    )

    L = (lambda ar, en: ar if is_ar else en)
    if p_info.get("role") == "Goalkeeper":
        c1, c2 = st.columns(2)
        with c1: stat_card(stats.get("total_saves", 0), L("إجمالي التصديات", "Total Saves"))
        with c2: stat_card(stats.get("total_passes", 0), L("إجمالي التمريرات", "Total Passes"))
    else:
        c1, c2 = st.columns(2)
        with c1: stat_card(stats.get("total_passes", 0), L("إجمالي التمريرات", "Total Passes"))
        with c2: stat_card(stats.get("goals", 0), L("الأهداف", "Goals"))
        c4, c5, c6 = st.columns(3)
        with c4: stat_card(stats.get("shots_total", 0), L("إجمالي التسديدات", "Total Shots"))
        with c5: stat_card(stats.get("shots_on_target", 0), L("على المرمى", "On Target"))
        with c6: stat_card(stats.get("defenders_packed", 0), L("مدافعين تم كسرهم", "Packed Def."))

    st.markdown(f"##### {L('🎬 اختر لقطة لتثبيت الكاميرا عليه:', '🎬 Select Highlight to Track:')}")

    t_actions, t_best, t_lapses = st.tabs([
        L("🎯 أهم لقطات اللاعب", "🎯 Key Actions"),
        L("🌟 التمركز والاستلام", "🌟 Positioning"),
        L("🛡️ المواقف الدفاعية", "🛡️ Defensive Actions"),
    ])

    def render_selector(m_list, prefix):
        if not m_list:
            st.caption(L("لا توجد لقطات مسجلة في هذه الفئة.", "No moments in this category."))
            return
        for idx, m in enumerate(m_list):
            title = m["title_ar"] if is_ar else m["title_en"]
            desc = m["desc_ar"] if is_ar else m["desc_en"]
            st.markdown(
                f"""
            <div class="play-card">
                <span style="background: #2b6cb0; color: white; padding: 2px 7px; border-radius: 4px; font-size: 11px;">⏱ {m['time_str']}</span>
                <strong style="color: #fff; margin-inline-start: 6px; font-size: 13.5px;">{title}</strong>
                <p style="margin: 4px 0 0 0; font-size: 12px; color: #cbd5e0; line-height: 1.4;">{desc}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )
            if st.button(f"▶ {L('تتبع تحركات اللاعب', 'Follow Player')}",
                         key=f"btn_{selected_player_key}_{prefix}_{idx}", use_container_width=True):
                st.session_state["current_spotlight_play"] = m
                st.rerun()

    with t_actions:
        render_selector(categories.get("key_actions", []), "act")
    with t_best:
        render_selector(categories.get("best_positioning", []), "best")
    with t_lapses:
        render_selector(categories.get("defensive_lapses", []), "lapse")

with col_pitch:
    active_m = st.session_state.get("current_spotlight_play")
    if not active_m:
        for cat in ("key_actions", "best_positioning", "defensive_lapses"):
            if categories.get(cat):
                active_m = categories[cat][0]
                break

    if active_m:
        m_title = active_m["title_ar"] if is_ar else active_m["title_en"]
        st.markdown(
            f"#### 🎥 {m_title} — <span style='color:#ecc94b;'>Player {p_info['player_id']}</span>",
            unsafe_allow_html=True,
        )
        seq_data = build_spotlight_sequence(
            frame_id=active_m["key_frame"],
            s_frame=int(active_m["start_frame"]),
            e_frame=int(active_m["end_frame"]),
            period=int(active_m["period"]),
            team=p_info["team"],
            spotlight_pid=p_info["player_id"],
        )
        if seq_data:
            if not seq_data["found"]:
                st.caption("⚠️ اللاعب مش موجود في الفريمات دي، الكاميرا بتتابع الكرة." if is_ar
                           else "⚠️ Player not tracked in these frames; camera follows the ball.")
            json_str = json.dumps(seq_data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
            components.html(VIEWER_HTML.replace("__DATA__", json_str), height=760, scrolling=False)
        else:
            st.warning("تعذر استخراج بيانات التتبع لهذه اللقطة." if is_ar else "No tracking data for this moment.")
    else:
        st.info("لا توجد لقطات لهذا اللاعب." if is_ar else "No moments for this player.")