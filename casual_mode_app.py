import os
import json
import html as html_lib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Casual Fan Drone Cam", layout="wide", initial_sidebar_state="collapsed")

# ==============================================================
# إعدادات
# ==============================================================
FPS = 25
PITCH_L, PITCH_W = 105.0, 68.0
STEP = 2                      # بناخد فريم من كل 2 (12.5 فريم/ثانية) والمتصفح بيعمل interpolation للنعومة

HOME_FILE = "Sample_Game_1_RawTrackingData_Home_Team.csv"
AWAY_FILE = "Sample_Game_1_RawTrackingData_Away_Team.csv"
MOMENTS_FILE = "casual_fan_moments.json"
ELITE_FILE = "casual_elite_11_moments.csv"
EVENTS_CANDIDATES = ["Sample_Game_1_RawEventsData.csv", "events.csv"]
PASSES_FILE = "calculated_packing_passes.csv"

st.markdown("""
<style>
    .fan-card {
        border: 1px solid #3182ce; border-radius: 12px; padding: 12px;
        background: linear-gradient(160deg, #1a202c, #12202f);
        text-align: center; min-height: 150px; margin-bottom: 6px;
        display: flex; flex-direction: column; justify-content: center;
    }
    .fan-chip { background:#2b6cb0; color:#fff; padding:2px 8px; border-radius:6px; font-size:11px; align-self:center; }
    .fan-badge { margin:8px 0 2px 0; font-size:12px; color:#63b3ed; font-weight:700; }
    .fan-title { margin:0; font-size:13px; color:#edf2f7; font-weight:500; }
</style>
""", unsafe_allow_html=True)


# ==============================================================
# 1. قراءة وتجهيز البيانات
# ==============================================================
def clean_tracking_columns(df, prefix):
    """إعادة تسمية أعمدة Metrica الخام بالترتيب: Prefix_PlayerN_x / _y و Ball_x / Ball_y"""
    cols = list(df.columns)
    new_cols = ["Period", "Frame", "Time [s]"]
    i = 3
    while i < len(cols) - 2:
        name = str(cols[i]).strip()
        new_cols.append(f"{prefix}_{name}_x")
        new_cols.append(f"{prefix}_{name}_y")
        i += 2
    new_cols += ["Ball_x", "Ball_y"]
    df.columns = new_cols
    return df


def read_tracking(path, prefix):
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    if "Period" not in df.columns:
        df = pd.read_csv(path, skiprows=2)
        df.columns = [str(c).strip() for c in df.columns]
    if not any(c.startswith(f"{prefix}_Player") for c in df.columns):
        df = clean_tracking_columns(df, prefix)
    return df


def tracking_to_meters(df):
    """0..1 -> أمتار (x: 0..105 ، y: 0..68 ومقلوب زي Step 1). لو متحول قبل كده بيسيبه."""
    x_cols = [c for c in df.columns if c.endswith("_x")]
    y_cols = [c for c in df.columns if c.endswith("_y")]
    if not x_cols:
        return df
    if np.nanmax(df[x_cols].to_numpy(dtype=float)) <= 1.5:
        for c in x_cols:
            df[c] = df[c] * PITCH_L
        for c in y_cols:
            df[c] = (1.0 - df[c]) * PITCH_W
    return df


def xy_to_meters(df):
    """نفس الفكرة لملفات الأحداث/التمريرات (Start X, Start Y, End X, End Y)."""
    df = df.copy()
    xs = [c for c in ("Start X", "End X") if c in df.columns]
    ys = [c for c in ("Start Y", "End Y") if c in df.columns]
    if xs and np.nanmax(df[xs].to_numpy(dtype=float)) <= 1.5:
        for c in xs:
            df[c] = df[c] * PITCH_L
        for c in ys:
            df[c] = (1.0 - df[c]) * PITCH_W
    return df


@st.cache_data(show_spinner="جاري تحميل بيانات المباراة...")
def load_all_data():
    with open(MOMENTS_FILE, "r", encoding="utf-8") as f:
        casual_json = json.load(f)

    df_elite = pd.read_csv(ELITE_FILE)
    df_elite.columns = df_elite.columns.str.strip().str.lstrip("-")

    df_home = tracking_to_meters(read_tracking(HOME_FILE, "Home")).set_index("Frame", drop=False)
    df_away = tracking_to_meters(read_tracking(AWAY_FILE, "Away")).set_index("Frame", drop=False)

    df_events = None
    for fn in EVENTS_CANDIDATES:
        if os.path.exists(fn):
            df_events = xy_to_meters(pd.read_csv(fn))
            break

    df_passes = xy_to_meters(pd.read_csv(PASSES_FILE)) if os.path.exists(PASSES_FILE) else None
    return casual_json, df_elite, df_home, df_away, df_events, df_passes


required = [HOME_FILE, AWAY_FILE, MOMENTS_FILE, ELITE_FILE]
missing = [f for f in required if not os.path.exists(f)]
if missing:
    st.error("ملفات ناقصة في الفولدر: " + " , ".join(missing))
    st.stop()

casual_data, df_elite, df_home, df_away, df_events, df_passes = load_all_data()


# ==============================================================
# 2. تحويل التتبع الحقيقي لكاميرا الدرون الرأسية
# ==============================================================
def attacks_positive_x(df_team, prefix, period):
    """يحدد اتجاه الهجوم من مراكز اللاعبين في أول لحظات الشوط (الفريق بيبدأ في نصفه)."""
    cols = [c for c in df_team.columns if c.startswith(f"{prefix}_Player") and c.endswith("_x")]
    sub = df_team[df_team["Period"] == period].head(50)
    if sub.empty or not cols:
        return period == 1 and prefix == "Home"
    mean_x = np.nanmean(sub[cols].to_numpy(dtype=float))
    return bool(mean_x < PITCH_L / 2)


def classify_shot(sub):
    sub = str(sub).upper()
    if "GOAL" in sub:
        return "GOAL"
    if "SAVED" in sub:
        return "SAVED"
    if "BLOCKED" in sub:
        return "BLOCKED"
    return "OFF_TARGET"


def _r(v):
    return round(float(v), 1)


@st.cache_data(show_spinner=False)
def build_sequence(frame_id, s_frame, e_frame, period, team, outcome_hint, packed):
    prefix = team if team in ("Home", "Away") else "Home"
    positive = attacks_positive_x(df_home if prefix == "Home" else df_away, prefix, period)

    def tr(x, y):
        # الهجوم دايماً لفوق (المرمى عند y=105). دوران صحيح (مش مرآة)
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

    n = len(h)
    fps_data = FPS / STEP
    T = (n - 1) / fps_data

    # ---- تمريرات الفريق المهاجم + التسديدة من ملف الأحداث ----
    passes, shot = [], None
    if df_events is not None and "Start Frame" in df_events.columns:
        ev = df_events[(df_events["Period"] == period) & (df_events["Team"].astype(str) == team) &
                       (df_events["Start Frame"] >= s0) & (df_events["Start Frame"] <= int(frames[-1]))]
        for _, r in ev.iterrows():
            typ = str(r.get("Type", "")).upper()
            if "PASS" in typ and pd.notna(r.get("End X")) and pd.notna(r.get("Start X")):
                x0, y0 = tr(float(r["Start X"]), float(r["Start Y"]))
                x1, y1 = tr(float(r["End X"]), float(r["End Y"]))
                t0 = (int(r["Start Frame"]) - s0) / FPS
                t1 = (int(r["End Frame"]) - s0) / FPS if pd.notna(r.get("End Frame")) else t0 + 0.8
                passes.append({"p": [_r(x0), _r(y0), _r(x1), _r(y1)], "t0": round(t0, 2), "t1": round(max(t1, t0 + 0.2), 2),
                               "key": int(r["Start Frame"]) == int(frame_id)})
            elif "SHOT" in typ and pd.notna(r.get("Start Frame")):
                sf = int(r["Start Frame"])
                if shot is None or (sf >= frame_id and shot["sf"] < frame_id):
                    ef = int(r["End Frame"]) if pd.notna(r.get("End Frame")) else sf + 20
                    shot = {"sf": sf, "ef": ef, "kind": classify_shot(r.get("Subtype", ""))}

    # لو التمريرة المفتاحية مش في ملف الأحداث، نجيبها من ملف الـ packing
    if not any(p["key"] for p in passes) and df_passes is not None and "Start Frame" in df_passes.columns:
        pm = df_passes[df_passes["Start Frame"] == frame_id]
        if not pm.empty:
            r = pm.iloc[0]
            x0, y0 = tr(float(r["Start X"]), float(r["Start Y"]))
            x1, y1 = tr(float(r["End X"]), float(r["End Y"]))
            t0 = (int(frame_id) - s0) / FPS
            t1 = (int(r["End Frame"]) - s0) / FPS if "End Frame" in pm.columns and pd.notna(r.get("End Frame")) else t0 + 0.8
            passes.append({"p": [_r(x0), _r(y0), _r(x1), _r(y1)], "t0": round(t0, 2), "t1": round(max(t1, t0 + 0.2), 2), "key": True})

    # ---- تأثير النتيجة (هدف / تصدي / بلوك / برا المرمى) ----
    effect = None
    if shot is not None:
        te = (shot["ef"] - s0) / FPS
        effect = {"t": round(min(max(te, 0.5), max(T - 0.4, 0.5)), 2), "kind": shot["kind"]}
    elif outcome_hint in ("GOAL", "SAVED", "BLOCKED", "OFF_TARGET"):
        effect = {"t": round(min(max((frame_id - s0) / FPS + 1.5, 0.5), max(T - 0.4, 0.5)), 2), "kind": outcome_hint}

    time_start = float(h["Time [s]"].iloc[0]) if "Time [s]" in h.columns else s0 / FPS

    return {
        "fps": fps_data, "n": n, "T": round(T, 2),
        "homeIds": h_ids, "awayIds": a_ids, "home": h_pos, "away": a_pos, "ball": ball,
        "passes": passes, "effect": effect, "packed": int(packed or 0),
        "attack": prefix, "clock0": round(time_start, 2),
        "keyT": round((frame_id - s0) / FPS, 2),
    }


def prep_steps(item, T):
    tl = item.get("commentator_timeline", []) or []
    offs = []
    for s in tl:
        try:
            offs.append(float(s.get("time_offset_s", 0) or 0))
        except (TypeError, ValueError):
            offs.append(0.0)
    mx = max(offs) if offs else 0.0
    if tl and mx <= 0:
        offs = list(np.linspace(0.3, T * 0.8, len(tl)))
    elif mx > T * 0.92:
        offs = [o * (T * 0.9 / mx) for o in offs]
    return [{"t": round(o, 2), "stage": s.get("stage", ""),
             "ar": s.get("text_ar", ""), "en": s.get("text_en", "")} for s, o in zip(tl, offs)]


# ==============================================================
# 3. الـ Viewer (HTML + Canvas): ملعب رأسي + معلق متزامن + تبديل لغة
# ==============================================================
VIEWER_HTML = r"""
<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  *{box-sizing:border-box}
  body{margin:0;background:#0c131a;color:#e2e8f0;font-family:"Segoe UI",Tahoma,Arial,sans-serif}
  .wrap{display:flex;gap:12px;flex-wrap:wrap;padding:6px}
  #cw{flex:1 1 430px;min-width:300px}
  canvas{display:block;border-radius:10px;border:1px solid #2d3748;background:#0f2a1a}
  .controls{display:flex;gap:8px;align-items:center;margin-top:8px;flex-wrap:wrap}
  button,select{background:#1f2a37;color:#e2e8f0;border:1px solid #3a4a5e;border-radius:8px;padding:7px 12px;font-size:13px;cursor:pointer}
  button:hover{background:#2b3a4d}
  #sl{flex:1 1 140px;accent-color:#ecc94b}
  #tm{font-size:12px;color:#a0aec0;min-width:70px;text-align:center}
  .panel{flex:1 1 260px;min-width:240px;background:#141d28;border:1px solid #2d3748;border-radius:10px;padding:10px;display:flex;flex-direction:column}
  .panel h4{margin:0 0 4px 0;font-size:15px;color:#ecc94b}
  .chain{font-size:11.5px;color:#a0aec0;margin-bottom:8px;line-height:1.5}
  #steps{overflow-y:auto;flex:1;max-height:520px}
  .step{background:#2d3748;border-radius:8px;padding:9px 10px;margin-bottom:7px;opacity:.55;border-inline-start:4px solid var(--c);transition:opacity .3s}
  .step.cur{opacity:1;background:#374357;animation:pop .35s ease}
  .step .ts{font-size:10.5px;color:#a0aec0}
  .step p{margin:3px 0 0 0;font-size:13.5px;line-height:1.5;color:#fff}
  @keyframes pop{from{transform:translateY(8px);opacity:0}to{transform:none;opacity:1}}
  .empty{color:#718096;font-size:12.5px;padding:8px}
</style></head>
<body>
<div class="wrap">
  <div id="cw">
    <canvas id="c"></canvas>
    <div class="controls">
      <button id="bp"></button>
      <input id="sl" type="range" min="0" max="1000" value="0">
      <span id="tm">0.0s</span>
      <select id="sp"><option value="0.5">0.5x</option><option value="1" selected>1.0x</option><option value="2">2.0x</option></select>
      <button id="bl">AR / EN</button>
    </div>
  </div>
  <div class="panel">
    <h4 id="ct"></h4>
    <div class="chain" id="chain"></div>
    <div id="steps"></div>
  </div>
</div>
<script>
const D = __DATA__;
const TXT = {
  ar:{play:"▶ تشغيل",pause:"⏸ إيقاف",replay:"↺ إعادة",comm:"🎙️ المعلق اللحظي",chain:"تسلسل اللعبة",
      lp:"تمريرة كاسرة للخطوط",def:"مدافعين",wait:"…المباراة بدأت، التعليق يظهر مع حركة الكرة",
      GOAL:"هدف!",SAVED:"تصدٍّ رائع!",BLOCKED:"تم حجب التسديدة",OFF_TARGET:"تسديدة خارج المرمى",dir:"rtl"},
  en:{play:"▶ Play",pause:"⏸ Pause",replay:"↺ Replay",comm:"🎙️ Live Commentary",chain:"Action chain",
      lp:"Line-breaking pass",def:"defenders",wait:"Kick-off… commentary follows the ball",
      GOAL:"GOAL!",SAVED:"GREAT SAVE!",BLOCKED:"SHOT BLOCKED",OFF_TARGET:"OFF TARGET",dir:"ltr"}
};
const EMO = {GOAL:"⚽",SAVED:"🧤",BLOCKED:"🛡️",OFF_TARGET:"🎯"};
const KCOL = {GOAL:"#38a169",SAVED:"#2b6cb0",BLOCKED:"#b7791f",OFF_TARGET:"#718096"};
const STAGE = {build_up:"#38b2ac",mid_action:"#ecc94b"};
let lang = D.lang, playing = true, t = 0, speed = 1, last = null;

const VIEW = {x0:-3, x1:71, y0:40, y1:110.5};
const cv = document.getElementById('c'), ctx = cv.getContext('2d');
const pitchCv = document.createElement('canvas');
let W=0,H=0,S=1,dpr=window.devicePixelRatio||1;
const X = x => (x-VIEW.x0)*S, Y = y => H-(y-VIEW.y0)*S;

function drawPitch(){
  pitchCv.width=W; pitchCv.height=H; const g=pitchCv.getContext('2d');
  g.fillStyle="#0e2418"; g.fillRect(0,0,W,H);
  for(let y=VIEW.y0, k=0; y<105; y+=6.5, k++){
    const y2=Math.min(y+6.5,105);
    g.fillStyle = k%2? "#1d5a34":"#1a5130";
    g.fillRect(X(0),Y(y2),X(68)-X(0),Y(y)-Y(y2));
  }
  g.strokeStyle="rgba(255,255,255,.9)"; g.lineWidth=Math.max(1.5,S*0.18); g.lineJoin="round";
  const line=(pts)=>{g.beginPath();pts.forEach((p,i)=>i?g.lineTo(X(p[0]),Y(p[1])):g.moveTo(X(p[0]),Y(p[1])));g.stroke();};
  line([[0,VIEW.y0],[0,105],[68,105],[68,VIEW.y0]]);
  line([[0,52.5],[68,52.5]]);
  line([[13.84,105],[13.84,88.5],[54.16,88.5],[54.16,105]]);
  line([[24.84,105],[24.84,99.5],[43.16,99.5],[43.16,105]]);
  // دايرة المنتصف
  g.beginPath(); g.arc(X(34),Y(52.5),9.15*S,0,Math.PI*2); g.stroke();
  // نقطة الجزاء + قوس الجزاء
  g.fillStyle="#fff"; g.beginPath(); g.arc(X(34),Y(94),Math.max(2,S*0.25),0,Math.PI*2); g.fill();
  let seg=[]; const arcs=[];
  for(let a=0;a<=Math.PI*2+0.05;a+=0.05){
    const px=34+9.15*Math.cos(a), py=94+9.15*Math.sin(a);
    if(py<88.5){seg.push([px,py]);} else if(seg.length){arcs.push(seg);seg=[];}
  }
  if(seg.length) arcs.push(seg);
  arcs.forEach(s=>{ if(s.length>1) line(s); });
  // المرمى
  g.fillStyle="rgba(255,255,255,.18)"; g.fillRect(X(30.34),Y(107.5),X(37.66)-X(30.34),Y(105)-Y(107.5));
  g.strokeStyle="#ffd700"; g.lineWidth=Math.max(2,S*0.3);
  line([[30.34,105],[30.34,107.5],[37.66,107.5],[37.66,105]]);
}

function resize(){
  const w=document.getElementById('cw').clientWidth;
  const h=w*(VIEW.y1-VIEW.y0)/(VIEW.x1-VIEW.x0);
  cv.style.width=w+'px'; cv.style.height=h+'px';
  cv.width=Math.round(w*dpr); cv.height=Math.round(h*dpr);
  W=cv.width; H=cv.height; S=W/(VIEW.x1-VIEW.x0);
  drawPitch(); draw();
}

function lerpPt(a,b,f){ if(a&&b) return [a[0]+(b[0]-a[0])*f, a[1]+(b[1]-a[1])*f]; return a||b; }

function drawPlayers(list, ids, color, i, i2, f, ringPt){
  const r=1.9*S;
  ctx.font="bold "+Math.max(9,r*0.95)+"px Arial"; ctx.textAlign="center"; ctx.textBaseline="middle";
  for(let k=0;k<ids.length;k++){
    const p=lerpPt(list[i][k],list[i2][k],f); if(!p) continue;
    if(p[1]<VIEW.y0-3||p[0]<VIEW.x0-3||p[0]>VIEW.x1+3) continue;
    const cx=X(p[0]), cy=Y(p[1]);
    ctx.beginPath(); ctx.arc(cx,cy,r,0,Math.PI*2);
    ctx.fillStyle=color; ctx.fill(); ctx.lineWidth=Math.max(1.5,r*0.14); ctx.strokeStyle="#fff"; ctx.stroke();
    ctx.fillStyle="#fff"; ctx.fillText(ids[k],cx,cy+r*0.05);
    if(ringPt && ringPt.k===k){ ctx.beginPath(); ctx.arc(cx,cy,r+S*0.7,0,Math.PI*2); ctx.strokeStyle="#ffd54a"; ctx.lineWidth=Math.max(2,S*0.28); ctx.stroke(); }
  }
}

function draw(){
  if(!W) return;
  const L=TXT[lang];
  ctx.clearRect(0,0,W,H); ctx.drawImage(pitchCv,0,0);
  const fi=Math.min(Math.max(t*D.fps,0),D.n-1), i=Math.floor(fi), i2=Math.min(i+1,D.n-1), f=fi-i;

  // ---- ممر الـ 8 متر + خطوط التمرير ----
  for(const p of D.passes){
    if(t<p.t0||t>p.t1+(p.key?3:1.8)) continue;
    const prog=Math.min(1,(t-p.t0)/Math.max(0.15,p.t1-p.t0));
    const fade=t>p.t1?Math.max(0,1-(t-p.t1)/(p.key?3:1.8)):1;
    const [x0,y0,x1,y1]=p.p; const xe=x0+(x1-x0)*prog, ye=y0+(y1-y0)*prog;
    if(p.key){
      const dx=x1-x0, dy=y1-y0, len=Math.hypot(dx,dy)||1, ux=-dy/len*4, uy=dx/len*4;
      ctx.beginPath();
      ctx.moveTo(X(x0+ux),Y(y0+uy)); ctx.lineTo(X(xe+ux),Y(ye+uy)); ctx.lineTo(X(xe-ux),Y(ye-uy)); ctx.lineTo(X(x0-ux),Y(y0-uy)); ctx.closePath();
      ctx.fillStyle="rgba(56,189,248,"+(0.22*fade)+")"; ctx.fill();
      ctx.save(); ctx.shadowColor="#00e5ff"; ctx.shadowBlur=18*dpr;
      ctx.strokeStyle="rgba(0,229,255,"+fade+")"; ctx.lineWidth=Math.max(3,S*0.45);
      ctx.beginPath(); ctx.moveTo(X(x0),Y(y0)); ctx.lineTo(X(xe),Y(ye)); ctx.stroke();
      ctx.strokeStyle="rgba(255,255,255,"+fade+")"; ctx.lineWidth=Math.max(1.5,S*0.16); ctx.stroke(); ctx.restore();
    } else {
      ctx.save(); ctx.setLineDash([S*0.8,S*0.6]); ctx.strokeStyle="rgba(255,255,255,"+(0.7*fade)+")"; ctx.lineWidth=Math.max(1.5,S*0.2);
      ctx.beginPath(); ctx.moveTo(X(x0),Y(y0)); ctx.lineTo(X(xe),Y(ye)); ctx.stroke(); ctx.restore();
    }
  }

  // ---- مسار الكرة ----
  const bp=lerpPt(D.ball[i],D.ball[i2],f);
  const j0=Math.max(0,i-14); const tr=[];
  for(let j=j0;j<=i;j++) tr.push(D.ball[j]); tr.push(bp);
  for(let j=1;j<tr.length;j++){
    const a=j/tr.length;
    ctx.beginPath(); ctx.moveTo(X(tr[j-1][0]),Y(tr[j-1][1])); ctx.lineTo(X(tr[j][0]),Y(tr[j][1]));
    ctx.strokeStyle="rgba(255,255,255,"+(0.55*a)+")"; ctx.lineWidth=Math.max(2,S*0.5*a); ctx.lineCap="round"; ctx.stroke();
  }

  // ---- اللاعبين (الخصم الأول ثم المهاجم) + حامل الكرة ----
  const atkHome = D.attack==="Home";
  const atkList = atkHome? D.home : D.away, atkIds = atkHome? D.homeIds : D.awayIds;
  let carrier=null, best=3.2;
  for(let k=0;k<atkIds.length;k++){ const p=lerpPt(atkList[i][k],atkList[i2][k],f); if(!p) continue;
    const d=Math.hypot(p[0]-bp[0],p[1]-bp[1]); if(d<best){best=d;carrier={k:k};} }
  drawPlayers(D.away,D.awayIds,"#da3633",i,i2,f, atkHome?null:carrier);
  drawPlayers(D.home,D.homeIds,"#1f6feb",i,i2,f, atkHome?carrier:null);

  // ---- الكرة ----
  ctx.save(); ctx.shadowColor="#ffffff"; ctx.shadowBlur=20*dpr;
  ctx.beginPath(); ctx.arc(X(bp[0]),Y(bp[1]),Math.max(4,1.15*S),0,Math.PI*2);
  ctx.fillStyle="#ffffff"; ctx.fill(); ctx.lineWidth=Math.max(1.5,S*0.18); ctx.strokeStyle="#111"; ctx.stroke(); ctx.restore();

  // ---- تأثير النتيجة ----
  if(D.effect){
    const dt=t-D.effect.t;
    if(dt>=0&&dt<3.2){
      const k=D.effect.kind, ease=Math.min(1,dt/0.3), al=dt<2.6?1:(3.2-dt)/0.6;
      if(k==="GOAL"&&dt<1.6){
        ctx.fillStyle="rgba(255,215,0,"+(0.32*(1-dt/1.6))+")"; ctx.fillRect(0,0,W,H);
        ctx.fillStyle="rgba(255,235,120,"+(0.8*(1-dt/1.6))+")"; ctx.fillRect(X(30.34),Y(107.5),X(37.66)-X(30.34),Y(105)-Y(107.5));
      }
      ctx.save(); ctx.globalAlpha=al;
      const bw=W*0.62*ease, bh=H*0.15*ease, cx=W/2, cy=H*0.5;
      ctx.fillStyle=KCOL[k]||"#444"; ctx.strokeStyle="#fff"; ctx.lineWidth=3*dpr;
      ctx.beginPath(); ctx.roundRect(cx-bw/2,cy-bh/2,bw,bh,14*dpr); ctx.fill(); ctx.stroke();
      ctx.fillStyle="#fff"; ctx.font="bold "+(W*0.062*ease)+"px Arial"; ctx.textAlign="center"; ctx.textBaseline="middle";
      ctx.fillText((EMO[k]||"")+" "+L[k], cx, cy+2); ctx.restore();
    }
  }

  // ---- HUD ----
  ctx.fillStyle="rgba(10,16,24,.72)"; ctx.beginPath(); ctx.roundRect(S*1.2,H-S*6.2,S*17,S*4.8,8*dpr); ctx.fill();
  const clk=D.clock0+t, mm=Math.floor(clk/60), ss=Math.floor(clk%60);
  ctx.fillStyle="#fff"; ctx.font="bold "+(S*2.2)+"px Arial"; ctx.textAlign="left"; ctx.textBaseline="middle";
  ctx.fillText((mm<10?"0":"")+mm+":"+(ss<10?"0":"")+ss, S*2.2, H-S*3.8);
  ctx.beginPath(); ctx.arc(S*13.2,H-S*3.8,S*0.9,0,Math.PI*2); ctx.fillStyle=atkHome?"#1f6feb":"#da3633"; ctx.fill();
  ctx.fillStyle="#cbd5e0"; ctx.font=(S*1.7)+"px Arial"; ctx.fillText(D.attack, S*14.6, H-S*3.8);

  const lp=D.passes.find(p=>p.key);
  if(lp && D.packed>0 && t>=lp.t0 && t<=lp.t1+3){
    const txt="🎯 "+L.lp+" — "+D.packed+" "+L.def;
    ctx.font="bold "+(S*1.6)+"px Arial"; const tw=ctx.measureText(txt).width+S*3;
    const bx0=W-tw-S*1.2;
    ctx.fillStyle="rgba(0,150,200,.88)"; ctx.beginPath(); ctx.roundRect(bx0,S*1.2,tw,S*3.6,8*dpr); ctx.fill();
    ctx.fillStyle="#fff"; ctx.textAlign="left"; ctx.textBaseline="middle"; ctx.fillText(txt,bx0+S*1.5,S*3.0);
  }
}

// ---------- المعلق ----------
let lastIdx=-2, lastLang="";
function renderComm(){
  let idx=-1; for(let k=0;k<D.steps.length;k++) if(D.steps[k].t<=t) idx=k;
  if(idx===lastIdx && lang===lastLang) return; lastIdx=idx; lastLang=lang;
  const L=TXT[lang], box=document.getElementById('steps');
  box.dir=L.dir; document.querySelector('.panel').dir=L.dir;
  document.getElementById('ct').textContent=L.comm;
  document.getElementById('chain').textContent=D.chain?(L.chain+": "+D.chain):"";
  if(idx<0){ box.innerHTML='<div class="empty">'+L.wait+'</div>'; return; }
  let h=""; for(let k=0;k<=idx;k++){ const s=D.steps[k];
    h+='<div class="step'+(k===idx?' cur':'')+'" style="--c:'+(STAGE[s.stage]||"#e53e3e")+'"><div class="ts">⏱ '+s.t.toFixed(1)+'s</div><p>'+(lang==="ar"?s.ar:s.en)+'</p></div>'; }
  box.innerHTML=h; box.scrollTop=box.scrollHeight;
}

function updateUI(){
  const L=TXT[lang];
  document.getElementById('bp').textContent = playing? L.pause : (t>=D.T-0.05? L.replay : L.play);
  document.getElementById('sl').value = Math.round(t/D.T*1000);
  document.getElementById('tm').textContent = t.toFixed(1)+"s / "+D.T.toFixed(1)+"s";
}

function loop(ts){
  if(last===null) last=ts;
  const dt=(ts-last)/1000; last=ts;
  if(playing){ t+=dt*speed; if(t>=D.T){ t=D.T; playing=false; } }
  draw(); renderComm(); updateUI(); requestAnimationFrame(loop);
}

document.getElementById('bp').onclick=()=>{ if(!playing && t>=D.T-0.05) t=0; playing=!playing; };
document.getElementById('sl').oninput=e=>{ t=e.target.value/1000*D.T; };
document.getElementById('sp').onchange=e=>{ speed=parseFloat(e.target.value); };
document.getElementById('bl').onclick=()=>{ lang=lang==="ar"?"en":"ar"; lastLang=""; };
window.addEventListener('resize',resize);
if(!CanvasRenderingContext2D.prototype.roundRect){
  CanvasRenderingContext2D.prototype.roundRect=function(x,y,w,h,r){this.moveTo(x+r,y);this.arcTo(x+w,y,x+w,y+h,r);this.arcTo(x+w,y+h,x,y+h,r);this.arcTo(x,y+h,x,y,r);this.arcTo(x,y,x+w,y,r);this.closePath();};
}
resize(); requestAnimationFrame(loop);
</script></body></html>
"""


def render_viewer(item, is_ar):
    """يرسم الكاميرا الرأسية + المعلق المتزامن لقطة واحدة."""
    fid = int(item["frame_id"])
    m = df_elite[df_elite["Key_Action_Frame"] == fid]
    if m.empty:
        s_frame, e_frame, outcome, packed = fid - 125, fid + 200, "", 0
    else:
        row = m.iloc[0]
        s_frame, e_frame = int(row["Seq_Start_Frame"]), int(row["Seq_End_Frame"])
        outcome = str(row.get("Outcome_Type", ""))
        packed = int(row.get("Danger_Value", 0) or 0)

    s_frame = max(s_frame, int(df_home["Frame"].min()))
    e_frame = min(e_frame, int(df_home["Frame"].max()))

    seq = build_sequence(fid, s_frame, e_frame, int(item["period"]), str(item["team"]), outcome, packed)
    if seq is None:
        st.warning("مفيش بيانات تتبع للفريمات دي.")
        return

    payload = dict(seq)
    payload["lang"] = "ar" if is_ar else "en"
    payload["steps"] = prep_steps(item, seq["T"])
    payload["chain"] = item.get("attack_chain", "")

    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    components.html(VIEWER_HTML.replace("__DATA__", data_json), height=680, scrolling=True)


# ==============================================================
# 4. الواجهة
# ==============================================================
col_title, col_lang = st.columns([8, 2])
with col_lang:
    lang = st.selectbox("🌐 Language / اللغة", ["العربية", "English"], index=0)
is_ar = (lang == "العربية")

with col_title:
    st.markdown(f"### {'🎬 استوديو المشجع: لقطات المباراة بتتبع الكاميرا الحقيقية' if is_ar else '🎬 Casual Fan Mode: Authentic Drone-Cam Tracking'}")

# نافذة منبثقة (Modal) لو نسخة Streamlit بتدعمها، وإلا بيتعرض تحت الكروت
_dialog = getattr(st, "dialog", None) or getattr(st, "experimental_dialog", None)
open_dialog = None
if _dialog is not None:
    try:
        @_dialog("🎥 لقطة المباراة / Match Moment", width="large")
        def open_dialog(item, is_ar_flag):
            t = item["card_title_ar"] if is_ar_flag else item["card_title_en"]
            b = item["card_badge_ar"] if is_ar_flag else item["card_badge_en"]
            st.markdown(f"**{t}** `[{item['time_str']}]` — {b}")
            render_viewer(item, is_ar_flag)
    except TypeError:
        @_dialog("🎥 لقطة المباراة / Match Moment")
        def open_dialog(item, is_ar_flag):
            t = item["card_title_ar"] if is_ar_flag else item["card_title_en"]
            st.markdown(f"**{t}** `[{item['time_str']}]`")
            render_viewer(item, is_ar_flag)

tab_labels = (["⚽ الأهداف ولحظات الحسم", "🧤 تصديات إعجازية", "🎯 تمريرات ليزرية"] if is_ar
              else ["⚽ Iconic Goals", "🧤 Super Saves", "🎯 Laser Passes"])
tab_goals, tab_saves, tab_passes = st.tabs(tab_labels)


def render_category_grid(category_key):
    moments = casual_data.get("categories", {}).get(category_key, [])
    if not moments:
        st.info("لا توجد لقطات في هذا القسم." if is_ar else "No moments in this category.")
        return
    per_row = 4
    for start in range(0, len(moments), per_row):
        chunk = moments[start:start + per_row]
        cols = st.columns(per_row)
        for i, item in enumerate(chunk):
            with cols[i]:
                title = html_lib.escape(item["card_title_ar"] if is_ar else item["card_title_en"])
                badge = html_lib.escape(item["card_badge_ar"] if is_ar else item["card_badge_en"])
                st.markdown(
                    f'<div class="fan-card"><span class="fan-chip">{html_lib.escape(str(item["time_str"]))}</span>'
                    f'<p class="fan-badge">{badge}</p><p class="fan-title">{title}</p></div>',
                    unsafe_allow_html=True)
                if st.button(f"▶ {'عرض اللقطة' if is_ar else 'View Play'}", key=f"btn_{category_key}_{item['event_id']}",
                             use_container_width=True):
                    if open_dialog is not None:
                        open_dialog(item, is_ar)
                    else:
                        st.session_state["active_moment"] = item


with tab_goals:
    render_category_grid("goals")
with tab_saves:
    render_category_grid("saves")
with tab_passes:
    render_category_grid("passes")

# بديل لو النسخة قديمة ومفيش dialog
if open_dialog is None and st.session_state.get("active_moment"):
    cur = st.session_state["active_moment"]
    st.markdown("---")
    c_head, c_close = st.columns([9, 1])
    with c_head:
        t = cur["card_title_ar"] if is_ar else cur["card_title_en"]
        st.markdown(f"#### 🎥 {t} `[{cur['time_str']}]`")
    with c_close:
        if st.button("✖ إغلاق" if is_ar else "✖ Close"):
            st.session_state["active_moment"] = None
            st.rerun()
    render_viewer(cur, is_ar)