import base64
import json
import os
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="AI Match Intelligence Broadcast",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==============================================================
# الثوابت
# ==============================================================
FPS = 25
STEP_P = 5        # اللاعبين: 5 فريم/ثانية (والمتصفح بيعمل interpolation لحد 60)
STEP_B = 2        # الكرة: 12.5 فريم/ثانية (أدق لأنها أسرع)
PITCH_L, PITCH_W = 105.0, 68.0
MISSING = -32768

HOME_FILE = "Sample_Game_1_RawTrackingData_Home_Team.csv"
AWAY_FILE = "Sample_Game_1_RawTrackingData_Away_Team.csv"
NARRATIVES_FILE = "live_broadcast_narratives.json"
EVENTS_CANDIDATES = ["Sample_Game_1_RawEventsData.csv", "events.csv"]

st.markdown(
    "<style>.block-container{padding-top:1.2rem;padding-bottom:0.5rem;}</style>",
    unsafe_allow_html=True,
)


# ==============================================================
# 1. تحميل البيانات وتجهيز المباراة كاملة (مرة واحدة ويتخزن في الكاش)
# ==============================================================
def read_tracking(csv_path, team_prefix):
    """قراءة ملف Metrica الخام: أعمدة Prefix_PlayerN_x/_y + Ball_x/Ball_y بالأمتار (y لفوق)."""
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
        num = str(row1[i]).strip() or str(row1[i - 1]).strip()
        col_names.append(f"{team_prefix}_Player{num}_x")
        col_names.append(f"{team_prefix}_Player{num}_y")
        i += 2

    df = pd.read_csv(csv_path, skiprows=3, header=None, names=col_names, low_memory=False)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    x_cols = [c for c in df.columns if c.endswith("_x")]
    y_cols = [c for c in df.columns if c.endswith("_y")]
    if x_cols and np.nanmax(df[x_cols].to_numpy(dtype=float)) <= 1.5:
        for c in x_cols:
            df[c] = df[c] * PITCH_L
        for c in y_cols:
            df[c] = (1.0 - df[c]) * PITCH_W
    return df


def _pack(arr):
    """float (متر) -> int16 (ديسيمتر) مع قيمة خاصة للمفقود -> base64"""
    a = np.round(arr * 10.0)
    a[np.isnan(a)] = MISSING
    a = np.clip(a, MISSING, 32767).astype("<i2")
    return base64.b64encode(a.tobytes()).decode("ascii")


@st.cache_data(show_spinner="جاري تجهيز المباراة كاملة (مرة واحدة فقط)...")
def build_match():
    df_home = read_tracking(HOME_FILE, "Home")
    df_away = read_tracking(AWAY_FILE, "Away")

    # ---- اللاعبين (كل المباراة بمعدل STEP_P) ----
    h = df_home.iloc[::STEP_P]
    a = df_away.set_index("Frame", drop=False).reindex(h["Frame"].to_numpy())
    h_x = [c for c in h.columns if c.startswith("Home_Player") and c.endswith("_x")]
    a_x = [c for c in a.columns if c.startswith("Away_Player") and c.endswith("_x")]
    h_ids = [c[len("Home_Player"):-2] for c in h_x]
    a_ids = [c[len("Away_Player"):-2] for c in a_x]

    HX = h[h_x].to_numpy(dtype=float); HY = h[[c[:-2] + "_y" for c in h_x]].to_numpy(dtype=float)
    AX = a[a_x].to_numpy(dtype=float); AY = a[[c[:-2] + "_y" for c in a_x]].to_numpy(dtype=float)
    X = np.concatenate([HX, AX], axis=1)
    Y = np.concatenate([HY, AY], axis=1)
    players = np.stack([X, Y], axis=2)                    # فريم × لاعب × (x,y)

    # ---- الكرة (معدل أعلى) ----
    b = df_home.iloc[::STEP_B]
    bx = pd.Series(b["Ball_x"].to_numpy(dtype=float)).ffill().bfill().fillna(PITCH_L / 2).to_numpy()
    by = pd.Series(b["Ball_y"].to_numpy(dtype=float)).ffill().bfill().fillna(PITCH_W / 2).to_numpy()
    ball = np.stack([bx, by], axis=1)

    t0 = float(df_home["Time [s]"].iloc[0])
    p2 = df_home.loc[df_home["Period"] == 2, "Time [s]"]
    p2start = float(p2.min()) if len(p2) else 1e9
    nf = players.shape[0]

    return {
        "P": _pack(players), "B": _pack(ball),
        "nf": int(nf), "nb": int(ball.shape[0]),
        "nh": len(h_ids), "na": len(a_ids), "np": len(h_ids) + len(a_ids),
        "fpsP": FPS / STEP_P, "fpsB": FPS / STEP_B,
        "homeIds": h_ids, "awayIds": a_ids,
        "gk": {"Home": h_ids[0] if h_ids else "", "Away": a_ids[0] if a_ids else ""},   # أول عمود في Metrica = الحارس
        "t0": round(t0, 2), "T": round((nf - 1) * STEP_P / FPS, 2), "p2start": round(p2start, 2),
    }


@st.cache_data(show_spinner=False)
def load_story():
    df_events = None
    for fn in EVENTS_CANDIDATES:
        if os.path.exists(fn):
            df_events = pd.read_csv(fn)
            break

    narratives = []
    if os.path.exists(NARRATIVES_FILE):
        with open(NARRATIVES_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for n in raw:
            try:
                s, e = float(n.get("start_s", 0)), float(n.get("end_s", 0))
            except (TypeError, ValueError):
                continue
            narratives.append({
                "player": str(n.get("player", "")), "team": str(n.get("team", "")),
                "ticker_ar": str(n.get("ticker_ar", "") or ""), "ticker_en": str(n.get("ticker_en", "") or ""),
                "s": s, "e": max(e, s + 1.0),
            })
        narratives.sort(key=lambda x: x["s"])

    goals = []
    if df_events is not None and {"Type", "Subtype", "Team"}.issubset(df_events.columns):
        g = df_events[df_events["Type"].astype(str).str.upper().str.contains("SHOT")
                      & df_events["Subtype"].astype(str).str.upper().str.contains("GOAL")]
        for _, r in g.iterrows():
            if pd.notna(r.get("End Frame")):
                gt = float(r["End Frame"]) / FPS
            elif pd.notna(r.get("End Time [s]")):
                gt = float(r["End Time [s]"])
            else:
                gt = float(r["Start Frame"]) / FPS
            goals.append({"t": round(gt, 2), "team": str(r["Team"]), "player": str(r.get("From", ""))})
        goals.sort(key=lambda x: x["t"])
    return narratives, goals


_missing = [f for f in (HOME_FILE, AWAY_FILE) if not os.path.exists(f)]
if _missing:
    st.error("ملفات ناقصة في الفولدر: " + " , ".join(_missing))
    st.stop()

match = build_match()
narratives, goals = load_story()


# ==============================================================
# 2. الشاشة: ملعب + تعليق جنبه + لوحة نتيجة (HTML/Canvas) - المباراة كاملة
# ==============================================================
VIEWER_HTML = r"""
<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  *{box-sizing:border-box}
  html,body{margin:0;height:100%;background:#0b0f19;color:#fff;overflow:hidden;
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Cairo",Tahoma,sans-serif}
  #shell{margin:0 auto;font-size:14px}
  #main{display:flex;gap:12px;align-items:flex-start;direction:ltr}
  #pcol{flex:none}
  #side{flex:none;display:flex;flex-direction:column;background:rgba(15,23,42,.92);border:1px solid rgba(255,255,255,.14);
    border-top:.35em solid #38bdf8;border-radius:.7em;padding:.8em;overflow:hidden}
  .badge{align-self:flex-start;background:#0284c7;color:#fff;font-size:.78em;font-weight:800;padding:.3em .7em;border-radius:.3em;white-space:nowrap;letter-spacing:.04em}
  #hist{flex:1;overflow:hidden;margin-top:.7em;display:flex;flex-direction:column;gap:.55em}
  .item{background:rgba(30,41,59,.7);border-radius:.55em;padding:.6em .75em;opacity:.45;border-inline-start:.3em solid var(--c)}
  .item.cur{opacity:1;background:rgba(51,65,85,.9);animation:fade .35s ease}
  .item.first .txt{font-size:1.08em}
  .chip{display:inline-block;font-size:.82em;font-weight:800;white-space:nowrap;padding:.2em .6em;border-radius:.4em;border:1px solid currentColor;direction:ltr;margin-bottom:.35em}
  .txt{line-height:1.5;font-weight:500;color:#f8fafc}
  .idle{color:#94a3b8;font-size:.95em;padding:.5em .2em}
  @keyframes fade{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:none}}
  #stage{position:relative;border-radius:.7em;overflow:hidden;border:1px solid #1e293b;box-shadow:0 10px 30px rgba(0,0,0,.6);background:#14401f}
  canvas{display:block;width:100%}
  #sb{position:absolute;top:.8em;left:.8em;display:flex;flex-direction:column;gap:.3em;z-index:5;direction:ltr}
  .sbrow{display:flex;align-items:stretch;background:rgba(15,23,42,.9);border-radius:.5em;overflow:hidden;
    border:1px solid rgba(255,255,255,.15);font-weight:800;box-shadow:0 4px 14px rgba(0,0,0,.5)}
  .tag{padding:.4em .8em;display:flex;align-items:center;gap:.4em;font-size:.95em}
  .score{padding:.3em .9em;font-size:1.35em;letter-spacing:.1em;background:#0f172a;display:flex;align-items:center}
  .tm{padding:.4em .7em;background:#1e293b;color:#38bdf8;font-family:Consolas,monospace;font-size:.9em;display:flex;align-items:center;gap:.5em}
  .dot{width:.55em;height:.55em;border-radius:50%;background:#ef4444;animation:pulse 1.2s infinite}
  @keyframes pulse{50%{opacity:.25}}
  .flash{animation:flash .9s ease 2}
  @keyframes flash{50%{background:#f59e0b}}
  #scorer{display:none;background:rgba(15,23,42,.9);border:1px solid #f59e0b;color:#fde68a;font-size:.85em;
    font-weight:700;padding:.3em .7em;border-radius:.4em;width:fit-content}
  #goal{position:absolute;top:42%;left:50%;transform:translate(-50%,-50%) scale(0);z-index:9;pointer-events:none;
    background:linear-gradient(135deg,#f59e0b,#ef4444);padding:.35em 1.5em;border-radius:.6em;text-align:center;
    box-shadow:0 0 50px rgba(239,68,68,.85);transition:transform .4s cubic-bezier(.175,.885,.32,1.275)}
  #goal.on{transform:translate(-50%,-50%) scale(1)}
  #goal b{display:block;font-size:2.6em;font-weight:900;letter-spacing:.12em;text-transform:uppercase}
  #goal span{font-size:1.05em;font-weight:700}
  #ctl{display:flex;gap:.45em;align-items:center;margin-top:.45em;flex-wrap:wrap}
  button,select{background:#1f2a37;color:#e2e8f0;border:1px solid #3a4a5e;border-radius:.5em;padding:.35em .7em;font-size:.88em;cursor:pointer}
  button:hover{background:#2b3a4d}
  #sl{flex:1 1 120px;min-width:90px;accent-color:#38bdf8}
  #tt{font-size:.8em;color:#94a3b8;min-width:7em;text-align:center;direction:ltr}
  #jump{max-width:11em}
</style></head>
<body>
<div id="shell">
 <div id="main">
  <div id="pcol">
   <div id="stage">
    <canvas id="c"></canvas>
    <div id="sb">
      <div class="sbrow">
        <div class="tag" id="th" style="background:#E63946">HOME</div>
        <div class="score" id="sc">0 - 0</div>
        <div class="tag" id="ta" style="background:#1D3557">AWAY</div>
        <div class="tm"><span class="dot"></span><span id="half">LIVE</span><span id="clk">00:00</span></div>
      </div>
      <div id="scorer"></div>
    </div>
    <div id="goal"><b id="gt">GOAL!</b><span id="gs"></span></div>
   </div>
   <div id="ctl">
    <button id="bp"></button><input id="sl" type="range" min="0" max="1000" step="1" value="0"><span id="tt"></span>
    <select id="sp"><option value="1" selected>1x</option><option value="2">2x</option><option value="4">4x</option><option value="8">8x</option><option value="16">16x</option></select>
    <select id="jump"></select>
    <button id="bl">AR / EN</button>
   </div>
  </div>
  <div id="side"><div class="badge" id="badge"></div><div id="hist"></div></div>
 </div>
</div>
<script>
const D = __DATA__;
function b64buf(s){ const bin=atob(s), n=bin.length, u=new Uint8Array(n); for(let i=0;i<n;i++) u[i]=bin.charCodeAt(i); return u.buffer; }
const P=new Int16Array(b64buf(D.P)), B=new Int16Array(b64buf(D.B)); D.P=null; D.B=null;
const MISS=-32768;
const TXT = {
  ar:{badge:"تحليل تكتيكي فوري",idle:"جاري قراءة أحداث ومسار اللعب...",play:"▶ تشغيل",pause:"⏸ إيقاف",replay:"↺ إعادة",
      live:"مباشر",h1:"الشوط الأول",h2:"الشوط الثاني",goal:"هدف!",dir:"rtl",jump:"⏩ اقفز إلى…",ko:"⏮ صافرة البداية",sh:"▶ الشوط الثاني",g:"⚽ الأهداف",m:"💬 لقطات التحليل"},
  en:{badge:"AI TACTICAL FEED",idle:"Monitoring tactical build-up...",play:"▶ Play",pause:"⏸ Pause",replay:"↺ Replay",
      live:"LIVE",h1:"1st HALF",h2:"2nd HALF",goal:"GOAL!",dir:"ltr",jump:"⏩ Jump to…",ko:"⏮ Kick-off",sh:"▶ 2nd half",g:"⚽ Goals",m:"💬 Key moments"}
};
const VX0=-6, VX1=111, VY1=70.5, VH=73, ASPECT=(VX1-VX0)/VH;
const GKC={Home:"#f6c945",Away:"#38d996"}, COL={Home:D.homeColor,Away:D.awayColor};
let lang=D.lang, playing=true, t=0, speed=1, last=null, W=0, H=0, S=1, dpr=window.devicePixelRatio||1;
const cv=document.getElementById('c'), ctx=cv.getContext('2d'), pitchCv=document.createElement('canvas');
const px=x=>(x-VX0)*S, py=y=>(VY1-y)*S;
const $=id=>document.getElementById(id);
$('th').style.background=COL.Home; $('ta').style.background=COL.Away;

function lerp(a,b,f){ if(a&&b) return [a[0]+(b[0]-a[0])*f,a[1]+(b[1]-a[1])*f]; return a||b; }
function fmt(s){ s=Math.max(0,s); const m=Math.floor(s/60), x=Math.floor(s%60); return (m<10?"0":"")+m+":"+(x<10?"0":"")+x; }
function clockOf(abs){ return abs<D.p2start? abs : 2700+(abs-D.p2start); }     // الشوط التاني يبدأ من 45:00
function pname(s){ const m=String(s).match(/\d+/); return m? "Player "+m[0] : String(s); }
function esc(x){return String(x).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
function pnum(s){ const m=String(s).match(/\d+/); return m? m[0] : ""; }
function bright(hex){ let h=String(hex).replace('#',''); if(h.length===3) h=h.split('').map(c=>c+c).join('');
  const r=parseInt(h.slice(0,2),16), g=parseInt(h.slice(2,4),16), b=parseInt(h.slice(4,6),16);
  if((0.299*r+0.587*g+0.114*b)/255>=0.45) return '#'+h;
  const m=v=>Math.round(v+(255-v)*0.6).toString(16).padStart(2,'0'); return '#'+m(r)+m(g)+m(b); }
function pl(k,i){ const o=(i*D.np+k)*2, x=P[o]; if(x===MISS) return null; return [x/10,P[o+1]/10]; }
function bl(j){ const o=j*2; return [B[o]/10,B[o+1]/10]; }

function drawPitch(){
  pitchCv.width=W; pitchCv.height=H; const g=pitchCv.getContext('2d');
  g.fillStyle="#14401f"; g.fillRect(0,0,W,H);
  for(let k=0;k<21;k++){ g.fillStyle=k%2?"#1d5a2a":"#195226"; const x0=k*5,x1=Math.min(105,x0+5); g.fillRect(px(x0),py(68),px(x1)-px(x0),py(0)-py(68)); }
  g.strokeStyle="rgba(255,255,255,.85)"; g.lineWidth=Math.max(1.5,S*.16); g.lineJoin="round";
  const line=pts=>{g.beginPath();pts.forEach((p,i)=>i?g.lineTo(px(p[0]),py(p[1])):g.moveTo(px(p[0]),py(p[1])));g.stroke();};
  line([[0,0],[105,0],[105,68],[0,68],[0,0]]); line([[52.5,0],[52.5,68]]);
  g.beginPath(); g.arc(px(52.5),py(34),9.15*S,0,Math.PI*2); g.stroke();
  const spot=(x,y)=>{g.fillStyle="#fff";g.beginPath();g.arc(px(x),py(y),Math.max(2,S*.28),0,Math.PI*2);g.fill();};
  spot(52.5,34);
  for(const [xg,d] of [[0,1],[105,-1]]){
    line([[xg,13.84],[xg+d*16.5,13.84],[xg+d*16.5,54.16],[xg,54.16]]);
    line([[xg,24.84],[xg+d*5.5,24.84],[xg+d*5.5,43.16],[xg,43.16]]);
    spot(xg+d*11,34);
    let seg=[],arcs=[];
    for(let a=0;a<=Math.PI*2+.05;a+=.05){ const x=xg+d*11+9.15*Math.cos(a), y=34+9.15*Math.sin(a);
      if((x-xg)*d>16.5) seg.push([x,y]); else if(seg.length){arcs.push(seg);seg=[];} }
    if(seg.length) arcs.push(seg); arcs.forEach(s=>{if(s.length>1) line(s);});
    const x1=xg-d*2.44; g.fillStyle="rgba(255,255,255,.2)";
    g.fillRect(px(Math.min(xg,x1)),py(37.66),Math.abs(px(xg)-px(x1)),py(30.34)-py(37.66));
    const sv=g.strokeStyle; g.strokeStyle="#ffd700"; g.lineWidth=Math.max(2,S*.3);
    line([[xg,30.34],[x1,30.34],[x1,37.66],[xg,37.66]]); g.strokeStyle=sv; g.lineWidth=Math.max(1.5,S*.16);
  }
  for(const [cx,cy,sx,sy] of [[0,0,1,1],[105,0,-1,1],[0,68,1,-1],[105,68,-1,-1]]){
    const pts=[]; for(let a=0;a<=Math.PI/2+.01;a+=.1) pts.push([cx+sx*Math.cos(a),cy+sy*Math.sin(a)]); line(pts);
  }
}

function layout(){
  const shell=$('shell'), parentW=document.body.clientWidth, gap=12, stack=parentW<720;
  const panelW=stack?parentW:Math.max(230,Math.min(340,parentW*.27));
  const ct=$('ctl').offsetHeight;
  const availW=stack?parentW:parentW-panelW-gap;
  const availH=window.innerHeight-ct-16-(stack?150+gap:0);
  const w=Math.max(300,Math.min(availW,availH*ASPECT));
  $('main').style.flexDirection=stack?'column':'row';
  shell.style.width=(stack?w:w+gap+panelW)+'px';
  shell.style.fontSize=Math.max(11,Math.min(16,w/62))+'px';
  $('pcol').style.width=w+'px';
  const h=w/ASPECT; cv.style.height=h+'px';
  cv.width=Math.round(w*dpr); cv.height=Math.round(h*dpr);
  W=cv.width; H=cv.height; S=W/(VX1-VX0);
  const side=$('side'); side.style.width=(stack?w:panelW)+'px';
  side.style.height=stack?'150px':($('pcol').offsetHeight+'px');
  drawPitch(); draw();
}

// ---- أحدث تعليق حتى الوقت الحالي (D.narr مرتبة بالوقت) ----
function lastNarr(abs){ let lo=0,hi=D.narr.length-1,r=-1; while(lo<=hi){ const m=(lo+hi)>>1; if(D.narr[m].s<=abs){r=m;lo=m+1;} else hi=m-1; } return r; }

function draw(){
  if(!W) return;
  const abs=D.t0+t, ri=lastNarr(abs), nr=ri>=0?D.narr[ri]:null, nar=(nr&&abs<=nr.e+3)?nr:null;
  const nid=nar?pnum(nar.player):"", nteam=nar?nar.team:"";
  const fp=Math.min(Math.max(t*D.fpsP,0),D.nf-1), i=Math.floor(fp), i2=Math.min(i+1,D.nf-1), f=fp-i;
  const fb=Math.min(Math.max(t*D.fpsB,0),D.nb-1), j=Math.floor(fb), j2=Math.min(j+1,D.nb-1);
  ctx.clearRect(0,0,W,H); ctx.drawImage(pitchCv,0,0);

  const bp=lerp(bl(j),bl(j2),fb-j), tr=[];
  for(let q=Math.max(0,j-10);q<=j;q++) tr.push(bl(q)); tr.push(bp);
  for(let q=1;q<tr.length;q++){ const a=q/tr.length; ctx.beginPath(); ctx.moveTo(px(tr[q-1][0]),py(tr[q-1][1])); ctx.lineTo(px(tr[q][0]),py(tr[q][1]));
    ctx.strokeStyle="rgba(255,255,255,"+(0.5*a)+")"; ctx.lineWidth=Math.max(2,S*.5*a); ctx.lineCap="round"; ctx.stroke(); }

  const r=Math.max(8*dpr,1.75*S);
  ctx.font="bold "+(r*1.0)+"px Arial"; ctx.textAlign="center"; ctx.textBaseline="middle";
  for(const team of ["Away","Home"]){
    const k0=team==="Home"?0:D.nh, n=team==="Home"?D.nh:D.na, ids=team==="Home"?D.homeIds:D.awayIds;
    for(let k=0;k<n;k++){
      const p=lerp(pl(k0+k,i),pl(k0+k,i2),f); if(!p) continue;
      const x=px(p[0]), y=py(p[1]), isGK=ids[k]===D.gk[team];
      if(nar && team===nteam && ids[k]===nid){ ctx.beginPath(); ctx.arc(x,y,r+S*.9,0,Math.PI*2); ctx.strokeStyle="#ffd54a"; ctx.lineWidth=Math.max(2,S*.3); ctx.shadowColor="#ffd54a"; ctx.shadowBlur=12*dpr; ctx.stroke(); ctx.shadowBlur=0; }
      ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2);
      ctx.fillStyle=isGK?GKC[team]:COL[team]; ctx.fill();
      ctx.lineWidth=Math.max(1.5,r*(isGK?.2:.13)); ctx.strokeStyle=isGK?COL[team]:"#fff"; ctx.stroke();
      ctx.fillStyle=isGK?"#111":"#fff"; ctx.fillText(ids[k],x,y+r*.05);
    }
  }
  const pulse=.5+.5*Math.sin(performance.now()/170), bx=px(bp[0]), by=py(bp[1]), br=Math.max(5*dpr,1.0*S);
  ctx.beginPath(); ctx.arc(bx,by,br*(1.9+.7*pulse),0,Math.PI*2); ctx.fillStyle="rgba(255,255,255,"+(0.16+0.1*pulse)+")"; ctx.fill();
  ctx.save(); ctx.shadowColor="#fff"; ctx.shadowBlur=16*dpr; ctx.beginPath(); ctx.arc(bx,by,br,0,Math.PI*2);
  ctx.fillStyle="#fff"; ctx.fill(); ctx.lineWidth=Math.max(1.5,S*.15); ctx.strokeStyle="#111"; ctx.stroke(); ctx.restore();
}

let lastKey="", bannerKey="";
function hud(){
  const L=TXT[lang], abs=D.t0+t;
  let hs=0, as=0, gl=null, ag=null;
  for(const g of D.goals){ if(g.t<=abs){ if(g.team==="Home") hs++; else as++; gl=g; if(abs<g.t+3.6) ag=g; } else break; }
  const sc=$('sc'), s=hs+" - "+as;
  if(sc.textContent!==s){ sc.textContent=s; sc.classList.remove('flash'); void sc.offsetWidth; sc.classList.add('flash'); }
  $('clk').textContent=fmt(clockOf(abs)); $('half').textContent=L.live+" · "+(abs<D.p2start?L.h1:L.h2);
  const sd=$('scorer');
  if(gl && abs-gl.t<10){ sd.style.display="block"; sd.textContent="⚽ "+pname(gl.player)+" · "+gl.team+" · "+fmt(clockOf(gl.t)); } else sd.style.display="none";
  const key=ag?ag.t+ag.team:"";
  if(key!==bannerKey){ bannerKey=key; $('goal').classList.toggle('on',!!ag);
    if(ag){ $('gt').textContent=L.goal+" ⚽"; $('gs').textContent=pname(ag.player)+" • "+ag.team; } }

  $('side').dir=L.dir; $('badge').textContent=L.badge;
  const ri=lastNarr(abs), live=ri>=0&&abs<=D.narr[ri].e+3, nk=ri+"|"+lang+(live?"A":"");
  if(nk!==lastKey){ lastKey=nk; const box=$('hist');
    if(ri<0){ box.innerHTML='<div class="idle">'+esc(L.idle)+'</div>'; }
    else { let h=""; for(let ix=0; ix<4 && ri-ix>=0; ix++){ const n=D.narr[ri-ix], c=bright(COL[n.team]||"#38bdf8");
      const tx=(lang==="ar"?(n.ticker_ar||n.ticker_en):(n.ticker_en||n.ticker_ar))||"";
      h+='<div class="item'+((ix===0&&live)?' cur':'')+(ix===0?' first':'')+'" style="--c:'+c+'"><span class="chip" style="color:'+c+';background:'+c+'22">'+esc(pname(n.player)+" • "+n.team)+'</span><div class="txt">'+esc(tx)+'</div></div>'; }
      box.innerHTML=h; } }
  $('bp').textContent=playing?L.pause:(t>=D.T-.05?L.replay:L.play);
  $('sl').value=Math.floor(t); $('tt').textContent=fmt(clockOf(D.t0+t))+" / "+fmt(clockOf(D.t0+D.T));
}

function buildJump(){
  const L=TXT[lang], sel=$('jump');
  let h='<option value="">'+L.jump+'</option><option value="0">'+L.ko+'</option>';
  if(D.p2start<1e8) h+='<option value="'+Math.max(0,D.p2start-D.t0)+'">'+L.sh+'</option>';
  if(D.goals.length){ h+='<optgroup label="'+L.g+'">'+D.goals.map(g=>'<option value="'+Math.max(0,g.t-D.t0-25)+'">'+esc(fmt(clockOf(g.t))+" · "+pname(g.player)+" ("+g.team+")")+'</option>').join("")+'</optgroup>'; }
  if(D.narr.length){ h+='<optgroup label="'+L.m+'">'+D.narr.slice(0,400).map(n=>'<option value="'+Math.max(0,n.s-D.t0-6)+'">'+esc(fmt(clockOf(n.s))+" · "+pname(n.player)+" ("+n.team+")")+'</option>').join("")+'</optgroup>'; }
  sel.innerHTML=h;
}

function loop(ts){
  if(last===null) last=ts; const dt=(ts-last)/1000; last=ts;
  if(playing){ t+=dt*speed; if(t>=D.T){ t=D.T; playing=false; } }
  draw(); hud(); requestAnimationFrame(loop);
}
$('sl').max=Math.floor(D.T);
$('bp').onclick=()=>{ if(!playing&&t>=D.T-.05) t=0; playing=!playing; };
$('sl').oninput=e=>{ t=parseFloat(e.target.value); };
$('sp').onchange=e=>{ speed=parseFloat(e.target.value); };
$('jump').onchange=e=>{ if(e.target.value!==""){ t=Math.min(D.T,parseFloat(e.target.value)); playing=true; } e.target.value=""; };
$('bl').onclick=()=>{ lang=lang==="ar"?"en":"ar"; lastKey=""; buildJump(); };
window.addEventListener('resize',layout);
buildJump(); layout(); requestAnimationFrame(loop);
</script></body></html>
"""

# ==============================================================
# 3. الشريط الجانبي
# ==============================================================
with st.sidebar:
    st.header("⚙️ Broadcast Settings")
    lang = st.radio("Language / اللغة", ["العربية", "English"], index=0)
    home_color = st.color_picker("Home Team Color", "#E63946")
    away_color = st.color_picker("Away Team Color", "#1D3557")
    st.caption("المباراة كاملة بتشتغل من صافرة البداية. السرعة والقفز لأي هدف أو لقطة من أزرار الشاشة نفسها.")


# ==============================================================
# 4. العرض
# ==============================================================
payload = dict(match)
payload.update({
    "lang": "ar" if lang == "العربية" else "en",
    "homeColor": home_color, "awayColor": away_color,
    "narr": narratives, "goals": goals,
})
data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
components.html(VIEWER_HTML.replace("__DATA__", data_json), height=700, scrolling=False)

def mmss(s):
    return f"{int(s // 60):02d}:{int(s % 60):02d}"

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Match Coverage", value=f"{len(narratives)} Key Moments", delta="LLM Annotated" if narratives else None)
with col2:
    st.metric(label="Goals Detected", value=f"{len(goals)}")
with col3:
    st.metric(label="Match Length", value=mmss(match["T"]))