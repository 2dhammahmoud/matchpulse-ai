import json
import os
import re
import runpy

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ==============================================================
# 1. إعداد الصفحة
# ==============================================================
st.set_page_config(
    page_title="MatchPulse AI | Microsoft Hackathon",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "current_view" not in st.session_state:
    st.session_state["current_view"] = "home_recap"

# ---------- الملفات ----------
HOME_FILE = "Sample_Game_1_RawTrackingData_Home_Team.csv"
AWAY_FILE = "Sample_Game_1_RawTrackingData_Away_Team.csv"
RECAP_FILE = "match_recap_data.json"
NARRATIVES_FILE = "live_broadcast_narratives.json"
EVENTS_CANDIDATES = ["Sample_Game_1_RawEventsData.csv", "events.csv"]

# ---------- الألوان ----------
HOME_C, AWAY_C = "#3b82f6", "#ef4444"
GOLD, ACCENT = "#eab308", "#38bdf8"
PITCH_L, PITCH_W = 105.0, 68.0

# ---------- الموديولات (بتدوّر على أول ملف موجود من القايمة) ----------
MODULES = {
    "app_analyst": dict(
        files=["app_analyst.py"], icon="🎯", color="#38bdf8",
        title_en="Tactical Video Room", title_ar="الغرفة التكتيكية للمحلل",
        desc_en="Elite attack sequences replayed with live 22-player tracking, the 8m line-breaking corridor, and bilingual AI tactical reports.",
        desc_ar="أخطر الهجمات بإعادة حية لـ 22 لاعباً مع ممر كسر الخطوط (8 متر) وتقارير تكتيكية بالذكاء الاصطناعي بلغتين.",
        btn_en="Launch Analyst Suite", btn_ar="افتح غرفة المحلل"),
    "casual_mode": dict(
        files=["casual_mode_app.py", "casual_fan_app.py"], icon="🎬", color="#f59e0b",
        title_en="Casual Fan Mode", title_ar="وضع المشجع",
        desc_en="Goals, saves and laser passes on a vertical drone-cam with synchronized, exciting live commentary.",
        desc_ar="الأهداف والتصديات والتمريرات الليزرية بكاميرا درون رأسية مع تعليق حماسي متزامن مع الحركة.",
        btn_en="Launch Fan Experience", btn_ar="افتح تجربة المشجع"),
    "player_spotlight": dict(
        files=["player_spotlight_app.py"], icon="🌟", color="#a855f7",
        title_en="Player Spotlight Camera", title_ar="كاميرا تتبع اللاعب",
        desc_en="Pick a team and a player: a follow-cam glides with them across the pitch, with their match stats and key actions.",
        desc_ar="اختار الفريق واللاعب والكاميرا تتحرك معاه في الملعب، مع إحصائياته ولقطاته المهمة.",
        btn_en="Launch Spotlight", btn_ar="افتح كاميرا اللاعب"),
    "broadcast_view": dict(
        files=["broadcast_view.py", "broadcast_app.py"], icon="📺", color="#10b981",
        title_en="Full Match Live Broadcast", title_ar="البث المباشر للمباراة كاملة",
        desc_en="The whole match as a TV broadcast: live scoreboard, goal animations and a synchronized AI tactical ticker.",
        desc_ar="المباراة كاملة كبث تلفزيوني: لوحة نتيجة حية وأنيميشن الأهداف وشريط تحليل تكتيكي متزامن.",
        btn_en="Launch Live Broadcast", btn_ar="افتح البث المباشر"),
}

# ==============================================================
# 2. التنسيق الجمالي
# ==============================================================
st.markdown("""
<style>
  #MainMenu, footer {visibility:hidden;}
  header[data-testid="stHeader"] {background:transparent;}
  .stApp {background: radial-gradient(1100px 520px at 12% -8%, #11304f 0%, #070a13 58%); color:#f1f5f9;}
  .block-container {max-width:1280px; padding-top:1.1rem; padding-bottom:3rem;}

  /* Hero */
  .hero {display:flex; align-items:center; gap:14px; margin:4px 0 14px 0;}
  .hero-logo {width:50px; height:50px; border-radius:14px; display:flex; align-items:center; justify-content:center;
    font-size:26px; background:linear-gradient(135deg,#0ea5e9,#6366f1); box-shadow:0 8px 24px rgba(14,165,233,.35);}
  .hero-title {font-size:1.7rem; font-weight:900; line-height:1.15; letter-spacing:.3px;
    background:linear-gradient(90deg,#f8fafc,#7dd3fc); -webkit-background-clip:text; -webkit-text-fill-color:transparent;}
  .hero-sub {font-size:.82rem; color:#94a3b8; letter-spacing:.4px;}

  /* Scoreboard */
  .sb-card {background:linear-gradient(135deg,rgba(15,23,42,.95),rgba(30,41,59,.9)); border:1px solid #334155; border-radius:18px;
    padding:20px 22px; margin-bottom:20px; box-shadow:0 14px 34px -10px rgba(0,0,0,.65);}
  .sb-meta {text-align:center; font-size:.74rem; color:#94a3b8; font-weight:700; letter-spacing:2px; text-transform:uppercase;}
  .sb-row {display:grid; grid-template-columns:1fr auto 1fr; align-items:center; gap:18px; margin:12px 0 10px 0; direction:ltr;}
  .sb-team {border-radius:12px; padding:14px 10px; text-align:center; font-weight:900; font-size:1.25rem; letter-spacing:3px;}
  .sb-home {background:linear-gradient(135deg,#1d4ed8,#3b82f6);} .sb-away {background:linear-gradient(135deg,#b91c1c,#ef4444);}
  .sb-score {font-size:3.4rem; font-weight:900; line-height:1; letter-spacing:6px; padding:0 8px; font-variant-numeric:tabular-nums;}
  .sb-goals {display:flex; justify-content:center; flex-wrap:wrap; gap:8px; margin-top:6px; direction:ltr;}
  .gchip {font-size:.82rem; font-weight:600; padding:4px 11px; border-radius:20px; border:1px solid;}
  .gchip.h {color:#93c5fd; border-color:rgba(59,130,246,.55); background:rgba(59,130,246,.12);}
  .gchip.a {color:#fca5a5; border-color:rgba(239,68,68,.55); background:rgba(239,68,68,.12);}

  /* KPI */
  .kpi {background:rgba(255,255,255,.035); border:1px solid rgba(255,255,255,.08); border-top:3px solid var(--c,#38bdf8);
    border-radius:14px; padding:14px 14px 12px 14px; height:100%;}
  .kpi-top {display:flex; align-items:center; justify-content:space-between;}
  .kpi-t {font-size:.72rem; color:#94a3b8; text-transform:uppercase; font-weight:800; letter-spacing:1px;}
  .kpi-i {font-size:1.25rem;}
  .kpi-v {font-size:1.9rem; font-weight:900; color:#fff; margin-top:4px; font-variant-numeric:tabular-nums;}
  .kpi-s {font-size:.76rem; color:var(--c,#38bdf8); margin-top:2px;}

  /* Section header */
  .sec {display:flex; align-items:center; gap:10px; margin:30px 0 4px 0; font-size:1.25rem; font-weight:800; color:#f8fafc;}
  .sec:before {content:""; width:5px; height:24px; border-radius:3px; background:linear-gradient(#38bdf8,#6366f1);}
  .sec-sub {font-size:.82rem; color:#94a3b8; margin:0 0 10px 15px;}

  /* AI studio */
  .ai-box {background:linear-gradient(90deg,rgba(14,165,233,.10),rgba(15,23,42,.65)); border:1px solid rgba(56,189,248,.18);
    border-inline-start:4px solid #0284c7; border-radius:14px; padding:18px 22px; margin:8px 0 6px 0;}
  .ai-badge {display:inline-block; background:#0284c7; color:#fff; font-size:.7rem; font-weight:800; padding:3px 10px; border-radius:5px; letter-spacing:1px;}
  .ai-text {font-size:1.02rem; color:#e2e8f0; line-height:1.75; margin-top:10px;}

  /* MOTM */
  .motm {background:linear-gradient(135deg,rgba(234,179,8,.10),rgba(15,23,42,.6)); border:1px solid rgba(234,179,8,.35);
    border-radius:16px; padding:18px 20px; margin-bottom:14px; display:flex; align-items:center; gap:16px; flex-wrap:wrap;}
  .motm-no {width:62px; height:62px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:1.7rem; font-weight:900;
    color:#1a1a1a; background:linear-gradient(135deg,#fde047,#ca8a04); box-shadow:0 0 0 4px rgba(234,179,8,.25);}
  .motm-badge {display:inline-block; background:linear-gradient(135deg,#eab308,#ca8a04); color:#000; font-weight:900; font-size:.7rem;
    padding:3px 11px; border-radius:20px; letter-spacing:1px;}
  .motm-name {font-size:1.6rem; font-weight:900; color:#f8fafc; line-height:1.2;}
  .motm-team {color:#94a3b8; font-size:.95rem;}
  [data-testid="stMetric"] {background:rgba(255,255,255,.035); border:1px solid rgba(255,255,255,.08); border-radius:12px; padding:10px 14px;}
  [data-testid="stMetricLabel"] p {color:#94a3b8 !important; font-size:.8rem !important;}
  [data-testid="stMetricValue"] {color:#f8fafc;}

  /* Leaderboard */
  table.lb {width:100%; border-collapse:separate; border-spacing:0 6px; direction:ltr;}
  table.lb th {font-size:.7rem; color:#94a3b8; text-transform:uppercase; letter-spacing:1px; text-align:left; padding:4px 12px; font-weight:800;}
  table.lb td {background:rgba(255,255,255,.04); padding:10px 12px; font-size:.92rem;}
  table.lb td:first-child {border-radius:10px 0 0 10px;} table.lb td:last-child {border-radius:0 10px 10px 0;}
  table.lb tr.top1 td {background:rgba(234,179,8,.12); border-top:1px solid rgba(234,179,8,.35); border-bottom:1px solid rgba(234,179,8,.35);}
  .rk {display:inline-flex; width:26px; height:26px; border-radius:50%; align-items:center; justify-content:center; font-weight:900; font-size:.8rem; background:#1e293b; color:#cbd5e1;}
  .top1 .rk {background:linear-gradient(135deg,#fde047,#ca8a04); color:#111;}
  .tbadge {font-size:.72rem; font-weight:800; padding:2px 9px; border-radius:12px;}
  .tbadge.h {background:rgba(59,130,246,.18); color:#93c5fd;} .tbadge.a {background:rgba(239,68,68,.18); color:#fca5a5;}

  /* Launchpad */
  .launch {background:linear-gradient(160deg,rgba(15,23,42,.85),rgba(15,23,42,.55)); border:1px solid #334155; border-top:3px solid var(--c);
    border-radius:16px; padding:20px 22px; min-height:150px; transition:all .2s ease; margin-bottom:8px;}
  .launch:hover {transform:translateY(-3px); border-color:var(--c); box-shadow:0 14px 30px -12px var(--c);}
  .launch-ic {width:48px; height:48px; border-radius:13px; display:flex; align-items:center; justify-content:center; font-size:1.6rem;
    background:color-mix(in srgb, var(--c) 18%, transparent); margin-bottom:12px;}
  .launch-t {font-size:1.12rem; font-weight:800; color:#fff;}
  .launch-d {font-size:.86rem; color:#94a3b8; line-height:1.55; margin-top:6px;}

  /* Buttons / segmented */
  .stButton>button {border-radius:11px; font-weight:700; border:1px solid #334155; background:#0f172a; color:#e2e8f0; transition:all .15s;}
  .stButton>button:hover {border-color:#38bdf8; color:#fff; background:#12223a;}
  .rtl {direction:rtl; text-align:right;}
</style>
""", unsafe_allow_html=True)


def H(html):
    st.markdown(html, unsafe_allow_html=True)


# ==============================================================
# 3. تحميل البيانات (حقيقية من الملفات)
# ==============================================================
@st.cache_data
def load_recap():
    if os.path.exists(RECAP_FILE):
        with open(RECAP_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def count_narratives():
    if os.path.exists(NARRATIVES_FILE):
        try:
            with open(NARRATIVES_FILE, "r", encoding="utf-8") as f:
                return len(json.load(f))
        except Exception:
            return 0
    return 0


@st.cache_data
def load_events():
    for fn in EVENTS_CANDIDATES:
        if os.path.exists(fn):
            df = pd.read_csv(fn)
            xs = [c for c in ("Start X", "End X") if c in df.columns]
            ys = [c for c in ("Start Y", "End Y") if c in df.columns]
            if xs and np.nanmax(df[xs].to_numpy(dtype=float)) <= 1.5:
                for c in xs:
                    df[c] = df[c] * PITCH_L
                for c in ys:
                    df[c] = (1.0 - df[c]) * PITCH_W
            return df
    return None


@st.cache_data(show_spinner=False)
def load_tracking_light(path, player_num=""):
    """قراءة خفيفة: الكرة + الحارس (لتحديد اتجاه الهجوم) + لاعب واحد اختياري."""
    with open(path, "r", encoding="utf-8") as f:
        _ = f.readline()
        row1 = f.readline().strip().split(",")
        _ = f.readline()
    ncols = len(row1)
    pcol = None
    for i in range(3, ncols - 2, 2):
        if str(row1[i]).strip() == str(player_num):
            pcol = i
            break
    use = {0, 1, 2, 3, 4, ncols - 2, ncols - 1}
    if pcol is not None:
        use |= {pcol, pcol + 1}
    df = pd.read_csv(path, skiprows=3, header=None, usecols=sorted(use), low_memory=False)
    df = df.apply(pd.to_numeric, errors="coerce")

    def sc(xi, yi):
        x, y = df[xi].to_numpy(dtype=float), df[yi].to_numpy(dtype=float)
        if np.nanmax(np.concatenate([x, y])) <= 1.5:
            x, y = x * PITCH_L, (1.0 - y) * PITCH_W
        return x, y

    bx, by = sc(ncols - 2, ncols - 1)
    gx, _gy = sc(3, 4)
    out = {"per": df[0].to_numpy(dtype=float), "t": df[2].to_numpy(dtype=float), "bx": bx, "by": by, "gkx": gx, "px": None, "py": None}
    if pcol is not None:
        out["px"], out["py"] = sc(pcol, pcol + 1)
    return out


def to_attack_right(x, y, per, gk_x):
    """نعكس اللقطات عشان الفريق يهاجم لليمين في الشوطين (حسب مكان الحارس)."""
    x, y = x.copy(), y.copy()
    for p in (1, 2):
        m = per == p
        if m.any() and np.isfinite(gk_x[m]).any():
            if np.nanmean(gk_x[m]) > PITCH_L / 2:        # بيدافع عن اليمين = بيهاجم لليسار
                x[m], y[m] = PITCH_L - x[m], PITCH_W - y[m]
    return x, y


def density(x, y, bins=(42, 28), sigma=1.5):
    ok = np.isfinite(x) & np.isfinite(y)
    Hh, _, _ = np.histogram2d(x[ok], y[ok], bins=bins, range=[[0, PITCH_L], [0, PITCH_W]])
    k = np.exp(-(np.arange(-4, 5) ** 2) / (2 * sigma ** 2)); k /= k.sum()
    Hh = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, Hh)
    Hh = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, Hh)
    if Hh.max() > 0:
        Hh = Hh / Hh.max()
    Hh[Hh < 0.04] = np.nan                                   # الأماكن الفاضية شفافة
    xc = (np.linspace(0, PITCH_L, bins[0] + 1)[:-1] + PITCH_L / bins[0] / 2)
    yc = (np.linspace(0, PITCH_W, bins[1] + 1)[:-1] + PITCH_W / bins[1] / 2)
    return Hh.T, xc, yc


def clock_of(t, per, p2start):
    if per == 2 and np.isfinite(p2start):
        t = 2700 + (t - p2start)
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


# ==============================================================
# 4. رسم الملعب (أرضية عشبية + خطوط بيضاء كاملة) بـ Plotly
# ==============================================================
HEAT_SCALE = [[0.0, "rgba(253,224,71,0.18)"], [0.35, "rgba(251,146,60,0.55)"],
              [0.7, "rgba(239,68,68,0.80)"], [1.0, "rgba(190,18,60,0.95)"]]


def _arc(cx, cy, r, a0, a1, n=30):
    a = np.linspace(a0, a1, n)
    return list(zip(cx + r * np.cos(a), cy + r * np.sin(a)))


def pitch_lines_trace():
    segs = [
        [(0, 0), (PITCH_L, 0), (PITCH_L, PITCH_W), (0, PITCH_W), (0, 0)],
        [(PITCH_L / 2, 0), (PITCH_L / 2, PITCH_W)],
        _arc(PITCH_L / 2, PITCH_W / 2, 9.15, 0, 2 * np.pi, 60),
    ]
    for xg, d in ((0, 1), (PITCH_L, -1)):
        segs.append([(xg, 13.84), (xg + d * 16.5, 13.84), (xg + d * 16.5, 54.16), (xg, 54.16)])
        segs.append([(xg, 24.84), (xg + d * 5.5, 24.84), (xg + d * 5.5, 43.16), (xg, 43.16)])
        cx = xg + d * 11
        pts = [p for p in _arc(cx, 34, 9.15, 0, 2 * np.pi, 90) if (p[0] - xg) * d > 16.5]
        if pts:                                              # قوس الجزاء (جزء واحد متصل)
            ang = sorted(pts, key=lambda p: np.arctan2(p[1] - 34, (p[0] - cx) * d))
            segs.append(ang)
        segs.append([(xg, 30.34), (xg - d * 2.44, 30.34), (xg - d * 2.44, 37.66), (xg, 37.66)])   # المرمى
    for cx, cy, sx, sy in ((0, 0, 1, 1), (PITCH_L, 0, -1, 1), (0, PITCH_W, 1, -1), (PITCH_L, PITCH_W, -1, -1)):
        segs.append([(cx + sx * np.cos(a), cy + sy * np.sin(a)) for a in np.linspace(0, np.pi / 2, 10)])
    xs, ys = [], []
    for s in segs:
        xs += [p[0] for p in s] + [None]
        ys += [p[1] for p in s] + [None]
    spots = go.Scatter(x=[PITCH_L / 2, 11, PITCH_L - 11], y=[34, 34, 34], mode="markers",
                       marker=dict(size=4, color="rgba(255,255,255,.9)"), hoverinfo="skip", showlegend=False)
    lines = go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="rgba(255,255,255,.88)", width=1.6),
                       hoverinfo="skip", showlegend=False)
    return lines, spots


def pitch_layout(fig, height):
    stripes = []
    for k in range(21):
        stripes.append(dict(type="rect", x0=k * 5, x1=min(PITCH_L, (k + 1) * 5), y0=0, y1=PITCH_W, layer="below",
                            line=dict(width=0), fillcolor="#1d5a2a" if k % 2 else "#195226"))
    fig.update_layout(
        shapes=stripes, height=height,
        xaxis=dict(range=[-4, PITCH_L + 4], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
        yaxis=dict(range=[-3, PITCH_W + 3], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True,
                   scaleanchor="x", scaleratio=1),
        plot_bgcolor="#0e2418", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=0, b=0),
        font=dict(color="#cbd5e1"), legend=dict(orientation="h", y=-0.04, x=0.5, xanchor="center", bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def show(fig):
    try:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    except TypeError:
        st.plotly_chart(fig, config={"displayModeBar": False})


def box():
    try:
        return st.container(border=True)
    except TypeError:
        return st.container()


def pick(options, key):
    if hasattr(st, "segmented_control"):
        return st.segmented_control("Perspective", options, default=options[0], label_visibility="collapsed", key=key) or options[0]
    return st.radio("Perspective", options, horizontal=True, label_visibility="collapsed", key=key)


# ==============================================================
# 5. الصفحة الرئيسية: Match Intelligence Recap
# ==============================================================
def render_main_recap():
    recap = load_recap()
    if not recap:
        st.error(f"ملف `{RECAP_FILE}` غير متوفر في المسار الحالي.")
        return

    sb, kpi = recap.get("scoreboard", {}), recap.get("kpis", {})
    radar, narrative = recap.get("radar_comparison", {}), recap.get("narrative", {})
    players = recap.get("top_impact_players", [])

    # ---- Hero + اللغة ----
    c_title, c_lang = st.columns([3.2, 1])
    with c_lang:
        lang = st.radio("Language / اللغة", ["English", "العربية"], horizontal=True, key="main_lang_toggle")
    is_ar = lang == "العربية"
    T = (lambda en, ar: ar if is_ar else en)
    with c_title:
        H(f"""<div class="hero"><div class="hero-logo">⚽</div><div>
            <div class="hero-title">{T("Match Intelligence Recap", "ملخص ذكاء المباراة")}</div>
            <div class="hero-sub">{T("AI-Powered Post-Match Executive Center • Spatial Analytics & Multimodal Synthesis",
                                     "مركز ما بعد المباراة بالذكاء الاصطناعي • تحليلات مكانية وسرد متعدد الوسائط")}</div>
        </div></div>""")

    # ---- لوحة النتيجة ----
    chips = "".join(
        f"<span class='gchip {'h' if str(g.get('team', '')).lower().startswith('h') else 'a'}'>⚽ {g.get('player', '')} · {g.get('clock', '')}</span>"
        for g in sb.get("goals", []))
    H(f"""<div class="sb-card">
        <div class="sb-meta">METRICA MATCH SYNTHESIS • FULL-TIME • {sb.get('match_duration_min', '–')} MIN ACTIVE</div>
        <div class="sb-row"><div class="sb-team sb-home">HOME</div>
            <div class="sb-score">{sb.get('home_score', 0)} - {sb.get('away_score', 0)}</div>
            <div class="sb-team sb-away">AWAY</div></div>
        <div class="sb-goals">{chips}</div></div>""")

    # ---- KPIs ----
    def kpi_card(col, icon, title, val, sub, color):
        with col:
            H(f"""<div class="kpi" style="--c:{color}"><div class="kpi-top"><div class="kpi-t">{title}</div><div class="kpi-i">{icon}</div></div>
                <div class="kpi-v">{val}</div><div class="kpi-s">{sub}</div></div>""")

    k1, k2, k3, k4 = st.columns(4)
    kpi_card(k1, "🏃", T("Total Distance", "إجمالي المسافة"), f"{kpi.get('total_distance_km', '–')} km",
             f"H: {kpi.get('home_distance_km', '–')} km  |  A: {kpi.get('away_distance_km', '–')} km", "#38bdf8")
    kpi_card(k2, "🗺️", T("Field Tilt (Final Third)", "السيطرة الميدانية (الثلث الأخير)"), f"{kpi.get('home_field_tilt', '–')}%",
             T("Home dominance vs Away", "هيمنة صاحب الأرض"), "#10b981")
    kpi_card(k3, "📐", T("Line-Breaking Passes", "تمريرات كسر الخطوط"), f"{kpi.get('total_line_breaks', '–')}",
             T("Vertical penetrations", "اختراقات رأسية"), "#a855f7")
    kpi_card(k4, "🛡️", T("Packing Index", "مؤشر الـ Packing"), f"{kpi.get('total_packing_events', '–')}",
             T("Defenders bypassed", "مدافعين تم تجاوزهم"), "#f59e0b")

    # ---- الاستوديو التحليلي ----
    H(f"<div class='sec'>🎙️ {T('AI Tactical Studio Narrative', 'الاستوديو التحليلي بالذكاء الاصطناعي')}</div>")
    opts = [T("Executive Match Narrative", "الموجز التكتيكي التنفيذي"),
            T("Decisive Turning Point", "نقطة التحول المفصلية"),
            T("Territorial & Progression Strategy", "استراتيجية السيطرة وكسر الخطوط")]
    chosen = pick(opts, key=f"persp_{'ar' if is_ar else 'en'}")
    idx = opts.index(chosen) if chosen in opts else 0
    tilt = kpi.get("home_field_tilt", "–")
    texts = [
        (narrative.get("summary_ar" if is_ar else "summary_en") or T("Analysis generated by the AI studio.", "تحليل مولّد بواسطة الاستوديو الذكي."), "MATCH OVERVIEW"),
        (narrative.get("turning_point_ar" if is_ar else "turning_point_en") or "–", "TURNING POINT"),
        (narrative.get("territorial_ar" if is_ar else "territorial_en") or (
            f"فرض الفريق صاحب الأرض هيمنة ميدانية بلغت {tilt}% على مجريات اللقاء، مستفيداً من كسر الخطوط المتكرر وتمركز اللعب في ثلث الخصم الأخير."
            if is_ar else
            f"Home engineered sustained territorial dominance ({tilt}% field tilt), pinning Away back and capitalizing on high-value line-breaking transitions."),
         "TACTICAL DRILLDOWN"),
    ]
    text_content, badge = texts[idx]
    H(f"""<div class="ai-box {'rtl' if is_ar else ''}"><span class="ai-badge">{badge}</span>
        <div class="ai-text">{text_content}</div></div>""")

    # ---- بيانات حقيقية: تتبع + أحداث ----
    events = load_events()
    have_tracking = os.path.exists(HOME_FILE)
    motm = players[-1] if players else None            # آخر عنصر = الأعلى تأثيراً
    motm_num = re.sub(r"\D", "", str(motm.get("player", ""))) if motm else ""
    motm_team = str(motm.get("team", "Home")) if motm else "Home"
    motm_file = AWAY_FILE if motm_team.lower().startswith("a") else HOME_FILE

    # ---- خريطة السيطرة الميدانية ----
    H(f"<div class='sec'>🗺️ {T('Spatial Territorial Dominance', 'خريطة السيطرة الميدانية والانتشار')}</div>")
    H(f"<div class='sec-sub'>{T('Real ball-position density, Home always attacking →  • Field tilt', 'كثافة مواقع الكرة الفعلية، وصاحب الأرض دايماً بيهاجم لليمين →  • السيطرة')} {tilt}%</div>")
    with box():
        try:
            if not have_tracking:
                raise FileNotFoundError(HOME_FILE)
            trk = load_tracking_light(HOME_FILE)
            bx, by = to_attack_right(trk["bx"][::5], trk["by"][::5], trk["per"][::5], trk["gkx"][::5])
            z, xc, yc = density(bx, by)
            fig = go.Figure()
            fig.add_trace(go.Heatmap(z=z, x=xc, y=yc, colorscale=HEAT_SCALE, showscale=False, zsmooth="best", hoverinfo="skip"))
            lines, spots = pitch_lines_trace()
            fig.add_trace(lines); fig.add_trace(spots)
            fig.add_trace(go.Scatter(x=[70, 70], y=[0, PITCH_W], mode="lines", hoverinfo="skip", showlegend=False,
                                     line=dict(color="rgba(250,204,21,.85)", width=2, dash="dash")))
            fig.add_annotation(x=87.5, y=PITCH_W - 3, text=T("FINAL THIRD", "الثلث الأخير"), showarrow=False,
                               font=dict(color="#fde047", size=12))
            show(pitch_layout(fig, 430))
        except Exception as e:
            st.info(T(f"Tracking data unavailable for the heatmap ({e}).", f"بيانات التتبع غير متاحة لرسم الخريطة ({e})."))

    # ---- رجل المباراة ----
    H(f"<div class='sec'>⭐ {T('Man of the Match', 'رجل المباراة')}</div>")
    if motm:
        pnum = motm_num or "?"
        stats = None
        if events is not None and motm_num:
            pat = rf"(?<!\d){motm_num}(?!\d)"
            me = events[(events["Team"].astype(str) == motm_team) & events["From"].astype(str).str.contains(pat, regex=True)]
            shots = me[me["Type"].astype(str).str.upper().str.contains("SHOT")]
            sub = shots["Subtype"].astype(str).str.upper()
            stats = dict(shots=len(shots), goals=int(sub.str.contains("GOAL").sum()),
                         on_target=int(sub.str.contains("ON TARGET|GOAL|SAVED", regex=True).sum()), df=shots)
        rating = motm.get("rating")
        H(f"""<div class="motm"><div class="motm-no">{pnum}</div><div style="flex:1">
            <span class="motm-badge">OFFICIAL MOTM</span>
            <div class="motm-name">{motm.get('player', '')} <span class="motm-team">({motm_team})</span></div></div>
            {f"<div style='font-size:1.15rem;font-weight:800;color:{GOLD}'>Rating {rating} / 10</div>" if rating else ""}</div>""")
        c_stats, c_map = st.columns([1, 2])
        with c_stats:
            if stats:
                g = stats["goals"]
                st.metric("⚽ " + T("Match Goals", "الأهداف"), f"{g}" + (" (Brace)" if g == 2 else " (Hat-trick)" if g >= 3 else ""))
                st.metric("🎯 " + T("Shots on Target", "التسديدات على المرمى"), f"{stats['on_target']} / {stats['shots']}")
            if motm.get("distance_m") is not None:
                st.metric("🏃 " + T("Distance Covered", "المسافة المقطوعة"), f"{float(motm['distance_m']) / 1000:.2f} km")
            if motm.get("packing_score") is not None:
                st.metric("⚡ " + T("Packing Score", "نقاط الـ Packing"), f"{motm['packing_score']} pts")
            if motm.get("involvements") is not None:
                st.metric("🔄 " + T("Touches / Actions", "اللمسات / التدخلات"), motm["involvements"])
        with c_map:
            with box():
                try:
                    trk = load_tracking_light(motm_file, motm_num)
                    if trk["px"] is None:
                        raise ValueError(f"Player {motm_num} not found")
                    px_, py_ = to_attack_right(trk["px"][::5], trk["py"][::5], trk["per"][::5], trk["gkx"][::5])
                    z, xc, yc = density(px_, py_, sigma=1.3)
                    fig = go.Figure()
                    fig.add_trace(go.Heatmap(z=z, x=xc, y=yc, colorscale=HEAT_SCALE, showscale=False, zsmooth="best", hoverinfo="skip"))
                    lines, spots = pitch_lines_trace()
                    fig.add_trace(lines); fig.add_trace(spots)
                    if stats is not None and len(stats["df"]):
                        gl = stats["df"][stats["df"]["Subtype"].astype(str).str.upper().str.contains("GOAL")]
                        if len(gl):
                            p2 = np.nanmin(trk["t"][trk["per"] == 2]) if (trk["per"] == 2).any() else np.nan
                            gx, gy, gt = [], [], []
                            for _, r in gl.iterrows():
                                end_x = r.get("End X", np.nan)
                                right = (end_x >= PITCH_L / 2) if pd.notna(end_x) else True
                                x0, y0 = float(r["Start X"]), float(r["Start Y"])
                                if not right:
                                    x0, y0 = PITCH_L - x0, PITCH_W - y0
                                gx.append(x0); gy.append(y0)
                                gt.append("⚽ " + clock_of(float(r["Start Time [s]"]) if "Start Time [s]" in r else float(r["Start Frame"]) / 25,
                                                           int(r["Period"]), p2))
                            fig.add_trace(go.Scatter(x=gx, y=gy, mode="markers+text", text=gt, textposition="top center",
                                                     textfont=dict(color="#fde047", size=12), showlegend=False,
                                                     marker=dict(size=17, color=GOLD, symbol="star", line=dict(color="#fff", width=1.5))))
                    show(pitch_layout(fig, 360))
                    st.caption(T(f"📍 Player {pnum} positional heatmap • real tracking data",
                                 f"📍 الخريطة الحرارية لتمركز اللاعب {pnum} • من بيانات التتبع الفعلية"))
                except Exception as e:
                    st.info(T(f"Player heatmap unavailable ({e}).", f"خريطة اللاعب غير متاحة ({e})."))

    # ---- التحليلات المعمقة ----
    H(f"<div class='sec'>📈 {T('Deep Tactical & Spatial Analytics', 'التحليلات التكتيكية والمكانية المعمقة')}</div>")
    col_radar, col_shots = st.columns(2)

    with col_radar:
        H(f"<div class='sec-sub'>📊 {T('5-Dimension Head-to-Head Radar', 'رادار المقارنة المباشرة (5 أبعاد)')}</div>")
        with box():
            if radar.get("metrics"):
                m, rh, ra = radar["metrics"], radar["home"], radar["away"]
                fig_r = go.Figure()
                fig_r.add_trace(go.Scatterpolar(r=rh + [rh[0]], theta=m + [m[0]], fill="toself", name="Home",
                                                line=dict(color=HOME_C, width=2.5), fillcolor="rgba(59,130,246,.28)"))
                fig_r.add_trace(go.Scatterpolar(r=ra + [ra[0]], theta=m + [m[0]], fill="toself", name="Away",
                                                line=dict(color=AWAY_C, width=2.5), fillcolor="rgba(239,68,68,.28)"))
                fig_r.update_layout(
                    polar=dict(bgcolor="rgba(255,255,255,.03)",
                               radialaxis=dict(visible=True, showticklabels=False, gridcolor="#334155", linecolor="#334155",
                                               range=[0, max(rh + ra) * 1.1]),
                               angularaxis=dict(gridcolor="#334155", linecolor="#334155", tickfont=dict(color="#e2e8f0", size=12))),
                    paper_bgcolor="rgba(0,0,0,0)", height=390, margin=dict(l=40, r=40, t=20, b=30),
                    font=dict(color="#cbd5e1"), legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"))
                show(fig_r)
            else:
                st.info(T("Radar data not available.", "بيانات الرادار غير متاحة."))

    with col_shots:
        n_h = n_a = 0
        fig_s = go.Figure()
        lines, spots = pitch_lines_trace()
        try:
            if events is None:
                raise FileNotFoundError("events file")
            sh = events[events["Type"].astype(str).str.upper().str.contains("SHOT")].copy()
            sub = sh["Subtype"].astype(str).str.upper()
            sh["goal"] = sub.str.contains("GOAL")
            sh["lab"] = np.where(sh["goal"], "Goal", np.where(sub.str.contains("SAVED"), "Saved",
                                 np.where(sub.str.contains("BLOCKED"), "Blocked", "Off target")))
            fig_s.add_trace(lines); fig_s.add_trace(spots)

            def norm(df, to_right):
                xs, ys = [], []
                for _, r in df.iterrows():
                    end_x = r.get("End X", np.nan)
                    right = (end_x >= PITCH_L / 2) if pd.notna(end_x) else (r["Start X"] >= PITCH_L / 2)
                    x0, y0 = float(r["Start X"]), float(r["Start Y"])
                    if right != to_right:
                        x0, y0 = PITCH_L - x0, PITCH_W - y0
                    xs.append(x0); ys.append(y0)
                return xs, ys

            for team, color, to_right in (("Home", HOME_C, True), ("Away", AWAY_C, False)):
                d = sh[(sh["Team"].astype(str) == team) & (~sh["goal"])]
                x_, y_ = norm(d, to_right)
                fig_s.add_trace(go.Scatter(x=x_, y=y_, mode="markers", name=f"{team} shots ({len(d) + int(sh[(sh['Team'].astype(str) == team) & sh['goal']].shape[0])})",
                                           text=[f"{r['From']} • {r['lab']}" for _, r in d.iterrows()], hoverinfo="text",
                                           marker=dict(size=11, color=color, opacity=.8, line=dict(color="#fff", width=1))))
            g = sh[sh["goal"]]
            gx, gy = [], []
            for team, to_right in (("Home", True), ("Away", False)):
                x_, y_ = norm(g[g["Team"].astype(str) == team], to_right)
                gx += x_; gy += y_
            fig_s.add_trace(go.Scatter(x=gx, y=gy, mode="markers", name=f"Goals ({len(g)})",
                                       text=[f"{r['From']} • Goal" for _, r in g.iterrows()], hoverinfo="text",
                                       marker=dict(size=18, color="#10b981", symbol="diamond", line=dict(color="#fff", width=1.6))))
            n_h = int((sh["Team"].astype(str) == "Home").sum()); n_a = int((sh["Team"].astype(str) == "Away").sum())
            H(f"<div class='sec-sub'>🎯 {T('Full Match Shot Map', 'خريطة تسديدات المباراة')} ({n_h} Home vs {n_a} Away)</div>")
            with box():
                show(pitch_layout(fig_s, 390))
        except Exception as e:
            H(f"<div class='sec-sub'>🎯 {T('Full Match Shot Map', 'خريطة تسديدات المباراة')}</div>")
            st.info(T(f"Shot map unavailable ({e}).", f"خريطة التسديدات غير متاحة ({e})."))

    # ---- أفضل 5 لاعبين تأثيراً ----
    if players:
        H(f"<div class='sec'>🏆 {T('Top 5 Impact Leaderboard', 'أفضل 5 لاعبين تأثيراً')}</div>")
        top5 = list(reversed(players))[:5]
        rows = ""
        for i, p in enumerate(top5, 1):
            tm = str(p.get("team", ""))
            cls = "h" if tm.lower().startswith("h") else "a"
            dist = f"{float(p['distance_m']) / 1000:.2f} km" if p.get("distance_m") is not None else "–"
            rows += (f"<tr class='{'top1' if i == 1 else ''}'><td><span class='rk'>{i}</span></td>"
                     f"<td><b>{p.get('player', '')}</b></td><td><span class='tbadge {cls}'>{tm}</span></td>"
                     f"<td>{dist}</td><td>{p.get('packing_score', '–')}</td><td>{p.get('involvements', '–')}</td></tr>")
        H(f"""<table class="lb"><tr><th>#</th><th>{T('Player', 'اللاعب')}</th><th>{T('Team', 'الفريق')}</th>
            <th>{T('Distance', 'المسافة')}</th><th>{T('Packing', 'Packing')}</th><th>{T('Actions', 'التدخلات')}</th></tr>{rows}</table>""")

    # ---- Launchpad ----
    H(f"<div class='sec'>🚀 {T('Specialized Experience Suites', 'منصات العرض والبث التخصصية')}</div>")
    H(f"<div class='sec-sub'>{T('Launch a dedicated module', 'اختار أي وحدة لتشغيلها')}</div>")
    n_moments = count_narratives()
    keys = list(MODULES.keys())
    for r in range(0, len(keys), 2):
        cols = st.columns(2)
        for c, key in zip(cols, keys[r:r + 2]):
            mod = MODULES[key]
            desc = T(mod["desc_en"], mod["desc_ar"])
            if key == "broadcast_view" and n_moments:
                desc += T(f" ({n_moments:,} analytical moments)", f" ({n_moments:,} لحظة تحليلية)")
            with c:
                H(f"""<div class="launch {'rtl' if is_ar else ''}" style="--c:{mod['color']}"><div class="launch-ic">{mod['icon']}</div>
                    <div class="launch-t">{T(mod['title_en'], mod['title_ar'])}</div><div class="launch-d">{desc}</div></div>""")
                if st.button(f"{T(mod['btn_en'], mod['btn_ar'])} ➔", key=f"btn_{key}", use_container_width=True):
                    st.session_state["current_view"] = key
                    st.rerun()


# ==============================================================
# 6. تشغيل الموديولات والرجوع للرئيسية
# ==============================================================
def render_sub_module(module_name):
    if st.button("⬅️ Back to Match Intelligence Center"):
        st.session_state["current_view"] = "home_recap"
        st.rerun()
    st.markdown("---")

    mod = MODULES.get(module_name)
    if not mod:
        st.error("Unknown module.")
        return
    path = next((f for f in mod["files"] if os.path.exists(f)), None)
    if path is None:
        st.error("File not found: " + " / ".join(mod["files"]))
        return

    # الموديولات بتنادي st.set_page_config من نفسها: نخليها no-op لأن الصفحة اتضبطت فوق
    original = st.set_page_config
    st.set_page_config = lambda *a, **k: None
    try:
        runpy.run_path(path, run_name="__main__")
    except Exception as e:
        st.error(f"Error executing {path}")
        st.exception(e)
    finally:
        st.set_page_config = original


# ==============================================================
# 7. الراوتر
# ==============================================================
if st.session_state["current_view"] == "home_recap":
    render_main_recap()
else:
    render_sub_module(st.session_state["current_view"])