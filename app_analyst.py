
import os
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches

st.set_page_config(
    page_title="Tactical Video Room | Live 22-Player Tracking",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #0b0e14; color: #c9d1d9; }
    .hud-banner {
        padding: 12px 24px; border-radius: 8px; text-align: center;
        font-size: 1.2rem; font-weight: 800; margin-bottom: 12px; letter-spacing: 0.5px;
    }
    .hud-goal { background: linear-gradient(90deg, #1b4b27, #238636); color: #fff; border: 1px solid #3fb950; }
    .hud-saved { background: linear-gradient(90deg, #0d3868, #1f6feb); color: #fff; border: 1px solid #58a6ff; }
    .hud-blocked { background: linear-gradient(90deg, #593f05, #9e6a03); color: #fff; border: 1px solid #d29922; }
    .hud-out { background: #161b22; color: #8b949e; border: 1px solid #30363d; }
    .hud-offside { background: linear-gradient(90deg, #5a1e1e, #da3633); color: #fff; border: 1px solid #f85149; }
    .chain-box {
        background-color: #161b22; padding: 9px 16px; border-radius: 6px;
        border-left: 4px solid #58a6ff; font-size: 0.95rem; margin-bottom: 14px; color: #e6edf3;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
FIELD_LEN = 105.0
FIELD_WID = 68.0

HOME_FILE = "Sample_Game_1_RawTrackingData_Home_Team.csv"
AWAY_FILE = "Sample_Game_1_RawTrackingData_Away_Team.csv"
SEQUENCES_FILE = "elite_sequences_18.csv"
FALLBACK_TOP_FILE = "top_danger_moments.csv"
ANALYST_FILE = "top_analyst_events.json"
PASSES_FILE = "calculated_packing_passes.csv"
EVENTS_CANDIDATES = ["events.csv", "Sample_Game_1_RawEventsData.csv"]

HOME_COLOR, HOME_TXT = "#1f6feb", "#a5d6ff"
AWAY_COLOR, AWAY_TXT = "#da3633", "#ffa198"

# ------------------------------------------------------------------
# 1. Data loading
# ------------------------------------------------------------------
def clean_tracking_columns(df, prefix):
    cols = list(df.columns)
    new_cols = ["Period", "Frame", "Time [s]"]
    i = 3
    while i < len(cols) - 2:
        name = cols[i]
        new_cols.append(f"{prefix}_{name}_x")
        new_cols.append(f"{prefix}_{name}_y")
        i += 2
    new_cols += ["Ball_x", "Ball_y"]
    df.columns = new_cols
    return df

def tracking_to_centered(df):
    for c in df.columns:
        if c.endswith("_x"):
            df[c] = (df[c] - 0.5) * FIELD_LEN
        elif c.endswith("_y"):
            df[c] = -(df[c] - 0.5) * FIELD_WID
    return df

def xy_to_centered(df):
    df = df.copy()
    x_cols = [c for c in ("Start X", "End X") if c in df.columns]
    y_cols = [c for c in ("Start Y", "End Y") if c in df.columns]
    if not x_cols:
        return df
    max_x = pd.concat([df[c] for c in x_cols]).max()
    min_x = pd.concat([df[c] for c in x_cols]).min()
    if max_x <= 1.5:
        for c in x_cols:
            df[c] = (df[c] - 0.5) * FIELD_LEN
        for c in y_cols:
            df[c] = -(df[c] - 0.5) * FIELD_WID
    elif min_x >= 0 and max_x <= FIELD_LEN + 0.5:
        for c in x_cols:
            df[c] = df[c] - FIELD_LEN / 2
        for c in y_cols:
            df[c] = df[c] - FIELD_WID / 2
    return df

@st.cache_resource
def load_datasets():
    seq_file = SEQUENCES_FILE if os.path.exists(SEQUENCES_FILE) else FALLBACK_TOP_FILE
    missing = [f for f in (HOME_FILE, AWAY_FILE, seq_file, ANALYST_FILE) if not os.path.exists(f)]
    if missing:
        return {"error": missing}

    df_home = clean_tracking_columns(pd.read_csv(HOME_FILE, skiprows=2), "Home")
    df_away = clean_tracking_columns(pd.read_csv(AWAY_FILE, skiprows=2), "Away")
    df_home = tracking_to_centered(df_home).set_index("Frame", drop=False)
    df_away = tracking_to_centered(df_away).set_index("Frame", drop=False)

    df_top = pd.read_csv(seq_file)
    with open(ANALYST_FILE, "r", encoding="utf-8") as f:
        analyst_db = json.load(f)

    df_passes = None
    if os.path.exists(PASSES_FILE):
        df_passes = xy_to_centered(pd.read_csv(PASSES_FILE))

    df_events = None
    for fn in EVENTS_CANDIDATES:
        if os.path.exists(fn):
            df_events = pd.read_csv(fn)
            break

    return {"home": df_home, "away": df_away, "top": df_top,
            "analyst": analyst_db, "passes": df_passes, "events": df_events}

data = load_datasets()
if "error" in data:
    st.error("ملفات ناقصة في الفولدر: " + " , ".join(data["error"]))
    st.stop()

df_home, df_away = data["home"], data["away"]
df_top, analyst_db = data["top"], data["analyst"]
df_passes, df_events = data["passes"], data["events"]

HOME_X = [c for c in df_home.columns if c.endswith("_x") and c != "Ball_x"]
HOME_Y = [c for c in df_home.columns if c.endswith("_y") and c != "Ball_y"]
AWAY_X = [c for c in df_away.columns if c.endswith("_x") and c != "Ball_x"]
AWAY_Y = [c for c in df_away.columns if c.endswith("_y") and c != "Ball_y"]

MIN_FRAME = int(df_home["Frame"].min())
MAX_FRAME = int(df_home["Frame"].max())

def short_id(col):
    return col.split("_")[1].replace("Player", "")

def player_name(col):
    return col.split("_")[1]

# ------------------------------------------------------------------
# 2. Pitch drawing
# ------------------------------------------------------------------
def draw_pitch_lines(ax):
    line = "#3d4650"
    hl, hw = FIELD_LEN / 2, FIELD_WID / 2
    ax.plot([-hl, hl, hl, -hl, -hl], [-hw, -hw, hw, hw, -hw], color=line, lw=2)
    ax.plot([0, 0], [-hw, hw], color=line, lw=1.5)
    ax.add_patch(plt.Circle((0, 0), 9.15, color=line, fill=False, lw=1.5))
    ax.scatter([0], [0], color=line, s=12)
    for side in (-1, 1):
        bx = side * hl
        ax.plot([bx, bx - side * 16.5, bx - side * 16.5, bx],
                [-20.15, -20.15, 20.15, 20.15], color=line, lw=1.5)
        ax.plot([bx, bx - side * 5.5, bx - side * 5.5, bx],
                [-9.16, -9.16, 9.16, 9.16], color=line, lw=1.5)
        ax.scatter([bx - side * 11], [0], color=line, s=12)
        ax.plot([bx + side * 1.5, bx + side * 1.5], [-3.66, 3.66], color="#8b949e", lw=3)
        ax.add_patch(patches.Arc((bx - side * 11, 0), 18.3, 18.3,
                                 theta1=(127 if side == 1 else -53),
                                 theta2=(233 if side == 1 else 53),
                                 color=line, lw=1.5))

def get_row(df, frame):
    if frame in df.index:
        r = df.loc[frame]
        return r.iloc[0] if isinstance(r, pd.DataFrame) else r
    return None

def draw_pitch_frame(frame_num, event_frame, pass_info, focus_player, focus_team,
                     show_ids=True, show_trail=True, hide_team=None):
    fig, ax = plt.subplots(figsize=(12, 7.5))
    fig.patch.set_facecolor("#0b0e14")
    ax.set_facecolor("#121820")
    draw_pitch_lines(ax)

    if pass_info is not None and frame_num >= event_frame:
        sx, sy = float(pass_info["Start X"]), float(pass_info["Start Y"])
        ex, ey = float(pass_info["End X"]), float(pass_info["End Y"])
        dx, dy = ex - sx, ey - sy
        dist = np.hypot(dx, dy)
        if dist > 0:
            ux, uy = -dy / dist, dx / dist
            w = 4.0
            poly = [(sx + ux * w, sy + uy * w), (ex + ux * w, ey + uy * w),
                    (ex - ux * w, ey - uy * w), (sx - ux * w, sy - uy * w)]
            ax.add_patch(patches.Polygon(poly, closed=True, color="#388bfd", alpha=0.22, zorder=3))
            ax.plot([sx, ex], [sy, ey], color="#58a6ff", linestyle="--", lw=2.5, alpha=0.85, zorder=4)

    home_row = get_row(df_home, frame_num)
    away_row = get_row(df_away, frame_num)

    def plot_team(row, xs, ys, color, txt_color, team_name):
        if row is None or team_name == hide_team:
            return
        for xc, yc in zip(xs, ys):
            px, py = row[xc], row[yc]
            if pd.isna(px) or pd.isna(py):
                continue
            if team_name == focus_team and player_name(xc) == focus_player:
                ax.scatter(px, py, s=520, facecolors="none", edgecolors="#e3b341", lw=2.5, zorder=6)
            ax.scatter(px, py, color=color, s=190, edgecolors="white", lw=1.5, zorder=7)
            if show_ids:
                ax.text(px, py + 2.0, short_id(xc), color=txt_color, fontsize=8,
                        weight="bold", ha="center", zorder=8)

    plot_team(home_row, HOME_X, HOME_Y, HOME_COLOR, HOME_TXT, "Home")
    plot_team(away_row, AWAY_X, AWAY_Y, AWAY_COLOR, AWAY_TXT, "Away")

    if home_row is not None:
        if show_trail:
            lo = max(MIN_FRAME, frame_num - 20)
            trail = df_home.loc[lo:frame_num, ["Ball_x", "Ball_y"]].dropna()
            if len(trail) > 1:
                ax.plot(trail["Ball_x"], trail["Ball_y"], color="#e3b341", lw=2, alpha=0.5, zorder=8)
        bx, by = home_row["Ball_x"], home_row["Ball_y"]
        if not pd.isna(bx) and not pd.isna(by):
            ax.scatter(bx, by, color="#e3b341", s=120, edgecolors="white", lw=2, zorder=9)

    t = float(home_row["Time [s]"]) if home_row is not None else 0.0
    ax.text(0, FIELD_WID / 2 + 2.5, f"{int(t // 60):02d}:{int(t % 60):02d}   |   Frame {frame_num}",
            color="#8b949e", fontsize=11, ha="center", va="center")

    ax.set_xlim(-FIELD_LEN / 2 - 4, FIELD_LEN / 2 + 4)
    ax.set_ylim(-FIELD_WID / 2 - 4, FIELD_WID / 2 + 5)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig

# ------------------------------------------------------------------
# 3. Dynamic HUD banner & Clean Attack Chain
# ------------------------------------------------------------------
def resolve_hud_banner(selected_row, events):
    start_frame = int(selected_row.get("Key_Action_Frame", selected_row["Start Frame"]))
    end_frame = start_frame + 200
    team = str(selected_row.get("Team", ""))
    cat = str(selected_row.get("Danger_Category", ""))
    player = str(selected_row.get("From", ""))

    if events is not None and "Start Frame" in events.columns:
        sub = events[(events["Start Frame"] >= start_frame - 25) & (events["Start Frame"] <= end_frame)]
        
        # 1. البحث الصريح عن أهداف الشباك
        goals = sub[(sub["Type"].astype(str).str.upper().str.contains("GOAL")) | 
                    (sub["Subtype"].astype(str).str.upper().str.contains("GOAL"))]
        if not goals.empty:
            g_row = goals.iloc[0]
            g_team = str(g_row.get("Team", team))
            g_scorer = str(g_row.get("From", player))
            return f"⚽ GOAL! - هدف محقق لصالح {g_team} بواسطة اللاعب ({g_scorer})", "hud-goal"

        # 2. البحث عن تصديات الحارس
        saved = sub[sub["Subtype"].astype(str).str.upper().str.contains("SAVED")]
        if not saved.empty:
            return f"🧤 SAVED! - الحارس يتصدى ويمنع هدفاً محققاً لـ {team}", "hud-saved"

        # 3. اعتراض دفاعي
        blocked = sub[sub["Subtype"].astype(str).str.upper().str.contains("BLOCKED")]
        if not blocked.empty:
            return "🛡️ BLOCKED! - استبسال دفاعي يمنع تسديدة المرمى", "hud-blocked"

        # 4. تسلل
        offside = sub[(sub["Type"].astype(str).str.upper().str.contains("OFFSIDE")) | 
                      (sub["Subtype"].astype(str).str.upper().str.contains("OFFSIDE"))]
        if not offside.empty:
            return "🚩 OFFSIDE! - راية التسلل تحسم نهاية الهجمة", "hud-offside"

        # 5. ركنية أو خارج الملعب
        corner = sub[sub["Subtype"].astype(str).str.upper().str.contains("CORNER")]
        if not corner.empty:
            return "📐 CORNER KICK! - ركنية بعد إبعاد دفاعي", "hud-saved"

        out_ev = sub[sub["Type"].astype(str).str.upper().str.contains("OUT")]
        if not out_ev.empty:
            return "⚪ OUT! - الكرة تغادر خطوط الملعب", "hud-out"

    # فحص القيمة المخزنة إن لم نجد أحداثاً دقيقة
    if "Outcome_Desc" in selected_row and pd.notna(selected_row["Outcome_Desc"]):
        desc = str(selected_row["Outcome_Desc"])
        t_type = str(selected_row.get("Outcome_Type", ""))
        if "GOAL" in desc.upper() or "GOAL" in t_type:
            return f"⚽ GOAL! - {desc}", "hud-goal"
        elif "SAVED" in desc.upper() or "SAVED" in t_type:
            return f"🧤 SAVED! - {desc}", "hud-saved"
        elif "BLOCKED" in desc.upper():
            return f"🛡️ BLOCKED! - {desc}", "hud-blocked"

    if "Shot" in cat:
        return "🧤 SHOT CONCLUDED - تسديدة خطيرة تنتهي بتغطية حارس المرمى", "hud-saved"
    return "⚡ BREAKTHROUGH COMPLETED - اختراق ناجح للثلث الأخير", "hud-goal"

def get_clean_attack_chain(selected_row, events):
    key_frame = int(selected_row.get("Key_Action_Frame", selected_row["Start Frame"]))
    team = str(selected_row.get("Team", ""))
    player = str(selected_row.get("From", ""))
    cat = str(selected_row.get("Danger_Category", ""))

    if events is not None and "Start Frame" in events.columns:
        sub = events[(events["Start Frame"] >= key_frame - 150) & (events["Start Frame"] <= key_frame + 125)]
        sub_team = sub[sub["Team"] == team]
        steps = []
        for _, ev in sub_team.iterrows():
            t = str(ev.get("Type", "")).upper()
            st_ = str(ev.get("Subtype", "")).upper()
            p = str(ev.get("From", ""))
            
            if "RECOVERY" in t or "RECOVERY" in st_:
                steps.append(f"Recovery ({p})")
            elif "PASS" in t:
                steps.append(f"Pass ({p})")
            elif "SHOT" in t:
                if "GOAL" in st_ or "GOAL" in t:
                    steps.append(f"Goal! ({p})")
                else:
                    steps.append(f"Shot ({p})")

        if steps:
            # تنظيف التكرارات المتتالية لاسم اللاعب
            clean_steps = [steps[0]]
            for s in steps[1:]:
                if s != clean_steps[-1]:
                    clean_steps.append(s)
            return " ➔ ".join(clean_steps)

    return f"Build-up ➔ {cat} ({player}) ➔ Final Outcome"

# ------------------------------------------------------------------
# 4. UI
# ------------------------------------------------------------------
st.title("🎯 Tactical Video Room | الغرفة التكتيكية الحية")
st.caption("عرض حي لسلاسل الهجمات مع ممر كسر الخطوط (8 متر).   🔵 Home   |   🔴 Away")
st.divider()

st.sidebar.header("🕹️ لوحة اختيار الهجمات")
event_labels = []
for _, r in df_top.iterrows():
    m = int(r["Start Time [s]"] // 60)
    s = int(r["Start Time [s]"] % 60)
    event_labels.append(f"[{m:02d}:{s:02d}] {r['Danger_Category']} ({r['Team']})")

chosen_idx = st.sidebar.selectbox("اختر الهجمة التكتيكية:", range(len(event_labels)),
                                  format_func=lambda i: event_labels[i])
st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ الفلاتر")
show_ids = st.sidebar.checkbox("إظهار أرقام اللاعبين", value=True)
show_trail = st.sidebar.checkbox("إظهار مسار الكرة", value=True)
hide_opp = st.sidebar.checkbox(" إخفاء لاعبي الخصم", value=False)

speed_mode = st.sidebar.radio("سرعة التشغيل:",
                              options=["1.0x (سرعة طبيعية)", "0.5x (عرض تكتيكي متأنٍ)"], index=0)
if "1.0x" in speed_mode:
    play_step, frame_delay = 25, 0.04
else:
    play_step = 15
    frame_delay = 0.08

selected_row = df_top.iloc[chosen_idx]
# الفريم الأصلي لمطابقة تقرير الـ JSON
orig_frame = int(selected_row["Start Frame"])
event_key = f"frame_{orig_frame}"

# فريم اللقطة للرسم والمشغل
event_frame = int(selected_row.get("Key_Action_Frame", orig_frame))
report = analyst_db.get(event_key, {}).get("tactical_analysis", {})

pass_info = None
if df_passes is not None and "Start Frame" in df_passes.columns:
    p_sub = df_passes[df_passes["Start Frame"] == event_frame]
    if not p_sub.empty:
        pass_info = p_sub.iloc[0]

focus_team = str(selected_row.get("Team", ""))
focus_player = str(selected_row.get("From", ""))
opp_team = {"Home": "Away", "Away": "Home"}.get(focus_team)
hide_team = opp_team if hide_opp else None

banner_text, banner_style = resolve_hud_banner(selected_row, df_events)
st.markdown(f'<div class="hud-banner {banner_style}">{banner_text}</div>', unsafe_allow_html=True)

chain_text = get_clean_attack_chain(selected_row, df_events)
st.markdown(f'<div class="chain-box">🔗 <b>سلسلة بناء الهجمة:</b> {chain_text}</div>', unsafe_allow_html=True)

# تمديد نهاية السلسلة 5 ثوانٍ كاملة (125 فريم) بعد اللقطة
START_WINDOW = int(selected_row.get("Seq_Start_Frame", max(MIN_FRAME, event_frame - 125)))
raw_end = int(selected_row.get("Seq_End_Frame", event_frame + 125))
END_WINDOW = min(MAX_FRAME, raw_end + 125)

slider_key = f"frame_slider_{event_frame}"
PLAYING, POS, OWNER, PENDING = "playing", "play_pos", "play_owner", "pending_slider"

if slider_key not in st.session_state:
    st.session_state[slider_key] = START_WINDOW

pend = st.session_state.pop(PENDING, None)
if pend is not None:
    st.session_state[slider_key] = pend

if st.session_state.get(OWNER) != event_frame:
    st.session_state[PLAYING] = False
    st.session_state[OWNER] = event_frame
    st.session_state[POS] = st.session_state[slider_key]

if st.session_state.get(PLAYING):
    st.session_state[POS] = min(max(st.session_state[POS], START_WINDOW), END_WINDOW)
    st.session_state[slider_key] = st.session_state[POS]

def _play():
    cur = st.session_state[slider_key]
    if cur >= END_WINDOW:
        cur = START_WINDOW
    st.session_state[POS] = cur
    st.session_state[PLAYING] = True

def _pause():
    st.session_state[PLAYING] = False
    st.session_state[slider_key] = st.session_state[POS]

def _freeze():
    st.session_state[PLAYING] = False
    st.session_state[POS] = event_frame
    st.session_state[slider_key] = event_frame

col_pitch, col_tactics = st.columns([1.35, 1.0])

with col_pitch:
    st.subheader("🎥 مشغل الهجمة الميداني (Live Possession Sequence)")

    c1, c2, c3 = st.columns(3)
    c1.button("▶ تشغيل", use_container_width=True, on_click=_play)
    c2.button("⏹ إيقاف مؤقت", use_container_width=True, on_click=_pause)
    c3.button("⏸️ لحظة الحدث", use_container_width=True, on_click=_freeze)

    is_playing = bool(st.session_state.get(PLAYING))
    current_frame = st.slider(
        "⏱️ مؤشر فريمات الهجمة:",
        min_value=START_WINDOW, max_value=END_WINDOW, step=1,
        key=slider_key, disabled=is_playing,
    )

    pitch_box = st.empty()

    if is_playing:
        for f in range(st.session_state[POS], END_WINDOW + 1, play_step):
            st.session_state[POS] = f
            fig = draw_pitch_frame(f, event_frame, pass_info, focus_player, focus_team,
                                   show_ids, show_trail, hide_team)
            pitch_box.pyplot(fig)
            plt.close(fig)
            time.sleep(frame_delay)
        st.session_state[PLAYING] = False
        st.session_state[PENDING] = END_WINDOW
        st.session_state[POS] = END_WINDOW
        st.rerun()
    else:
        st.session_state[POS] = current_frame
        fig = draw_pitch_frame(current_frame, event_frame, pass_info, focus_player, focus_team,
                               show_ids, show_trail, hide_team)
        pitch_box.pyplot(fig)
        plt.close(fig)

    m1, m2, m3 = st.columns(3)
    m1.metric("الفريق المستحوذ", focus_team)
    m2.metric("اللاعب المنفذ", focus_player)
    dv = selected_row.get("Danger_Value", None)
    try:
        m3.metric("مؤشر الكسر التكتيكي", f"{float(dv):.0f} مدافعين")
    except (TypeError, ValueError):
        m3.metric("مؤشر الكسر التكتيكي", "-")

with col_tactics:
    st.subheader("📋 تقرير الأداء الفني والتكتيكي")
    if not report:
        st.warning(f"مفيش تقرير في ملف JSON للمفتاح: {event_key}")

    tab_ar, tab_en = st.tabs(["العربية 🇸🇦", "English 🇬🇧"])

    with tab_ar:
        st.markdown(f"**التحليل الفني للحدث:**\n\n{report.get('analysis_ar', 'لا يوجد تقرير متاح')}")
        st.markdown("---")
        st.info(f"🛡️ **الهيكل الدفاعي للخصم:**\n\n{report.get('defensive_structure_ar', '-')}")
        st.success(f"⚡ **تقييم الممر المستغل (8m Corridor):**\n\n{report.get('corridor_evaluation_ar', '-')}")
        st.warning(f"📈 **القيمة التكتيكية المضافة:**\n\n{report.get('tactical_value_ar', '-')}")

    with tab_en:
        st.markdown(f"**Technical Performance Analysis:**\n\n{report.get('analysis_en', 'No analysis available')}")
        st.markdown("---")
        st.info(f"🛡️ **Defensive Structure Evaluation:**\n\n{report.get('defensive_structure_en', '-')}")
        st.success(f"⚡ **8m Corridor Assessment:**\n\n{report.get('corridor_evaluation_en', '-')}")
        st.warning(f"📈 **Tactical Value Created:**\n\n{report.get('tactical_value_en', '-')}")