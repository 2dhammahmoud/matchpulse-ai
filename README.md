<div align="center">

# ⚽ MatchPulse AI: End-to-End Multi-Modal Football Tactical Intelligence Platform

### Next-Gen Spatio-Temporal Tracking, Automated Possession Analytics & Generative TV Broadcast Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://matchpulse-ai.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](#)
[![Metrica Sports](https://img.shields.io/badge/Data-Metrica_Sports_Optical_Tracking-success.svg)](#)
[![LLM Orchestration](https://img.shields.io/badge/GenAI-Groq_API_•_Allam_•_Qwen-orange.svg)](#)

**[🚀 Launch Live Production Application on Streamlit Cloud](https://matchpulse-ai.streamlit.app/)**

</div>

---

## 📌 Executive Summary

**MatchPulse AI** is an enterprise-grade football tactical analytics platform that bridges raw tracking telemetry with generative broadcast engineering.

**It ingests:**

1. **High-Frequency Spatio-Temporal Tracking Data (25 FPS):** complete 22-player and ball tracking spanning **145,006 frames**.
2. **Discrete Event Stream:** **1,745+** in-match actions (passes, shots, recoveries, tackles, saves).

**It produces:**

3. **Generative Tactical LLM Reports (via Groq API):** automated bilingual tactical commentary (Arabic & English) synchronized to on-pitch events.
4. **Hybrid Client-Side Canvas Engines:** sub-sampled, quantized binary streams delivering 60 FPS in-browser visualizations without memory overhead.

---

## 🛠️ End-to-End Engineering & Tactical Data Pipeline

```text
                [Raw Metrica Tracking & Events]
                              │
                              ▼  (1. Ingestion & Pitch Normalization)
        [Metric Coordinates & 8m Corridor Packing Geometry]
                              │
                              ▼  (2. Attack Lifecycle & Temporal Debouncing)
        [16 Elite Attack Chains Catalog] ──► [Curated 11 Casual Fan Moments]
                    │                                   │
                    ▼  (3. Zero-Hardcoding               ▼  (4. Generative AI
                       Player Profiling)                    Commentary)
        [818 Player Actions Master DB]          [Allam-2-7b Tactical Feed]
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      │
                                      ▼  (5. Production Deployment)
          [4 Interactive Streamlit Suites + Post-Match Analytical Recap]
```

---

## Step 1: Spatial Geometry & Corridor Packing Engine

### Mathematical Formulation & Implementation

- **Metric Coordinate Normalization:** raw Metrica relative coordinates ($[0, 1]$) are mapped to standard FIFA pitch dimensions ($105\text{ m} \times 68\text{ m}$), centered at $(0, 0)$ such that $X \in [-52.5, +52.5]$ and $Y \in [-34.0, +34.0]$.
- **Dynamic 8 m Spatial Corridor:** line-breaking passes are evaluated using an 8-meter perpendicular corridor around the pass vector (**4 m orthogonal on each side**). Defenders situated inside this geometric envelope, between the start and end points of the pass, are quantified as **"Packed Defenders"**.

```text
                 [Start (x0, y0)]
                  /           \
   +4 m vector   /             \   -4 m vector
                /               \     Width = 8 meters
               /                 \
              /                   \
                 [End (x1, y1)]  ──►  Packed Defenders intersected
```

### Core Outputs

| File | Content |
|------|---------|
| `danger_moments.csv` | 156 dangerous moments identified |
| `calculated_packing_passes.csv` | 218 line-breaking passes breaking $\ge 1$ opponent line |
| `calculated_shots.csv` | 24 shots parameterized with trigonometric distance and goal-mouth angle |

### Engineering Challenge & Resolution

- **The Problem:** orientation inversion between halves. Teams switch sides at half-time, which caused line-breaking vectors in the second half to register in the reverse direction.
- **The Solution:** an automated geometric heuristic (`attacks_positive_x`) inspects the mean $X$ position of each team across the **first 50 frames of each half** and applies a dynamic coordinate rotation, standardizing attack direction upwards / forward.

---

## Step 2: Possession Lifecycle Reconstruction & Temporal Windowing

### Mathematical Formulation & Implementation

Instead of analyzing isolated events, an automated **Possession Lifecycle Tracing** algorithm was implemented:

- **Backward Trace (up to 15 seconds):** backtracks through the event stack to identify the origin of possession (e.g., `RECOVERY` or an intercepted free ball).
- **Forward Trace (up to 15 seconds):** follows the possession flow to capture terminal actions (`GOAL`, `SAVED`, `BLOCKED`, `OFF_TARGET`, `TURNOVER`).
- **Automated Chain String Generation (`Attack_Chain`):**

$$\text{Recovery (Player6)} \to \text{Pass (Player10)} \to \text{Shot} \to \text{Goal ⚽ (Player9)}$$

### Engineering Challenge & The Minute 66 Rebound Edge-Case

- **The Problem:**
  1. Identical attacks were generating duplicate rows across sub-second frames.
  2. At minute 66 (Frame `98958` / second `3958.32`), `Player4` executed a **5-man line-breaking pass** leading to `Player5`'s shot, a goalkeeper save, and a rebound finished by `Player9`. Standard discrete event filtering detached the line-breaking pass from the rebound goal, attributing the goal without recognizing its playmaker.
- **The Solution:**
  1. Applied an adaptive **4.5-second temporal debouncing window** to merge overlapping team events into unified sequence intervals (`Seq_Start_Frame` → `Seq_End_Frame`).
  2. Implemented strict **Tactical Reconciliation Logic**, so the pass by `Player4` retains primary tactical attribution while the terminal status is locked to `GOAL`:

     $$\text{Pass (Player8)} \to \text{Line-Break Pass (Player4)} \to \text{Shot (Player5)} \to \text{Goal ⚽ (Player9)}$$

  3. Reconciled the output across `filtered_elite_attacks.csv` and `elite_sequences_16_ready.csv` to exactly **16 verified elite sequences**.

---

## Step 3: Zero-Hardcoding Profiling & Curation

### Mathematical Formulation & Implementation

- **Casual Fan Curation** (`casual_elite_11_moments.csv` & `casual_fan_moments.json`): the 16 sequences were distilled into **11 highlight moments**:
  - 3 confirmed goals (`GOAL`)
  - 4 top goalkeeper saves (`SAVED`)
  - 4 high-value laser passes (line-breaking passes packing 3 to 5 defenders)
- **818-Action Player Master Extraction** (`all_players_master_events.csv`): all player events were extracted into a **4-tier functional taxonomy**: `ATTACK`, `GOALKEEPER`, `DEFENSE`, and `PLAYMAKING` / `POSITIONING`.
- **Dynamic Goalkeeper Inference:** rather than hardcoding jersey numbers, a geometric proximity model computed the mean distance to the defending goal line across tracking frames:
  - **Home GK:** automatically identified as `Player 11`
  - **Away GK:** automatically identified as `Player 25`
- **Sequential Possession Pass Accuracy:** Metrica raw events omit explicit pass-completion flags. A state-verification loop assesses the *subsequent* event: if possession remains with the passing team without an infraction, the pass is marked complete; if it is intercepted or goes out of bounds, it is marked incomplete.

$$\text{Pass Accuracy \%} = \frac{\text{Completed Passes}}{\text{Total Passes}} \times 100$$

- All metrics were synthesized into `players_unfiltered_raw_profiles.json`, followed by an automated extraction of the **top 3 highlights per role** into `players_spotlight_profiles.json`.

---

## Step 4: Generative LLM Broadcast Pipeline (Groq API)

### Mathematical Formulation & Implementation

Orchestrated the **Groq API** with the **`allam-2-7b`** model to process **1,358 active motion events**, transforming event metrics into lower-third broadcast tickers (`live_broadcast_narratives.json`), constrained to **under 12 words per language**:

```json
{
  "ticker_ar": "تمريرة طولية تخترق عمق دفاع الخصم وتصنع فرصة خطيرة",
  "ticker_en": "Precision long-range pass slicing through the defensive block"
}
```

### Engineering Challenge & Rate-Limit Management

- **The Problem:** batch-processing 1,358 continuous API completions frequently hit network interruptions and HTTP `429 Too Many Requests` rate limits.
- **The Solution:**
  - **Fault-Tolerant Resumption:** an automatic checkpointing loop reads existing entries from `live_broadcast_narratives.json` on startup and skips previously processed event indices.
  - **Incremental Disk Commits:** results are written to disk every 5 events (`SAVE_INTERVAL = 5`).
  - **Adaptive Exponential Backoff:** an automatic 15-second cooldown on rate limits, paired with a mandatory 1.5-second pacing delay between calls.

---

## Step 5: Post-Match Analytical Recap Engine

### Mathematical Formulation & Implementation

Multi-source metrics are aggregated into a unified post-match summary in `match_recap_data.json`:

- **Physical Workload & Speed Filtering:**

$$d = \sqrt{(\Delta x)^2 + (\Delta y)^2}$$

  Non-physical jumps ($d \ge 25\text{ m}$ per 5-frame step, equivalent to $> 50\text{ m/s}$) are filtered out before computing total team distances.

- **Pitch Territory Thirds:** ball presence is evaluated across all frames:

| Zone | Condition |
|------|-----------|
| Defensive Third | $X < 35\text{ m}$ |
| Middle Third | $35\text{ m} \le X \le 70\text{ m}$ |
| Attacking Third | $X > 70\text{ m}$ |

- **Field Tilt Percentage:**

$$\text{Field Tilt \%} = \frac{\text{Team Final Third Passes}}{\text{Total Final Third Passes}} \times 100$$

- **Tactical 5-Axis Radar Matrix:** direct cross-team comparison across **Passes, Line Breaks, Shots, Recoveries, and Final Third Tilt**.

---

## 🖥️ Interactive Application Architecture (Streamlit Suites)

```text
┌───────────────────────────────────────────────────────────────────────────────────┐
│                                MATCHPULSE AI SUITE                                │
├─────────────────────┬─────────────────────┬───────────────────┬───────────────────┤
│ 1. Analysis Suite   │ 2. Casual Fan Mode  │ 3. Spotlight Cam  │ 4. Broadcast Room │
│ 2D Pitch + Corridor │ Vertical Drone View │ Dynamic Follow    │ 90-Min Full Match │
│ Dual-Language LLM   │ Sync Commentary     │ Radar Minimap     │ TV Live Ticker    │
└─────────────────────┴─────────────────────┴───────────────────┴───────────────────┘
```

### 1. Tactical Video Room (Analysis Suite)

- Interactive 2D pitch tracking with adjustable playback speeds (**0.5x** and **1.0x**).
- Real-time rendering of the **8 m Spatial Corridor polygon**, displaying the exploited passing lanes.
- Dynamic HUD banner indicating the play resolution (`GOAL`, `SAVED`, `BLOCKED`, `OFFSIDE`, `OUT`) and the build-up chain.
- Bilingual technical evaluation tabs: *Defensive Structure*, *Corridor Assessment*, *Tactical Value*.

### 2. Casual Fan Mode (Drone Cam Experience)

- Top-down **vertical drone orientation**, so the attacking trajectory always flows upwards toward the goal ($Y \to 105\text{ m}$).
- Synchronized live-commentary ticker displaying play descriptions in real time.
- Modal dialog playback (`st.dialog`) with celebratory goal flashes and save overlays.
- Grid layout across curated categories: **Goals**, **Saves**, and **Laser Passes**.

### 3. Player Spotlight Tracker (Dynamic Follow Cam)

- Dynamic camera tracking that applies a **temporal moving-average filter** ($\text{LEAD} = 4\text{ m}$) to center on the designated player **without artificial zooming**.
- Pulsing target beacon ($\text{pulse} = 1 + 0.18 \times \sin(7t)$) and a directional locator triangle above the player.
- **Radar Minimap HUD** overlay displaying the pitch-wide positional distribution.
- **Off-screen ball pointer** showing an arrow and a ⚽ icon when play moves outside the camera bounds.
- Player profile card highlighting full-match statistics and the top 3 highlights per role.

### 4. Live Broadcast Studio (Full Match AI Broadcast)

- Continuous **90-minute full-match replay** powered by **Base64-encoded `int16` decimeter quantization**.
- **Dual-rate interpolation** (5 FPS for players, 12.5 FPS for the ball) rendered at **60 FPS** on an HTML5 Canvas.
- Dynamic TV scoreboard that increments the score at the exact goal timestamps, with goal banners and scorer tags.
- Stacked lower-third tactical ticker displaying LLM-generated tactical insights.
- **Teleportation engine** allowing instant jumps to kick-offs, goals, and key moments.

---

## 📂 Master Assets Matrix

| Asset Filename | Format | Primary Module | Engineering & Tactical Purpose |
|----------------|:------:|----------------|--------------------------------|
| `Sample_Game_1_RawTrackingData_Home_Team.csv` | CSV | Core Engine | Raw optical tracking coordinates for the Home team and ball (145,006 frames). |
| `Sample_Game_1_RawTrackingData_Away_Team.csv` | CSV | Core Engine | Raw optical tracking coordinates for the Away team (145,006 frames). |
| `Sample_Game_1_RawEventsData.csv` | CSV | Core Engine | Discrete in-match events stream (1,745 events). |
| `danger_moments.csv` | CSV | Data Pipeline | 156 high-danger transition moments. |
| `calculated_packing_passes.csv` | CSV | Tactical Analytics | 218 line-breaking passes with packed-defenders metrics ($\ge 1$). |
| `calculated_shots.csv` | CSV | Tactical Analytics | 24 shots parameterized with distance and goal-mouth angle. |
| `filtered_elite_attacks.csv` | CSV | Tactical Analytics | 16 debounced elite sequences with reconstructed attack chains. |
| `elite_sequences_16_ready.csv` | CSV | Tactical Analytics | Production-validated file of the 16 elite sequences. |
| `elite_attacks_catalog.json` | JSON | Data Pipeline | 3-layer structured JSON catalog of the 16 elite attacks. |
| `top_analyst_events.json` | JSON | Analysis Room | Bilingual LLM tactical evaluations across all 16 elite attacks. |
| `casual_elite_11_moments.csv` | CSV | Casual Fan Mode | Curated top 11 match moments (3 goals, 4 saves, 4 laser passes). |
| `casual_fan_moments.json` | JSON | Casual Fan Mode | Narrative cards and synchronized commentary steps for the 11 moments. |
| `all_players_master_events.csv` | CSV | Spotlight Suite | 818 individual actions categorized across all 22 players. |
| `players_unfiltered_raw_profiles.json` | JSON | Spotlight Suite | Full-match profiles with sequential pass accuracy and inferred roles. |
| `players_spotlight_profiles.json` | JSON | Spotlight Suite | Filtered player cards containing the top 3 moments per category. |
| `live_broadcast_narratives.json` | JSON | Broadcast Room | 1,358 LLM-generated TV lower-third tickers generated via Allam-2-7b. |
| `match_recap_data.json` | JSON | Recap Suite | Post-match summary, physical distance metrics, and 5-axis radar data. |

---

```text
                       ┌─────────────────────────────────────────────────────────┐
                       │          Metrica Sports Raw Tracking & Events           │
                       └────────────────────────────┬────────────────────────────┘
                                                    │
                                                    ▼
                       ┌─────────────────────────────────────────────────────────┐
                       │   Data Processing, Metric Conversion & Pitch Centering  │
                       └────────────────────────────┬────────────────────────────┘
                                                    │
            ┌───────────────────────────────────────┼───────────────────────────────────────┐
            │                                       │                                       │
            ▼                                       ▼                                       ▼
┌───────────────────────┐               ┌───────────────────────┐               ┌───────────────────────┐
│ Tactical Analysis     │               │ Casual Fan Drone-Cam  │               │ Player Spotlight      │
├───────────────────────┤               ├───────────────────────┤               ├───────────────────────┤
│ • elite_sequences_16  │               │ • casual_elite_11_    │               │ • all_players_master_ │
│   _ready.csv          │               │   moments.csv         │               │   events.csv          │
│ • top_analyst_events  │               │ • casual_fan_moments  │               │ • players_spotlight_  │
│   .json               │               │   .json               │               │   profiles.json       │
│ • calculated_packing_ │               │ • Dynamic Drone View  │               │ • Sliding Camera &    │
│   passes.csv          │               │   & Sync Commentary   │               │   Minimap HUD         │
└───────────────────────┘               └───────────────────────┘               └───────────────────────┘
                                                    │
                                                    ▼
                                ┌───────────────────────────────────────┐
                                │ Full Match Broadcast Suite            │
                                ├───────────────────────────────────────┤
                                │ • live_broadcast_narratives.json      │
                                │ • Binary Quantization (Base64)        │
                                │ • 60 FPS Multi-Rate Interpolation     │
                                │ • Live TV Scoreboard & Jump Engine    │
                                └───────────────────────────────────────┘
```
  
## ▶️ Run Locally

```bash
pip install streamlit pandas numpy plotly groq
streamlit run app.py
```

Place the data files from the **Master Assets Matrix** in the same folder as `app.py`.

---

<div align="center">

**Developed with precision for Football Data Analysts, Coaches, and Modern Fans.**

*MatchPulse AI © 2026 — Pioneering Spatio-Temporal Sports Analytics.*

</div>
