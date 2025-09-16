
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Poultry Digital Twin Demo", layout="wide")

# -----------------------------
# Helpers and core simulation
# -----------------------------
def clip(arr, lo, hi):
    return np.minimum(np.maximum(arr, lo), hi)

def compute_kpis(h, temp, ven):
    """Map environment series -> KPI proxies (illustrative)."""
    thi = 0.7*temp + 0.003*h*temp
    stress = 20 + 2.2*np.maximum(0, thi-25) + 0.04*np.maximum(0, 22-temp) + 0.02*(h-55)**2 - 0.1*ven
    stress = clip(stress, 0, 100)
    mortality = 2.0 + 0.02*(stress - 35) + 0.01*np.maximum(0, h-75) - 0.005*(ven-50)
    mortality = clip(mortality, 0.5, 8.0)
    stability = 100 - (np.abs(temp - 23)*3 + np.abs(h - 60)*0.6 + 0.2*np.abs(ven - 55))
    meat_quality = 60 + 0.15*stability - 0.25*(stress - 35)
    meat_quality = clip(meat_quality, 0, 100)
    fcr = 1.5 + 0.003*(stress - 35) + 0.002*np.abs(temp - 23) + 0.0008*np.abs(h - 60) - 0.0008*(ven - 55)
    fcr = clip(fcr, 1.3, 1.9)
    return mortality, meat_quality, stress, fcr

def daily_cycle(T, base, amp, phase=0, noise=0.0):
    t = np.arange(T)
    return base + amp*np.sin(2*np.pi*(t+phase)/24.0) + np.random.normal(0, noise, T)

def schedule_series(T, kind="A"):
    t = np.arange(T)
    if kind == "A":
        hum = clip(daily_cycle(T, 60, 8, phase=0, noise=1.0), 40, 85)
        tmp = clip(daily_cycle(T, 22, 5, phase=-4, noise=0.5), 18, 32)
        ven = clip(daily_cycle(T, 40, 10, phase=6, noise=1.5), 20, 100)
    elif kind == "B":
        hum = clip(daily_cycle(T, 58, 5, phase=0, noise=1.0), 40, 85)
        tmp = clip(daily_cycle(T, 23, 3, phase=-2, noise=0.5), 18, 32)
        ven = clip(daily_cycle(T, 55, 8, phase=6, noise=1.5), 20, 100)
    elif kind == "C":
        hum = clip(daily_cycle(T, 62, 10, phase=0, noise=1.2) + (np.random.rand(T) < 0.08)*10, 40, 95)
        tmp = clip(daily_cycle(T, 21, 6, phase=-5, noise=0.6) - (np.random.rand(T) < 0.05)*2, 16, 34)
        ven = clip(daily_cycle(T, 35, 12, phase=6, noise=2.0), 15, 100)
    else: # Custom baseline-ish
        hum = clip(daily_cycle(T, 60, 7, phase=0, noise=0.8), 40, 90)
        tmp = clip(daily_cycle(T, 22, 4, phase=-3, noise=0.4), 18, 32)
        ven = clip(daily_cycle(T, 45, 9, phase=6, noise=1.0), 20, 100)
    return hum, tmp, ven

def apply_policy_delta(h, t, v, dh=0.0, dt=0.0, dv=0.0):
    return clip(h+dh, 35, 95), clip(t+dt, 15, 35), clip(v+dv, 10, 100)

def summarize_kpis(kpis):
    mort, mq, stress, fcr = kpis
    return {
        "mortality_pct": float(np.mean(mort)),
        "meat_quality_idx": float(np.mean(mq)),
        "stress_idx": float(np.mean(stress)),
        "fcr": float(np.mean(fcr)),
    }

def uplift_table(summary_df):
    oriented = summary_df.copy()
    oriented["mortality_pct"] = -oriented["mortality_pct"]
    oriented["stress_idx"] = -oriented["stress_idx"]
    oriented["fcr"] = -oriented["fcr"]
    for col in ["mortality_pct", "meat_quality_idx", "stress_idx", "fcr"]:
        best = oriented[col].max()
        worst = oriented[col].min()
        denom = abs(best) + 1e-9
        oriented[col] = (oriented[col] - worst) / denom
    oriented = oriented.rename(columns={
        "mortality_pct":"Mortality (uplift)",
        "meat_quality_idx":"Meat Quality (uplift)",
        "stress_idx":"Stress (uplift)",
        "fcr":"FCR (uplift)"
    })
    return oriented

# Litter effects (synthetic)
LITTERS = [
    "Chopped straw (rapeseed)",
    "Wheat straw",
    "Wood shavings",
    "Sawdust",
    "Chopped flax",
    "Rice husks (hulls)",
    "Sunflower shells",
]

BASELINE_KPI = {"mortality_pct":3.5, "meat_quality_idx":70.0, "stress_idx":40.0, "fcr":1.60}
LITTER_EFFECT = {
    "Chopped straw (rapeseed)": {"mortality_pct": -0.3, "meat_quality_idx": +2.0, "stress_idx": -2.0, "fcr": -0.02},
    "Wheat straw":               {"mortality_pct": +0.2, "meat_quality_idx": -1.0, "stress_idx": +1.5, "fcr": +0.01},
    "Wood shavings":             {"mortality_pct": -0.4, "meat_quality_idx": +3.0, "stress_idx": -3.0, "fcr": -0.03},
    "Sawdust":                   {"mortality_pct": +0.4, "meat_quality_idx": -2.0, "stress_idx": +2.5, "fcr": +0.03},
    "Chopped flax":              {"mortality_pct": -0.2, "meat_quality_idx": +1.0, "stress_idx": -1.0, "fcr": -0.01},
    "Rice husks (hulls)":        {"mortality_pct": +0.1, "meat_quality_idx": -0.5, "stress_idx": +0.5, "fcr": +0.00},
    "Sunflower shells":          {"mortality_pct": -0.1, "meat_quality_idx": +0.5, "stress_idx": -0.5, "fcr": -0.005},
}

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("Poultry Digital Twin")
page = st.sidebar.radio("Go to", ["Dashboard", "Anomaly Monitor", "Litter Uplift (Causal)", "Simulator & Policies"])

# -----------------------------
# Dashboard (quick overview)
# -----------------------------
if page == "Dashboard":
    st.title("Poultry Digital Twin — Demo")
    st.write("This demo showcases three components:")
    st.markdown("""
    1. **Real-time anomaly detection** on multi-sensor streams (refreshes every ~10s).
    2. **Causal evaluation of litter types** on key KPIs with confounder adjustment.
    3. **Environment simulator & policy evaluator** to compare control schedules.
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Current KPI Benchmarks (illustrative)")
        st.json(BASELINE_KPI)
    with col2:
        st.subheader("Available Litters")
        st.write(", ".join(LITTERS))

# -----------------------------
# Anomaly Monitor (Streaming)
# -----------------------------
if page == "Anomaly Monitor":
    st.title("Real-time Anomaly Detection")
    st.caption("Auto-refresh every ~10 seconds. Synthetic stream for demo purposes.")

    refresh = st.sidebar.slider("Refresh interval (seconds)", 5, 30, 10)
    window = st.sidebar.slider("Rolling window (samples)", 24, 240, 96, step=12)
    z_thr = st.sidebar.slider("Z-score threshold", 1.0, 4.0, 2.0, step=0.1)
    inject_now = st.sidebar.button("Inject anomaly now")
    keep_points = st.sidebar.slider("Points to keep in view", 120, 720, 360, step=60)

    # Initialize session state stream
    if "stream_df" not in st.session_state:
        N0 = 180
        idx = pd.RangeIndex(N0)
        def seasonal_noise(base, amp, period, noise):
            t = np.arange(N0)
            return base + amp*np.sin(2*np.pi*t/period) + np.random.normal(0, noise, N0)
        st.session_state.stream_df = pd.DataFrame({
            "feed_water_ratio": seasonal_noise(1.7, 0.1, 48, 0.02),
            "temperature": seasonal_noise(23.0, 3.0, 96, 0.3),
            "humidity": seasonal_noise(60.0, 6.0, 96, 1.2),
            "draft": seasonal_noise(0.2, 0.1, 60, 0.02),
            "daily_weight_gain": seasonal_noise(55.0, 5.0, 96, 0.6),
            "mobility": seasonal_noise(85.0, 4.0, 96, 0.8),
            "audio_noise": seasonal_noise(30.0, 6.0, 96, 1.5),
        }, index=idx)
        st.session_state.last_update = time.time()

    # Update stream with a few new points each refresh
    now = time.time()
    if now - st.session_state.last_update > refresh:
        k = int((now - st.session_state.last_update) // refresh)
        for _ in range(max(1, k)):
            t = len(st.session_state.stream_df)
            new = st.session_state.stream_df.iloc[-1] + np.random.normal(0, st.session_state.stream_df.std()/50.0)
            st.session_state.stream_df.loc[t] = new.values
        st.session_state.last_update = now

    df = st.session_state.stream_df.copy()

    # Optional anomaly injection (fan failure + humidity spike etc.)
    if inject_now:
        start = max(len(df)-int(keep_points/3), 0)
        end = min(start + int(keep_points/4), len(df)-1)
        df.loc[start:end, "temperature"] -= 4.0
        df.loc[start:end, "humidity"] += 12.0
        df.loc[start:end, "draft"] -= 0.12
        df.loc[start:end, "audio_noise"] += 10.0
        df.loc[start:end, "daily_weight_gain"] -= 12.0
        df.loc[start:end, "mobility"] -= 10.0
        df.loc[start:end, "feed_water_ratio"] += 0.25
        st.session_state.stream_df = df

    # Compute anomaly score
    roll_mean = df.rolling(window, min_periods=12).mean()
    roll_std = df.rolling(window, min_periods=12).std().replace(0, 1e-6)
    z = (df - roll_mean) / roll_std
    anomaly_score = z.abs().mean(axis=1)
    thr = anomaly_score.mean() + z_thr*anomaly_score.std()

    view = df.tail(keep_points)
    score_view = anomaly_score.tail(keep_points)
    x = view.index.values

    # Plot multi-signal scaled lines
    scaled = (view - view.mean()) / (view.std() + 1e-9)
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    for col in scaled.columns:
        ax1.plot(x, scaled[col], label=col)
    ax1.set_xlabel("Time index")
    ax1.set_ylabel("Scaled Signal")
    ax1.set_title("Multi-Signal Monitoring")
    ax1.legend(ncol=3, fontsize=8)
    st.pyplot(fig1)

    # Plot anomaly score with threshold
    fig2, ax2 = plt.subplots(figsize=(12, 3.5))
    ax2.plot(score_view.index.values, score_view.values, label="Anomaly Score")
    ax2.axhline(thr, linestyle="--", label="Threshold")
    ax2.set_xlabel("Time index")
    ax2.set_ylabel("Score")
    ax2.set_title("Anomaly Score")
    ax2.legend()
    st.pyplot(fig2)

    st.info(f"Current anomaly score: {float(score_view.iloc[-1]):.3f} | Threshold: {float(thr):.3f}")

# -----------------------------
# Litter Uplift (Causal-adjusted)
# -----------------------------
if page == "Litter Uplift (Causal)":
    st.title("Causal Evaluation — Litter → KPI (with confounder adjustment)")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        avg_temp = st.slider("Avg Temperature (°C)", 18.0, 32.0, 23.0, 0.5)
    with c2:
        avg_hum = st.slider("Avg Humidity (%)", 40.0, 90.0, 60.0, 1.0)
    with c3:
        avg_ven = st.slider("Avg Ventilation (%)", 15.0, 100.0, 55.0, 1.0)
    with c4:
        confounder_strength = st.slider("Confounder Effect Strength", 0.0, 2.0, 0.7, 0.05)

    # Confounder adjustment factor: down-weight litter benefits when far from ideal env
    def confounder_factor(temp, hum, ven, strength=0.7):
        d = np.sqrt(((temp-23)/6.0)**2 + ((hum-60)/12.0)**2 + ((ven-55)/15.0)**2)
        return np.exp(-strength * d)

    adj = confounder_factor(avg_temp, avg_hum, avg_ven, confounder_strength)

    # Build oriented % improvement heatmap (higher=better)
    rows = []
    for l in LITTERS:
        eff = LITTER_EFFECT[l]
        mort = BASELINE_KPI["mortality_pct"] + eff["mortality_pct"]*adj
        mq   = BASELINE_KPI["meat_quality_idx"] + eff["meat_quality_idx"]*adj
        stress = BASELINE_KPI["stress_idx"] + eff["stress_idx"]*adj
        fcr  = BASELINE_KPI["fcr"] + eff["fcr"]*adj

        def pct_improve(value, base, higher_is_better=True):
            if higher_is_better:
                return (value - base) / base * 100.0
            else:
                return (base - value) / base * 100.0

        rows.append({
            "Litter": l,
            "Mortality": pct_improve(mort, BASELINE_KPI["mortality_pct"], higher_is_better=False),
            "Meat Quality": pct_improve(mq, BASELINE_KPI["meat_quality_idx"], higher_is_better=True),
            "Stress": pct_improve(stress, BASELINE_KPI["stress_idx"], higher_is_better=False),
            "FCR": pct_improve(fcr, BASELINE_KPI["fcr"], higher_is_better=False),
        })
    heat = pd.DataFrame(rows).set_index("Litter")

    # Heatmap
    data = heat.values
    fig, ax = plt.subplots(figsize=(10, 5.5))
    im = ax.imshow(data, aspect="auto")
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(heat.columns, rotation=0)
    ax.set_yticklabels(heat.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:+.1f}%", ha="center", va="center", fontsize=9)
    ax.set_title("Litter Effects on KPIs — Oriented % Improvement (confounder-adjusted)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Improvement vs Baseline (%)")
    st.pyplot(fig)

    st.dataframe(heat.round(2))

# -----------------------------
# Simulator & Policy Evaluator
# -----------------------------
if page == "Simulator & Policies":
    st.title("Environment Simulator & Policy Evaluator")

    T = st.sidebar.slider("Horizon (hours)", 48, 240, 96, step=24)

    # Baseline schedules
    schedules = {"Schedule A":"A", "Schedule B":"B", "Schedule C":"C"}
    cho = st.multiselect("Choose baseline schedules to compare", list(schedules.keys()), default=list(schedules.keys()))

    # Custom policy deltas
    st.subheader("Custom Policy (delta applied to a baseline-like cycle)")
    colA, colB, colC, colD = st.columns(4)
    with colA:
        dh = st.slider("Δ Humidity", -15.0, 15.0, 0.0, 0.5)
    with colB:
        dt = st.slider("Δ Temperature", -6.0, 6.0, 0.0, 0.5)
    with colC:
        dv = st.slider("Δ Ventilation", -20.0, 20.0, 0.0, 1.0)
    with colD:
        policy_name = st.text_input("Policy name", "Custom")

    add_policy = st.button("Add/Update Custom Policy")

    # Maintain saved policies in session state
    if "policies" not in st.session_state:
        st.session_state.policies = {}

    if add_policy:
        st.session_state.policies[policy_name] = {"dh":dh, "dt":dt, "dv":dv}

    # Show selected + any saved policies
    all_candidates = {k:v for k,v in schedules.items() if k in cho}
    for pname, deltas in st.session_state.policies.items():
        all_candidates[pname] = "CUSTOM"

    # Build series per schedule/policy
    series_map = {}
    for name, kind in all_candidates.items():
        base = schedule_series(T, "A" if kind=="CUSTOM" else schedules.get(name, "A"))
        if kind == "CUSTOM":
            d = st.session_state.policies[name]
            series_map[name] = apply_policy_delta(*base, dh=d["dh"], dt=d["dt"], dv=d["dv"])
        else:
            series_map[name] = base

    # Show environment cycles (3 separate line charts)
    x = np.arange(T)
    for metric_idx, metric_name in enumerate(["Humidity (%)", "Temperature (°C)", "Ventilation (%)"]):
        fig, ax = plt.subplots(figsize=(10, 4))
        for name, (h, t, v) in series_map.items():
            y = [h, t, v][metric_idx]
            ax.plot(x, y, label=name)
        ax.set_xlabel("Hour")
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name} Across Schedules/Policies")
        ax.legend(ncol=3, fontsize=8)
        st.pyplot(fig)

    # Compute KPI summaries & uplift
    summaries = []
    for name, (h, t, v) in series_map.items():
        kpis = compute_kpis(h, t, v)
        summaries.append({"schedule": name, **summarize_kpis(kpis)})
    summary_df = pd.DataFrame(summaries)
    uplift_df = uplift_table(summary_df.set_index("schedule")).reset_index()

    st.subheader("Policy Results")
    st.dataframe(summary_df.set_index("schedule").round(3))

    # Grouped bar: uplift per KPI per schedule
    melted = uplift_df.melt(id_vars=["schedule"], var_name="KPI", value_name="uplift")
    pivot = melted.pivot(index="KPI", columns="schedule", values="uplift")
    fig_u, ax_u = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax_u)
    ax_u.set_ylabel("Relative Uplift (higher is better)")
    ax_u.set_title("Simulated Uplift by Schedule/Policy and KPI")
    st.pyplot(fig_u)

    st.info("Tip: Save multiple custom policies and compare their KPI uplift vs the baselines.")
