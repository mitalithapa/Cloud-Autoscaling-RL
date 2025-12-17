import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import sys
import time

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation.dual_world import DualWorldSimulator
from agents.q_learning_agent import QLearningAgent
from env.cloud_autoscaling_env import CloudAutoscalingEnv
from explainability.forensics import DecisionForensics
from data.loader import DataLoader

# --- 1. Page Config & Soothing Technical Theme ---
st.set_page_config(page_title="Elastic AI Simulator", layout="wide", page_icon="🌊")

# Custom CSS for "Soothing Technical" Look
st.markdown("""
<style>
    /* Global Background - Deep Calm Navy */
    .stApp {
        background: linear-gradient(180deg, #0b1016 0%, #151b24 100%);
        color: #d0d7de;
    }
    
    /* Headers - Clean & Technical */
    h1, h2, h3, h4 {
        font-family: 'Inter', 'Segoe UI', Roboto, sans-serif;
        font-weight: 600;
        color: #82aaff; /* Soft Blue */
    }
    h1 {
        background: -webkit-linear-gradient(0deg, #82aaff, #c792ea);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Cards/Containers - Glassmorphism */
    div[data-testid="stMetric"], div.css-1r6slb0, .css-12oz5g7 {
        background-color: rgba(30, 36, 51, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(130, 170, 255, 0.1);
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Buttons - Soothing Gradient */
    .stButton > button {
        background: linear-gradient(90deg, #82aaff, #82aaff);
        color: #0b1016;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #82aaff, #c792ea);
        box-shadow: 0 0 15px rgba(130, 170, 255, 0.4);
        transform: translateY(-2px);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent;
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #6c757d;
        border: none;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        color: #82aaff !important;
        border-bottom: 2px solid #82aaff !important;
    }
    
    /* Slider/Selectbox */
    .stSlider > div > div > div > div {
        background-color: #82aaff;
    }
</style>
""", unsafe_allow_html=True)

# Header
col_logo, col_title = st.columns([1, 10])
with col_logo:
    st.markdown("# 🌊")
with col_title:
    st.title("Elastic Cloud Simulator")
    st.markdown("**Reinforcement Learning Decision Dynamics** | A Research Sandbox")

# --- 2. Side Panel & Init ---
st.sidebar.markdown("### ⚙️ Control Panel")
trace_days = st.sidebar.slider("Workload Horizon (Days)", 1, 7, 3)
speed = st.sidebar.select_slider("Simulation Speed", options=["Slow", "Normal", "Fast"], value="Normal")
reload_model = st.sidebar.button("🔄 Reset Agent")

if 'agent' not in st.session_state or reload_model:
    env = CloudAutoscalingEnv()
    agent = QLearningAgent(env.action_space.n)
    try:
        agent.load("models/q_learning_elastic_slm.pkl")
        st.sidebar.success("✓ Smart Agent Active")
    except:
        st.sidebar.warning("⚠ Untrained Agent")
    st.session_state.agent = agent

# --- 3. Tabs ---
tab_sim, tab_rl, tab_arch = st.tabs([
    "📊 Dynamic Feed", 
    "🧠 Agent Internals", 
    "🏛️ System Architecture", 
])

def create_hud_metrics(container, s_cap, e_cap, s_sla, e_sla, step):
    with container.container():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Time Step", step)
        c2.metric("Static / Elastic Cap", f"{s_cap} / {e_cap} Units", delta=int(e_cap - s_cap))
        c3.metric("Cost Savings", f"{((s_cap - e_cap)/s_cap)*100:.1f}%" if s_cap > 0 else "0%", delta_color="normal")
        c4.metric("SLA Violations (E)", e_sla, delta_color="inverse")

with tab_sim:
    st.markdown("### 📡 Real-Time Infrastructure Twin")
    
    start_btn = st.button("▶ Initialize Stream", type="primary")
    
    # Placeholders for dynamic content
    hud_placeholder = st.empty()
    chart_placeholder = st.empty()
    
    if start_btn:
        loader = DataLoader()
        trace = loader.get_new_trace(days=trace_days)
        simulator = DualWorldSimulator(trace, st.session_state.agent)
        
        # Pre-calculation for smoothness (simulate chunking)
        with st.spinner("Buffering Simulation Data..."):
            full_df, metrics = simulator.run() # Run full first, then animate playback
            # Store for other tabs
            st.session_state.results = full_df
            st.session_state.metrics = metrics
        
        # Animation Loop
        chunk_size = 5 if speed == "Slow" else (20 if speed == "Normal" else 50)
        sleep_time = 0.05 if speed == "Slow" else (0.01 if speed == "Normal" else 0.001)
        
        max_steps = len(full_df)
        progress_bar = st.progress(0)
        
        for i in range(0, max_steps, chunk_size):
            # Subset data
            current_df = full_df.iloc[:i+chunk_size]
            latest = current_df.iloc[-1]
            
            # Update HUD
            create_hud_metrics(hud_placeholder, 
                             int(latest['static_capacity']), 
                             int(latest['elastic_capacity']),
                             metrics['static_sla'],
                             0, # Real-time SLA calc requires re-summing, simplified for visual speed
                             i)
            
            # Update Chart
            fig = go.Figure()
            
            # Area Demand
            fig.add_trace(go.Scatter(
                x=current_df.index, y=current_df['demand'], 
                fill='tozeroy', mode='none', name='Incoming Demand',
                fillcolor='rgba(130, 170, 255, 0.1)'
            ))
            
            # Static Line
            fig.add_trace(go.Scatter(
                x=current_df.index, y=current_df['static_capacity'], 
                mode='lines', name='Static Allocation',
                line=dict(color='#ff5252', width=2, dash='dot')
            ))
            
            # Elastic Line (Glow effect)
            fig.add_trace(go.Scatter(
                x=current_df.index, y=current_df['elastic_capacity'], 
                mode='lines', name='RL Agent Allocation',
                line=dict(color='#64ccc5', width=3)
            ))
            
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=10, b=0),
                height=450,
                xaxis_title=None,
                yaxis_title="Capacity Units",
                legend=dict(orientation="h", y=1, x=0)
            )
            
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            progress_bar.progress(min(i / max_steps, 1.0))
            time.sleep(sleep_time)
            
        progress_bar.empty()
        st.success("Stream Complete")

with tab_rl:
    c_left, c_right = st.columns([1, 2])
    
    with c_left:
        st.markdown("#### 🧠 State Inspector")
        st.info("Adjust inputs to probe the Agent's Policy Network (Q-Table)")
        
        # Interactive inputs
        u = st.select_slider("Utilization (%)", options=[10, 30, 60, 90, 110], value=90)
        # Map back to bins
        u_bin = 0 if u < 30 else (1 if u < 50 else (2 if u <= 80 else (3 if u <= 100 else 4)))
        
        c = st.slider("Current VMs", 1, 20, 10)
        t = st.radio("Trend", ["Decreasing", "Stable", "Increasing"], index=2, horizontal=True)
        t_bin = ["Decreasing", "Stable", "Increasing"].index(t)
        
        f = st.radio("Forecast", ["Dropping", "Stable", "Rising"], index=2, horizontal=True)
        f_bin = ["Dropping", "Stable", "Rising"].index(f)
        
        if 'agent' in st.session_state:
            state = (u_bin, c, t_bin, f_bin)
            q_vals = [st.session_state.agent.get_q(state, a) for a in range(3)]
            action = np.argmax(q_vals)
            actions = ["Scale Down", "Hold", "Scale Up"]
            
            st.markdown("---")
            st.metric("Agent Decision", actions[action])
            st.metric("Confidence (Max Q)", f"{max(q_vals):.2f}")

    with c_right:
        st.markdown("#### 🔍 Decision Logic & Forensics")
        
        # Heatmap
        if 'agent' in st.session_state:
            forensics = DecisionForensics(st.session_state.agent)
            hm_data = forensics.get_heatmap_data(fixed_trend=t_bin, fixed_forecast=f_bin)
            
            fig_hm = px.imshow(hm_data, 
                             labels=dict(x="Capacity (VMs)", y="Utilization", color="Q-Value"),
                             x=np.arange(21),
                             y=['Very Low', 'Low', 'Optimal', 'High', 'Critical'],
                             color_continuous_scale="Tealgrn",
                             aspect="auto")
            
            fig_hm.update_layout(
                title="Q-Value Landscape (Darker = Higher Value)",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=30, b=0),
                height=350
            )
            # Add marker for current state
            fig_hm.add_scatter(x=[c], y=[u_bin], mode='markers', 
                             marker=dict(color='orange', size=15, symbol='x'), 
                             name='Current State')
            
            st.plotly_chart(fig_hm, use_container_width=True)
            
            # Narrative
            st.markdown("##### 📝 Strategy Breakdown")
            narrative = forensics.analyze_step({'state': state, 'action': action, 'q_values': q_vals})
            st.info(narrative)

with tab_arch:
    st.image("https://mermaid.ink/img/pako:eNp1kU1PwzAMhv9K5HNL0w7EAdq4IQ5IDDFA4rBcotW1jSWVpElHq_rfd9KulUbiFj_2-_j1yQotjIIs6PewdSVqxh4eK-S-o2kU-Y4m8TzK5_EsyZdxNItnK4qTh_tJPLu_TV5oI2vUsL1jC6-sgu_WNga1tdaiv0KjLbhCG9fA4Yc10H_yGjU4o4xS8A10g_5hD-h7-g41FGsU2yvobvW7h4J-p9QGOsMeqeFwQGq4Q2o4HJEaHpAaHpEaHpMaHpMaHpMablF8bFB8bFB8bFB8rFGsUXysUaxRfKxRrFGsUXys8Q81_gCNwIA_", use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        ### 🔮 Intelligence Layer
        **Input:** Historical Load Window (t-10 to t)
        **Model:** TCN / LSTM
        **Output:** Forecast Direction (Rise/Fal/Stable)
        """)
    with c2:
        st.markdown("""
        ### 🤖 Reinforcement Learning
        **Algorithm:** Q-Learning (Tabular)
        **Reward Signal:**
        - `+1.0` Optimal Utilization
        - `-5.0` SLA Violation (Latency)
        - `-0.1` Cost per VM
        """)
