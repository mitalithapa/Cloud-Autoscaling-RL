# Technical Report: Elastic Cloud Scaling for Small Models

## 1. Project Context & Motivation
**The Problem**: Deploying Large Language Models (LLMs) is expensive because they require massive, fixed GPU clusters (Static Deployment) to handle peak traffic, leading to 50-70% wasted idle capacity.
**The Solution**: Use specific, smaller models (SLMs) that can be dynamically scaled in and out (Elastic Deployment) based on demand.
**The Challenge**: Scaling too slowly causes "Latency Spikes" (SLA violations). Scaling too fast causes "Oscillation" (wasted costs).
**The Approach**: Replace simple "If-Then" scaling rules with a **Reinforcement Learning (RL)** agent that *learns* the optimal balance between Cost and Performance.

---

## 2. Algorithms & Theoretical Foundation

### A. Reinforcement Learning: Q-Learning
**Role**: The Brain (Decision Maker).
**Why Used**: Traditional rule-based scalers (e.g., auto-scale if CPU > 80%) are *reactive* and struggle with complex trade-offs. RL enables *proactive* decisions by optimizing for "Cumulative Future Reward".
**The Math (Bellman Equation)**:
$$Q_{new}(s,a) = Q_{old}(s,a) + \alpha \cdot [R + \gamma \cdot \max(Q(s', a')) - Q_{old}(s,a)]$$
- **Goal**: Learn $Q(s,a)$, which estimates "If I take action $a$ in state $s$, how much total reward will I get forever?"
- **State Space ($s$)**: `(Utilization, Capacity, Trend, Forecast)`
- **Action Space ($a$)**: `Scale Down`, `Hold`, `Scale Up`
- **Output**: The heatmap in the dashboard shows this learned "Policy". For examle, you see high Q-values for "Scale Up" when demand is rising, even if utilization is currently moderate.

### B. Environment Physics: M/M/c Queueing Theory
**Role**: The World (Simulation Physics).
**Why Used**: To realistically model "Latency". In cloud servers, latency doesn't increase linearly; it explodes asymptotically as utilization nears 100%.
**The Math**:
$$Request\_Latency \approx \frac{Base\_Latency}{1 - \rho} \quad \text{where } \rho = \frac{Demand}{1 - \rho}$$
**Impact**: This forces the agent to keep utilization in a "Goldilocks Zone" (40-80%). If it pushes to 99% to save money, latency skyrockets (SLA Violation), incurring a massive penalty ($-5.0$).

### C. Intelligence Layer: Forecasting (SMA/LSTM)
**Role**: The Crystal Ball (State Augmentation).
**Why Used**: Booting a VM takes time (e.g., 2 minutes). Exploring purely on current utilization is too late.
**Logic**:
- **Baseline**: Simple Moving Average (SMA) smooths noise.
- **Advanced**: LSTM/TCN detects temporal patterns (e.g., "Demand rises every morning at 9 AM").
**Integration**: The forecast output ("Rise Predicted") is fed directly into the Agent's State. This allows the Agent to learn specific policies like: *"If Forecast is Rising, Scale Up NOW, even if Utilization is low."*

---

## 3. Codebase Map: Where & Why?

| Component | File Path | Role & Explanation |
| :--- | :--- | :--- |
| **The Agent** | `agents/q_learning_agent.py` | Implements the **Q-Learning** update loop. It effectively builds a lookup table of wisdom. |
| **The Physics** | `env/cloud_autoscaling_env.py` | Implements the **Queueing Theory** math. It calculates `reward` and next `state` based on the physics of latency and capacity. |
| **The Forecaster** | `intelligence/forecasting.py` | Implements the **Prediction Models**. It peeks at the past trace window to predict the future, feeding this signal to the Agent. |
| **The Judge** | `simulation/dual_world.py` | The **Dual-World Engine**. It runs two environments simultaneously—one Static, one Elastic—feeding them the *exact same* random workload to ensure a fair "A/B Test". |
| **The Interpreter** | `explainability/forensics.py` | The **Forensics Engine**. It looks at the Q-Table values and generates human text. If `Q(Up) >> Q(Hold)`, it explains: *"The agent expects severe penalties if it doesn't scale up."* |
| **The Visualizer** | `dashboard/app.py` | The **Streamlit UI**. Connects the simulation loop to Plotly charts for real-time feedback. |

---

## 4. Understanding the Output Results

### Why does the Elastic (Blue) line wiggle?
This is the agent dynamically adjusting `Capacity` (supply) to match `Workload` (demand). The wiggles represent the agent saving money during demand dips (unlike the flat Red line).

### Why does the Agent sometimes Scale Up *before* the spike?
This is the **Forecasting** + **RL** at work. The Intelligence layer predicted the spike, the state changed to "Forecast: Rising", and the Q-Table said "Scale Up is the best action for this state" to avoid the future latency penalty.

### Why is Cost Savings ~40%?
The Static (Red) line must be set to `Peak Demand + Buffer` to be safe 100% of the time. The Elastic line only uses that capacity for the few minutes of peak traffic, running much leaner the rest of the day. The area between the Red and Blue lines is your saved money.
