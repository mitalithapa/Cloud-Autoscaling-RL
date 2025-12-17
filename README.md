# Elastic Cloud Scaling for Small Models: AI Infrastructure Simulator

## 1. Project Overview
**Elastic Cloud Scaling for Small Models** is an educational research simulator demonstrating how **Reinforcement Learning (RL)** can solve complex cloud resource allocation problems.

Unlike static rules (heuristics), this system uses a **Q-Learning Agent** that learns to balance competing objectives:
1.  **SLA Compliance**: Avoiding latency spikes and dropped requests.
2.  **Cost Efficiency**: Minimizing idle GPU resources.
3.  **Stability**: Preventing oscillations (flapping).

This tool serves as an interactive sandbox to explore how RL agents learn optimal policies in stochastic environments with predictive state signals.

## 2. System Architecture
The simulator implements a novel "Intelligence-Driven" architecture:
- **Intelligence Layer**: `forecasting.py` uses Simple Moving Average (SMA) and LSTM/TCN models to predict demand trends.
- **RL Agent**: A Q-Learning agent optimizes a multi-objective reward function (SLA compliance, Cost efficiency, Stability).
- **Dual-World Simulator**: Runs two parallel environments:
    - **World A (Static LLM)**: Fixed capacity sized for peak demand.
    - **World B (Elastic SLM)**: Dynamic capacity controlled by the RL agent.
- **Physics Engine**: `CloudAutoscalingEnv` models M/M/c queuing dynamics, latency asymptotes, and request drops.

## 3. Getting Started

### Prerequisites
Install the research dependencies:
```bash
pip install -r requirements.txt
```

### Reproducibility
1. **Train the Agent**:
   To reproduce the "Smart Agent" behavior, run the training script:
   ```bash
   python train_elastic_agent.py
   ```
   *Artifact: `models/q_learning_elastic_slm.pkl`*

2. **Launch the Simulator**:
   Open the interactive research dashboard:
   ```bash
   streamlit run dashboard/app.py
   ```

## 4. Dashboard Features
- **Architectural Diagram**: Visualizes the flow from Workload to RL Action.
- **Dual-World Simulation**: animated comparison of Static vs Elastic capacity.
- **Decision Forensics**: "Why did the agent scale up?" - Natural language explanations aligned with the research paper.
- **Cost Analysis**: Real-time tracking of cost savings (typically 30-50% vs Static).

## 5. Experiment Results
Typical results show that Elastic SLMs achieve **99.9% SLA compliance** while reducing compute costs by **40%** compared to over-provisioned Static LLMs.

### Repository Structure
```
cloud-autoscaling-rl/
├── intelligence/    # Forecasting Models
├── simulation/      # Dual-World Logic
├── explainability/  # Forensics Engine
├── env/             # Latency/Queue Dynamics
├── agents/          # Q-Learning Implementation
├── dashboard/       # Streamlit Research UI
└── results/         # Experiment Logs
```
