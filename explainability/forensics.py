import numpy as np
from env.cloud_autoscaling_env import CloudAutoscalingEnv

class DecisionForensics:
    def __init__(self, agent):
        self.agent = agent
        # Create a shadow env for simulations
        self.sim_env = CloudAutoscalingEnv()
        
    def analyze_step(self, interaction_data):
        """
        interaction_data: {state, action, q_values, ...}
        """
        state = interaction_data['state']
        chosen_action = interaction_data['action']
        q_values = interaction_data['q_values']
        
        # 1. Counterfactual Analysis based on Q-values
        # "If I acted differently, the expected long-term return would be X less"
        best_q = max(q_values)
        missed_values = [best_q - q for q in q_values]
        
        # 2. Heuristic "Why" based on State Components
        # State: (Util, Cap, Trend, Forecast)
        util_bin, cap_bin, trend_bin, forecast_bin = state
        
        narrative = ""
        
        # Context
        util_str = ["Very Low", "Low", "Optimal", "High", "Critical (SLA Violation)"][util_bin]
        trend_str = ["Decreasing", "Stable", "Increasing"][trend_bin]
        forecast_str = ["Drop Predicted", "Stable", "Rise Predicted"][forecast_bin]
        
        narrative += f"Context: Utilization was {util_str} with {trend_str} demand trend. "
        narrative += f"Forecast indicated demand would {forecast_str}.\n\n"
        
        # Decision
        action_names = ["Scale Down", "Hold", "Scale Up"]
        narrative += f"Decision: The agent chose to {action_names[chosen_action]}.\n\n"
        
        # Quantitative Justification (RL Logic)
        narrative += f"RL Agent Logic:\n"
        narrative += f"- Q-Value Maximization: The agent selected the action with the highest expected discounted return (Q={q_values[chosen_action]:.2f}).\n"
        narrative += f"- Value Difference: This action is expected to yield {abs(q_values[chosen_action] - min(q_values)):.2f} more cumulative reward than the worst alternative.\n\n"
        
        alternatives = []
        for a in range(3):
            if a != chosen_action:
                diff = q_values[chosen_action] - q_values[a]
                status = "lower"
                alternatives.append(f"{action_names[a]} had a Q-value of {q_values[a]:.2f} ({diff:.2f} {status})")
        
        narrative += f"- Alternative Values: {', '.join(alternatives)}.\n\n"
        
        # Paper-Aligned Auto-Explanation (Academic + RL)
        narrative += "Reinforcement Learning Interpretation:\n"
        
        if chosen_action == 2: # Up
            if util_bin >= 3: 
                narrative += f"SLA Penalty Avoidance: The agent learned that the immediate cost of scaling up is lower than the long-term penalty of an SLA violation ($R_{{SLA}} \\ll R_{{Cost}}$)."
            elif forecast_bin == 2:
                narrative += f"Proactive Value Estimation: The agent's Q-function $Q(s, a)$ has captured the temporal correlation between 'Rising Forecast' state and future demand spikes, triggering an early scale-up to maximize utility."
            elif trend_bin == 2:
                narrative += "Agent is acting to preserve future rewards by mitigating rising trend latency risks."
            else:
                narrative += "Exploratory or conservative scale up to ensure stability."
                
        elif chosen_action == 0: # Down
            if util_bin <= 1:
                narrative += f"Reward Maximization (Efficiency): With utilization low ({util_str}), the agent predicts that scaling down will increase the net reward by reducing the Cost Term ($R_{{Cost}}$) without triggering an SLA penalty."
            elif forecast_bin == 0:
                narrative += "Proactive downscaling: The agent anticipates a drop in demand, acting to minimize idle resource costs."
            else:
                narrative += "Aggressive cost-saving move identified as optimal policy."
                
        else: # Hold
            narrative += "Stability Optimization: The agent determined that the 'Change Penalty' ($R_{{Change}}$) combined with current optimality outweighs the marginal utility of scaling."

        return narrative

    def get_heatmap_data(self, fixed_trend=1, fixed_forecast=1):
        """
        Generates Q-Value grid for (Utilization x Capacity)
        """
        data = np.zeros((5, 21))
        for u in range(5):
            for c in range(21):
                state = (u, c, fixed_trend, fixed_forecast)
                # Max Q
                qs = [self.agent.get_q(state, a) for a in range(3)]
                data[u, c] = max(qs)
        return data
