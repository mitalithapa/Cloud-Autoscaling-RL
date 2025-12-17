import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.cloud_autoscaling_env import CloudAutoscalingEnv
from agents.q_learning_agent import QLearningAgent
from data.loader import DataLoader

def train_agent():
    print("Starting Agent Retraining for Elastic SLM Environment...")
    
    # 1. Setup Env with specific trace types (Training on diverse traces)
    loader = DataLoader()
    traces = [loader.get_new_trace(days=3) for _ in range(5)] # 5 variations
    
    env = CloudAutoscalingEnv(workload_trace=traces[0])
    
    # 2. Init Agent
    # New state space size is 4 tuples. 
    # Tuple((5, 21, 3, 3)) -> Size is huge if flattened? 
    # Agent uses dictionary Q-table, so sparse is fine.
    agent = QLearningAgent(env.action_space.n, epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.05)
    
    episodes = 200
    
    for ep in range(episodes):
        # Rotate traces
        trace = traces[ep % len(traces)]
        env = CloudAutoscalingEnv(workload_trace=trace) # Re-init to inject trace cleanly
        
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward
            
        agent.decay_epsilon()
        
        if (ep+1) % 20 == 0:
            print(f"Episode {ep+1}/{episodes} | Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f}")
            
    # Save
    os.makedirs("models", exist_ok=True)
    agent.save("models/q_learning_elastic_slm.pkl")
    print("Retraining Complete. Model saved to models/q_learning_elastic_slm.pkl")

if __name__ == "__main__":
    train_agent()
