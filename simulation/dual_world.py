import numpy as np
import pandas as pd
from env.cloud_autoscaling_env import CloudAutoscalingEnv
from intelligence.forecasting import SimpleMovingAverage

class DualWorldSimulator:
    def __init__(self, trace, agent):
        self.trace = trace
        self.agent = agent
        
        # World A: Static
        self.env_static = CloudAutoscalingEnv(workload_trace=trace)
        
        # World B: Elastic
        self.env_elastic = CloudAutoscalingEnv(workload_trace=trace)
        
        # Configure Static World Capacity
        # Peak demand * 1.2 safety factor
        peak_demand = np.max(trace)
        # Assuming normalized trace 0-1 matches capacity units roughly? 
        # In env, cap=1 handles load=1 roughly?
        # Env logic: util = load / cap.
        # So to keep util < 0.8 (optimal max), cap should be >= load / 0.8.
        # Static Cap = Peak / 0.8
        static_cap = int(np.ceil(peak_demand / 0.7)) # Conservative static provisioning (aiming for 70% max util)
        static_cap = max(static_cap, 5) # Minimum baseline
        
        self.env_static.current_capacity = static_cap
        
    def run(self):
        steps = len(self.trace)
        
        results = []
        
        obs_elastic, _ = self.env_elastic.reset()
        _ = self.env_static.reset()
        # Force static capacity again after reset (reset defaults to 5)
        # We need to ensure reset doesn't overwrite if we want custom static.
        # Actually reset DOES reset capacity to 5.
        # We need to override it manually after reset.
        static_cap_target = self.env_static.current_capacity # Wait, I set it in __init__ but reset wipes it.
        # Recalculate
        peak_demand = np.max(self.trace)
        static_cap_target = int(np.ceil(peak_demand / 0.7))
        static_cap_target = max(static_cap_target, 5)
        self.env_static.current_capacity = static_cap_target
        
        elastic_sla = 0
        static_sla = 0
        
        for t in range(steps):
            # --- Elastic World Step ---
            action = self.agent.get_action(obs_elastic)
            next_obs, reward, done, _, info_elastic = self.env_elastic.step(action)
            obs_elastic = next_obs
            
            # --- Static World Step ---
            # Action: 1 (Hold) always, AND force capacity to stay static if logic drifts?
            # Env step logic: 0->-1, 2->+1. 1->Hold. 
            # So just sending 1 keeps capacity constant.
            _, _, _, _, info_static = self.env_static.step(1)
            
            # Aggregate metrics
            row = {
                'step': t,
                'demand': info_elastic['load'], # Same for both
                'static_capacity': info_static['capacity'],
                'elastic_capacity': info_elastic['capacity'],
                'static_util': info_static['utilization'],
                'elastic_util': info_elastic['utilization'],
                'static_latency': info_static['latency'],
                'elastic_latency': info_elastic['latency'],
                'static_queue': info_static['queue'],
                'elastic_queue': info_elastic['queue'],
                'static_dropped': info_static['dropped'],
                'elastic_dropped': info_elastic['dropped']
            }
            results.append(row)
            
            if info_static['dropped'] > 0 or info_static['latency'] > 0.2:
                static_sla += 1
            if info_elastic['dropped'] > 0 or info_elastic['latency'] > 0.2:
                elastic_sla += 1
                
        return pd.DataFrame(results), {'static_sla': static_sla, 'elastic_sla': elastic_sla}
