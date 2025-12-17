import gymnasium as gym
from gymnasium import spaces
import numpy as np
from intelligence.forecasting import SimpleMovingAverage # Default forecaster

class CloudAutoscalingEnv(gym.Env):
    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 4}

    def __init__(self, workload_trace=None, forecaster=None):
        super(CloudAutoscalingEnv, self).__init__()
        
        # --- Configurable Parameters ---
        self.min_capacity = 1
        self.max_capacity = 20
        self.queue_capacity = 100 # Max requests in queue before drop
        self.base_latency = 0.05  # 50ms base latency
        self.sla_latency = 0.20   # 200ms SLA limit
        
        # Penalties/Rewards
        self.reward_success = 1.0     
        self.penalty_sla_violation = -5.0 # High latency or drops
        self.penalty_cost = -0.1      
        self.penalty_change = -0.2    
        self.penalty_waste = -0.5     

        # --- Workload & Intelligence ---
        self.full_trace = workload_trace if workload_trace is not None else np.zeros(100)
        self.forecaster = forecaster if forecaster else SimpleMovingAverage()
        
        self.current_step = 0
        self.max_steps = len(self.full_trace)

        # --- Dynamic State ---
        self.current_capacity = 5 # Start mid-range
        self.current_load = 0.0
        self.current_queue = 0.0
        
        # --- State Space ---
        # (Util_Bin, Cap_Bin, Trend_Bin, Forecast_Bin)
        # Forecast Bin: 0=Drop Predicted, 1=Stable, 2=Rise Predicted
        self.observation_space = spaces.Tuple((
            spaces.Discrete(5),            # Utilization
            spaces.Discrete(self.max_capacity + 1), # Capacity
            spaces.Discrete(3),            # Trend
            spaces.Discrete(3)             # Forecast Direction
        ))

        # --- Action Space ---
        # 0: Down, 1: Hold, 2: Up
        self.action_space = spaces.Discrete(3)
        
        self.history = {
            'demand': [],
            'capacity': [],
            'utilization': [],
            'latency': [],
            'queue': [],
            'dropped': [],
            'reward': [],
            'sla_violations': 0
        }

    def _get_obs(self):
        # 1. Utilization
        # Effective load = Incoming + Queue
        total_work = self.current_load + self.current_queue
        # Capacity processing power = capacity * 1.0 (normalized)
        # Utilization based on *processing* capability vs demand
        # Check instantaneous utilization
        if self.current_capacity == 0:
            util = 2.0
        else:
            util = self.current_load / self.current_capacity
        
        if util < 0.3: util_bin = 0
        elif util < 0.5: util_bin = 1
        elif util <= 0.8: util_bin = 2
        elif util <= 1.0: util_bin = 3
        else: util_bin = 4

        # 2. Capacity
        cap_bin = self.current_capacity

        # 3. Trend (Past)
        if self.current_step > 0:
            prev = self.full_trace[self.current_step - 1]
            diff = self.current_load - prev
            trend_bin = 2 if diff > 0.02 else (0 if diff < -0.02 else 1)
        else:
            trend_bin = 1

        # 4. Forecast (Future)
        # Predict next 5 steps
        history_window = self.full_trace[max(0, self.current_step-10):self.current_step+1]
        pred = self.forecaster.predict(history_window, steps=5)
        avg_pred = np.mean(pred)
        
        # Compare prediction to current
        if avg_pred > self.current_load * 1.05:
            forecast_bin = 2 # Rising
        elif avg_pred < self.current_load * 0.95:
            forecast_bin = 0 # Falling
        else:
            forecast_bin = 1 # Stable

        return (util_bin, cap_bin, trend_bin, forecast_bin)

    def _get_info(self):
        return {
            "load": self.current_load,
            "capacity": self.current_capacity,
            "utilization": self.current_load / max(1, self.current_capacity),
            "latency": self._calculate_latency(),
            "queue": self.current_queue,
            "dropped": 0 # updated in step
        }

    def _calculate_latency(self):
        # M/M/c queue approx or simple asymptote
        # Latency = Base / (1 - rho) where rho is util
        # Soft cap rho at 0.99 to avoid div zero
        rho = self.current_load / max(1, self.current_capacity)
        rho = min(rho, 0.99)
        return self.base_latency / (1 - rho)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options and 'trace' in options:
            self.full_trace = options['trace']
            self.max_steps = len(self.full_trace)
        
        self.current_step = 0
        self.current_capacity = 5
        self.current_load = self.full_trace[0] if len(self.full_trace) > 0 else 0
        self.current_queue = 0
        
        self.history = {k: [] for k in self.history}
        self.history['sla_violations'] = 0
        
        return self._get_obs(), self._get_info()

    def step(self, action):
        # 1. Action
        if action == 0: self.current_capacity = max(self.min_capacity, self.current_capacity - 1)
        elif action == 2: self.current_capacity = min(self.max_capacity, self.current_capacity + 1)
        
        # 2. Dynamics (Process Requests)
        # Incoming load
        incoming = self.full_trace[self.current_step] if self.current_step < len(self.full_trace) else 0
        
        # Processed
        processed = min(incoming + self.current_queue, self.current_capacity)
        
        # Remaining -> Queue
        new_queue = (incoming + self.current_queue) - processed
        
        # Drop logic
        dropped = max(0, new_queue - self.queue_capacity)
        self.current_queue = min(new_queue, self.queue_capacity)
        
        # Update observable load (incoming)
        self.current_load = incoming
        
        # Latency calc
        latency = self._calculate_latency()
        # Add queueing delay
        queue_delay = (self.current_queue / max(1, self.current_capacity)) * 0.05 # rough 50ms per item in queue/cap
        total_latency = latency + queue_delay

        # 3. Reward
        reward = 0
        
        # SLA Violations (Latency > 200ms OR Drops > 0)
        sla_violation = False
        if total_latency > self.sla_latency or dropped > 0:
            reward += self.penalty_sla_violation
            sla_violation = True
            self.history['sla_violations'] += 1
        else:
            reward += self.reward_success # Good service
            
        # Cost
        reward += self.penalty_cost * self.current_capacity
        
        # Waste
        util = incoming / max(1, self.current_capacity)
        if util < 0.3:
            reward += self.penalty_waste

        # 4. Step
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Log
        self.history['demand'].append(incoming)
        self.history['capacity'].append(self.current_capacity)
        self.history['utilization'].append(util)
        self.history['latency'].append(total_latency)
        self.history['queue'].append(self.current_queue)
        self.history['dropped'].append(dropped)
        self.history['reward'].append(reward)

        return self._get_obs(), reward, terminated, truncated, self._get_info()
