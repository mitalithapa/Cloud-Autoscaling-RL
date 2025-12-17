import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class WorkloadGenerator:
    def __init__(self, duration=1000, sample_rate=1):
        """
        Args:
            duration (int): Total duration of the workload in minutes/steps.
            sample_rate (int): Steps per minute.
        """
        self.duration = duration
        self.sample_rate = sample_rate
        self.time_steps = np.arange(0, duration, 1/sample_rate)

    def generate_sinusoidal(self, period=1440, amplitude=0.3, baseline=0.5, noise_level=0.05):
        """
        Generates a sinusoidal workload pattern (e.g., daily cycle).
        
        Args:
            period (int): Period of the sine wave (e.g., 1440 mins for daily).
            amplitude (float): Amplitude of oscillation.
            baseline (float): Baseline load level (0.0 to 1.0).
            noise_level (float): Std dev of Gaussian noise.
            
        Returns:
            np.array: Normalized workload trace.
        """
        # Basic Sine Wave
        workload = baseline + amplitude * np.sin(2 * np.pi * self.time_steps / period)
        
        # Add Noise
        noise = np.random.normal(0, noise_level, size=len(self.time_steps))
        workload += noise
        
        # Clip to [0, 1]
        return np.clip(workload, 0, 1)

    def add_spikes(self, workload, spike_prob=0.01, spike_magnitude=0.3):
        """
        Adds random spikes to the workload.
        """
        spikes = np.random.choice([0, spike_magnitude], size=len(workload), p=[1-spike_prob, spike_prob])
        return np.clip(workload + spikes, 0, 1)

    def generate_full_trace(self, days=1):
        """
        Generates a full synthetic trace including daily patterns and spikes.
        """
        duration = days * 1440
        self.time_steps = np.arange(duration)
        
        # Daily Pattern (24h = 1440m)
        trace = self.generate_sinusoidal(period=1440, amplitude=0.3, baseline=0.4, noise_level=0.05)
        
        # Add Spikes
        trace = self.add_spikes(trace, spike_prob=0.005, spike_magnitude=0.4)
        
        return trace

if __name__ == "__main__":
    # Test and Visualize
    gen = WorkloadGenerator()
    trace = gen.generate_full_trace(days=3)
    
    plt.figure(figsize=(12, 4))
    plt.plot(trace, label="Synthetic Workload", alpha=0.8)
    plt.title("Synthetic Cloud Workload Trace (3 Days)")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Normalized Demand (CPU)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
