import numpy as np
from .workload_generator import WorkloadGenerator

class DataLoader:
    def __init__(self, trace_type='synthetic', **kwargs):
        """
        Args:
            trace_type (str): 'synthetic' or 'real' (not implemented yet).
            **kwargs: Arguments for WorkloadGenerator.
        """
        self.trace_type = trace_type
        self.generator = WorkloadGenerator(**kwargs)
        self.current_trace = None
    
    def get_new_trace(self, days=1):
        """
        Returns a new normalized demand trace.
        """
        if self.trace_type == 'synthetic':
            return self.generator.generate_full_trace(days=days)
        else:
            raise NotImplementedError("Real traces not yet implemented.")

    def get_batch_traces(self, num_traces=10, days=1):
        """
        Returns a batch of traces for training.
        """
        return np.array([self.get_new_trace(days) for _ in range(num_traces)])
