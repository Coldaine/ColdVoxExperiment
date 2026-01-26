import time
import psutil
import os
from typing import Dict, Any
try:
    import jiwer
except ImportError:
    jiwer = None

def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate using jiwer."""
    if jiwer is None:
        return -1.0
    if not reference or not hypothesis:
        return 1.0 if reference != hypothesis else 0.0
    return jiwer.wer(reference, hypothesis)

class PerformanceTracker:
    """Tracks timing and memory usage during inference."""
    
    def __init__(self):
        self.start_time = 0
        self.end_time = 0
        self.process = psutil.Process(os.getpid())
        self.start_mem = 0
        self.peak_mem = 0

    def start(self):
        self.start_time = time.perf_counter()
        self.start_mem = self.process.memory_info().rss
        self.peak_mem = self.start_mem

    def update_peak_mem(self):
        current_mem = self.process.memory_info().rss
        if current_mem > self.peak_mem:
            self.peak_mem = current_mem

    def stop(self) -> Dict[str, float]:
        self.end_time = time.perf_counter()
        duration = self.end_time - self.start_time
        return {
            "duration": duration,
            "peak_ram_gb": self.peak_mem / (1024 ** 3),
            "ram_delta_gb": (self.peak_mem - self.start_mem) / (1024 ** 3)
        }
