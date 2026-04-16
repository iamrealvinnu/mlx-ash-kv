import threading
import time
from typing import Any, Optional, List
from .cache import ASHCache

class AdaptiveSensitivity:
    """
    Autonomous Calibration Agent for ASH-KV.
    Adjusts the 'sensitivity' threshold based on historical AVD scores
    to balance model performance and integrity.
    """
    def __init__(self, initial_sensitivity: float = 0.85):
        self.sensitivity = initial_sensitivity
        self.history: List[float] = []
        self._lock = threading.Lock()
        
    def record_score(self, score: float):
        with self._lock:
            self.history.append(score)
            if len(self.history) > 50:
                self.history.pop(0)
            
            # Simple Adaptation Logic:
            # If the last 5 scores are consistently high but stable, 
            # we might be over-pruning. Relax the threshold.
            if len(self.history) >= 10:
                recent = self.history[-5:]
                avg_recent = sum(recent) / len(recent)
                variance = sum((x - avg_recent)**2 for x in recent) / len(recent)
                
                # If variance is low but score is high, the model is 'confidently drifting'
                # or we are in a high-complexity domain (like medical/legal).
                if avg_recent > 0.75 and variance < 0.01:
                    self.sensitivity = min(0.95, self.sensitivity + 0.01)
                elif avg_recent > 0.9: # Extreme drift
                    self.sensitivity = max(0.5, self.sensitivity - 0.05)

    @property
    def current_threshold(self) -> float:
        with self._lock:
            return self.sensitivity

def protect(model: Any, sensitivity: float = 0.85, critic_model_path: Optional[str] = None):
    """
    Wraps a model with ASH-KV protection.
    
    1. Initializes the ASHCache.
    2. Starts the Adaptive Sensitivity Agent.
    3. Starts the Asynchronous Verification Daemon (AVD).
    """
    cache = ASHCache(critic_model_path=critic_model_path)
    adapter = AdaptiveSensitivity(initial_sensitivity=sensitivity)
    
    def avd_daemon():
        print("[AVD] Asynchronous Verification Daemon Started.")
        while True:
            # Simulate analyzing recent chunks
            if cache.seq_len > 128:
                score = cache.analyze_manifold_chunk(max(0, cache.seq_len - 128))
                if score is not None:
                    adapter.record_score(score)
                    if score > adapter.current_threshold:
                        # Auto-flag drift if we cross the adaptive threshold
                        cache.flag_logical_drift(cache.seq_len - 1, severity_score=score)
            
            time.sleep(1.0)
            
    daemon_thread = threading.Thread(target=avd_daemon, daemon=True)
    daemon_thread.start()
    
    print(f"ASH-KV Protection Active (Adaptive Initial: {sensitivity})")
    return model, cache, adapter
