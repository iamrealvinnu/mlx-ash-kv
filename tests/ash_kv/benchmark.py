"""
Diagnostic TUI and Integration Benchmark for ASH-KV (v4.1.0).

Provides a brutalist, high-density Terminal UI to visualize the 
Asynchronous Self-Healing Key-Value Cache in action.

Updates: Integrated Live ANE-Daemon verification (Hardware Parallelism).
"""

import mlx.core as mx
import time
import threading
import random
import math
import os
from collections import deque
from mlx_ash_kv.cache import ASHCache

from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.console import Group
from rich.align import Align

# Configuration: Typical Transformer Head Dimensions
BATCH_SIZE = 1
NUM_HEADS = 32
HEAD_DIM = 128
TARGET_TOKENS = 3000
ANE_MODEL_PATH = "models/mock_critic.mlpackage"

class DiagnosticMonitor:
    def __init__(self):
        # Initialize ASHCache with the ANE model if it exists
        model_exists = os.path.exists(ANE_MODEL_PATH)
        self.cache = ASHCache(critic_model_path=ANE_MODEL_PATH if model_exists else None)
        self.current_token = 0
        self.excision_log = deque(maxlen=20)
        self.start_time = time.time()
        self.is_running = True
        self.ane_active = model_exists
        self.last_ane_score = 0.0
        
    def generate_barcode(self) -> Group:
        """Renders the attention manifold as a dense 1D spectral barcode with Gaussian gradients."""
        blocks = 120  # Console width target
        tokens_per_block = TARGET_TOKENS / blocks
        
        barcode_lines = []
        
        for _ in range(5):
            line = Text()
            for i in range(blocks):
                block_start = i * tokens_per_block
                
                if block_start > self.current_token:
                    line.append(" ", style="on black")
                    continue
                
                if block_start == 0:
                    line.append("█", style="bold green") # BOS Sink
                    continue

                max_penalty = 0.0
                for strike in self.cache.strikes:
                    mu = strike["index"]
                    sigma = strike["sigma"]
                    if block_start >= mu:
                        dist_sq = (block_start - mu)**2
                        penalty = math.exp(-dist_sq / (2 * sigma**2))
                        max_penalty = max(max_penalty, penalty)
                
                if max_penalty > 0.95:
                    line.append(" ", style="on black") # Full excision
                elif max_penalty > 0.1:
                    shade = int(255 * (1.0 - max_penalty))
                    line.append("█", style=f"rgb({shade},{shade},{shade})")
                else:
                    intensity = random.choice(["bold white", "white", "bright_white"])
                    line.append("█", style=intensity)
            barcode_lines.append(Align.center(line))
            
        return Group(
            Text("\n\n"),
            *barcode_lines,
            Text("\n[ 1D SPECTRAL ATTENTION MANIFOLD // ANE-VERIFICATION ACTIVE ]", style="bold white", justify="center")
        )

    def generate_telemetry(self) -> Table:
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="bold white")
        table.add_column("Value", style="white")
        
        elapsed = time.time() - self.start_time
        tps = self.current_token / elapsed if elapsed > 0 else 0
        mem_mb = (BATCH_SIZE * NUM_HEADS * self.current_token * HEAD_DIM * 2) / (1024 * 1024)
        
        table.add_row("KV Tensor Shape", f"[{BATCH_SIZE}, {NUM_HEADS}, {self.current_token}, {HEAD_DIM}]")
        table.add_row("Manifold Depth", f"{self.current_token} / {TARGET_TOKENS} tokens")
        table.add_row("Throughput", f"{tps:.2f} tok/s")
        table.add_row("Unified Memory", f"{mem_mb:.2f} MB Allocated")
        table.add_row("Verification Engine", "Apple Neural Engine (ANE)" if self.ane_active else "Python Thread (Fallback)")
        table.add_row("ANE Critic Score", f"{self.last_ane_score:.4f}")
        table.add_row("Hardware Isolation", "[bold green]ENABLED[/]" if self.ane_active else "[bold yellow]NONE[/]")
        
        return table

    def generate_logs(self) -> Group:
        log_texts = [Text.from_markup(log) for log in self.excision_log]
        return Group(*log_texts)

    def make_layout(self) -> Layout:
        layout = Layout(name="root")
        layout.split(
            Layout(name="header", size=3),
            Layout(name="main")
        )
        layout["main"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=2)
        )
        layout["left"].split_column(
            Layout(name="telemetry", size=10),
            Layout(name="logs")
        )
        
        header_text = Text(" ASH-KV V4.1.0 // ANE-DAEMON HARDWARE MONITOR ", style="bold black on green", justify="center")
        layout["header"].update(Panel(header_text, style="green"))
        
        layout["left"]["telemetry"].update(Panel(self.generate_telemetry(), title="[bold]HARDWARE TELEMETRY[/]", border_style="green"))
        layout["left"]["logs"].update(Panel(self.generate_logs(), title="[bold]ANE GHOST CRITIC LOGS[/]", border_style="green"))
        layout["right"].update(Panel(self.generate_barcode(), title="[bold]TENSOR MANIFOLD STATUS[/]", border_style="green"))
        
        return layout

def primary_generation_loop(monitor: DiagnosticMonitor):
    for i in range(1, TARGET_TOKENS + 1):
        if not monitor.is_running:
            break
            
        new_k = mx.random.uniform(shape=(BATCH_SIZE, NUM_HEADS, 1, HEAD_DIM), dtype=mx.float16)
        new_v = mx.random.uniform(shape=(BATCH_SIZE, NUM_HEADS, 1, HEAD_DIM), dtype=mx.float16)
        
        k, v, mask = monitor.cache.update(new_k, new_v)
        
        if i % 100 == 0:
            # Route evaluation through the sync gate to prevent Metal buffer collisions
            monitor.cache.sync_eval(k, v, mask)
            
        monitor.current_token = i
        time.sleep(0.003)
        
    monitor.is_running = False

def ghost_critic_loop(monitor: DiagnosticMonitor):
    # Analyzing in 128-token chunks as defined in the Tracer Bullet model
    chunk_size = 128
    last_analyzed_idx = 0
    
    while monitor.is_running and monitor.current_token < TARGET_TOKENS:
        time.sleep(0.2)
        
        current_len = monitor.cache.seq_len
        # Only analyze if we have a new full chunk
        if current_len >= last_analyzed_idx + chunk_size:
            start_idx = last_analyzed_idx
            
            # Dispatch to ANE
            severity = monitor.cache.analyze_manifold_chunk(start_idx, chunk_size)
            
            if severity is not None:
                monitor.last_ane_score = severity
                
                # Strike condition: Simulated logical drift threshold
                if severity > 0.8:
                    strike_idx = start_idx + random.randint(0, chunk_size - 1)
                    if strike_idx > 0:
                        monitor.cache.flag_hallucination(strike_idx, severity_score=severity)
                        timestamp = time.strftime("%H:%M:%S")
                        log_entry = f"[{timestamp}] [bold green]ANE STRIKE[/] @ {strike_idx:04d} | SEV: {severity:.4f}"
                        monitor.excision_log.append(log_entry)
                
                last_analyzed_idx += chunk_size // 2 # Overlapping analysis

def run_integration_benchmark():
    monitor = DiagnosticMonitor()
    if monitor.ane_active:
        monitor.excision_log.append("[SYSTEM] ANE-DAEMON INITIALIZED AND ARMED.")
    else:
        monitor.excision_log.append("[WARNING] ANE MODEL NOT FOUND. FALLING BACK TO CPU.")
    
    gen_thread = threading.Thread(target=primary_generation_loop, args=(monitor,))
    critic_thread = threading.Thread(target=ghost_critic_loop, args=(monitor,))
    
    gen_thread.start()
    critic_thread.start()
    
    try:
        with Live(monitor.make_layout(), refresh_per_second=20, screen=True) as live:
            while monitor.is_running:
                live.update(monitor.make_layout())
                time.sleep(0.05)
    except KeyboardInterrupt:
        monitor.is_running = False
        
    gen_thread.join()
    critic_thread.join()

if __name__ == "__main__":
    run_integration_benchmark()
