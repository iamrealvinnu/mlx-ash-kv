"""
Diagnostic TUI and Integration Benchmark for ASH-KV (v4.0.0).

Provides a brutalist, high-density Terminal UI to visualize the 
Asynchronous Self-Healing Key-Value Cache in action.

Updates: Visualizes Phase 1 Gaussian Decay (Temporal Rollback).
"""

import mlx.core as mx
import time
import threading
import random
import math
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

class DiagnosticMonitor:
    def __init__(self):
        self.cache = ASHCache()
        self.current_token = 0
        self.excision_log = deque(maxlen=20)
        self.start_time = time.time()
        self.is_running = True
        
    def generate_barcode(self) -> Group:
        """Renders the attention manifold as a dense 1D spectral barcode with Gaussian gradients."""
        blocks = 120  # Console width target
        tokens_per_block = TARGET_TOKENS / blocks
        
        barcode_lines = []
        
        # Pre-calculate the mask for visualization if possible
        # Since we want to show the gradient, we simulate the Gaussian effect here
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

                # Calculate max penalty for this block across all strikes
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
                    # Gradient representation using grayscale intensity
                    # 0.1 to 0.95 penalty -> brighter to darker
                    shade = int(255 * (1.0 - max_penalty))
                    line.append("█", style=f"rgb({shade},{shade},{shade})")
                else:
                    # Healthy context
                    intensity = random.choice(["bold white", "white", "bright_white"])
                    line.append("█", style=intensity)
            barcode_lines.append(Align.center(line))
            
        return Group(
            Text("\n\n"),
            *barcode_lines,
            Text("\n[ 1D SPECTRAL ATTENTION MANIFOLD // GAUSSIAN DECAY ENABLED ]", style="bold white", justify="center")
        )

    def generate_telemetry(self) -> Table:
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="bold white")
        table.add_column("Value", style="white")
        
        elapsed = time.time() - self.start_time
        tps = self.current_token / elapsed if elapsed > 0 else 0
        
        # Simulating memory readouts based on shapes (float16 = 2 bytes)
        mem_mb = (BATCH_SIZE * NUM_HEADS * self.current_token * HEAD_DIM * 2) / (1024 * 1024)
        
        table.add_row("KV Tensor Shape", f"[{BATCH_SIZE}, {NUM_HEADS}, {self.current_token}, {HEAD_DIM}]")
        table.add_row("Manifold Depth", f"{self.current_token} / {TARGET_TOKENS} tokens")
        table.add_row("Throughput", f"{tps:.2f} tok/s")
        table.add_row("Unified Memory", f"{mem_mb:.2f} MB Allocated")
        table.add_row("Active Strikes", f"{len(self.cache.strikes)}")
        table.add_row("Correction Mode", "Gaussian Causal Decay (V4.0)")
        table.add_row("Sink Preservation", "Index 0 [bos] Locked")
        
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
        
        header_text = Text(" ASH-KV V4.0.0 // TEMPORAL ROLLBACK MONITOR ", style="bold black on cyan", justify="center")
        layout["header"].update(Panel(header_text, style="cyan"))
        
        layout["left"]["telemetry"].update(Panel(self.generate_telemetry(), title="[bold]SYSTEM TELEMETRY[/]", border_style="cyan"))
        layout["left"]["logs"].update(Panel(self.generate_logs(), title="[bold]GHOST CRITIC INTERCEPTS[/]", border_style="cyan"))
        layout["right"].update(Panel(self.generate_barcode(), title="[bold]TENSOR MANIFOLD STATUS[/]", border_style="cyan"))
        
        return layout

def primary_generation_loop(monitor: DiagnosticMonitor):
    for i in range(1, TARGET_TOKENS + 1):
        if not monitor.is_running:
            break
            
        new_k = mx.random.uniform(shape=(BATCH_SIZE, NUM_HEADS, 1, HEAD_DIM), dtype=mx.float16)
        new_v = mx.random.uniform(shape=(BATCH_SIZE, NUM_HEADS, 1, HEAD_DIM), dtype=mx.float16)
        
        k, v, mask = monitor.cache.update(new_k, new_v)
        
        # Periodic eval to force Metal execution
        if i % 100 == 0:
            mx.eval(k, v, mask)
            
        monitor.current_token = i
        time.sleep(0.003)
        
    monitor.is_running = False

def ghost_critic_loop(monitor: DiagnosticMonitor):
    while monitor.is_running and monitor.current_token < TARGET_TOKENS:
        time.sleep(0.5)
        
        current_len = monitor.cache.seq_len
        if current_len > 200 and random.random() < 0.3:
            target_idx = current_len - random.randint(50, 150)
            
            # Severity score: 0.1 (surgical) to 0.9 (broad)
            severity = random.uniform(0.1, 0.9)
            
            if target_idx > 0 and not any(s["index"] == target_idx for s in monitor.cache.strikes):
                monitor.cache.flag_hallucination(target_idx, severity_score=severity)
                timestamp = time.strftime("%H:%M:%S")
                log_entry = f"[{timestamp}] STRIKE @ {target_idx:04d} | SEV: {severity:.2f} -> [bold cyan]GAUSSIAN RADIUS APPLIED[/]"
                monitor.excision_log.append(log_entry)

def run_integration_benchmark():
    monitor = DiagnosticMonitor()
    monitor.excision_log.append("[SYSTEM] GHOST CRITIC V4.0 (TEMPORAL) INITIALIZED.")
    
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
