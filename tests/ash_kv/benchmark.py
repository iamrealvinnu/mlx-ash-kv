"""
Diagnostic TUI and Integration Benchmark for ASH-KV (v5.0.2).

Provides a brutalist, high-density Terminal UI to visualize the 
Asynchronous Self-Healing Key-Value Cache in action.

Updates: v5.0.2 High-Entropy Patch (No-Stutter Compaction).
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

# Configuration
BATCH_SIZE = 1
NUM_HEADS = 32
HEAD_DIM = 128
TARGET_TOKENS = 3000
ANE_MODEL_PATH = "models/mock_critic.mlpackage"

class DiagnosticMonitor:
    def __init__(self):
        model_exists = os.path.exists(ANE_MODEL_PATH)
        self.cache = ASHCache(critic_model_path=ANE_MODEL_PATH if model_exists else None)
        self.excision_log = deque(maxlen=20)
        self.start_time = time.time()
        self.is_running = True
        self.ane_active = model_exists
        self.last_ane_score = 0.0
        
    def generate_barcode(self) -> Group:
        blocks = 120
        tokens_per_block = TARGET_TOKENS / blocks
        barcode_lines = []
        
        # Snapshot manifold state
        current_len = self.cache.seq_len
        strikes = list(self.cache.strikes)
        
        for _ in range(5):
            line = Text()
            for i in range(blocks):
                block_start = i * tokens_per_block
                
                if block_start > current_len:
                    line.append(" ", style="on black")
                    continue
                
                if block_start == 0:
                    line.append("█", style="bold green")
                    continue

                max_penalty = 0.0
                for strike in strikes:
                    mu = strike["index"]
                    sigma = strike["sigma"]
                    if block_start >= mu:
                        dist_sq = (block_start - mu)**2
                        penalty = math.exp(-dist_sq / (2 * sigma**2))
                        max_penalty = max(max_penalty, penalty)
                
                if max_penalty > 0.95:
                    line.append(" ", style="on black")
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
            Text("\n[ 1D SPECTRAL ATTENTION MANIFOLD // HIGH-ENTROPY MODE ]", style="bold white", justify="center")
        )

    def generate_telemetry(self) -> Table:
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="bold white")
        table.add_column("Value", style="white")
        
        elapsed = time.time() - self.start_time
        current_len = self.cache.seq_len
        tps = current_len / elapsed if elapsed > 0 else 0
        mem_mb = (BATCH_SIZE * NUM_HEADS * current_len * HEAD_DIM * 2) / (1024 * 1024)
        
        table.add_row("KV Tensor Shape", f"[{BATCH_SIZE}, {NUM_HEADS}, {current_len}, {HEAD_DIM}]")
        table.add_row("Manifold Depth", f"{current_len} / {TARGET_TOKENS} tokens")
        table.add_row("Throughput", f"{tps:.2f} tok/s")
        table.add_row("Unified Memory", f"{mem_mb:.2f} MB Allocated")
        table.add_row("Correction Mode", "[bold magenta]Semantic Compaction (V5.0)[/]")
        table.add_row("ANE Critic Score", f"{self.last_ane_score:.4f}")
        table.add_row("Hardware Isolation", "[bold green]ENABLED[/]" if self.ane_active else "[bold yellow]NONE[/]")
        
        return table

    def generate_logs(self) -> Group:
        log_texts = [Text.from_markup(log) for log in list(self.excision_log)]
        return Group(*log_texts)

    def make_layout(self) -> Layout:
        layout = Layout(name="root")
        layout.split(Layout(name="header", size=3), Layout(name="main"))
        layout["main"].split_row(Layout(name="left", ratio=1), Layout(name="right", ratio=2))
        layout["left"].split_column(Layout(name="telemetry", size=10), Layout(name="logs"))
        
        header_text = Text(" ASH-KV V5.0.2 // IMMORTAL MANIFOLD (HIGH-ENTROPY) ", style="bold black on magenta", justify="center")
        layout["header"].update(Panel(header_text, style="magenta"))
        
        layout["left"]["telemetry"].update(Panel(self.generate_telemetry(), title="[bold]HARDWARE TELEMETRY[/]", border_style="magenta"))
        layout["left"]["logs"].update(Panel(self.generate_logs(), title="[bold]IMMORTAL DAEMON LOGS[/]", border_style="magenta"))
        layout["right"].update(Panel(self.generate_barcode(), title="[bold]TENSOR MANIFOLD STATUS[/]", border_style="magenta"))
        return layout

def primary_generation_loop(monitor: DiagnosticMonitor):
    for i in range(1, TARGET_TOKENS + 1):
        if not monitor.is_running:
            break
            
        new_k = mx.random.uniform(shape=(BATCH_SIZE, NUM_HEADS, 1, HEAD_DIM), dtype=mx.float16)
        new_v = mx.random.uniform(shape=(BATCH_SIZE, NUM_HEADS, 1, HEAD_DIM), dtype=mx.float16)
        
        k, v, mask = monitor.cache.update(new_k, new_v)
        
        # Lightweight periodic eval
        if i % 100 == 0:
            monitor.cache.sync_eval(k, v, mask)
            
        time.sleep(0.003)
        
        # 🚀 PHASE 3 COMPACTION (Gated by token threshold)
        current_len = monitor.cache.seq_len
        if current_len > 0 and current_len % 600 == 0:
            # We log before attempting to ensure we see the status
            monitor.excision_log.append(f"\n[bold magenta][SYSTEM] ⏸️ User Pause. Compacting...[/]")
            
            # The function now handles the quick-exit if no strikes exist
            freed = monitor.cache.compact_manifold(threshold=-9000.0)
            
            if freed > 0:
                monitor.excision_log.append(f"[bold magenta]↳ SUCCESS: {freed} tokens reclaimed.[/]\n")
            else:
                monitor.excision_log.append(f"[bold magenta]↳ OPTIMAL: Cache is high-entropy.[/]\n")
        
    monitor.is_running = False

def ghost_critic_loop(monitor: DiagnosticMonitor):
    chunk_size = 128
    last_analyzed_idx = 0
    
    while monitor.is_running:
        time.sleep(0.4) # Throttle ANE access to reduce lock contention
        
        current_len = monitor.cache.seq_len
        if last_analyzed_idx > current_len:
            last_analyzed_idx = 0
            
        if current_len >= last_analyzed_idx + chunk_size:
            severity = monitor.cache.analyze_manifold_chunk(last_analyzed_idx, chunk_size)
            if severity is not None:
                monitor.last_ane_score = severity
                if severity > 0.8:
                    strike_idx = last_analyzed_idx + random.randint(0, chunk_size - 1)
                    if strike_idx > 0:
                        monitor.cache.flag_hallucination(strike_idx, severity_score=severity)
                        monitor.excision_log.append(f"[{time.strftime('%H:%M:%S')}] [bold green]ANE STRIKE[/] @ {strike_idx:04d}")
                last_analyzed_idx += chunk_size // 2

def run_integration_benchmark():
    monitor = DiagnosticMonitor()
    threading.Thread(target=primary_generation_loop, args=(monitor,)).start()
    threading.Thread(target=ghost_critic_loop, args=(monitor,)).start()
    
    try:
        # Reduced FPS to 10 to give Python threads more air
        with Live(monitor.make_layout(), refresh_per_second=10, screen=True) as live:
            while monitor.is_running:
                live.update(monitor.make_layout())
                time.sleep(0.1)
    except KeyboardInterrupt:
        monitor.is_running = False

if __name__ == "__main__":
    run_integration_benchmark()
