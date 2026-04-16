from rich.table import Table
from rich.live import Live
from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
import mlx.core as mx
from .cache import ASHCache

class LiveDiagnosticTUI:
    """
    Live Diagnostic Terminal UI for ASH-KV.
    Visualizes layer-wise health and attention manifold distribution.
    """
    def __init__(self, cache: ASHCache):
        self.cache = cache
        self.console = Console()

    def generate_table(self) -> Table:
        table = Table(
            show_header=True, 
            header_style="bold cyan", 
            border_style="dim",
            show_footer=True
        )
        table.add_column("Layer Index", justify="center", style="cyan")
        table.add_column("Memory Tier", justify="center")
        table.add_column("Attention Variance", justify="right", style="magenta")
        table.add_column("Health Status", justify="center")

        total_vram_saved_mb = 0
        
        # Use the cache lock to safely access and evaluate layer data
        with self.cache._lock:
            has_strikes = len(self.cache.strikes) > 0
            display_layers = min(self.cache.num_layers, 32)
            
            # Prepare all variance calculations to evaluate them in one go
            variances = []
            valid_indices = []
            for i in range(display_layers):
                if self.cache.layer_keys[i] is not None:
                    variances.append(mx.var(self.cache.layer_keys[i]))
                    valid_indices.append(i)
            
            # Evaluate all variances together to minimize synchronization overhead
            if variances:
                mx.eval(*variances)
                var_values = [v.item() for v in variances]
            else:
                var_values = []

            var_map = dict(zip(valid_indices, var_values))

            for i in range(display_layers):
                var = var_map.get(i, 0.0)
                var_str = f"{var:.6f}"
                
                # Health Status
                status = "[bold green]Healthy[/]" if not has_strikes else "[bold yellow]Healing[/]"
                
                # Memory Tier Logic
                is_paged = len(self.cache.page_map[i]) > 0
                if is_paged:
                    tier = "[bold yellow]WARM (NVMe)[/]"
                    # Calculate saved VRAM for this layer (simplified)
                    # paged_tokens * heads * head_dim * 2 bytes
                    paged_len = sum(p["len"] for p in self.cache.page_map[i])
                    layer_saved_mb = (paged_len * self.cache.num_heads * 128 * 2) / (1024 * 1024)
                    total_vram_saved_mb += layer_saved_mb
                else:
                    tier = "[bold red]HOT (VRAM)[/]"
                
                table.add_row(f"{i+1:02d}", tier, var_str, status)
            
            # Footer
            table.columns[0].footer = Text("SYSTEM TOTAL", style="bold white")
            table.columns[1].footer = Text(f"{total_vram_saved_mb/1024:.2f} GB SAVED", style="bold green")
            
        return table

    def get_layout(self) -> Layout:
        layout = Layout()
        layout.update(
            Panel(
                self.generate_table(),
                title="[bold white]ASH-KV MULTI-LAYER HYPERVISOR[/]",
                border_style="cyan"
            )
        )
        return layout
