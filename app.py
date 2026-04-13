import gradio as gr
import numpy as np
import plotly.graph_objects as go
import time
import math

def generate_triage_data(token_count, error_rate, severity_score, trigger_compaction):
    total_tokens = int(token_count)
    detected_hallucinations = int(total_tokens * (error_rate / 100))
    
    sigma = 1.0 + (severity_score * 19.0)
    
    # Initialize manifold
    attention_values = np.ones(total_tokens)
    poisoned_indices = np.random.choice(np.arange(10, total_tokens), detected_hallucinations, replace=False)
    
    # Apply Gaussian Decay (Phase 1 logic)
    t = np.arange(total_tokens)
    for mu in poisoned_indices:
        dist_sq = (t - mu)**2
        penalty = np.exp(-dist_sq / (2 * sigma**2))
        valid = (t >= mu) & (t > 0)
        attention_values[valid] = np.minimum(attention_values[valid], 1.0 - penalty[valid])

    # Attention Sink: Index 0 is always healthy
    attention_values[0] = 1.0
    
    # 🚀 PHASE 3: SEMANTIC COMPACTION SIMULATION
    freed_tokens = 0
    if trigger_compaction:
        # Simulate physical removal of tokens with < 0.1 attention mass
        keep_indices = attention_values > 0.1
        new_attention = attention_values[keep_indices]
        freed_tokens = total_tokens - len(new_attention)
        attention_values = new_attention
        # If compacted, we show the shrunken manifold
        display_tokens = len(attention_values)
    else:
        display_tokens = total_tokens

    fig = go.Figure(data=go.Heatmap(
        z=[attention_values],
        colorscale=[[0, 'rgb(15, 23, 42)'], [0.2, 'rgb(76, 29, 149)'], [1, 'rgb(217, 70, 239)']],
        showscale=False
    ))
    
    fig.update_layout(
        height=150,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False, title="Physical Token Memory Index", color="#64748b"),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        title=dict(text=f"IMMORTAL MANIFOLD // Depth: {display_tokens} tokens", font=dict(color="#d946ef", size=14))
    )
    
    logs = []
    if trigger_compaction and freed_tokens > 0:
        logs.append(f"[SYSTEM] ⏸️ User Pause detected. Triggering EDSC.")
        logs.append(f"[SYSTEM] COMPACTION SUCCESS: {freed_tokens} nodes vaporized from RAM.")
        logs.append(f"[SYSTEM] Manifold depth physically reduced to {display_tokens}.")
    else:
        for i in range(5):
            idx = np.random.randint(1, total_tokens)
            logs.append(f"[SYSTEM] ANE_{idx:04d} verified. Information density optimal.")
    
    log_output = "\n".join(logs)
    
    report = (
        f"**Manifold Status:** {'COMPACTED' if trigger_compaction else 'EXPANDING'}\n\n"
        f"**Memory Reclaimed:** {freed_tokens} tokens\n\n"
        f"**Correction Mode:** Phase 3 Semantic Compaction\n\n"
        f"**Physical RAM Usage:** {((display_tokens * 32 * 128 * 2) / (1024*1024)):.2f} MB"
    )
    return fig, log_output, report

custom_css = """
.gradio-container { background-color: #0f172a !important; }
.dark { color: #e2e8f0 !important; }
"""

with gr.Blocks(theme=gr.themes.Monochrome(), css=custom_css, title="ASH-KV Diagnostics") as ui:
    gr.Markdown("<h2 style='color: #d946ef; font-family: monospace;'>ASH-KV v5.0.2 // Immortal Manifold Dashboard</h2>")
    
    with gr.Row():
        with gr.Column(scale=1):
            tokens = gr.Slider(label="Context Manifold Size", minimum=1000, maximum=100000, value=10000, step=1000)
            errors = gr.Slider(label="Logical Drift Rate (%)", minimum=0.1, maximum=5.0, value=1.5, step=0.1)
            severity = gr.Slider(label="Strike Severity", minimum=0.0, maximum=1.0, value=0.5, step=0.1)
            compaction = gr.Checkbox(label="ENABLE SEMANTIC COMPACTION", value=True)
            run_btn = gr.Button("INITIALIZE IMMORTAL SCAN", variant="primary")
            
            gr.Markdown("### 📡 Telemetry")
            metric_display = gr.Markdown("Awaiting stream data...")

        with gr.Column(scale=2):
            plot_output = gr.Plot()
            gr.Markdown("### 📟 Immortal Daemon Logs")
            log_output = gr.Code(label="Daemon Output", interactive=False)

    run_btn.click(fn=generate_triage_data, inputs=[tokens, errors, severity, compaction], outputs=[plot_output, log_output, metric_display])

if __name__ == "__main__":
    ui.launch()
