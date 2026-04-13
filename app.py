import gradio as gr
import numpy as np
import plotly.graph_objects as go
import time
import math

def simulate_hive_mind(token_count, error_rate, severity_score, run_compaction):
    total_tokens = int(token_count)
    num_heads = 32
    detected_hallucinations = int(total_tokens * (error_rate / 100))
    sigma = 1.0 + (severity_score * 19.0)
    
    # --- AGENT A GENERATION ---
    attention_values = np.ones((num_heads, total_tokens))
    poisoned_indices = np.random.choice(np.arange(10, total_tokens), detected_hallucinations, replace=False)
    
    t = np.arange(total_tokens)
    for mu in poisoned_indices:
        dist_sq = (t - mu)**2
        penalty = np.exp(-dist_sq / (2 * sigma**2))
        valid = (t >= mu) & (t > 0)
        # Apply only to reasoning heads
        for h in range(16):
            attention_values[h, valid] = np.minimum(attention_values[h, valid], 1.0 - penalty[valid])
            
    attention_values[:, 0] = 1.0 # Sink preservation
    
    freed_tokens = 0
    if run_compaction:
        keep_indices = attention_values[0] > 0.1 
        attention_values = attention_values[:, keep_indices]
        freed_tokens = total_tokens - attention_values.shape[1]
        display_tokens = attention_values.shape[1]
    else:
        display_tokens = total_tokens

    # --- THE HANDOFF (Agent A to Agent B) ---
    # In v7.0.0, this is zero-copy pointer referencing
    handoff_start = time.perf_counter()
    agent_b_manifold = attention_values # Simulation of pointer reference
    handoff_latency = (time.perf_counter() - handoff_start) * 1000 # In ms

    fig = go.Figure()
    
    # Plot Agent A (Left)
    fig.add_trace(go.Heatmap(
        z=attention_values,
        colorscale=[[0, 'rgb(15, 23, 42)'], [0.2, 'rgb(76, 29, 149)'], [1, 'rgb(217, 70, 239)']],
        showscale=False,
        name="Agent A Manifold"
    ))
    
    fig.update_layout(
        margin=dict(l=40, r=10, t=30, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False, title="Unified Memory Address Space", color="#64748b"),
        yaxis=dict(showgrid=False, zeroline=False, title="Attention Heads", color="#64748b"),
        title=dict(text=f"HIVE MIND PROTOCOL // Zero-Copy Handoff Active", font=dict(color="#d946ef", size=14))
    )
    
    logs = [
        f"[SYSTEM] Agent_A (Researcher) context generation complete.",
        f"[SYSTEM] AVD flagged {detected_hallucinations} causal contaminated nodes.",
        f"[SYSTEM] EDCC Compaction: {freed_tokens} tokens deallocated.",
        f"[SYSTEM] 🐝 HIVE MIND TRIGGERED: Exporting pointer to Agent_B (Evaluator).",
        f"[SYSTEM] SUCCESS: Agent_B mounted manifold in {handoff_latency:.6f} ms.",
        f"[SYSTEM] Agent_B TTFT: 0.00 ms (Zero pre-fill required)."
    ]
    
    log_output = "\n".join(logs)
    report = (
        f"**Hive Mind Status:** LINKED\n\n"
        f"**Handoff Latency:** {handoff_latency:.6f} ms\n\n"
        f"**TTFT (Agent B):** 0.00 ms\n\n"
        f"**Architecture:** v7.0.0 Hive Mind Protocol"
    )
    return fig, log_output, report

custom_css = ".dark { color: #e2e8f0 !important; }"

with gr.Blocks(theme=gr.themes.Monochrome(), css=custom_css, title="ASH-KV v7.0 Hive Mind") as ui:
    gr.Markdown("<h2 style='color: #d946ef; font-family: monospace;'>ASH-KV v7.0.0 // Hive Mind Zero-Copy Dashboard</h2>")
    
    with gr.Row():
        with gr.Column(scale=1):
            tokens = gr.Slider(label="Context Manifold Size", minimum=1000, maximum=100000, value=10000, step=1000)
            errors = gr.Slider(label="Logical Drift Rate (%)", minimum=0.1, maximum=5.0, value=1.5, step=0.1)
            severity = gr.Slider(label="Strike Severity", minimum=0.0, maximum=1.0, value=0.5, step=0.1)
            compaction = gr.Checkbox(label="ENABLE EDCC (COMPACTION)", value=True)
            run_btn = gr.Button("TRIGGER ZERO-COPY HANDOFF", variant="primary")
            
            gr.Markdown("### 📡 Silicon Telemetry")
            metric_display = gr.Markdown("Awaiting handoff...")
            
            gr.Markdown("### 📟 Hive Mind Logs")
            log_output = gr.Code(label="AVD Daemon Output", interactive=False)
            
        with gr.Column(scale=2):
            plot_output = gr.Plot()
            gr.Markdown("<p style='text-align: center; color: #64748b;'>Visualizing instantaneous brain-state transfer between distinct agent instances.</p>")
            
    run_btn.click(fn=simulate_hive_mind, inputs=[tokens, errors, severity, compaction], outputs=[plot_output, log_output, metric_display])

if __name__ == "__main__":
    ui.launch()
