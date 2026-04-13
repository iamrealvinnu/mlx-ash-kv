import gradio as gr
import numpy as np
import plotly.graph_objects as go
import math

def generate_triage_data(token_count, error_rate, severity_score, trigger_compaction):
    total_tokens = int(token_count)
    num_heads = 32
    detected_hallucinations = int(total_tokens * (error_rate / 100))
    sigma = 1.0 + (severity_score * 19.0)
    
    # Initialize 2D manifold (Heads, Tokens)
    attention_values = np.ones((num_heads, total_tokens))
    poisoned_indices = np.random.choice(np.arange(10, total_tokens), detected_hallucinations, replace=False)
    
    t = np.arange(total_tokens)
    for mu in poisoned_indices:
        dist_sq = (t - mu)**2
        penalty = np.exp(-dist_sq / (2 * sigma**2))
        valid = (t >= mu) & (t > 0)
        
        # Apply penalty ONLY to reasoning heads (0-15)
        for h in range(16):
            attention_values[h, valid] = np.minimum(attention_values[h, valid], 1.0 - penalty[valid])
            
    attention_values[:, 0] = 1.0 # Sink preservation
    
    freed_tokens = 0
    logs = []
    
    if trigger_compaction:
        # EDCC Compaction Logic (Tokens dead in all reasoning heads)
        keep_indices = attention_values[0] > 0.1 
        attention_values = attention_values[:, keep_indices]
        freed_tokens = total_tokens - attention_values.shape[1]
        display_tokens = attention_values.shape[1]
        
        if freed_tokens > 0:
            logs.append(f"[SYSTEM] ⏸️ AVD triggered EDCC.")
            logs.append(f"[SYSTEM] COMPACTION SUCCESS: {freed_tokens} nodes deallocated from RAM.")
            logs.append(f"[SYSTEM] Manifold depth physically reduced to {display_tokens}.")
    else:
        display_tokens = total_tokens
        
    if not logs:
        for i in range(3):
            idx = np.random.randint(1, total_tokens)
            logs.append(f"[SYSTEM] AVD_{idx:04d} verified. Information density optimal.")

    fig = go.Figure(data=go.Heatmap(
        z=attention_values,
        colorscale=[[0, 'rgb(15, 23, 42)'], [0.2, 'rgb(76, 29, 149)'], [1, 'rgb(217, 70, 239)']],
        showscale=False
    ))
    
    fig.update_layout(
        margin=dict(l=40, r=10, t=30, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False, title="Context Index", color="#64748b"),
        yaxis=dict(showgrid=False, zeroline=False, title="Attention Heads", color="#64748b"),
        title=dict(text=f"2D HEAD-SPECIFIC MANIFOLD // Depth: {display_tokens}", font=dict(color="#d946ef", size=14))
    )
    
    log_output = "\n".join(logs)
    report = (
        f"**Manifold Status:** {'COMPACTED' if trigger_compaction and freed_tokens > 0 else 'STABLE'}\n\n"
        f"**Memory Reclaimed:** {freed_tokens} tokens\n\n"
        f"**Correction Mode:** Head-Specific Causal Pruning (V6.0)\n\n"
        f"**Physical RAM Usage:** {((display_tokens * 32 * 128 * 2) / (1024*1024)):.2f} MB"
    )
    return fig, log_output, report

custom_css = """
.dark { color: #e2e8f0 !important; }
"""

with gr.Blocks(theme=gr.themes.Monochrome(), css=custom_css, title="ASH-KV v6.0") as ui:
    gr.Markdown("<h2 style='color: #d946ef; font-family: monospace;'>ASH-KV v6.0.0 // Multi-Head EDCC Dashboard</h2>")
    with gr.Row():
        with gr.Column(scale=1):
            tokens = gr.Slider(label="Context Manifold Size", minimum=1000, maximum=100000, value=10000, step=1000)
            errors = gr.Slider(label="Logical Drift Rate (%)", minimum=0.1, maximum=5.0, value=1.5, step=0.1)
            severity = gr.Slider(label="Strike Severity", minimum=0.0, maximum=1.0, value=0.5, step=0.1)
            compaction = gr.Checkbox(label="ENABLE EDCC (COMPACTION)", value=True)
            run_btn = gr.Button("INITIALIZE AVD SCAN", variant="primary")
            
            gr.Markdown("### 📡 Telemetry")
            metric_display = gr.Markdown("Awaiting stream data...")
            
            gr.Markdown("### 📟 AVD System Logs")
            log_output = gr.Code(label="Daemon Output", interactive=False)
            
        with gr.Column(scale=2):
            plot_output = gr.Plot()
            
    run_btn.click(fn=generate_triage_data, inputs=[tokens, errors, severity, compaction], outputs=[plot_output, log_output, metric_display])

if __name__ == "__main__":
    ui.launch()
