import gradio as gr
import numpy as np
import plotly.graph_objects as go
import time
import math

def generate_triage_data(token_count, error_rate, severity_score):
    total_tokens = int(token_count)
    detected_hallucinations = int(total_tokens * (error_rate / 100))
    
    # 0.01 to 1.0 -> Sigma 1 to 20
    sigma = 1.0 + (severity_score * 19.0)
    
    # Initialize manifold
    attention_values = np.ones(total_tokens)
    poisoned_indices = np.random.choice(np.arange(10, total_tokens), detected_hallucinations, replace=False)
    
    # Apply Gaussian Decay
    t = np.arange(total_tokens)
    for mu in poisoned_indices:
        # Causal Gaussian Decay: penalty = exp(-(t-mu)^2 / (2*sigma^2))
        dist_sq = (t - mu)**2
        penalty = np.exp(-dist_sq / (2 * sigma**2))
        
        # Apply causal constraint and sink preservation (i > 0)
        valid = (t >= mu) & (t > 0)
        # We simulate the -inf mask by dropping attention to near-zero
        attention_values[valid] = np.minimum(attention_values[valid], 1.0 - penalty[valid])

    # Attention Sink: Index 0 is always healthy
    attention_values[0] = 1.0
    
    fig = go.Figure(data=go.Heatmap(
        z=[attention_values],
        colorscale=[[0, 'rgb(15, 23, 42)'], [0.2, 'rgb(30, 41, 59)'], [1, 'rgb(34, 197, 94)']],
        showscale=False
    ))
    
    fig.update_layout(
        height=150,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False, title="Token Context Index", color="#64748b"),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        title=dict(text=f"1D Spectral Attention Manifold // Sigma: {sigma:.1f}", font=dict(color="#22c55e", size=14))
    )
    
    # Generate the "Reasoning Stream" Log
    logs = []
    for i in range(7):
        idx = np.random.randint(1, total_tokens)
        if any(idx >= mu for mu in poisoned_indices):
            logs.append(f"[SYSTEM] Node_{idx:04d} influenced by causal drift. Applying soft correction.")
        else:
            logs.append(f"[SYSTEM] Node_{idx:04d} verified. Manifold geometry stable.")
    
    log_output = "\n".join(logs)
    
    report = (
        f"**Manifold Health:** {100 - error_rate:.1f}% Purity\n\n"
        f"**Correction Mode:** Phase 1 Temporal Rollback\n\n"
        f"**Sink Status:** Locked (Index 0 Protected)\n\n"
        f"**Critic Action:** {detected_hallucinations} Gaussian strikes executed."
    )
    return fig, log_output, report

custom_css = """
.gradio-container { background-color: #0f172a !important; }
.dark { color: #e2e8f0 !important; }
"""

with gr.Blocks(theme=gr.themes.Monochrome(), css=custom_css, title="ASH-KV Diagnostics") as ui:
    gr.Markdown("<h2 style='color: #22c55e; font-family: monospace;'>ASH-KV v4.0.0 // Temporal Rollback Dashboard</h2>")
    
    with gr.Row():
        with gr.Column(scale=1):
            tokens = gr.Slider(label="Context Manifold Size", minimum=1000, maximum=100000, value=10000, step=1000)
            errors = gr.Slider(label="Logical Drift Rate (%)", minimum=0.1, maximum=5.0, value=1.5, step=0.1)
            severity = gr.Slider(label="Decay Severity (Sigma)", minimum=0.0, maximum=1.0, value=0.5, step=0.1)
            run_btn = gr.Button("INITIALIZE GHOST CRITIC SCAN", variant="primary")
            
            gr.Markdown("### 📡 Telemetry")
            metric_display = gr.Markdown("Awaiting stream data...")

        with gr.Column(scale=2):
            plot_output = gr.Plot()
            gr.Markdown("### 📟 Ghost Critic Daemon Logs")
            log_output = gr.Code(label="Daemon Output", interactive=False)

    run_btn.click(fn=generate_triage_data, inputs=[tokens, errors, severity], outputs=[plot_output, log_output, metric_display])

if __name__ == "__main__":
    ui.launch()
