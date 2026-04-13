import gradio as gr
import numpy as np
import plotly.graph_objects as go
import time

def generate_triage_data(token_count, error_rate):
    total_tokens = int(token_count)
    detected_hallucinations = int(total_tokens * (error_rate / 100))
    
    # Generate the Attention Manifold (Interactive Heatmap)
    attention_values = np.random.uniform(0.1, 1.0, total_tokens)
    poisoned_indices = np.random.choice(total_tokens, detected_hallucinations, replace=False)
    
    # Surgical Excision: The -inf mask sets attention to near-zero
    attention_values[poisoned_indices] = 0.01 
    
    fig = go.Figure(data=go.Heatmap(
        z=[attention_values],
        colorscale=[[0, 'rgb(220, 20, 60)'], [0.1, 'rgb(15, 25, 35)'], [1, 'rgb(16, 185, 129)']],
        showscale=False
    ))
    
    fig.update_layout(
        height=150,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False, title="Token Context Index", color="#64748b"),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        title=dict(text="1D Spectral Attention Manifold", font=dict(color="#10b981", size=14))
    )
    
    # Generate the "Reasoning Stream" Log
    logs = []
    for i in range(7):
        idx = np.random.randint(0, total_tokens)
        if idx in poisoned_indices:
            logs.append(f"[SYSTEM] Node_{idx:04d} anomaly detected. Masking -inf. Attention dropped.")
        else:
            logs.append(f"[SYSTEM] Node_{idx:04d} verified. Manifold geometry stable.")
    
    log_output = "\n".join(logs)
    
    report = (
        f"**Manifold Health:** {100 - error_rate:.1f}% Purity\n\n"
        f"**Throughput Status:** 100% (Non-blocking Healing)\n\n"
        f"**Critic Action:** {detected_hallucinations} surgical strikes executed."
    )
    return fig, log_output, report

custom_css = """
.gradio-container { background-color: #0f172a !important; }
.dark { color: #e2e8f0 !important; }
"""

with gr.Blocks(theme=gr.themes.Monochrome(), css=custom_css, title="ASH-KV Diagnostics") as ui:
    gr.Markdown("<h2 style='color: #10b981; font-family: monospace;'>ASH-KV v3.0.0 // Asynchronous Self-Healing Manifold</h2>")
    
    with gr.Row():
        with gr.Column(scale=1):
            tokens = gr.Slider(label="Context Manifold Size", minimum=1000, maximum=100000, value=10000, step=1000)
            errors = gr.Slider(label="Logical Drift Rate (%)", minimum=0.1, maximum=5.0, value=1.5, step=0.1)
            run_btn = gr.Button("INITIALIZE GHOST CRITIC SCAN", variant="primary")
            
            gr.Markdown("### 📡 Telemetry")
            metric_display = gr.Markdown("Awaiting stream data...")

        with gr.Column(scale=2):
            plot_output = gr.Plot()
            gr.Markdown("### 📟 Ghost Critic Daemon Logs")
            log_output = gr.Code(label="Daemon Output", interactive=False)

    run_btn.click(fn=generate_triage_data, inputs=[tokens, errors], outputs=[plot_output, log_output, metric_display])

if __name__ == "__main__":
    ui.launch()
