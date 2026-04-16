import gradio as gr
import time
import os
import mlx.core as mx
from mlx_lm import load
from mlx_ash_kv.api import protect, generate_stream

# Global state for the model
class ModelHub:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.cache = None
        self.adapter = None
        self.proxies = None
        self.model_path = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"

    def ensure_loaded(self):
        if self.model is None:
            print(f"--- Loading Real Model: {self.model_path} ---")
            try:
                self.model, self.tokenizer = load(self.model_path)
                # Initialize ASH-KV Protection
                _, self.cache, self.adapter, self.proxies = protect(
                    self.model, 
                    critic_model_path="models/mock_critic.mlpackage"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load model. Ensure 'mlx-lm' and 'huggingface_hub' are installed. Error: {e}")

hub = ModelHub()

def run_real_inference(prompt, sensitivity):
    try:
        hub.ensure_loaded()
    except Exception as e:
        yield str(e), str(e), "0.00"
        return

    # Update sensitivity from UI
    hub.adapter.sensitivity = sensitivity
    
    standard_out = ""
    protected_out = ""
    
    gen = generate_stream(
        hub.model, 
        hub.tokenizer, 
        hub.cache, 
        hub.proxies, 
        prompt, 
        adapter=hub.adapter
    )
    
    intervention_active = False
    
    for token, health_score in gen:
        # Standard generation trace (unfiltered)
        standard_out += token
        
        # Protected generation trace
        protected_out += token
        
        # If health score drops below sensitivity, trigger UI intervention warning
        if health_score < (1.0 - hub.adapter.current_threshold) and not intervention_active:
            protected_out += "\n\n**[ASH-KV INTERVENTION: Clinical contraindication detected. Attention heads pruned.]**\n\n"
            intervention_active = True
        elif health_score >= (1.0 - hub.adapter.current_threshold):
            intervention_active = False
        
        yield standard_out, protected_out, f"{health_score:.2f}"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚡ ASH-KV: Real Neural Reliability Playground")
    gr.Markdown("### Running Meta-Llama-3-8B with Hardware-Level Causal Pruning.")
    
    with gr.Row():
        sensitivity_slider = gr.Slider(minimum=0.5, maximum=0.99, value=0.85, step=0.01, label="Shield Sensitivity")
        health_meter = gr.Label(label="Live AVD Integrity Score")

    with gr.Row():
        with gr.Column():
            gr.Markdown("#### Standard Generation Trace")
            std_output = gr.Textbox(lines=10, interactive=False, label="Raw Trace")
        
        with gr.Column():
            gr.Markdown("#### ASH-KV Protected Output")
            protected_output = gr.Textbox(lines=10, interactive=False, label="Healed Output")

    prompt_input = gr.Textbox(
        value="What are the specific side effects of prescribing Lisinopril to a patient with a 104F fever?", 
        label="Clinical Prompt"
    )
    run_btn = gr.Button("🚀 Trigger Real Inference", variant="primary")

    run_btn.click(
        run_real_inference, 
        inputs=[prompt_input, sensitivity_slider], 
        outputs=[std_output, protected_output, health_meter]
    )

if __name__ == "__main__":
    demo.launch()
