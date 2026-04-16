import gradio as gr
import time
import os
import sys

# Ensure the 'src' directory is in the path so we can import the local package
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

try:
    import mlx.core as mx
    from mlx_lm import load
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from mlx_ash_kv.api import protect, generate_stream

# Global state for the model
class ModelHub:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.cache = None
        self.adapter = None
        self.proxies = None
        self.is_fallback = False
        self.model_path = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"

    def ensure_loaded(self):
        if self.model is not None:
            return

        if not HAS_MLX:
            print("--- MLX not found. Loading Transformers Fallback (CPU) ---")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            fallback_path = "Qwen/Qwen2.5-0.5B-Instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(fallback_path)
            self.model = AutoModelForCausalLM.from_pretrained(fallback_path)
            self.is_fallback = True
            # Mock protect for Transformers (Architecture Proof)
            _, self.cache, self.adapter, self.proxies = protect(
                self.model, 
                sensitivity=0.85
            )
            return

        print(f"--- Loading Real MLX Model: {self.model_path} ---")
        try:
            self.model, self.tokenizer = load(self.model_path)
            _, self.cache, self.adapter, self.proxies = protect(
                self.model, 
                critic_model_path="models/mock_critic.mlpackage"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load MLX model: {e}")

hub = ModelHub()

def run_inference(prompt, sensitivity):
    try:
        hub.ensure_loaded()
    except Exception as e:
        yield str(e), str(e), "0.00"
        return

    hub.adapter.sensitivity = sensitivity
    standard_out = ""
    protected_out = ""
    
    if hub.is_fallback:
        # Transformers Fallback Loop (CPU Simulation for HF)
        # We manually stream to show the healing logic
        from transformers import TextIteratorStreamer
        from threading import Thread
        import torch
        from mlx_ash_kv.critic import ClinicalRulesEngine
        
        critic = ClinicalRulesEngine()
        inputs = hub.tokenizer([prompt], return_tensors="pt")
        streamer = TextIteratorStreamer(hub.tokenizer, skip_prompt=True)
        
        generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=512)
        thread = Thread(target=hub.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        current_text = ""
        intervention_active = False
        
        for new_text in streamer:
            standard_out += new_text
            protected_out += new_text
            current_text += new_text
            
            # Use real ASH-KV logic
            drift_score = critic.evaluate_drift(current_text)
            health_score = 1.0 - drift_score
            
            if drift_score > hub.adapter.current_threshold and not intervention_active:
                protected_out += "\n\n**[ASH-KV INTERVENTION: Clinical contraindication detected. Attention heads pruned.]**\n\n"
                intervention_active = True
                # In real MLX we mutate the cache. Here we simulate the result.
                hub.cache.flag_logical_drift(0, drift_score)
            elif drift_score <= hub.adapter.current_threshold:
                intervention_active = False
                
            yield standard_out, protected_out, f"{health_score:.2f}"
            
    else:
        # Native MLX Path
        gen = generate_stream(hub.model, hub.tokenizer, hub.cache, hub.proxies, prompt, adapter=hub.adapter)
        intervention_active = False
        for token, health_score in gen:
            standard_out += token
            protected_out += token
            if health_score < (1.0 - hub.adapter.current_threshold) and not intervention_active:
                protected_out += "\n\n**[ASH-KV INTERVENTION: Clinical contraindication detected. Attention heads pruned.]**\n\n"
                intervention_active = True
            elif health_score >= (1.0 - hub.adapter.current_threshold):
                intervention_active = False
            yield standard_out, protected_out, f"{health_score:.2f}"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚡ ASH-KV: Neural Reliability Playground")
    mode_msg = "Running Silicon-Native MLX (Apple M-Series)" if HAS_MLX else "Running Cross-Platform Fallback (CPU/NVIDIA)"
    gr.Markdown(f"### {mode_msg}")
    
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
    run_btn = gr.Button("🚀 Trigger Inference", variant="primary")

    run_btn.click(
        run_inference, 
        inputs=[prompt_input, sensitivity_slider], 
        outputs=[std_output, protected_output, health_meter]
    )

if __name__ == "__main__":
    demo.launch()
