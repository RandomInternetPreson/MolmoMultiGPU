import gradio as gr
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import torch
import inspect

# Load the processor and model
model_path = '/home/myself/Desktop/molmo/Molmo-7B-D-0924/'

processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True,
)

# Load the model in full precision (fp32) and distribute it across GPUs
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float32,
    device_map="auto",
    low_cpu_mem_usage=True,
)

# Monkey patch the _scaled_dot_product_attention method to ensure same device
def patched_scaled_dot_product_attention(self, *args, **kwargs):
    # Get the device of the first tensor argument
    device = next(arg for arg in args if isinstance(arg, torch.Tensor)).device
    
    # Move all tensor arguments to the same device
    args = [arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in args]
    kwargs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
    
    # Call the original method with the moved arguments
    return self._old_scaled_dot_product_attention(*args, **kwargs)

# Apply the patch to the model
for module in model.modules():
    if hasattr(module, '_scaled_dot_product_attention'):
        module._old_scaled_dot_product_attention = module._scaled_dot_product_attention
        module._scaled_dot_product_attention = patched_scaled_dot_product_attention.__get__(module)

def process_image_and_text(image, text):
    # Process the image and text
    inputs = processor.process(
        images=[Image.fromarray(image)],
        text=text
    )

    # Move inputs to the correct device and make a batch of size 1
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    # Generate output
    with torch.no_grad():
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer
        )

    # Only get generated tokens; decode them to text
    generated_tokens = output[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return generated_text

def chatbot(image, text, history):
    if image is None:
        return history + [("Please upload an image first.", None)]

    try:
        response = process_image_and_text(image, text)
        history.append((text, response))
    except Exception as e:
        history.append((text, f"An error occurred: {str(e)}"))
    
    return history

# Define the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Image Chatbot with Molmo-7B-D-0924")
    
    with gr.Row():
        image_input = gr.Image(type="numpy")
        chatbot_output = gr.Chatbot()
    
    text_input = gr.Textbox(placeholder="Ask a question about the image...")
    submit_button = gr.Button("Submit")

    state = gr.State([])

    submit_button.click(
        chatbot,
        inputs=[image_input, text_input, state],
        outputs=[chatbot_output]
    )

    text_input.submit(
        chatbot,
        inputs=[image_input, text_input, state],
        outputs=[chatbot_output]
    )

if __name__ == "__main__":
    demo.launch()
