import gradio as gr
import re
import threading
from mmif import Mmif, DocumentTypes

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
quant_config = BitsAndBytesConfig(load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=quant_config
)

MMIF_PATH = "/home/kmlynch/clams_apps/cas-pipeline/6-llama_summary/Eyewitness_News_at_11pm_-_Boston_Blizzard_of_78_-_WBZ-TV_Complete_Broadcast_2_7_1978.mmif"

def get_mmif_object():
    with open(MMIF_PATH, 'r') as file:
        return Mmif(file.read())

def load_mmif_annotations():
    mmif_obj = get_mmif_object()
    global annotations
    annotations = []
    for view in mmif_obj.views:
        annotations.extend(view.annotations)
    # print("Annotations loaded", annotations)

threading.Thread(target=load_mmif_annotations, daemon=True).start()

def chat_respond(message, chat_history):
    if chat_history is None:
        chat_history = []
    chat_history.append(("User", message))
    # Construct the prompt instructing the model to generate timestamps in mm:ss format
    prompt = f"Please generate timestamps in the format mm:ss where appropriate in your response.\nUser: {message}\nAssistant:"
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def replace_ts(match):
        ts = match.group(0)
        parts = ts.split(":")
        total_seconds = int(parts[0]) * 60 + int(parts[1])
        return f'<a href="#" onclick=\'const v=document.getElementsByTagName("video")[0]; if(v){{ v.currentTime={total_seconds}; v.play(); }} return false;\'>{ts}</a>'
    
    processed_response = re.sub(r'(\d{1,2}:\d{2})', replace_ts, response)
    chat_history.append(("Assistant", processed_response))
    chat_html = "".join([f'<p><strong>{sender}:</strong> {msg}</p>' for sender, msg in chat_history])
    return chat_html, chat_history

def get_video_url():
    mmif_obj = get_mmif_object()
    video_docs = mmif_obj.get_documents_by_type(DocumentTypes.VideoDocument)
    if video_docs:
        video_doc = video_docs[0]
         # remove first 7 characters of the url if necessary
        return video_doc.get_property("location")[7:]
    return None

with gr.Blocks() as demo:
    gr.Markdown("# Chatbot and Video Player")
    with gr.Row():
        with gr.Column():
            chat_history_state = gr.State([])
            chat_display = gr.HTML()
            message_box = gr.Textbox(placeholder="Type your message here...")
            message_box.submit(chat_respond, inputs=[message_box, chat_history_state], outputs=[chat_display, chat_history_state])
        with gr.Column():
            gr.Markdown("## Video Player")
            video_url = get_video_url()
            print("Video URL", video_url)
            video_component = gr.Video(value=video_url, show_label=False, elem_id="video_player")
    demo.launch()