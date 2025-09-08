import gradio as gr
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from PIL import Image
from pydub import AudioSegment
from io import BytesIO
from typing import Dict, Any

load_dotenv()
# Paths to default fallback assets
DEFAULT_IMAGE_PATH = "astronaut_riding_a_horse.png"
DEFAULT_VIDEO_PATH = "a_young_man_walking_on_the_street.mp4"

# On startup, ensure default image and video exist (create simple placeholders if not)
def ensure_default_assets():
    if not os.path.exists(DEFAULT_IMAGE_PATH):
        # Create a simple blank image as fallback
        img = Image.new("RGB", (512, 512), color=(200, 200, 200))
        img.save(DEFAULT_IMAGE_PATH)
    if not os.path.exists(DEFAULT_VIDEO_PATH):
        # Create a 1-second blank video using ffmpeg if available, else just touch the file
        try:
            import subprocess
            subprocess.run([
                "ffmpeg", "-f", "lavfi", "-i", "color=c=gray:s=512x512:d=1", "-c:v", "libx264",
                "-t", "1", "-pix_fmt", "yuv420p", DEFAULT_VIDEO_PATH, "-y"
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            # If ffmpeg not available, just create an empty file
            with open(DEFAULT_VIDEO_PATH, "wb") as f:
                f.write(b"")

ensure_default_assets()

def set_hf_token(token):
    os.environ["HF_TOKEN"] = token

def get_hf_token():
    return os.getenv("HF_TOKEN")

def get_client():
    return InferenceClient(
        provider="fal-ai",
        api_key=get_hf_token()
    )

def text_to_image(prompt):
    client = get_client()
    try:
        image = client.text_to_image(prompt, model="Qwen/Qwen-Image")
        # Save image to a temporary file and return the file path for Gradio
        temp_path = "generated_image.png"
        image.save(temp_path)
        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            return temp_path
        else:
            return DEFAULT_IMAGE_PATH
    except Exception:
        return DEFAULT_IMAGE_PATH

def text_to_video(prompt):
    client = get_client()
    try:
        video_bytes = client.text_to_video(prompt, model="Wan-AI/Wan2.2-T2V-A14B")
        # Save video to a temporary file and return the file path for Gradio
        temp_path = "generated_video.mp4"
        with open(temp_path, "wb") as f:
            f.write(video_bytes)
        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            return temp_path
        else:
            return DEFAULT_VIDEO_PATH
    except Exception:
        return DEFAULT_VIDEO_PATH

def automatic_speech_recognition(audio_file):
    client = get_client()
    # audio_file is a tuple (filepath, numpy array) or just a filepath
    if isinstance(audio_file, tuple):
        audio_path = audio_file[0]
    else:
        audio_path = audio_file
    try:
        output = client.automatic_speech_recognition(audio_path, model="openai/whisper-large-v3")
        # output is a dict, get the transcription
        if isinstance(output, dict):
            return output.get("text", str(output))
        return str(output)
    except Exception as e:
        return f"Error: {e}"

with gr.Blocks() as demo:
    with gr.Row():
        hf_token = gr.Textbox(label="HF Token", value=get_hf_token())
        confirm_btn = gr.Button("Confirm")
        confirm_btn.click(set_hf_token, inputs=[hf_token])

    with gr.Row():
        text_to_image_prompt = gr.Textbox(label="Text to Image Prompt", value="Astronaut riding a horse")
        text_to_image_btn = gr.Button("Generate")
        text_to_image_output = gr.Image(label="Image Output", value=DEFAULT_IMAGE_PATH)
        text_to_image_btn.click(text_to_image, inputs=[text_to_image_prompt], outputs=[text_to_image_output])
        

    with gr.Row():
        text_to_video_prompt = gr.Textbox(label="Text to Video Prompt", value="A young man walking on the street")
        text_to_video_btn = gr.Button("Generate")
        text_to_video_output = gr.Video(label="Video Output", value=DEFAULT_VIDEO_PATH)
        text_to_video_btn.click(text_to_video, inputs=[text_to_video_prompt], outputs=[text_to_video_output])

    with gr.Row():
        automatic_speech_recognition_audio = gr.Audio(label="Automatic Speech Recognition Audio", type="filepath")
        automatic_speech_recognition_btn = gr.Button("Generate")
        automatic_speech_recognition_output = gr.Textbox(label="Automatic Speech Recognition Output")
        automatic_speech_recognition_btn.click(
            automatic_speech_recognition,
            inputs=[automatic_speech_recognition_audio],
            outputs=[automatic_speech_recognition_output]
        )

if __name__ == "__main__":
    demo.launch()
