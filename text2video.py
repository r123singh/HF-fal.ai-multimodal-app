import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
load_dotenv()

client = InferenceClient(
    provider="fal-ai",
    api_key=os.getenv("HF_TOKEN"),
)

video = client.text_to_video(
    "A young man walking on the street",
    model="Wan-AI/Wan2.2-T2V-A14B",
)

with open("a_young_man_walking_on_the_street.mp4", "wb") as f:
    f.write(video)