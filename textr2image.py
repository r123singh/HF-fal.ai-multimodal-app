import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

# initialize the client
client = InferenceClient(
    provider="fal-ai",
    api_key=os.getenv("HF_TOKEN"),
)

# output is a PIL.Image object
image = client.text_to_image(
    "Astronaut riding a horse",
    model="Qwen/Qwen-Image",
)

# image.save("astronaut_riding_a_horse.png")