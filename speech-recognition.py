import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
load_dotenv()

# initialize the client
client = InferenceClient(
    provider="fal-ai",
    api_key=os.getenv("HF_TOKEN"),
)

# output is a dict
output = client.automatic_speech_recognition("earnings_call.wav", model="openai/whisper-large-v3")

print(output)