# HF-fal.ai App Usage Guide

This document explains how to use the `app.py` Gradio application for multimodal AI inference (text-to-image, text-to-video, and speech recognition) using the Hugging Face Hub and the fal.ai provider.

---

## 1. Prerequisites

- Python 3.8+
- Install dependencies:
  ```
  pip install gradio huggingface_hub python-dotenv pillow pydub
  ```
- (Optional) For video fallback, `ffmpeg` is recommended.

- You need a Hugging Face token with access to the fal.ai provider.  
  Get one from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

---

## 2. Environment Setup

- Create a `.env` file in the project root with your Hugging Face token:
  ```
  HF_TOKEN=your_huggingface_token_here
  ```

---

## 3. Running the App

Start the Gradio app with:
```bash
python app.py
```

---

## 4. Usage

1. Set your Hugging Face token in the textbox.
2. Enter a prompt for text-to-image, text-to-video, or speech recognition. 
3. Click the "Generate" button to see the results.
