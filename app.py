"""
app.py — Hugging Face Spaces entry point
==========================================
This file wraps the FastAPI app so Hugging Face can discover and run it.
Hugging Face Spaces with SDK=docker or SDK=gradio need a specific entry point.

For Hugging Face with FastAPI:
  - Set SDK to "docker" in README.md metadata, OR
  - Use this file directly with: uvicorn app:app

Deploy steps:
  1. Create a new Space on huggingface.co (SDK: Docker or Gradio)
  2. Push your code including this file and the trained models/
  3. Set the start command to: uvicorn app:app --host 0.0.0.0 --port 7860
"""

from api.main import app   # re-export so HF / uvicorn finds it

# HF Spaces exposes port 7860 by default
# Run: uvicorn app:app --host 0.0.0.0 --port 7860
