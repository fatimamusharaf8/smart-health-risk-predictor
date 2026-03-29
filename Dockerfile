# ── Hugging Face Spaces Dockerfile ───────────────────────────────────────────
# Exposes the FastAPI app on port 7860 (HF default)
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached if requirements.txt unchanged)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Hugging Face Spaces exposes port 7860
EXPOSE 7860

# Start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
