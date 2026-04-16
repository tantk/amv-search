FROM python:3.11-slim

# Install ffmpeg + audio libs for librosa
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py pipeline.py search.py download.py ./
COPY static/ static/

# Create jobs dir
RUN mkdir -p jobs

# HF Spaces expects port 7860
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
