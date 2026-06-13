FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# System deps: ffmpeg for audio conversion, git for pip installs if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fonts-liberation \
    pipx \
    && rm -rf /var/lib/apt/lists/*

# Install spotdl in isolation (it conflicts with modern FastAPI)
RUN pipx install spotdl && ln -s /root/.local/bin/spotdl /usr/local/bin/spotdl

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
