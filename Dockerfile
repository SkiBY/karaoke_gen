FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# System deps: ffmpeg for audio conversion, pipx for spotdl.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fonts-liberation \
    pipx \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Modern Node.js (v20) — yt-dlp's JS runtime for solving YouTube's signature
# and n-param challenges. The distro's default node (v12) is far too old to run
# the EJS challenge solver, which leaves only image formats available.
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && node --version

# Install spotdl in isolation (it conflicts with modern FastAPI)
RUN pipx install spotdl && ln -s /root/.local/bin/spotdl /usr/local/bin/spotdl

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
