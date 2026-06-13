"""
Karaoke Generator — FastAPI backend
Handles file upload + YouTube download, vocal separation (Demucs),
transcription (faster-whisper), and ASS karaoke subtitle generation.
"""
import os
import re
import sys
import subprocess
import urllib.parse
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="Karaoke Generator")

WORK_DIR = Path("work")
WORK_DIR.mkdir(exist_ok=True)

# Initialize persistent catalog
from catalog import init_db, upsert_song, get_song, list_songs, count_songs, delete_song, _parse_artist_title  # noqa: E402, PLC0415
init_db()

FFMPEG = "/usr/bin/ffmpeg"

# In-memory job store (replace with Redis for production)
jobs: Dict[str, Dict[str, Any]] = {}


def _set(job_id: str, **kwargs) -> None:
    jobs[job_id].update(kwargs)


def _run(cmd: list, **kwargs) -> str:
    """Run a subprocess; raise RuntimeError with stderr on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-3000:] or result.stdout[-3000:])
    return result.stdout


# ── Routes ────────────────────────────────────────────────────────────────────

_TITLE_JUNK = re.compile(
    r"\s*[\(\[](official|music|video|audio|lyrics|hd|hq|mv|clip|live|feat\.?.*|"
    r"official\s+\w+\s+video|4k|full)[\)\]]",
    re.IGNORECASE,
)


def _clean_title(title: str) -> str:
    """Strip YouTube/video junk from title to improve lyrics search."""
    return _TITLE_JUNK.sub("", title).strip(" -–—")


def _fetch_lrclib(title: str) -> str:
    """Query lrclib.net — free API, good Cyrillic coverage."""
    try:
        import urllib.request, json  # noqa: PLC0415
        # Try to split "Artist - Title" for a better query
        parts = title.split(" - ", 1)
        if len(parts) == 2:
            artist, track = parts
        else:
            artist, track = "", title
        params = urllib.parse.urlencode({"artist_name": artist, "track_name": track})
        req = urllib.request.Request(
            f"https://lrclib.net/api/search?{params}",
            headers={"User-Agent": "karaoke-gen/1.0"},
        )
        with urllib.request.urlopen(req, timeout=8) as r:
            results = json.loads(r.read())
        for item in results:
            plain = (item.get("plainLyrics") or "").strip()
            if plain:
                return plain
    except Exception:
        pass
    return ""


def _parse_lrc(lrc_text: str):
    """Parse LRC text into list of (start_seconds, text) tuples.

    Returns (synced_lines, plain_text) where synced_lines is a list of
    (start_sec, line_text) or None if no timestamps found.
    """
    lines = []
    for line in lrc_text.splitlines():
        m = re.match(r"\[(\d+):(\d+(?:\.\d+)?)\]\s*(.*)", line)
        if m:
            mins, secs, text = int(m.group(1)), float(m.group(2)), m.group(3).strip()
            if text:
                lines.append((mins * 60 + secs, text))
    if lines:
        lines.sort(key=lambda x: x[0])
        plain = "\n".join(text for _, text in lines)
        return lines, plain
    # No timestamps found — return as plain text
    plain = re.sub(r"\[[\d:.]+\]", "", lrc_text).strip()
    return None, plain


def _fetch_yandex_lyrics(title: str) -> str:
    """Try to fetch lyrics from Yandex Music search API (no auth needed for search)."""
    try:
        import urllib.request, json  # noqa: PLC0415
        parts = title.split(" - ", 1)
        if len(parts) == 2:
            query = f"{parts[0]} {parts[1]}"
        else:
            query = title
        params = urllib.parse.urlencode({"text": query, "type": "track", "page": 0})
        req = urllib.request.Request(
            f"https://music.yandex.ru/handlers/music-search.jsx?{params}",
            headers={"User-Agent": "karaoke-gen/1.0", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read())
        tracks = data.get("tracks", {}).get("items", [])
        if not tracks:
            return ""
        track_id = tracks[0].get("id")
        album_id = tracks[0].get("albums", [{}])[0].get("id", "")
        if not track_id:
            return ""
        # Fetch lyrics supplement
        lyric_req = urllib.request.Request(
            f"https://music.yandex.ru/api/v2.1/handlers/track/{track_id}:{album_id}/lyrics/json",
            headers={"User-Agent": "karaoke-gen/1.0", "Accept": "application/json"},
        )
        with urllib.request.urlopen(lyric_req, timeout=8) as r:
            lyric_data = json.loads(r.read())
        text = lyric_data.get("lyrics", {}).get("fullLyrics", "")
        if text:
            return text.strip()
    except Exception:
        pass
    return ""


def _fetch_genius(title: str) -> str:
    """Fetch plain lyrics from Genius via lyricsgenius."""
    try:
        import lyricsgenius  # noqa: PLC0415
        # Uses GENIUS_ACCESS_TOKEN env var, or skip if not set
        token = os.environ.get("GENIUS_ACCESS_TOKEN", "")
        if not token:
            return ""
        genius = lyricsgenius.Genius(token, verbose=False, timeout=10, retries=1)
        genius.remove_section_headers = True

        parts = title.split(" - ", 1)
        if len(parts) == 2:
            song = genius.search_song(parts[1].strip(), parts[0].strip())
        else:
            song = genius.search_song(title)
        if song and song.lyrics:
            # Strip the trailing "...Embed" junk Genius appends
            text = re.sub(r"\d*Embed$", "", song.lyrics).strip()
            # Strip the title header line if present
            lines = text.split("\n")
            if lines and lines[0].endswith("Lyrics"):
                lines = lines[1:]
            return "\n".join(lines).strip()
    except Exception:
        pass
    return ""


def _fetch_lyrics(title: str):
    """Fetch lyrics from multiple sources.

    Returns (synced_lines, plain_text) where synced_lines is a list of
    (start_sec, line_text) or None if only plain text available.

    Source chain:
      1. syncedlyrics (Spotify, Musixmatch, Genius, NetEase)
      2. lrclib.net (free API, good Cyrillic coverage)
      3. Genius via lyricsgenius (plain text fallback)
    """
    clean = _clean_title(title)
    # 1. syncedlyrics (Spotify, Musixmatch, Genius, NetEase…)
    try:
        import syncedlyrics  # noqa: PLC0415
        lrc = syncedlyrics.search(clean) or syncedlyrics.search(title)
        if lrc:
            synced, plain = _parse_lrc(lrc)
            if plain:
                return synced, plain
    except Exception:
        pass
    # 2. lrclib.net fallback
    for q in (clean, title):
        plain = _fetch_lrclib(q)
        if plain:
            return None, plain
    # 3. Yandex Music (good for Russian/Belarusian)
    for q in (clean, title):
        plain = _fetch_yandex_lyrics(q)
        if plain:
            return None, plain
    # 4. Genius (plain text only)
    for q in (clean, title):
        plain = _fetch_genius(q)
        if plain:
            return None, plain
    return None, ""


def _detect_chorus(lines: list[str], min_block: int = 2, min_repeats: int = 2) -> list[tuple[int, int]]:
    """Detect chorus blocks — groups of consecutive lines that repeat in the lyrics.

    Returns list of (start_line_idx, end_line_idx) for chorus occurrences
    AFTER the first one (i.e. the repeated copies to remove).
    """
    if len(lines) < min_block * min_repeats:
        return []

    norm_lines = [re.sub(r"[^\w\s]", "", l.lower()).strip() for l in lines]

    # Try block sizes from largest to smallest
    found_ranges: list[tuple[int, int]] = []
    used = set()

    for block_size in range(min(8, len(lines) // 2), min_block - 1, -1):
        for i in range(len(norm_lines) - block_size + 1):
            if any(j in used for j in range(i, i + block_size)):
                continue
            block = tuple(norm_lines[i:i + block_size])
            if not any(b for b in block):  # skip empty blocks
                continue

            # Find all occurrences of this block
            occurrences = []
            for j in range(len(norm_lines) - block_size + 1):
                if tuple(norm_lines[j:j + block_size]) == block:
                    occurrences.append(j)

            if len(occurrences) >= min_repeats:
                # Mark first occurrence as "keep", rest as chorus repeats
                for idx, occ in enumerate(occurrences):
                    occ_range = range(occ, occ + block_size)
                    if any(j in used for j in occ_range):
                        continue
                    for j in occ_range:
                        used.add(j)
                    if idx > 0:  # skip first occurrence
                        found_ranges.append((occ, occ + block_size))

    return sorted(found_ranges)


def _remove_chorus_lines(lyrics: str) -> str:
    """Remove repeated chorus blocks from lyrics, keeping only the first occurrence."""
    lines = [l for l in lyrics.splitlines() if l.strip()]
    if not lines:
        return lyrics

    chorus_ranges = _detect_chorus(lines)
    if not chorus_ranges:
        return lyrics

    remove_idxs = set()
    for start, end in chorus_ranges:
        for i in range(start, end):
            remove_idxs.add(i)

    return "\n".join(l for i, l in enumerate(lines) if i not in remove_idxs)


def _mark_chorus_lines(lyrics: str) -> tuple[str, list[tuple[int, int]]]:
    """Return lyrics with chorus lines marked and the chorus ranges.

    Returns (annotated_lyrics, chorus_ranges) where chorus_ranges
    are the indices of repeated blocks.
    """
    lines = [l for l in lyrics.splitlines() if l.strip()]
    chorus_ranges = _detect_chorus(lines)
    return lyrics, chorus_ranges


@app.post("/api/jobs")
async def create_job(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    youtube_url: Optional[str] = Form(None),
    model: str = Form("medium"),
    language: str = Form("auto"),
    lyrics_hint: Optional[str] = Form(None),
    word_timing: bool = Form(True),
    static_video: bool = Form(False),
    keep_chorus: bool = Form(True),
    show_bg_lyrics: bool = Form(False),
    display_mode: str = Form("subtitles"),  # "subtitles", "background", "both"
    video_bg: str = Form("color"),  # "color" (dark bg), "original" (keep YT video), "cover" (thumbnail intro)
    cover_image: Optional[UploadFile] = File(None),
):
    url = (youtube_url or "").strip()
    if not file and not url:
        raise HTTPException(400, "Provide a file or a URL (YouTube, Spotify, SoundCloud)")

    job_id = str(uuid.uuid4())
    job_dir = WORK_DIR / job_id
    job_dir.mkdir()
    jobs[job_id] = {"status": "pending", "step": "Queued", "pct": 0, "error": "", "files": {}}

    hint = (lyrics_hint or "").strip()

    # Save cover image if uploaded
    cover_path = ""
    if cover_image and cover_image.filename:
        cover_ext = Path(cover_image.filename).suffix or ".jpg"
        cover_p = job_dir / f"cover{cover_ext}"
        cover_p.write_bytes(await cover_image.read())
        cover_path = str(cover_p)

    if file and file.filename:
        suffix = Path(file.filename).suffix or ".mp3"
        input_path = job_dir / f"input{suffix}"
        input_path.write_bytes(await file.read())
        file_title = Path(file.filename).stem
        _set(job_id, title=file_title)
        artist, song_title = _parse_artist_title(file_title)
        upsert_song(job_id, title=file_title, artist=artist, lyrics=hint)
        background_tasks.add_task(process_audio, job_id, str(input_path), model, language, hint, word_timing, static_video, keep_chorus, display_mode, video_bg, cover_path)
    else:
        _set(job_id, title="track", youtube_url=url)
        upsert_song(job_id, title="track", source_url=url, lyrics=hint)
        background_tasks.add_task(process_url, job_id, url, model, language, hint, word_timing, static_video, keep_chorus, display_mode, video_bg, cover_path)

    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


@app.get("/api/jobs/{job_id}/download/{file_key}")
def download_file(job_id: str, file_key: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    path_str = jobs[job_id].get("files", {}).get(file_key)
    if not path_str:
        raise HTTPException(404, "File not ready")
    p = Path(path_str)
    return FileResponse(str(p), filename=p.name, media_type="application/octet-stream")


def _detect_source(url: str) -> str:
    """Detect the source platform from URL."""
    u = url.lower()
    if "spotify.com" in u or "open.spotify" in u:
        return "spotify"
    if "soundcloud.com" in u:
        return "soundcloud"
    if "music.yandex" in u:
        return "yandex"
    return "youtube"


# ── Background processors ─────────────────────────────────────────────────────

def _fetch_yt_subtitles(url: str, job_dir: Path) -> str:
    """Try to download subtitles from a YouTube video via yt-dlp. Returns plain text or ''."""
    try:
        sub_tpl = str(job_dir / "subs.%(ext)s")
        _run([
            "yt-dlp", "--skip-download",
            "--write-subs", "--write-auto-subs",
            "--sub-langs", "ru,be,uk,en",
            "--sub-format", "srv3/vtt/srt/best",
            "--convert-subs", "srt",
            "-o", sub_tpl, url,
        ])
        for f in sorted(job_dir.glob("subs.*.srt")):
            raw = f.read_text(encoding="utf-8", errors="ignore")
            # Strip SRT timecodes and indices, keep only text
            text = re.sub(r"\d+\n\d{2}:\d{2}:\d{2},\d+ --> [^\n]+\n", "", raw)
            text = re.sub(r"<[^>]+>", "", text)   # strip HTML tags
            text = re.sub(r"\n{2,}", "\n", text).strip()
            if text:
                return text
    except Exception:
        pass
    return ""


def _download_spotify(url: str, job_dir: Path) -> tuple[str, str]:
    """Download audio from Spotify via spotdl CLI. Returns (audio_path, title)."""
    _run([
        "spotdl",
        "--output", str(job_dir / "{artists} - {title}.{output-ext}"),
        "--format", "mp3",
        "--threads", "1",
        url,
    ])
    mp3_files = list(job_dir.glob("*.mp3"))
    if not mp3_files:
        raise RuntimeError("spotdl produced no output file")
    audio_path = mp3_files[0]
    title = audio_path.stem  # "Artists - Title" from spotdl naming
    return str(audio_path), title


def _download_yt_dlp(url: str, job_dir: Path) -> tuple[str, str]:
    """Download audio via yt-dlp (YouTube, SoundCloud, etc). Returns (audio_path, title)."""
    output_tpl = str(job_dir / "input.%(ext)s")
    _run(["yt-dlp", "-x", "--audio-format", "mp3", "-o", output_tpl, url])

    # Best-effort title
    title = ""
    try:
        title = _run(["yt-dlp", "--get-title", "--no-playlist", url]).strip()
    except Exception:
        pass

    audio_files = list(job_dir.glob("input.*"))
    if not audio_files:
        raise RuntimeError("yt-dlp produced no output file")
    return str(audio_files[0]), title


def _download_yt_video(url: str, job_dir: Path) -> str:
    """Download full video from YouTube. Returns path to the video file."""
    output_tpl = str(job_dir / "original_video.%(ext)s")
    _run(["yt-dlp", "-f", "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
          "--merge-output-format", "mp4", "-o", output_tpl, url])
    for f in job_dir.glob("original_video.*"):
        return str(f)
    return ""


def _fetch_yt_thumbnail(url: str, job_dir: Path) -> str:
    """Download YouTube thumbnail. Returns path or ''."""
    try:
        _run(["yt-dlp", "--skip-download", "--write-thumbnail",
              "--convert-thumbnails", "jpg",
              "-o", str(job_dir / "yt_thumb.%(ext)s"), url])
        for f in job_dir.glob("yt_thumb*.jpg"):
            return str(f)
    except Exception:
        pass
    return ""


def process_url(job_id: str, url: str, model: str, language: str = "auto", lyrics_hint: str = "", word_timing: bool = True, static_video: bool = False, keep_chorus: bool = True, display_mode: str = "subtitles", video_bg: str = "color", cover_path: str = "") -> None:
    job_dir = WORK_DIR / job_id
    source = _detect_source(url)
    try:
        _set(job_id, status="running",
             step=f"Downloading audio from {source.title()}...")

        if source == "spotify":
            audio_path, title = _download_spotify(url, job_dir)
        else:
            audio_path, title = _download_yt_dlp(url, job_dir)

        if title:
            _set(job_id, title=title)
            artist, song_title = _parse_artist_title(title)
            upsert_song(job_id, title=title, artist=artist, source_url=url)

        # Download full video if user wants original video background
        original_video = ""
        if video_bg == "original" and source == "youtube":
            _set(job_id, step="Downloading full video from YouTube...")
            original_video = _download_yt_video(url, job_dir)

        # Fetch YouTube thumbnail as cover if no custom cover provided
        if not cover_path and source == "youtube" and video_bg == "cover":
            _set(job_id, step="Fetching YouTube thumbnail...")
            cover_path = _fetch_yt_thumbnail(url, job_dir)

        # Try to get subtitles (YouTube/YT Music only)
        if not lyrics_hint and source == "youtube":
            _set(job_id, step="Checking YouTube for subtitles...")
            lyrics_hint = _fetch_yt_subtitles(url, job_dir)
            if lyrics_hint:
                _set(job_id, lyrics_found=True, lyrics_text=lyrics_hint)

        process_audio(job_id, audio_path, model, language, lyrics_hint, word_timing, static_video, keep_chorus, display_mode, video_bg, cover_path, original_video)
    except Exception as exc:
        _set(job_id, status="error", step="Failed", error=str(exc))


def process_audio(job_id: str, input_path: str, model: str, language: str = "auto", lyrics_hint: str = "", word_timing: bool = True, static_video: bool = False, keep_chorus: bool = True, display_mode: str = "subtitles", video_bg: str = "color", cover_path: str = "", original_video: str = "") -> None:
    job_dir = WORK_DIR / job_id
    title = jobs[job_id].get("title", "track")
    safe = "".join(c for c in title if c.isalnum() or c in " -_").strip() or "track"

    try:
        # ── 1. Vocal separation ───────────────────────────────────────────────
        _set(job_id, status="running", step="Separating vocals with Demucs (0%)...", pct=5)

        input_p = Path(input_path)
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(
            [sys.executable, "-m", "demucs",
             "--two-stems", "vocals", "-n", "htdemucs",
             "--out", str(job_dir), input_path],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )
        output_buf = ""
        all_output = []
        while True:
            chunk = proc.stdout.read(256)
            if not chunk:
                break
            all_output.append(chunk)
            output_buf += chunk
            # tqdm uses \r to overwrite lines
            for part in re.split(r"[\r\n]", output_buf):
                m = re.search(r"(\d+)%\|", part)
                if m:
                    demucs_pct = int(m.group(1))
                    _set(job_id, step=f"Separating vocals with Demucs ({demucs_pct}%)...",
                         pct=5 + int(demucs_pct * 0.35))
            output_buf = re.split(r"[\r\n]", output_buf)[-1]
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError("".join(all_output)[-3000:])

        # Locate Demucs outputs (nested: <out>/<model>/<stem>/{vocals,no_vocals}.wav)
        no_vocals_candidates = list(job_dir.rglob("no_vocals.wav"))
        if not no_vocals_candidates:
            raise RuntimeError("Demucs no_vocals.wav not found in output")
        no_vocals_wav = no_vocals_candidates[0]
        vocals_candidates = list(job_dir.rglob("vocals.wav"))
        vocals_wav = vocals_candidates[0] if vocals_candidates else None

        # Convert WAV → MP3 (requires ffmpeg)
        minus_path = job_dir / f"{safe}_minus.mp3"
        _run([FFMPEG, "-i", str(no_vocals_wav), "-q:a", "2", str(minus_path), "-y"])
        jobs[job_id]["files"]["minus"] = str(minus_path)

        # ── 2a. Static text video (skip transcription entirely) ───────────────
        if (static_video or display_mode == "background") and lyrics_hint:
            _set(job_id, step="Generating static lyrics video...", pct=50)
            title = jobs[job_id].get("title", "")
            safe2 = "".join(c for c in title if c.isalnum() or c in " -_").strip() or "track"
            from ass_gen import generate_static_ass  # noqa: PLC0415
            import subprocess as _sp  # noqa: PLC0415
            # Get audio duration via ffprobe
            dur_out = _run([
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", input_path,
            ])
            duration = float(dur_out.strip() or "0") or 300.0
            ass_path = job_dir / f"{safe2}_karaoke.ass"
            generate_static_ass(lyrics_hint, duration, str(ass_path))
            jobs[job_id]["files"]["ass"] = str(ass_path)
            _set(job_id, step="Rendering video...", pct=70)
            video_path = job_dir / f"{safe2}_karaoke.mp4"
            _run([
                FFMPEG,
                "-f", "lavfi", "-i", "color=c=0x0d0d1a:size=1920x1080:rate=25",
                "-i", str(minus_path),
                "-vf", (
                    "ass={ass},"
                    "drawtext=text='{txt}'"
                    ":fontfile=/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
                    ":fontsize=36:fontcolor=white@0.7"
                    ":x=(w-text_w)/2:y=30"
                    ":shadowcolor=black@0.6:shadowx=2:shadowy=2"
                ).format(
                    ass=ass_path,
                    txt=title.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:"),
                ),
                "-shortest",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.1",
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                str(video_path), "-y",
            ])
            jobs[job_id]["files"]["video"] = str(video_path)
            _set(job_id, status="done", step="Done!", pct=100)
            title = jobs[job_id].get("title", "track")
            artist, _ = _parse_artist_title(title)
            upsert_song(job_id, title=title, artist=artist, status="done",
                        video_path=str(video_path), minus_path=str(minus_path),
                        ass_path=str(ass_path))
            return

        # ── 2. Lyrics + transcription + render ───────────────────────────────
        title = jobs[job_id].get("title", "")
        _set(job_id, step="Searching for lyrics online...", pct=42)
        synced_lines = None
        if lyrics_hint:
            lyrics = lyrics_hint
        else:
            synced_lines, lyrics = _fetch_lyrics(title)
        if lyrics:
            # Detect chorus in lyrics
            lines_for_detect = [l for l in lyrics.splitlines() if l.strip()]
            chorus_ranges = _detect_chorus(lines_for_detect)
            _set(job_id, lyrics_found=True, lyrics_text=lyrics,
                 chorus_detected=len(chorus_ranges) > 0,
                 chorus_count=len(chorus_ranges))

            if not keep_chorus and lyrics:
                lyrics = _remove_chorus_lines(lyrics)
                if lyrics_hint:
                    lyrics_hint = lyrics  # update hint too
                if synced_lines:
                    # Filter synced lines to match remaining lyrics
                    remaining = set(re.sub(r"[^\w\s]", "", l.lower()).strip()
                                    for l in lyrics.splitlines() if l.strip())
                    synced_lines = [(t, text) for t, text in synced_lines
                                   if re.sub(r"[^\w\s]", "", text.lower()).strip() in remaining]

        # Store params so retries can skip Demucs
        _set(job_id,
             _input_path=input_path,
             _vocals_wav=str(vocals_wav) if vocals_wav else None,
             _minus_path=str(minus_path),
             _model=model,
             _language=language,
             _lyrics_hint=lyrics_hint,
             _lyrics=lyrics,
             _synced_lines=synced_lines,
             _safe=safe,
             _word_timing=word_timing,
             _keep_chorus=keep_chorus,
             _display_mode=display_mode,
             _video_bg=video_bg,
             _cover_path=cover_path,
             _original_video=original_video,
             retry_count=0,
        )

        _run_transcription_and_render(
            job_id=job_id,
            input_path=input_path,
            vocals_wav=str(vocals_wav) if vocals_wav else None,
            minus_path=str(minus_path),
            model=model,
            language=language,
            lyrics_hint=lyrics_hint,
            lyrics=lyrics,
            synced_lines=synced_lines,
            safe=safe,
            title=title,
            job_dir=job_dir,
            whisper_settings=RETRY_SETTINGS[0],
            word_timing=word_timing,
            display_mode=display_mode,
            video_bg=video_bg,
            cover_path=cover_path,
            original_video=original_video,
        )

    except Exception as exc:
        _set(job_id, status="error", step="Failed", error=str(exc))


# ── Retry settings progression ────────────────────────────────────────────────
RETRY_SETTINGS = [
    {"beam_size": 5,  "no_speech_threshold": 0.6, "temperature": 0.0},
    {"beam_size": 10, "no_speech_threshold": 0.4, "temperature": 0.0},
    {"beam_size": 10, "no_speech_threshold": 0.3, "temperature": [0.0, 0.2, 0.4]},
]


class LyricsPayload(BaseModel):
    lyrics: str


@app.post("/api/jobs/{job_id}/update-lyrics")
async def update_lyrics(job_id: str, payload: LyricsPayload, background_tasks: BackgroundTasks):
    """Update lyrics and re-run transcription with user-provided text."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job.get("status") not in ("done", "error"):
        raise HTTPException(400, "Job still running")

    new_lyrics = payload.lyrics.strip()
    _set(job_id, lyrics_text=new_lyrics, _lyrics_hint=new_lyrics, _lyrics=new_lyrics,
         retry_count=0, status="running", step="Re-running with updated lyrics...", pct=45,
         error="", text_stars=None, video_stars=None)
    upsert_song(job_id, lyrics=new_lyrics)
    background_tasks.add_task(retry_transcription, job_id)
    return {"retrying": True}


class RatingPayload(BaseModel):
    text_stars: int
    video_stars: int


@app.post("/api/jobs/{job_id}/rate")
async def rate_job(job_id: str, payload: RatingPayload, background_tasks: BackgroundTasks):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job.get("status") != "done":
        raise HTTPException(400, "Job not done yet")

    _set(job_id, text_stars=payload.text_stars, video_stars=payload.video_stars)

    retry_count = job.get("retry_count", 0)
    if (payload.text_stars < 4 or payload.video_stars < 4) and retry_count < len(RETRY_SETTINGS) - 1:
        next_retry = retry_count + 1
        _set(job_id, retry_count=next_retry)
        background_tasks.add_task(retry_transcription, job_id)
        return {"retrying": True, "attempt": next_retry}

    return {"retrying": False}


def retry_transcription(job_id: str) -> None:
    """Re-run transcription + ASS + video with next retry settings. Skips Demucs."""
    job = jobs[job_id]
    retry_count = job.get("retry_count", 1)
    settings = RETRY_SETTINGS[min(retry_count, len(RETRY_SETTINGS) - 1)]

    input_path = job.get("_input_path")
    vocals_wav = job.get("_vocals_wav")
    minus_path = job.get("_minus_path")
    model = job.get("_model")
    language = job.get("_language", "auto")
    lyrics_hint = job.get("_lyrics_hint", "")
    lyrics = job.get("_lyrics", "")
    synced_lines = job.get("_synced_lines")
    safe = job.get("_safe", "track")
    title = job.get("title", "")
    word_timing = job.get("_word_timing", False)
    display_mode = job.get("_display_mode", "subtitles")
    video_bg = job.get("_video_bg", "color")
    cover_path = job.get("_cover_path", "")
    original_video = job.get("_original_video", "")

    if not input_path or not minus_path:
        _set(job_id, status="error", step="Failed", error="Missing job params for retry")
        return

    job_dir = WORK_DIR / job_id
    _set(job_id, status="running", step=f"Retrying transcription (attempt {retry_count})...", pct=45,
         error="", text_stars=None, video_stars=None)

    try:
        _run_transcription_and_render(
            job_id=job_id,
            input_path=input_path,
            vocals_wav=vocals_wav,
            minus_path=minus_path,
            model=model,
            language=language,
            lyrics_hint=lyrics_hint,
            lyrics=lyrics,
            synced_lines=synced_lines,
            safe=safe,
            title=title,
            job_dir=job_dir,
            whisper_settings=settings,
            word_timing=word_timing,
            display_mode=display_mode,
            video_bg=video_bg,
            cover_path=cover_path,
            original_video=original_video,
        )
    except Exception as exc:
        _set(job_id, status="error", step="Failed", error=str(exc))


def _transfer_whisper_timing(seg, lyric_words: list[str]):
    """Try to map Whisper's word timestamps onto lyric words via greedy alignment.

    Returns a list of SimpleNamespace words with timing, or None if alignment fails.
    """
    from types import SimpleNamespace  # noqa: PLC0415

    whisper_words = getattr(seg, "words", None)
    if not whisper_words or len(whisper_words) == 0:
        return None

    # Normalize for comparison
    def _nw(s):
        return re.sub(r"[^\w]", "", s.lower())

    w_list = [(_nw(w.word), w) for w in whisper_words]
    l_list = [(_nw(w), w) for w in lyric_words]

    if not w_list or not l_list:
        return None

    # Greedy alignment: for each lyric word, find the best matching Whisper word
    # that hasn't been used yet and is in roughly the right position
    used = set()
    aligned = []
    for li, (l_norm, l_text) in enumerate(l_list):
        best_wi, best_score = None, -1
        # Search window: allow some drift but prefer positional match
        expected_pos = li / len(l_list) * len(w_list)
        for wi, (w_norm, w_obj) in enumerate(w_list):
            if wi in used:
                continue
            # Exact or fuzzy match
            if l_norm == w_norm:
                score = 2.0
            elif l_norm in w_norm or w_norm in l_norm:
                score = 1.0
            elif len(l_norm) > 2 and len(w_norm) > 2 and (l_norm[:3] == w_norm[:3]):
                score = 0.5
            else:
                continue
            # Penalize position distance
            pos_penalty = abs(wi - expected_pos) / max(len(w_list), 1)
            score -= pos_penalty * 0.3
            if score > best_score:
                best_score, best_wi = score, wi

        if best_wi is not None:
            used.add(best_wi)
            w_obj = w_list[best_wi][1]
            aligned.append(SimpleNamespace(
                word=f" {l_text}", start=w_obj.start, end=w_obj.end, probability=1.0,
            ))
        else:
            aligned.append(None)  # No match — will interpolate

    # Fill gaps by interpolation from neighbors
    for i, a in enumerate(aligned):
        if a is not None:
            continue
        # Find nearest matched neighbors
        prev_end = seg.start
        next_start = seg.end
        for j in range(i - 1, -1, -1):
            if aligned[j] is not None:
                prev_end = aligned[j].end
                break
        for j in range(i + 1, len(aligned)):
            if aligned[j] is not None:
                next_start = aligned[j].start
                break
        # Count unmatched words in this gap
        gap_count = 0
        gap_start_idx = i
        for j in range(i, len(aligned)):
            if aligned[j] is None:
                gap_count += 1
            else:
                break
        gap_dur = max(next_start - prev_end, 0.01)
        char_lens = [max(len(lyric_words[gap_start_idx + k]), 1) for k in range(gap_count)]
        total_c = sum(char_lens)
        cum = 0
        for k in range(gap_count):
            idx = gap_start_idx + k
            if aligned[idx] is not None:
                continue
            frac_s = cum / total_c
            cum += char_lens[k]
            frac_e = cum / total_c
            aligned[idx] = SimpleNamespace(
                word=f" {lyric_words[idx]}",
                start=prev_end + gap_dur * frac_s,
                end=prev_end + gap_dur * frac_e,
                probability=0.5,
            )

    # If too few words matched directly, don't trust this alignment
    matched = sum(1 for a in aligned if a is not None and a.probability == 1.0)
    if matched < len(lyric_words) * 0.3:
        return None

    return aligned


def _align_to_lyrics(segments, lyric_lines: list[str]):
    """Replace each segment's text with the closest matching lyric line.

    Keeps Whisper's word-level timestamps intact; only replaces the text so
    the output uses exact spelling/wording from the user-provided lyrics.
    Words are redistributed proportionally across the segment's time range.
    """
    import difflib  # noqa: PLC0415
    from types import SimpleNamespace  # noqa: PLC0415

    used: set[int] = set()
    result = []

    for seg in segments:
        seg_words = re.findall(r"\S+", seg.text.strip())
        if not seg_words:
            result.append(seg)
            continue

        # Find the best matching unused lyric line
        best_idx, best_score = None, -1.0
        for i, line in enumerate(lyric_lines):
            if i in used:
                continue
            score = difflib.SequenceMatcher(
                None,
                _norm_align(seg.text),
                _norm_align(line),
            ).ratio()
            if score > best_score:
                best_score, best_idx = score, i

        if best_idx is None or best_score < 0.15:
            result.append(seg)
            continue

        used.add(best_idx)
        lyric_text = lyric_lines[best_idx]
        lyric_words = re.findall(r"\S+", lyric_text)

        # Redistribute lyric words across segment time range, weighted by character length
        seg_start = seg.start
        seg_end = seg.end
        seg_dur = max(seg_end - seg_start, 0.01)

        # If we have Whisper word timestamps, try to align word-by-word
        new_words = _transfer_whisper_timing(seg, lyric_words)
        if not new_words:
            # Fallback: distribute by character length (longer words get more time)
            char_lengths = [max(len(w), 1) for w in lyric_words]
            total_chars = sum(char_lengths)
            cumulative = 0
            new_words = []
            for wi, word in enumerate(lyric_words):
                frac_start = cumulative / total_chars
                cumulative += char_lengths[wi]
                frac_end = cumulative / total_chars
                w_start = seg_start + seg_dur * frac_start
                w_end = seg_start + seg_dur * frac_end
                new_words.append(SimpleNamespace(word=f" {word}", start=w_start, end=w_end,
                                                 probability=1.0))

        new_seg = SimpleNamespace(
            start=seg_start, end=seg_end,
            text=f" {lyric_text}",
            words=new_words,
        )
        result.append(new_seg)

    return result


def _norm_align(s: str) -> str:
    """Normalise text for alignment comparison."""
    _CYR = str.maketrans("ўіІЎёЁ", "уиИУеЕ")
    return re.sub(r"[^\w\s]", "", s.lower().translate(_CYR))


def _run_transcription_and_render(
    job_id, input_path, vocals_wav, minus_path, model, language,
    lyrics_hint, lyrics, safe, title, job_dir, whisper_settings=None, word_timing=False,
    synced_lines=None, display_mode="subtitles", video_bg="color", cover_path="", original_video="",
):
    if whisper_settings is None:
        whisper_settings = RETRY_SETTINGS[0]

    lang_label = language if language != "auto" else "auto-detect"
    _set(job_id, step=f"Transcribing with Whisper ({model}, {lang_label})...", pct=45)

    from faster_whisper import WhisperModel  # noqa: PLC0415

    device = os.environ.get("WHISPER_DEVICE", "cuda")
    compute_type = "int8_float16" if device == "cuda" else "int8"
    try:
        wm = WhisperModel(model, device=device, compute_type=compute_type)
    except Exception:
        device = "cpu"
        wm = WhisperModel(model, device="cpu", compute_type="int8")

    BE_PROMPT = "Беларуская мова. Словы песні па-беларуску."
    BE_PROMPT_WORDS = set(re.findall(r"\w+", BE_PROMPT.lower()))
    if language == "be":
        initial_prompt = lyrics_hint[:1000] if lyrics_hint else BE_PROMPT
    else:
        initial_prompt = lyrics[:1000] if lyrics else None

    transcribe_path = vocals_wav if vocals_wav else input_path

    def _do_transcribe(w):
        gen, inf = w.transcribe(
            transcribe_path,
            word_timestamps=True,
            language=None if language == "auto" else language,
            initial_prompt=initial_prompt,
            condition_on_previous_text=False,
            **whisper_settings,
        )
        return gen, inf

    try:
        segments_gen, info = _do_transcribe(wm)
        segments_list_raw = list(segments_gen)
    except Exception as e:
        if "out of memory" in str(e).lower() and device == "cuda":
            _set(job_id, step=f"GPU OOM — retrying on CPU ({model})...", pct=45)
            wm = WhisperModel(model, device="cpu", compute_type="int8")
            segments_gen, info = _do_transcribe(wm)
            segments_list_raw = list(segments_gen)
        else:
            raise

    _CYR_NORM = str.maketrans("ўіІЎёЁ", "уиИУеЕ")

    def _norm(s):
        return s.lower().translate(_CYR_NORM)

    lyric_words: set[str] = set()
    if lyrics:
        for w in re.findall(r"\w+", _norm(lyrics)):
            lyric_words.add(w)

    def _is_lyric_segment(text):
        words = re.findall(r"\w+", _norm(text))
        if not words:
            return False
        if BE_PROMPT_WORDS and all(w in BE_PROMPT_WORDS for w in re.findall(r"\w+", text.lower())):
            return False
        if not lyric_words:
            return True
        threshold = 0.25 if lyrics_hint else 0.3
        overlap = sum(1 for w in words if w in lyric_words) / len(words)
        return overlap >= threshold

    filtered = []
    duration = info.duration or 1
    text_counts: dict[str, int] = {}
    for seg in segments_list_raw:
        normalized = seg.text.strip().lower()
        text_counts[normalized] = text_counts.get(normalized, 0) + 1
        if text_counts[normalized] >= 3:
            continue
        if not _is_lyric_segment(seg.text):
            continue
        filtered.append(seg)
        seg_pct = min(99, int(seg.end / duration * 100))
        _set(job_id, step=f"Transcribing with Whisper ({model}, {lang_label}) — {seg_pct}%...",
             pct=45 + int(seg_pct * 0.35))

    # If we have synced LRC lines, use them as segment anchors (better timing than Whisper segments)
    if synced_lines and not lyrics_hint:
        from types import SimpleNamespace  # noqa: PLC0415
        segments = []
        for i, (start_sec, line_text) in enumerate(synced_lines):
            end_sec = synced_lines[i + 1][0] if i + 1 < len(synced_lines) else (
                filtered[-1].end if filtered else start_sec + 5.0
            )
            words = re.findall(r"\S+", line_text)
            if not words:
                continue
            # Try to find a Whisper segment near this time to borrow word timestamps
            best_seg = None
            best_overlap = 0
            for seg in filtered:
                overlap = min(seg.end, end_sec) - max(seg.start, start_sec)
                if overlap > best_overlap:
                    best_overlap, best_seg = overlap, seg

            new_words = None
            if best_seg:
                new_words = _transfer_whisper_timing(best_seg, words)
            if not new_words:
                # Character-weighted fallback
                dur = max(end_sec - start_sec, 0.1)
                char_lens = [max(len(w), 1) for w in words]
                total_c = sum(char_lens)
                cum = 0
                new_words = []
                for w in words:
                    frac_s = cum / total_c
                    cum += max(len(w), 1)
                    frac_e = cum / total_c
                    new_words.append(SimpleNamespace(
                        word=f" {w}", start=start_sec + dur * frac_s,
                        end=start_sec + dur * frac_e, probability=1.0,
                    ))
            segments.append(SimpleNamespace(
                start=start_sec, end=end_sec, text=f" {line_text}", words=new_words,
            ))
    elif lyrics_hint and filtered:
        # User provided lyrics — replace text with exact lyric lines (keep timestamps)
        lyric_lines = [l.strip() for l in lyrics_hint.splitlines() if l.strip()]
        segments = _align_to_lyrics(filtered, lyric_lines)
    else:
        segments = filtered

    _set(job_id, step="Generating karaoke subtitles...", pct=82)
    from ass_gen import generate_ass  # noqa: PLC0415
    ass_path = job_dir / f"{safe}_karaoke.ass"
    bg_lyrics = (lyrics or lyrics_hint or "") if display_mode == "both" else ""
    duration = info.duration if display_mode == "both" else 0
    generate_ass(segments, str(ass_path), word_timing=word_timing,
                 background_lyrics=bg_lyrics, duration=duration)
    jobs[job_id]["files"]["ass"] = str(ass_path)

    _set(job_id, step="Rendering karaoke video...", pct=88)
    video_path = job_dir / f"{safe}_karaoke.mp4"
    safe_title = title.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")

    title_filter = (
        "drawtext=text='{txt}'"
        ":fontfile=/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
        ":fontsize=36:fontcolor=white@0.7"
        ":x=(w-text_w)/2:y=30"
        ":shadowcolor=black@0.6:shadowx=2:shadowy=2"
    ).format(txt=safe_title)

    # Build FFmpeg command based on video background mode
    if video_bg == "original" and original_video and Path(original_video).exists():
        # Use original YouTube video as background, overlay subtitles
        ffmpeg_cmd = [
            FFMPEG,
            "-i", original_video,
            "-i", str(minus_path),
            "-vf", f"scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,ass={ass_path},{title_filter}",
            "-map", "0:v", "-map", "1:a",
            "-shortest",
        ]
    elif video_bg == "cover" and cover_path and Path(cover_path).exists():
        # 5-second cover intro, then dark background for the rest
        ffmpeg_cmd = [
            FFMPEG,
            "-loop", "1", "-t", "5", "-i", cover_path,
            "-f", "lavfi", "-i", "color=c=0x0d0d1a:size=1920x1080:rate=25",
            "-i", str(minus_path),
            "-filter_complex",
            f"[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[cover];"
            f"[1:v]trim=duration=1,setpts=PTS-STARTPTS[bg];"
            f"[cover][bg]concat=n=2:v=1:a=0[base];"
            f"[base]ass={ass_path},{title_filter}[vout]",
            "-map", "[vout]", "-map", "2:a",
            "-shortest",
        ]
    else:
        # Default: solid dark background
        ffmpeg_cmd = [
            FFMPEG,
            "-f", "lavfi", "-i", "color=c=0x0d0d1a:size=1920x1080:rate=25",
            "-i", str(minus_path),
            "-vf", f"ass={ass_path},{title_filter}",
            "-shortest",
        ]

    ffmpeg_cmd.extend([
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.1",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(video_path), "-y",
    ])
    _run(ffmpeg_cmd)
    jobs[job_id]["files"]["video"] = str(video_path)
    _set(job_id, status="done", step="Done!", pct=100)
    artist, _ = _parse_artist_title(title)
    upsert_song(job_id, title=title, artist=artist, status="done",
                video_path=str(video_path), minus_path=str(minus_path),
                ass_path=str(ass_path), lyrics=lyrics or lyrics_hint or "")


# ── Catalog API ──────────────────────────────────────────────────────────────

@app.get("/api/catalog")
def api_catalog(search: str = "", limit: int = 50, offset: int = 0):
    songs = list_songs(search=search, limit=limit, offset=offset)
    total = count_songs(search=search)
    # Check that files still exist on disk
    for song in songs:
        for key in ("video_path", "minus_path", "ass_path", "thumbnail_path"):
            if song.get(key) and not Path(song[key]).exists():
                song[key] = ""
    return {"songs": songs, "total": total}


@app.get("/api/catalog/{job_id}")
def api_catalog_song(job_id: str):
    song = get_song(job_id)
    if not song:
        raise HTTPException(404, "Song not found in catalog")
    return song


@app.delete("/api/catalog/{job_id}")
def api_catalog_delete(job_id: str):
    song = get_song(job_id)
    if not song:
        raise HTTPException(404, "Song not found")
    # Remove files from disk
    import shutil  # noqa: PLC0415
    job_dir = WORK_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)
    delete_song(job_id)
    # Remove from in-memory store too
    jobs.pop(job_id, None)
    return {"deleted": True}


# ── YouTube preparation ──────────────────────────────────────────────────────

def _generate_thumbnail(title: str, output_path: str) -> None:
    """Generate a 1280x720 thumbnail image with the song title."""
    safe_title = title.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")
    # Split into artist/title for two-line layout
    artist, song = _parse_artist_title(title)
    if artist:
        text_lines = f"{artist}\\n{song}"
    else:
        text_lines = safe_title

    _run([
        FFMPEG,
        "-f", "lavfi", "-i", "color=c=0x0d0d1a:size=1280x720:d=1",
        "-vf", (
            "drawtext=text='{txt}'"
            ":fontfile=/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
            ":fontsize=64:fontcolor=white"
            ":x=(w-text_w)/2:y=(h-text_h)/2-40"
            ":shadowcolor=black:shadowx=3:shadowy=3,"
            "drawtext=text='KARAOKE'"
            ":fontfile=/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
            ":fontsize=36:fontcolor=yellow"
            ":x=(w-text_w)/2:y=(h/2)+50"
            ":shadowcolor=black:shadowx=2:shadowy=2"
        ).format(txt=text_lines),
        "-frames:v", "1",
        "-update", "1",
        str(output_path), "-y",
    ])


def _generate_youtube_metadata(job_id: str, title: str, lyrics: str, output_path: str) -> None:
    """Generate a JSON metadata file for YouTube upload."""
    import json  # noqa: PLC0415
    artist, song = _parse_artist_title(title)
    if artist:
        yt_title = f"{artist} - {song} (Karaoke)"
    else:
        yt_title = f"{title} (Karaoke)"

    description_parts = [
        f"{yt_title}",
        "",
        "Karaoke version with word-by-word highlighting.",
        "Generated with Karaoke Generator.",
        "",
    ]
    if lyrics:
        description_parts.append("--- Lyrics ---")
        description_parts.append(lyrics[:4500])  # YouTube description limit ~5000 chars

    tags = ["karaoke", "instrumental", "sing along", "lyrics"]
    if artist:
        tags.extend([artist.lower(), song.lower()])

    metadata = {
        "title": yt_title,
        "description": "\n".join(description_parts),
        "tags": tags,
        "category": "10",  # Music
        "privacy": "public",
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


@app.post("/api/catalog/{job_id}/prepare-youtube")
async def api_prepare_youtube(job_id: str):
    """Generate thumbnail + metadata and bundle into a ZIP for YouTube upload."""
    import zipfile  # noqa: PLC0415

    song = get_song(job_id)
    if not song:
        raise HTTPException(404, "Song not found")
    if song["status"] != "done":
        raise HTTPException(400, "Song not ready yet")

    job_dir = WORK_DIR / job_id
    title = song["title"] or "track"
    lyrics = song.get("lyrics", "")

    # Generate thumbnail
    thumb_path = job_dir / "thumbnail.jpg"
    try:
        _generate_thumbnail(title, str(thumb_path))
    except Exception:
        thumb_path = None

    # Generate metadata
    meta_path = job_dir / "youtube_metadata.json"
    _generate_youtube_metadata(job_id, title, lyrics, str(meta_path))

    # Bundle into ZIP
    safe = "".join(c for c in title if c.isalnum() or c in " -_").strip() or "track"
    zip_path = job_dir / f"{safe}_youtube.zip"
    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
        if song.get("video_path") and Path(song["video_path"]).exists():
            zf.write(song["video_path"], f"{safe}_karaoke.mp4")
        if song.get("minus_path") and Path(song["minus_path"]).exists():
            zf.write(song["minus_path"], f"{safe}_minus.mp3")
        if song.get("ass_path") and Path(song["ass_path"]).exists():
            zf.write(song["ass_path"], f"{safe}_karaoke.ass")
        if thumb_path and thumb_path.exists():
            zf.write(str(thumb_path), "thumbnail.jpg")
        if meta_path.exists():
            zf.write(str(meta_path), "youtube_metadata.json")

    # Update catalog
    upsert_song(job_id, youtube_ready=1,
                thumbnail_path=str(thumb_path) if thumb_path and thumb_path.exists() else "")
    jobs.setdefault(job_id, {}).setdefault("files", {})["youtube_zip"] = str(zip_path)

    return {
        "ready": True,
        "download_url": f"/api/jobs/{job_id}/download/youtube_zip",
    }


# ── Static frontend (must be last) ────────────────────────────────────────────
app.mount("/", StaticFiles(directory="static", html=True), name="static")
