"""
ASS karaoke subtitle generator.
Converts faster-whisper segments (with word timestamps) to an ASS file
with \\kf (karaoke fill) tags for word-by-word highlighting.

Color convention (AABBGGRR in ASS):
  PrimaryColour   = &H0000FFFF  — yellow  (word currently/already sung)
  SecondaryColour = &H00FFFFFF  — white   (word not yet sung)
  OutlineColour   = &H00000000  — black outline
  BackColour      = &H80000000  — semi-transparent shadow
"""

_ASS_HEADER = """\
[Script Info]
Title: Karaoke
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke,Arial,72,&H0000FFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,2,0,1,4,2,2,10,10,80,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

_ASS_HEADER_DUAL = """\
[Script Info]
Title: Karaoke
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke,Arial,72,&H0000FFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,2,0,1,4,2,2,10,10,80,1
Style: BgLyrics,Arial,28,&H50FFFFFF,&H50FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,1,0,1,2,1,8,60,60,30,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _fmt_time(seconds: float) -> str:
    """Format seconds as ASS timestamp H:MM:SS.cc"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _segment_to_dialogue(segment, word_timing: bool = True) -> str:
    """Build one ASS Dialogue line from a faster-whisper segment.

    word_timing=True  — word-by-word \\kf highlighting
    word_timing=False — plain text, whole line appears at once
    """
    start = _fmt_time(segment.start)
    end = _fmt_time(segment.end)

    if not word_timing:
        text = segment.text.strip()
        if not text:
            return ""
        return f"Dialogue: 0,{start},{end},Karaoke,,0,0,0,,{text}"

    words = segment.words or []
    if not words:
        return ""

    MIN_WORD_CS = 15  # minimum 150ms per word to avoid flicker
    GAP_THRESHOLD = 0.10  # ignore gaps shorter than 100ms (measurement noise)

    parts: list[str] = []
    prev_end = segment.start

    for word in words:
        gap_s = word.start - prev_end
        if gap_s > GAP_THRESHOLD:
            parts.append(f"{{\\kf{max(1, int(round(gap_s * 100)))}}}")
        elif gap_s > 0:
            # Small gap — absorb into word duration instead of creating a flicker gap
            pass
        dur_cs = max(MIN_WORD_CS, int(round((word.end - word.start) * 100)))
        parts.append(f"{{\\kf{dur_cs}}}{word.word}")
        prev_end = word.end

    text = "".join(parts)
    return f"Dialogue: 0,{start},{end},Karaoke,,0,0,0,,{text}"


def generate_static_ass(lyrics: str, duration: float, output_path: str) -> None:
    """Create an ASS file showing the full lyrics as static text for the entire song."""
    _STATIC_HEADER = """\
[Script Info]
Title: Karaoke
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke,Arial,40,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,2,0,1,3,2,5,80,80,80,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [l.strip() for l in lyrics.splitlines() if l.strip()]
    if not lines:
        return

    start = _fmt_time(0)
    end = _fmt_time(duration)
    text = r"\N".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(_STATIC_HEADER)
        f.write(f"Dialogue: 0,{start},{end},Karaoke,,0,0,0,,{text}\n")


def generate_ass(segments, output_path: str, word_timing: bool = True,
                 background_lyrics: str = "", duration: float = 0) -> None:
    """Write an ASS karaoke file from a list of faster-whisper segments.

    If background_lyrics is provided, the full lyrics are shown as dimmed static
    text in the upper area (BgLyrics style) while karaoke highlighting runs at the bottom.
    """
    use_dual = bool(background_lyrics and background_lyrics.strip())
    lines = [_ASS_HEADER_DUAL if use_dual else _ASS_HEADER]

    # Background lyrics — static full text for entire duration
    if use_dual:
        bg_lines = [l.strip() for l in background_lyrics.splitlines() if l.strip()]
        if bg_lines:
            if duration <= 0 and segments:
                duration = max(seg.end for seg in segments) + 1
            start = _fmt_time(0)
            end = _fmt_time(duration or 300)
            text = r"\N".join(bg_lines)
            lines.append(f"Dialogue: 0,{start},{end},BgLyrics,,0,0,0,,{text}\n")

    # Karaoke lines
    for seg in segments:
        line = _segment_to_dialogue(seg, word_timing=word_timing)
        if line:
            lines.append(line + "\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
