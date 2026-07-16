"""
CD+G (``.cdg``) karaoke subtitle generator.

CD+G is *the* format real karaoke machines and players (MP3+G) understand.
This module renders timed lyric segments — the same word-level data used for
the ASS subtitles — into a standards-compliant ``.cdg`` stream with a
word/character highlight "wipe".

Format primer
-------------
CD+G rides in the CD subcode channel as a stream of 24-byte packets at
**300 packets/second**. A packet whose first byte is ``0x09`` carries a
graphics instruction; anything else is a no-op the player skips. The visible
screen is 300x216 px, addressed as a grid of **50 x 18 tiles of 6 x 12 px**.

We only need a handful of instructions:
  * MEMORY_PRESET (1)   — clear the screen to a colour
  * BORDER_PRESET (2)   — set the border colour
  * TILE_BLOCK (6)      — paint one 6x12 tile as a 2-colour bitmap
  * LOAD_CLUT_LOW (30)  — load palette entries 0-7 (12-bit RGB)

Highlighting trick: every glyph is drawn into its own tile. To "sing" a
character we simply re-emit the *same* tile bitmap but swap the foreground
palette index from the unsung colour to the sung colour. Scheduling those
re-emits across each word's duration produces a left-to-right wipe.

Pairs with the instrumental MP3 (same basename) to form an MP3+G set.
"""

# ── CD+G constants ──────────────────────────────────────────────────────────
CDG_COMMAND = 0x09
INST_MEMORY_PRESET = 1
INST_BORDER_PRESET = 2
INST_TILE_BLOCK = 6
INST_LOAD_CLUT_LOW = 30

PACKETS_PER_SEC = 300
TILE_W, TILE_H = 6, 12
COLS, ROWS = 50, 18            # 300x216 px screen in tiles

# Palette (4 bits per channel, 0-15). Indices must match the tile colours below.
COLOR_BG = 0        # dark blue background — matches the video's 0x0d0d1a
COLOR_TEXT = 1      # white — unsung lyric text
COLOR_SUNG = 2      # yellow — text already sung (the wipe colour)
_PALETTE = [
    (1, 1, 3),      # 0 background
    (15, 15, 15),   # 1 white
    (15, 15, 0),    # 2 yellow
    (0, 0, 0),      # 3 (unused)
    (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),
]

_EMPTY_PACKET = bytes(24)

DEFAULT_FONT = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"


# ── Low-level packet builders ────────────────────────────────────────────────
def _packet(instruction: int, data) -> bytes:
    p = bytearray(24)
    p[0] = CDG_COMMAND & 0x3F
    p[1] = instruction & 0x3F
    # p[2:4] parity Q, p[20:24] parity P — left zero, players ignore them.
    for i in range(16):
        p[4 + i] = (data[i] if i < len(data) else 0) & 0x3F
    return bytes(p)


def _clut_low_packet(palette) -> bytes:
    data = [0] * 16
    for i in range(8):
        r, g, b = palette[i] if i < len(palette) else (0, 0, 0)
        data[2 * i] = ((r & 0x0F) << 2) | ((g & 0x0F) >> 2)
        data[2 * i + 1] = ((g & 0x03) << 4) | (b & 0x0F)
    return _packet(INST_LOAD_CLUT_LOW, data)


def _memory_preset_packet(color: int, repeat: int = 0) -> bytes:
    return _packet(INST_MEMORY_PRESET, [color & 0x0F, repeat & 0x0F])


def _border_preset_packet(color: int) -> bytes:
    return _packet(INST_BORDER_PRESET, [color & 0x0F])


def _tile_packet(row: int, col: int, color0: int, color1: int, bitmap) -> bytes:
    data = [0] * 16
    data[0] = color0 & 0x0F
    data[1] = color1 & 0x0F
    data[2] = row & 0x1F
    data[3] = col & 0x3F
    for i in range(12):
        data[4 + i] = bitmap[i] & 0x3F
    return _packet(INST_TILE_BLOCK, data)


# ── Font rasterisation ───────────────────────────────────────────────────────
def _load_font(font_path: str):
    from PIL import ImageFont  # noqa: PLC0415
    for size in (12, 11, 13):
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _char_bitmap(ch: str, font, cache: dict, thr: int = 100):
    """Rasterise ``ch`` into a 6x12 tile: list of 12 six-bit rows (bit5=left)."""
    if ch in cache:
        return cache[ch]
    from PIL import Image, ImageDraw  # noqa: PLC0415
    img = Image.new("L", (TILE_W, TILE_H), 0)
    d = ImageDraw.Draw(img)
    try:
        bbox = d.textbbox((0, 0), ch, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (TILE_W - w) // 2 - bbox[0]
        y = (TILE_H - h) // 2 - bbox[1]
    except Exception:
        x, y = 0, 0
    d.text((x, y), ch, fill=255, font=font)
    px = img.load()
    rows = []
    for yy in range(TILE_H):
        bits = 0
        for xx in range(TILE_W):
            if px[xx, yy] >= thr:
                bits |= 1 << (5 - xx)
        rows.append(bits)
    cache[ch] = rows
    return rows


# ── Layout ───────────────────────────────────────────────────────────────────
def _segments_to_lines(segments):
    """Turn segments into ``[[(char_text, start, end), ...per word], ...]``."""
    lines = []
    for seg in segments:
        words = []
        for w in (getattr(seg, "words", None) or []):
            txt = (w.word or "").strip()
            if txt:
                words.append((txt, float(w.start), float(w.end)))
        if not words:
            txt = (getattr(seg, "text", "") or "").strip()
            if txt:
                words = [(txt, float(seg.start), float(seg.end))]
        if words:
            lines.append(words)
    return lines


def _group_pages(lines, lines_per_page: int, max_gap: float):
    """Group consecutive lyric lines into screen pages."""
    pages, cur = [], []
    for wline in lines:
        if cur:
            prev_end = cur[-1][-1][2]
            gap = wline[0][1] - prev_end
            if len(cur) >= lines_per_page or gap > max_gap:
                pages.append(cur)
                cur = []
        cur.append(wline)
    if cur:
        pages.append(cur)
    return pages


# ── Main entry point ─────────────────────────────────────────────────────────
def generate_cdg(segments, output_path: str, duration: float = 0.0,
                 font_path: str = DEFAULT_FONT, lines_per_page: int = 4,
                 max_gap: float = 6.0, lead_in: float = 1.2) -> None:
    """Render timed ``segments`` to a ``.cdg`` file at ``output_path``.

    ``segments`` are the same namespaces produced for ASS generation (each with
    ``.words`` carrying ``.word``/``.start``/``.end``). Raises ``ValueError`` if
    there is nothing to render.
    """
    lines = _segments_to_lines(segments)
    if not lines:
        raise ValueError("no lyric lines to render as CDG")

    font = _load_font(font_path)
    char_cache: dict = {}
    pages = _group_pages(lines, lines_per_page, max_gap)

    # events: (target_time_sec, packet_bytes) — placed onto the 300Hz grid later.
    events: list[tuple[float, bytes]] = []
    prev_page_end = 0.0

    for page in pages:
        n = len(page)
        span_rows = (n - 1) * 3 + 1
        start_row = max(0, (ROWS - span_rows) // 2)

        first_word_start = page[0][0][1]
        # Draw a comfortable lead ahead of the first word, but never clear the
        # screen before the previous page has finished singing.
        show_time = max(first_word_start - lead_in, prev_page_end + 0.05)

        draw_tiles = []              # (row, col, bitmap) painted unsung
        highlights = []              # (time, row, col, bitmap) repainted sung

        for k, wline in enumerate(page):
            row = start_row + k * 3
            # Flatten the line to characters, remembering per-word char spans.
            line_chars = []          # (char, word_index)
            for wi, (text, _ws, _we) in enumerate(wline):
                if wi > 0:
                    line_chars.append((" ", -1))
                for ch in text:
                    line_chars.append((ch, wi))
            n_chars = len(line_chars)
            start_col = max(0, (COLS - n_chars) // 2)

            # Bucket the visible character columns of each word.
            word_cols: dict[int, list[int]] = {}
            for ci, (ch, wi) in enumerate(line_chars):
                col = start_col + ci
                if col >= COLS:
                    break
                bmp = _char_bitmap(ch, font, char_cache)
                if any(bmp):
                    draw_tiles.append((row, col, bmp))
                if wi >= 0 and ch != " " and any(bmp):
                    word_cols.setdefault(wi, []).append((col, bmp))

            # Schedule the wipe: spread each word's chars across its duration.
            for wi, (text, ws, we) in enumerate(wline):
                cols = word_cols.get(wi, [])
                if not cols:
                    continue
                dur = max(we - ws, 0.05)
                m = len(cols)
                for idx, (col, bmp) in enumerate(cols):
                    t = ws + dur * (idx / m)
                    highlights.append((t, row, col, bmp))

        # Emit: clear screen, paint the page unsung, then the timed wipes.
        for _ in range(4):
            events.append((show_time, _memory_preset_packet(COLOR_BG)))
        for row, col, bmp in draw_tiles:
            events.append((show_time, _tile_packet(row, col, COLOR_BG, COLOR_TEXT, bmp)))
        for t, row, col, bmp in highlights:
            events.append((t, _tile_packet(row, col, COLOR_BG, COLOR_SUNG, bmp)))

        prev_page_end = max(prev_page_end, page[-1][-1][2])

    _write_stream(events, output_path, duration, prev_page_end)


def _write_stream(events, output_path, duration, content_end):
    """Place events on the 300Hz packet grid (monotonic) and write the file."""
    # Header: palette + initial clear + border, all at t=0.
    header = [
        _clut_low_packet(_PALETTE),
        _border_preset_packet(COLOR_BG),
    ]
    for _ in range(16):
        header.append(_memory_preset_packet(COLOR_BG))

    packets = list(header)
    for target_time, pkt in sorted(events, key=lambda e: e[0]):
        target_idx = int(target_time * PACKETS_PER_SEC)
        # Can't travel back in time; late-but-clustered events serialise after
        # the ones already placed (this is the real 300 packet/s throughput cap).
        if target_idx < len(packets):
            target_idx = len(packets)
        packets.extend([_EMPTY_PACKET] * (target_idx - len(packets)))
        packets.append(pkt)

    total_packets = int(max(duration, content_end + 2.0) * PACKETS_PER_SEC)
    if len(packets) < total_packets:
        packets.extend([_EMPTY_PACKET] * (total_packets - len(packets)))

    with open(output_path, "wb") as f:
        f.write(b"".join(packets))
