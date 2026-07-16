"""
Anchor-based lyrics correction.

Borrows the core idea from nomadkaraoke's `lyrics-transcriber`: rather than
trusting Whisper's *text*, we align the Whisper word stream against reference
lyrics fetched online (or pasted by the user). Words that match between the
two streams become **anchors** — we keep Whisper's timing on them and use the
reference spelling. The mismatching spans *between* anchors are **gaps**: there
we drop Whisper's (often wrong) words, substitute the reference words, and
interpolate their timing from the surrounding anchors.

This is a dependency-free reimplementation of that anchor/gap strategy built on
``difflib.SequenceMatcher`` — its "equal" opcodes are the anchor sequences and
the "replace"/"insert"/"delete" opcodes are the gaps.

The result is more principled than line-by-line greedy matching: it works at
the word level across the whole song, so it survives Whisper segmenting the
audio differently from how the lyrics are line-broken.

Output is a list of ``SimpleNamespace`` segments (one per lyric line), each with
``.start``, ``.end``, ``.text`` and ``.words`` — the exact shape ``ass_gen`` and
``cdg_gen`` consume.
"""
import difflib
import re
from types import SimpleNamespace

# Fold Belarusian/Russian Cyrillic variants together before comparing so that
# ў/у, і/и, ё/е don't block an otherwise-correct match. Mirrors main.py.
_CYR = str.maketrans("ўіІЎёЁ", "уиИУеЕ")


def _norm(s: str) -> str:
    """Lowercase, fold Cyrillic variants, strip punctuation for comparison."""
    return re.sub(r"[^\w]", "", (s or "").lower().translate(_CYR))


def _flatten_whisper(segments):
    """Flatten segment word lists into one ``[(norm, word_obj), ...]`` stream."""
    flat = []
    for seg in segments:
        for w in (getattr(seg, "words", None) or []):
            norm = _norm(w.word)
            if norm:
                flat.append((norm, w))
    return flat


def _reference_words(lyrics: str):
    """Tokenise reference lyrics into ``[(norm, original, line_idx), ...]``."""
    ref = []
    line_idx = 0
    lines = []
    for line in lyrics.splitlines():
        if not line.strip():
            continue
        toks = re.findall(r"\S+", line)
        emitted = False
        for tok in toks:
            norm = _norm(tok)
            if norm:
                ref.append((norm, tok, line_idx))
                emitted = True
        if emitted:
            lines.append(line.strip())
            line_idx += 1
    return ref, lines


def correct_segments(whisper_segments, lyrics: str, min_coverage: float = 0.20):
    """Correct a Whisper transcription against reference ``lyrics``.

    Returns a list of segment namespaces (one per lyric line) with word-level
    timing, or ``None`` if correction isn't worthwhile (no word timestamps, no
    usable lyrics, or too few anchors to trust the alignment) — in which case
    the caller should fall back to its existing behaviour.
    """
    flat = _flatten_whisper(whisper_segments)
    ref, ref_lines = _reference_words(lyrics)
    if not flat or not ref:
        return None

    w_norm = [x[0] for x in flat]
    r_norm = [x[0] for x in ref]

    sm = difflib.SequenceMatcher(None, w_norm, r_norm, autojunk=False)
    opcodes = sm.get_opcodes()

    # times[j] = [start, end] for reference word j; None until assigned.
    times = [None] * len(ref)
    anchor_count = 0

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            # Anchor run: reference word j1+k inherits whisper word i1+k timing.
            for k in range(j2 - j1):
                wobj = flat[i1 + k][1]
                times[j1 + k] = [float(wobj.start), float(wobj.end)]
                anchor_count += 1
        elif tag == "replace" and i2 > i1:
            # Gap with whisper words present: spread the reference words across
            # the whisper time span, weighted by character length.
            span_start = float(flat[i1][1].start)
            span_end = float(flat[i2 - 1][1].end)
            _spread(ref, times, j1, j2, span_start, span_end)
        # "insert" (reference words, no whisper) and "delete" (whisper words,
        # no reference — dropped) are left for neighbour interpolation below.

    if anchor_count < max(2, int(len(ref) * min_coverage)):
        return None

    _interpolate_gaps(ref, times)

    return _build_segments(ref, ref_lines, times)


def _spread(ref, times, j1, j2, span_start, span_end):
    """Distribute reference words ``j1..j2`` across a time span by char weight."""
    span = max(span_end - span_start, 0.01)
    weights = [max(len(ref[j][1]), 1) for j in range(j1, j2)]
    total = sum(weights)
    cum = 0
    for idx, j in enumerate(range(j1, j2)):
        frac_s = cum / total
        cum += weights[idx]
        frac_e = cum / total
        times[j] = [span_start + span * frac_s, span_start + span * frac_e]


def _interpolate_gaps(ref, times):
    """Fill any still-empty reference-word times from neighbouring anchors."""
    n = len(times)
    i = 0
    while i < n:
        if times[i] is not None:
            i += 1
            continue
        # Find the contiguous run of missing words [i, k).
        k = i
        while k < n and times[k] is None:
            k += 1
        prev_end = times[i - 1][1] if i > 0 and times[i - 1] else None
        next_start = times[k][0] if k < n and times[k] else None
        if prev_end is None and next_start is None:
            # No anchors at all — degenerate; give everything a tiny slot.
            prev_end, next_start = 0.0, float(k - i) * 0.3
        elif prev_end is None:
            prev_end = max(0.0, next_start - (k - i) * 0.3)
        elif next_start is None:
            next_start = prev_end + (k - i) * 0.3
        if next_start < prev_end:
            next_start = prev_end
        _spread(ref, times, i, k, prev_end, next_start)
        i = k


def _build_segments(ref, ref_lines, times):
    """Group timed reference words back into per-line segments."""
    # Bucket words by their line index.
    lines: dict[int, list] = {}
    for j, (_norm_tok, original, line_idx) in enumerate(ref):
        t = times[j]
        if t is None:
            continue
        lines.setdefault(line_idx, []).append((original, t[0], t[1]))

    segments = []
    for line_idx in sorted(lines):
        words = lines[line_idx]
        if not words:
            continue
        # Enforce monotonic, non-zero-length word timing within the line.
        clean = []
        prev_end = words[0][1]
        for original, ws, we in words:
            ws = max(ws, prev_end)
            we = max(we, ws + 0.05)
            clean.append(SimpleNamespace(
                word=f" {original}", start=ws, end=we, probability=1.0,
            ))
            prev_end = we
        seg_text = ref_lines[line_idx] if line_idx < len(ref_lines) else \
            " ".join(w.word.strip() for w in clean)
        segments.append(SimpleNamespace(
            start=clean[0].start, end=clean[-1].end,
            text=f" {seg_text}", words=clean,
        ))

    segments.sort(key=lambda s: s.start)
    return segments or None
