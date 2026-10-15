"""Planning: master BPM/key, stretch + pitch decisions, arrangement timeline."""
import json
import os
from statistics import median

import numpy as np

BAR_BEATS = 4


def effective_rate(song_tempo: float, master_bpm: float):
    """Limit stretch to +/-12%; use half/double time if needed.

    Tries interpreting the song at half/double time (multipliers 0.5/1/2),
    picks the interpretation needing the least stretch, then clamps to
    [0.88, 1.12]. Returns (rate, effective_bpm, clamped).
    effective_bpm is the song's actual resulting tempo (may differ slightly
    from master_bpm when clamping was needed).
    """
    best = None
    for m in (0.5, 1.0, 2.0):
        raw = float(master_bpm) / (song_tempo * m)
        if best is None or abs(raw - 1.0) < abs(best - 1.0):
            best, best_m = raw, m
    rate = min(1.12, max(0.88, best))
    clamped = abs(rate - best) > 1e-9
    effective_bpm = song_tempo * best_m * rate
    return rate, effective_bpm, clamped


def _transpose_camelot(root: int, mode: str, shift: int):
    from .analysis import key_to_camelot
    return key_to_camelot((root + shift) % 12, mode)


def camelot_distance(a: str, b: str) -> int:
    """0 same, 1 adjacent-or-mode-swap, 2 else. Lower is more compatible."""
    if a == b:
        return 0
    na, la = int(a[:-1]), a[-1]
    nb, lb = int(b[:-1]), b[-1]
    if la == lb and min(abs(na - nb), 12 - abs(na - nb)) == 1:
        return 1
    if na == nb and la != lb:
        return 1
    return 2


def best_pitch_shift(song_root: int, song_mode: str, master_camelot: str) -> int:
    """Pick semitone shift in [-3..3] minimizing camelot distance, then |shift|."""
    cands = []
    for s in range(-3, 4):
        c = _transpose_camelot(song_root, song_mode, s)
        cands.append((camelot_distance(c, master_camelot), abs(s), s, c))
    cands.sort()
    return cands[0][2]


def build_plan(analyses: list, master_bpm: float | None = None, phrase_bars: int = 8) -> dict:
    tempos = [a["tempo"] for a in analyses]
    if master_bpm is None:
        master_bpm = float(median(tempos))
    # master key = key of highest-energy song
    master_idx = int(np.argmax([a["mean_energy"] for a in analyses]))
    master_key = analyses[master_idx]["key"]
    master_camelot = analyses[master_idx]["camelot"]

    # energy order low -> high (peak last = last third for N>=2)
    order = sorted(range(len(analyses)), key=lambda i: analyses[i]["mean_energy"])
    # ensure global max is in last third (ascending sort already guarantees last)
    songs = []
    for i in order:
        a = analyses[i]
        rate, eff, clamped = effective_rate(a["tempo"], master_bpm)
        if clamped:
            print(f"warning: {os.path.basename(a['path'])} needs >12% stretch "
                  f"even at half/double time; clamped to {rate:.3f} "
                  f"(result {eff:.1f} BPM vs master {master_bpm:.1f} BPM, grids will drift)")
        shift = best_pitch_shift(a["key_root"], a["key_mode"], master_camelot)
        beat_dur = 60.0 / eff  # effective grid after half/double handling
        bar_dur = beat_dur * BAR_BEATS
        songs.append({
            "index": i,
            "path": a["path"],
            "tempo": a["tempo"],
            "rate": rate,
            "effective_bpm": eff,
            "pitch_shift": shift,
            "key": a["key"],
            "camelot": a["camelot"],
            "transposed_camelot": _transpose_camelot(a["key_root"], a["key_mode"], shift),
            "mean_energy": a["mean_energy"],
            "beat_dur": beat_dur,
            "bar_dur": bar_dur,
        })

    # arrangement: 8-bar phrases (16 for the peak song if it has enough downbeats)
    segments = []
    transitions = []
    t = 0.0
    for si, s in enumerate(songs):
        a = analyses[s["index"]]
        rate = s["rate"]
        # downbeats in stretched time
        db = [d / rate for d in a["downbeats"]]
        bars_avail = max(1, len(db) - 1)
        is_peak = (si == len(songs) - 1)
        want_bars = 16 if (is_peak and bars_avail >= 16) else phrase_bars
        use_bars = min(want_bars, bars_avail)
        # start at first downbeat (cut only on downbeats)
        start_in_song = float(db[0]) * rate  # back in original-time seconds for reference
        seg_dur = use_bars * s["bar_dur"]
        segments.append({
            "song_index": s["index"],
            "song_path": s["path"],
            "start_in_song_stretched": float(db[0]),
            "start_in_song_orig": start_in_song,
            "bars": use_bars,
            "duration": seg_dur,
            "start_in_mashup": t,
            "rate": rate,
            "pitch_shift": s["pitch_shift"],
        })
        # transition to next: overlap 2 bars (1-4 allowed), shrink if segments short
        if si < len(songs) - 1:
            xbars = int(min(4, max(1, 2 if use_bars >= 6 else 1)))
            xdur = xbars * s["bar_dur"]
            transitions.append({
                "from": si, "to": si + 1,
                "bars": xbars, "duration": xdur,
                "at_time": t + seg_dur - xdur,
                "type": "bass-swap",
            })
            t = t + seg_dur - xdur  # overlap (crossfade, not butt joint)
        else:
            t = t + seg_dur

    plan = {
        "master_bpm": master_bpm,
        "master_key": master_key,
        "master_camelot": master_camelot,
        "phrase_bars": phrase_bars,
        "songs": songs,
        "segments": segments,
        "transitions": transitions,
        "total_duration": t,
    }
    return plan


def save_plan(plan: dict, path: str):
    with open(path, "w") as f:
        json.dump(plan, f, indent=1)


def load_plan(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
