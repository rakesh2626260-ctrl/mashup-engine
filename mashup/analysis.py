"""Per-song analysis with librosa. Cached as JSON."""
import hashlib
import json
import os
from pathlib import Path

import librosa
import numpy as np

ANALYSIS_SR = 44100

# Krumhansl-Schmuckler key profiles
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# pitch class -> Camelot code. Major = B, minor = m -> A.
_MAJOR_CAMELOT = {0: "8B", 7: "9B", 2: "10B", 9: "11B", 4: "12B", 11: "1B",
                  6: "2B", 1: "3B", 8: "4B", 3: "5B", 10: "6B", 5: "7B"}
_MINOR_CAMELOT = {9: "8A", 4: "9A", 11: "10A", 6: "11A", 1: "12A", 8: "1A",
                  3: "2A", 10: "3A", 5: "4A", 0: "5A", 7: "6A", 2: "7A"}


def key_to_camelot(root_pc: int, mode: str) -> str:
    if mode == "major":
        return _MAJOR_CAMELOT[int(root_pc) % 12]
    return _MINOR_CAMELOT[int(root_pc) % 12]


def camelot_to_sort_key(code: str):
    # "8A" -> (8, 0); "8B" -> (8, 1)
    return (int(code[:-1]), 0 if code[-1] == "A" else 1)


def estimate_key(chroma_mean: np.ndarray):
    """Correlate mean chroma against rotated KS profiles."""
    best = (-2.0, 0, "major")
    for shift in range(12):
        maj = np.roll(KS_MAJOR, shift)
        min_ = np.roll(KS_MINOR, shift)
        cm = float(np.corrcoef(chroma_mean, maj)[0, 1]) if np.std(chroma_mean) > 1e-9 else 0.0
        cn = float(np.corrcoef(chroma_mean, min_)[0, 1]) if np.std(chroma_mean) > 1e-9 else 0.0
        if cm > best[0]:
            best = (cm, shift, "major")
        if cn > best[0]:
            best = (cn, shift, "minor")
    _, root, mode = best
    key_name = f"{NOTE_NAMES[root]} {'maj' if mode == 'major' else 'min'}"
    camelot = key_to_camelot(root, mode)
    return root, mode, key_name, camelot


def _cache_path(audio_path: str, cache_dir: str) -> str:
    h = hashlib.md5(f"{os.path.abspath(audio_path)}:{os.path.getsize(audio_path)}:{int(os.path.getmtime(audio_path))}".encode()).hexdigest()[:12]
    base = Path(audio_path).stem
    return os.path.join(cache_dir, f"{base}_{h}.json")


def analyze_song(audio_path: str, cache_dir: str = ".mashup_cache") -> dict:
    """Analyze one song. Returns dict and writes JSON cache."""
    os.makedirs(cache_dir, exist_ok=True)
    cp = _cache_path(audio_path, cache_dir)
    if os.path.exists(cp):
        try:
            with open(cp) as f:
                d = json.load(f)
            # validate essential keys
            if all(k in d for k in ("tempo", "beats", "downbeats", "key")):
                return d
        except Exception:
            pass

    y, sr = librosa.load(audio_path, sr=ANALYSIS_SR, mono=True)
    duration = float(len(y) / sr)

    # tempo + beats
    tempo_arr, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    tempo = float(np.atleast_1d(tempo_arr)[0])
    if tempo <= 0 or len(beat_frames) < 8:
        # fallback: onset-based default
        tempo = 120.0
        step = 60.0 / tempo
        beat_frames = librosa.time_to_frames(np.arange(0, duration, step), sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).astype(float)
    beat_times = [float(t) for t in beat_times if float(t) < duration]

    # downbeats: every 4th beat with strongest onset phase
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    scores = []
    for off in range(4):
        idx = beat_frames[off::4]
        idx = idx[idx < len(onset_env)]
        scores.append(float(onset_env[idx].sum()) if len(idx) else -1.0)
    best_off = int(np.argmax(scores)) if scores else 0
    db_frames = beat_frames[best_off::4]
    downbeats = [float(t) for t in librosa.frames_to_time(db_frames, sr=sr) if float(t) < duration]
    if not downbeats:
        downbeats = beat_times[::4] or beat_times[:1]

    # key/mode via chroma
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    root, mode, key_name, camelot = estimate_key(chroma_mean)
    root_pc = int(root)

    # RMS energy curve (downsampled to ~100 points + full res times)
    rms = librosa.feature.rms(y=y)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
    # full curve for grid (cap at 500 pts)
    n = len(rms)
    stride = max(1, n // 500)
    energy_curve = [{"t": float(rms_times[i]), "v": float(rms[i])} for i in range(0, n, stride)]
    mean_energy = float(rms.mean())
    # normalize 0..1 for planning
    vmax = float(rms.max()) if float(rms.max()) > 0 else 1.0
    energy_norm = (rms / vmax).astype(float)

    # sections: novelty from chroma self-distance, peak-pick, snap to bar lines (downbeats)
    sections = _sections_from_novelty(chroma, downbeats, duration, sr)

    result = {
        "path": os.path.abspath(audio_path),
        "sr": sr,
        "duration": duration,
        "tempo": tempo,
        "beats": beat_times,
        "downbeats": downbeats,
        "downbeat_offset": best_off,
        "key_root": root_pc,
        "key_mode": mode,
        "key": key_name,
        "camelot": camelot,
        "mean_energy": mean_energy,
        "energy_curve": energy_curve,
        "energy_norm_peak": float(energy_norm.max()),
        "sections": sections,
    }
    with open(cp, "w") as f:
        json.dump(result, f, indent=1)
    return result


def _sections_from_novelty(chroma: np.ndarray, downbeats: list, duration: float, sr: int) -> list:
    try:
        # coarse novelty: distance between successive pooled chroma columns
        pool = 8
        T = chroma.shape[1]
        nb = max(1, T // pool)
        pooled = np.stack([chroma[:, i * pool:(i + 1) * pool].mean(axis=1) for i in range(nb)], axis=1)
        pooled = pooled / (np.linalg.norm(pooled, axis=0, keepdims=True) + 1e-9)
        nov = np.zeros(nb)
        nov[1:] = np.linalg.norm(np.diff(pooled, axis=1), axis=0)
        # time axis for pooled frames
        hop = 512  # chroma_cqt default hop
        times = librosa.frames_to_time(np.arange(nb) * pool, sr=sr, hop_length=hop)
        # peak pick
        peaks = librosa.util.peak_pick(nov, pre_max=5, post_max=5, pre_avg=10, post_avg=10, delta=float(nov.mean() * 0.5), wait=10)
        bounds = sorted({float(times[p]) for p in peaks if float(times[p]) < duration})
    except Exception:
        bounds = []
    # snap each bound to nearest downbeat
    snapped = []
    db = np.array(downbeats) if downbeats else np.array([0.0])
    for b in bounds:
        j = int(np.argmin(np.abs(db - b)))
        snapped.append(float(db[j]))
    # always include 0 and duration snapped to grid ends
    pts = sorted(set([0.0] + snapped + [duration]))
    # drop near-duplicates (<2s apart)
    filt = [pts[0]]
    for p in pts[1:]:
        if p - filt[-1] >= 2.0:
            filt.append(p)
    if filt[-1] < duration - 0.5:
        pass
    else:
        filt[-1] = duration
    sections = [{"start": filt[i], "end": filt[i + 1]} for i in range(len(filt) - 1)]
    return sections
