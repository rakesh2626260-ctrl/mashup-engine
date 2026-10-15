"""Optional Demucs stems (htdemucs, CPU) behind --stems."""
import os
import subprocess
import sys

import librosa
import numpy as np
import soundfile as sf

from .build import SR, pitch_shift, time_stretch


def separate_songs(paths: list, out_dir: str, model: str = "htdemucs", log=print) -> dict:
    """Run demucs on each song. Returns {song_index: stem_dir}."""
    try:
        import demucs  # noqa: F401
    except ImportError as e:
        raise RuntimeError("Demucs is not installed. pip install torch demucs (CPU) or re-run with --no-stems.") from e
    os.makedirs(out_dir, exist_ok=True)
    stem_dirs = {}
    for i, p in enumerate(paths):
        log(f"  demucs [{i+1}/{len(paths)}] {os.path.basename(p)} ({model}, CPU)...")
        # Use the same interpreter running us (venv-safe): bare `python3`
        # may resolve to a system python without demucs installed.
        # NOTE: `--mp3` is deliberate, not just size: demucs saves .wav via
        # torchaudio, whose 2.9+ saver requires torchcodec — whose native
        # image lib needs CUDA (libnvrtc) and fails to import on CPU-only
        # boxes (local env and CI runners alike). The .mp3 path uses lameenc
        # (pure CPU) and demucs loads input via ffmpeg, so torchaudio is
        # never touched. 320 kbps is transparent for stem mixing.
        cmd = [sys.executable, "-m", "demucs.separate", "-n", model, "-d", "cpu",
               "--mp3", "--mp3-bitrate", "320",
               "--out", out_dir, p]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"Demucs failed on {p}:\n{r.stderr[-3000:]}")
        # demucs writes <out>/<model>/<track>/vocals.wav etc.
        track = os.path.splitext(os.path.basename(p))[0]
        stem_dirs[i] = os.path.join(out_dir, model, track)
    return stem_dirs


def _load(path: str, sr: int):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)


def mixed_stem_audio(song_index: int, plan: dict, stems_root: str, sr: int = SR, log=print) -> np.ndarray:
    """For stems mode: vocals of the highest-energy song over instrumental of others.

    Convention: the peak (last, highest-energy) song contributes vocals;
    every other song contributes no_vocals (= instrumental). The peak song's own
    segment keeps its vocals + instrumental mixed (i.e. full), others are instrumental-only.
    Each stem is pitch/time processed with the song's rate/shift.
    """
    import glob
    s = next(x for x in plan["songs"] if x["index"] == song_index)
    # find stem dir: <root>/htdemucs/<track>/
    cands = glob.glob(os.path.join(stems_root, "*", "*"))
    track = os.path.splitext(os.path.basename(s["path"]))[0]
    d = next((c for c in cands if os.path.basename(c) == track), None)
    if d is None:
        # fallback: assume direct layout
        d = os.path.join(stems_root, track)
    # demucs --mp3 writes .mp3 (lameenc, CPU-safe); accept legacy .wav too.
    vocals = next((os.path.join(d, f"vocals.{e}") for e in ("mp3", "wav")
                   if os.path.exists(os.path.join(d, f"vocals.{e}"))), os.path.join(d, "vocals.mp3"))
    novox = next((os.path.join(d, f"no_vocals.{e}") for e in ("mp3", "wav")
                  if os.path.exists(os.path.join(d, f"no_vocals.{e}"))), os.path.join(d, "no_vocals.mp3"))
    is_peak = (plan["segments"][-1]["song_index"] == song_index)
    parts = []
    if os.path.exists(vocals) and is_peak:
        v = _load(vocals, sr)
        parts.append(time_stretch(pitch_shift(v, sr, s["pitch_shift"]), sr, s["rate"]))
    if os.path.exists(novox):
        nv = _load(novox, sr)
        parts.append(time_stretch(pitch_shift(nv, sr, s["pitch_shift"]), sr, s["rate"]))
    if not parts:
        # fallback to full mix
        if log:
            log(f"  warning: stems missing for {track}, using full mix")
        from .build import process_song
        return process_song(s["path"], s["rate"], s["pitch_shift"], sr)
    n = max(len(p) for p in parts)
    mix = np.zeros(n, dtype=np.float32)
    for p in parts:
        mix[:len(p)] += p
    mix /= max(1, len(parts)) ** 0.5
    return mix.astype(np.float32)
