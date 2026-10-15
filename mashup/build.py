"""Render the arrangement to audio: stretch, pitch, bass-swap crossfades."""
import os

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

SR = 44100


def _butter_filter(y: np.ndarray, sr: int, kind: str, cutoff: float, order: int = 4):
    sos = butter(order, cutoff, btype=kind, fs=sr, output="sos")
    return sosfiltfilt(sos, y).astype(np.float32)


def highpass(y, sr, cutoff=150.0):
    return _butter_filter(y, sr, "highpass", cutoff)


def lowpass(y, sr, cutoff=5000.0):
    return _butter_filter(y, sr, "lowpass", cutoff)


def time_stretch(y: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """rate>1 speeds up. Uses pyrubberband, falls back to librosa."""
    if abs(rate - 1.0) < 1e-6:
        return y.astype(np.float32)
    try:
        import pyrubberband as pyrb
        return pyrb.time_stretch(y.astype(np.float32), sr, rate).astype(np.float32)
    except Exception:
        return librosa.effects.time_stretch(y.astype(np.float32), rate=rate).astype(np.float32)


def pitch_shift(y: np.ndarray, sr: int, n_steps: int) -> np.ndarray:
    if n_steps == 0:
        return y.astype(np.float32)
    return librosa.effects.pitch_shift(y.astype(np.float32), sr=sr, n_steps=int(n_steps)).astype(np.float32)


def sine_riser(n: int, sr: int, f0: float = 200.0, f1: float = 4000.0, gain: float = 0.12):
    t = np.arange(n) / sr
    dur = n / sr
    freq = f0 * (f1 / f0) ** (t / max(dur, 1e-6))
    phase = 2 * np.pi * np.cumsum(freq) / sr
    env = (t / max(dur, 1e-6)) ** 2
    return (gain * env * np.sin(phase)).astype(np.float32)


def process_song(path: str, rate: float, shift: int, sr: int = SR) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    y = pitch_shift(y, sr, shift)
    y = time_stretch(y, sr, rate)
    return y.astype(np.float32)


def render_plan(plan: dict, analyses: list, out_wav: str, use_stems: bool = False,
               stems_dir: str | None = None, log=print) -> np.ndarray:
    """Render plan to a mono float32 mix. Returns mix."""
    sr = SR
    # Prepare per-song processed audio
    proc = {}  # song_index -> audio
    for s in plan["songs"]:
        idx = s["index"]
        if use_stems and stems_dir:
            # stems mode: audio comes from separated files, handled by stems module
            from .stems import mixed_stem_audio
            y = mixed_stem_audio(idx, plan, stems_dir, sr, log=log)
        else:
            log(f"  stretch+shift {os.path.basename(s['path'])}: rate={s['rate']:.3f} shift={s['pitch_shift']:+d}st")
            y = process_song(s["path"], s["rate"], s["pitch_shift"], sr)
        proc[idx] = y

    # Cut segments (only on downbeats -> already aligned in plan)
    segs = []
    for seg in plan["segments"]:
        y = proc[seg["song_index"]]
        start = int(seg["start_in_song_stretched"] * sr)
        want = int(seg["duration"] * sr)
        chunk = y[start:start + want]
        if len(chunk) < want:
            chunk = np.pad(chunk, (0, want - len(chunk)))
        segs.append(chunk)

    # Join with bass-swap crossfades
    mix = segs[0]
    for k, tr in enumerate(plan["transitions"]):
        nxt = segs[k + 1]
        xdur = tr["duration"]
        xn = int(xdur * sr)
        xn = min(xn, len(mix), len(nxt))
        if xn < 64:
            mix = np.concatenate([mix, nxt])
            continue
        out_tail = mix[-xn:]
        in_head = nxt[:xn]
        # bass-swap: high-pass outgoing, low-pass incoming
        out_f = highpass(out_tail, sr, 150.0)
        in_f = lowpass(in_head, sr, 5000.0)
        # equal-power crossfade
        fade_out = np.cos(0.5 * np.pi * np.linspace(0, 1, xn)).astype(np.float32)
        fade_in = np.sin(0.5 * np.pi * np.linspace(0, 1, xn)).astype(np.float32)
        x = out_f * fade_out + in_f * fade_in
        # optional riser + reverse-tail on longer transitions (>=2 bars)
        if tr["bars"] >= 2 and len(x) > sr:
            rlen = min(int(sr * 1.0), len(x) // 2)
            riser = sine_riser(rlen, sr)
            x[-rlen:] = x[-rlen:] + riser * 0.5
            # reverse-tail: reversed fade of incoming head's first 0.25s
            tl = min(int(sr * 0.25), len(x) // 4)
            tail = in_head[:tl][::-1] * np.linspace(0, 0.4, tl).astype(np.float32)
            x[:tl] = x[:tl] + tail
        mix = np.concatenate([mix[:-xn], x, nxt[xn:]])
        log(f"  transition {k}: bass-swap {tr['bars']} bars ({xdur:.2f}s) at {tr['at_time']:.2f}s")
    # normalize headroom before mastering
    peak = float(np.max(np.abs(mix))) if len(mix) else 0.0
    if peak > 0.95:
        mix = (mix * (0.89 / peak)).astype(np.float32)
    sf.write(out_wav, mix, sr)
    return mix
