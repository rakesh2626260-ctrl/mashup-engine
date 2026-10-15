"""Mastering: loudnorm to -14 LUFS + limiter, export wav + mp3."""
import json
import subprocess


def _run(cmd: list) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def master(in_wav: str, out_wav: str, out_mp3: str, target_lufs: float = -14.0, log=print):
    # Pass 1: measure
    m = _run(["ffmpeg", "-y", "-i", in_wav, "-filter:a",
              f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"])
    info = None
    try:
        js = m.stderr[m.stderr.index("{"):m.stderr.rindex("}") + 1]
        info = json.loads(js)
    except Exception:
        info = None
    if info:
        filt = (f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11:"
                f"measured_I={info['input_i']}:measured_TP={info['input_tp']}:"
                f"measured_LRA={info['input_lra']}:measured_thresh={info['input_thresh']}:"
                f"offset={info['target_offset']}:linear=true,alimiter=limit=0.95")
    else:
        if log:
            log("  loudnorm measure failed, using single-pass")
        filt = f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11,alimiter=limit=0.95"
    r1 = _run(["ffmpeg", "-y", "-i", in_wav, "-filter:a", filt, "-ar", "44100", out_wav])
    if r1.returncode != 0:
        raise RuntimeError(f"mastering wav failed:\n{r1.stderr[-3000:]}")
    r2 = _run(["ffmpeg", "-y", "-i", out_wav, "-codec:a", "libmp3lame", "-b:a", "192k", out_mp3])
    if r2.returncode != 0:
        raise RuntimeError(f"mp3 export failed:\n{r2.stderr[-3000:]}")
    if log:
        log(f"  mastered to {target_lufs} LUFS -> {out_wav}, {out_mp3}")
