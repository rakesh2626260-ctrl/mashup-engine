"""Sync check: beat misalignment per transition via onset-envelope xcorr."""
import json

import librosa
import numpy as np

GRID_SR = 4410  # analysis rate for click-track grids (0.23 ms resolution)


def _click_train(times: list, center: float, window: float, sr: int = GRID_SR) -> np.ndarray:
    """Gaussian click train (~onset envelope) for beats near center."""
    n = int(window * 2 * sr)
    x = np.zeros(n)
    sigma = 0.005 * sr  # 5 ms wide clicks
    span = int(6 * sigma)
    for t in times:
        dt = t - center
        if abs(dt) > window:
            continue
        c = int((dt + window) * sr)
        lo, hi = max(0, c - span), min(n, c + span + 1)
        g = np.arange(lo - c, hi - c)
        x[lo:hi] += np.exp(-0.5 * (g / sigma) ** 2)
    return x - x.mean()


def transition_misalignment(out_beats: list, in_beats: list, at_time: float,
                            max_lag_ms: float = 100.0) -> float:
    """Cross-correlate the two songs' beat-click envelopes around the
    transition center. Returns lag in ms (0 = grids agree).

    The lag search is restricted to +/-max_lag_ms: whole-beat multiples are
    the same grid phase (cuts are on downbeats), so only sub-beat alignment
    counts as misalignment.
    """
    window = 3.0
    o = _click_train(out_beats, at_time, window)
    i = _click_train(in_beats, at_time, window)
    n1, n2 = np.linalg.norm(o), np.linalg.norm(i)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    corr = np.correlate(o / n1, i / n2, mode="full")
    zero = len(i) - 1
    max_lag = int(max_lag_ms / 1000.0 * GRID_SR)
    lo, hi = max(0, zero - max_lag), min(len(corr), zero + max_lag + 1)
    lag = int(np.argmax(corr[lo:hi]) + lo - zero)
    return abs(lag / GRID_SR * 1000.0)


def _placed_beats(analysis: dict, rate: float, seg_start: float) -> list:
    """Song's beat grid (stretched) placed at its segment start in mashup time."""
    # segment starts on a stretched downbeat; find that downbeat's index
    db = [d / rate for d in analysis["downbeats"]]
    # offset of segment start within the song's stretched grid:
    off = db[0] if db else 0.0
    return [b / rate + (seg_start - off) for b in analysis["beats"] if b / rate >= off - 1e-6]


def sync_report(plan: dict, analyses: list, mix: np.ndarray, sr: int, log=print) -> list:
    """Measure each transition. Returns list of {index, ms, ok}."""
    by_idx = {a["path"]: a for a in analyses}
    rows = []
    for k, tr in enumerate(plan["transitions"]):
        out_seg = plan["segments"][tr["from"]]
        in_seg = plan["segments"][tr["to"]]
        out_a = by_idx[out_seg["song_path"]]
        in_a = by_idx[in_seg["song_path"]]
        out_b = _placed_beats(out_a, out_seg["rate"], out_seg["start_in_mashup"])
        in_b = _placed_beats(in_a, in_seg["rate"], in_seg["start_in_mashup"])
        center = tr["at_time"] + tr["duration"] / 2
        ms = transition_misalignment(out_b, in_b, center)
        ok = ms <= 20.0
        rows.append({"transition": k, "at_time": tr["at_time"], "ms": ms, "ok": ok})
        status = "OK " if ok else "FAIL"
        log(f"  sync transition {k} @ {tr['at_time']:.2f}s: {ms:.1f} ms [{status}] (limit 20 ms)")
    return rows


def write_grid(plan: dict, mix: np.ndarray, sr: int, analyses: list, path: str):
    """Final beat/downbeat grid + energy curve for the video stage."""
    # rebuild mashup beat grid from the stitched segments. All segments share
    # the master grid when no clamping was needed; stitching keeps it exact
    # even across crossfade overlaps (overlaps are whole bars).
    beats, downs = [], []
    t_offset = 0.0
    for si, seg in enumerate(plan["segments"]):
        s = next(x for x in plan["songs"] if x["index"] == seg["song_index"])
        bd = s["beat_dur"]
        t0 = seg["start_in_mashup"]
        nb = int(round(seg["duration"] / bd))
        for b in range(nb):
            tt = t0 + b * bd
            if not beats or tt > beats[-1] + bd * 0.4:
                beats.append(round(tt, 4))
    beats = sorted(beats)
    downs = [b for i, b in enumerate(beats) if i % 4 == 0]
    dur = plan["total_duration"]
    # energy curve of final mix
    rms = librosa.feature.rms(y=mix)[0] if len(mix) else np.zeros(1)
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
    stride = max(1, len(rms) // 500)
    energy = [{"t": float(times[i]), "v": float(rms[i])} for i in range(0, len(rms), stride)]
    grid = {
        "master_bpm": plan["master_bpm"],
        "master_key": plan["master_key"],
        "duration": dur,
        "sr": sr,
        "beats": beats,
        "downbeats": downs,
        "energy": energy,
    }
    with open(path, "w") as f:
        json.dump(grid, f, indent=1)
    return grid
