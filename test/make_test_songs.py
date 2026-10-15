"""Generate two short synthetic test songs (different tempos and keys).

Song A: 120 BPM, A major-ish (A2/E3 chord arps + kick on beats).
Song B: 128 BPM, C major-ish (C3/G3 chord arps + kick on beats).
Both include transients (for beat tracking) and tonal content (for key).
"""
import numpy as np
import soundfile as sf

SR = 44100


def _sine(freq, n):
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * freq * t).astype(np.float64)


def _note(freq, dur, decay=6.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    env = np.exp(-t * decay)
    return (np.sin(2 * np.pi * freq * t) * env).astype(np.float64)


def make_song(bpm, roots, out, bars=16, beats_per_bar=4):
    beat = 60.0 / bpm
    bar = beat * beats_per_bar
    total = bars * bar
    n = int(total * SR)
    y = np.zeros(n)
    # kick on every beat (short decaying sine at 60Hz + click)
    for b in range(bars * beats_per_bar):
        t0 = int(b * beat * SR)
        k = _note(60.0, 0.25, decay=18.0)
        y[t0:t0 + len(k)] += k * 0.9
        # hats on off-beats
        h0 = int((b + 0.5) * beat * SR)
        h = np.random.RandomState(b).randn(int(0.05 * SR)) * np.exp(-np.arange(int(0.05 * SR)) / (0.008 * SR)) * 0.15
        y[h0:h0 + len(h)] += h
    # bass + arp chord tones per bar (roots cycle)
    for m in range(bars):
        f = roots[m % len(roots)]
        for k, mult in enumerate([1, 2, 3, 4]):
            t0 = int((m * bar + k * beat) * SR)
            nt = _note(f * mult / 2, beat * 0.95, decay=4.0)
            y[t0:t0 + len(nt)] += nt * (0.35 / k + 0.15) if k else nt * 0.4
    y /= max(1e-9, np.abs(y).max())
    sf.write(out, (y * 0.89).astype(np.float32), SR)
    print(f"wrote {out} ({total:.1f}s @ {bpm} BPM)")


if __name__ == "__main__":
    import os
    os.makedirs("test_assets/songs", exist_ok=True)
    # A major-ish @ 120 BPM; C major-ish @ 128 BPM.
    # (Kept within +/-12% of their median 124 BPM so one exact master grid fits.)
    make_song(120, [110.0, 164.81, 138.59, 110.0], "test_assets/songs/songA_120bpm_Amaj.wav")
    make_song(128, [130.81, 98.0, 110.0, 87.31], "test_assets/songs/songB_128bpm_Cmaj.wav")
