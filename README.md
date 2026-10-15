# mashup-engine
Open-source Python CLI that makes a mashup from 2+ songs, then edits videos to match it.

## Quick start

```bash
pip install -r requirements.txt
# full mix, no stems (fast, no torch needed)
mashup make songs/ --bpm auto --no-stems --out out/
mashup video clips/ --audio out/mashup.wav --grid out/mashup_grid.json --out out/mashup.mp4
mashup all songs/ clips/ --out out/ --no-stems
```

With stems (vocals of one song over instrumental of another, needs torch + Demucs):

```bash
mashup make songs/ --bpm auto --stems --out out/
mashup all songs/ clips/ --out out/ --stems
```

Edit the arrangement by hand:

```bash
mashup make songs/ --out out/ --no-stems
# edit out/plan.json, then re-render only:
mashup make songs/ --out out/ --no-stems --plan out/plan.json
```

## CLI

```
mashup make songs/ --bpm auto --stems --out out/
mashup video clips/ --audio out/mashup.wav --grid out/mashup_grid.json --out out/mashup.mp4
mashup all songs/ clips/ --out out/
```

Options:
- `make`: `--bpm auto|FLOAT` (default auto = median), `--stems` / `--no-stems` (default no-stems fast mode), `--out DIR`, `--plan plan.json` (edit arrangement by hand), `--cache-dir DIR`
- `video`: `--audio WAV`, `--grid JSON`, `--out MP4`, `--vertical` (1080x1920), `--fps 30|60`
- `all`: runs make + video.

## How it works

Audio (`mashup/` package):
1. Per-song analysis with librosa (tempo, beat grid, downbeats every 4th beat with strongest onset phase, key/mode via chroma + Krumhansl-Schmuckler mapped to Camelot, RMS energy, sections from novelty snapped to bar lines). Cached as JSON.
2. Master BPM = median (or `--bpm`). Time-stretch with pyrubberband, limited to +/-12% (half/double time if needed). Pitch-shift to a Camelot-compatible key, max +/-3 semitones, smallest shift wins.
3. `--stems`: Demucs htdemucs on CPU (vocals of one song over instrumental of another). Without it: full-mix crossfades.
4. Arrangement from 8/16-bar phrases, cuts only on downbeats, bass-swap crossfades (1-4 bars), riser/reverse-tail, energy low->high with peak in last third. `--plan plan.json` for hand edits.
5. Mastering: FFmpeg `loudnorm=I=-14:TP=-1.5:LRA=11` + `alimiter`, exports `mashup.wav` + `mashup.mp3`.
6. Sync check: cross-correlate onset envelopes at each transition, report ms, warn if >20ms. Writes `mashup_grid.json` (beats, downbeats, energy).

Video (`mashup/video.py`, `videoedit.py` shim):
1. PySceneDetect shots, ranked by OpenCV motion + brightness.
2. Cuts on beats from `mashup_grid.json` (high energy every 1-2 beats, low every 4-8), strongest-motion shots on downbeats/drops, zoom-punch/white flash on downbeats, speed ramps at drops, crossfades when calm, no repeat twice in a row.
3. 1920x1080 (`--vertical` for 1080x1920), 30/60fps, H.264 + AAC, length == audio. Verified with ffprobe.

## GitHub Actions

`.github/workflows/mashup.yml` (workflow_dispatch): installs ffmpeg, rubberband-cli, CPU torch, runs `mashup all inputs/songs inputs/clips --stems`, uploads `out/` (3 days).

## Deps

Pinned in `requirements.txt`. Heavy/optional: `torch`, `demucs` (only needed for `--stems`).
