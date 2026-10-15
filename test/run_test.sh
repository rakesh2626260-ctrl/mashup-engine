#!/bin/bash
# End-to-end test: synthetic songs + clips -> mashup all --no-stems -> verify.
# Usage: bash test/run_test.sh  (run from repo root with .venv active or VENV set)
set -u
VENV="${VENV:-.venv}"
PY="$VENV/bin/python"
OUT="${OUT:-out_test}"

"$PY" test/make_test_songs.py
bash test/make_test_clips.sh
rm -rf "$OUT"
"$PY" mashup.py all test_assets/songs test_assets/clips --out "$OUT" --no-stems
echo "=== plan ==="
cat "$OUT/plan.json"
echo "=== grid (head) ==="
"$PY" -c "import json; g=json.load(open('$OUT/mashup_grid.json')); print({k:(len(v) if isinstance(v,list) else v) for k,v in g.items()})"
echo "=== ffprobe audio ==="
ffprobe -v error -show_entries format=duration,size -show_entries stream=codec_type,codec_name,sample_rate -of default=noprint_wrappers=1 "$OUT/mashup.wav"
ffprobe -v error -show_entries format=duration,size -show_entries stream=codec_type,codec_name -of default=noprint_wrappers=1 "$OUT/mashup.mp3"
echo "=== ffprobe video ==="
ffprobe -v error -show_entries format=duration,size -show_entries stream=codec_type,codec_name,width,height,avg_frame_rate -of default=noprint_wrappers=1 "$OUT/mashup.mp4"
