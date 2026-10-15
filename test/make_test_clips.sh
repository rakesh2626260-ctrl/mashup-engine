#!/bin/bash
# Generate two tiny test clips with FFmpeg (no image viewing needed).
set -e
mkdir -p test_assets/clips
ffmpeg -y -f lavfi -i "testsrc=size=640x360:rate=30:duration=12" \
  -f lavfi -i "sine=frequency=440:duration=12" \
  -c:v libx264 -preset ultrafast -pix_fmt yuv420p -c:a aac -shortest \
  test_assets/clips/clipA.mp4
ffmpeg -y -f lavfi -i "mandelbrot=size=640x360:rate=30:maxiter=20:end_pts=360" \
  -f lavfi -i "sine=frequency=220:duration=12" \
  -c:v libx264 -preset ultrafast -pix_fmt yuv420p -c:a aac -shortest \
  test_assets/clips/clipB.mp4
ls -la test_assets/clips/
