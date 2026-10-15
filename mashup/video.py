"""Video pipeline: shot detect, rank by motion/brightness, cut on beats, render."""
import glob
import json
import os
import subprocess

import cv2
import numpy as np

TARGET_W, TARGET_H = 1920, 1080
VERT_W, VERT_H = 1080, 1920


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def ffprobe_info(path: str) -> dict:
    r = _run(["ffprobe", "-v", "error", "-print_format", "json",
              "-show_format", "-show_streams", path])
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {r.stderr[-2000:]}")
    return json.loads(r.stdout)


def list_clips(inp: str) -> list:
    if os.path.isfile(inp):
        return [inp]
    exts = ("*.mp4", "*.mov", "*.mkv", "*.avi", "*.webm", "*.m4v", "*.MP4", "*.MOV")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(inp, e)))
    files = sorted(files)
    if not files:
        raise RuntimeError(f"No clips found in {inp} (looked for {exts})")
    return files


def detect_shots(clip: str, log=print) -> list:
    """PySceneDetect shots -> [(start, end)]. Fallback: whole clip."""
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector
        video = open_video(clip)
        mgr = SceneManager()
        mgr.add_detector(ContentDetector(threshold=27.0))
        mgr.detect_scenes(video=video, show_progress=False)
        scenes = mgr.get_scene_list()
        if scenes:
            out = [(s[0].get_seconds(), s[1].get_seconds()) for s in scenes]
            log(f"  {os.path.basename(clip)}: {len(out)} shots")
            return out
    except Exception as e:
        log(f"  scene detect fallback for {os.path.basename(clip)}: {e}")
    info = ffprobe_info(clip)
    dur = float(info["format"].get("duration", 5.0))
    return [(0.0, dur)]


def rank_shot(clip: str, start: float, end: float, n_samples: int = 12) -> dict:
    """Motion = mean abs frame difference; brightness = mean gray level."""
    cap = cv2.VideoCapture(clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = max(0.1, end - start)
    times = np.linspace(start + 0.05, max(start + 0.06, end - 0.05), n_samples)
    prev = None
    mot, bright, cnt = 0.0, 0.0, 0
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok or fr is None:
            continue
        g = cv2.cvtColor(cv2.resize(fr, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        bright += float(g.mean())
        if prev is not None:
            mot += float(np.abs(g - prev).mean())
        prev = g
        cnt += 1
    cap.release()
    if cnt == 0:
        return {"motion": 0.0, "brightness": 0.5, "fps": fps}
    n = max(1, cnt - 1)
    return {"motion": mot / n, "brightness": (bright / cnt) / 255.0, "fps": fps}


def plan_cuts(grid: dict) -> list:
    """Beat-aligned cut points. High energy every 2 beats, low every 4-6 beats."""
    beats = grid["beats"]
    downs = set(grid["downbeats"])
    energy = grid.get("energy", [])
    if not beats:
        raise RuntimeError("Empty beat grid")
    ev = np.array([e["v"] for e in energy]) if energy else np.array([0.5])
    et = np.array([e["t"] for e in energy]) if energy else np.array([0.0])
    med = float(np.median(ev)) if len(ev) else 0.5
    emax = float(ev.max()) if len(ev) else 1.0

    def e_at(t):
        if not len(et):
            return 0.5
        j = int(np.argmin(np.abs(et - t)))
        return float(ev[j])

    cuts = [0.0]
    i = 0
    while i < len(beats):
        t = beats[i]
        e = e_at(t)
        hi = e >= med
        if hi:
            step = 1 if e >= med + 0.25 * (emax - med + 1e-9) else 2
        else:
            step = 4 if e >= 0.5 * med else 6
            step = min(step, 8)
        j = min(len(beats) - 1, i + step)
        if j <= i:
            break
        cuts.append(beats[j])
        i = j
        if cuts[-1] >= grid["duration"] - 0.05:
            break
    if cuts[-1] < grid["duration"] - 0.05:
        cuts.append(grid["duration"])
    return sorted(set(round(c, 4) for c in cuts if c <= grid["duration"] + 1e-6))


def _xd_for(e_prev: float, e_next: float, d_prev: float, d_next: float,
            tcur: float, med: float) -> float:
    calm = e_prev < med and e_next < med
    xd = 0.6 if calm else 0.15
    return max(0.05, min(xd, d_prev / 2, d_next / 2, tcur / 2))


def _chain_len(segs: list, med: float) -> tuple:
    """Returns (video_len, xds). xfade overlaps consume sum(xds)."""
    if not segs:
        return 0.0, []
    tcur = segs[0]["dur"]
    xds = []
    for k in range(1, len(segs)):
        xd = _xd_for(segs[k - 1]["energy"], segs[k]["energy"],
                     segs[k - 1]["dur"], segs[k]["dur"], tcur, med)
        xds.append(xd)
        tcur = tcur + segs[k]["dur"] - xd
    return tcur, xds


def build_video(clips_in: str, audio_path: str, grid_path: str, out_mp4: str,
                vertical: bool = False, fps: int = 30, log=print) -> str:
    with open(grid_path) as f:
        grid = json.load(f)
    TW, TH = (VERT_W, VERT_H) if vertical else (TARGET_W, TARGET_H)
    clips = list_clips(clips_in)
    log(f"shots: scanning {len(clips)} clip(s)...")
    shots = []  # {clip,start,end,motion,brightness}
    for c in clips:
        for (s, e) in detect_shots(c, log=log):
            if e - s < 0.3:
                continue
            r = rank_shot(c, s, e)
            shots.append({"clip": c, "start": s, "end": e, **r})
    if not shots:
        raise RuntimeError("No usable shots found")
    # rank: motion primary, brightness tiebreak (avoid all-dark)
    shots.sort(key=lambda s: (s["motion"], s["brightness"]), reverse=True)
    log(f"  ranked {len(shots)} shots (top motion={shots[0]['motion']:.2f})")

    cuts = plan_cuts(grid)
    downs = set(grid["downbeats"])
    dur = grid["duration"]
    log(f"  cuts: {len(cuts)-1} segments from {len(grid['beats'])} beats")

    # energy lookup for drop detection
    ev = np.array([e["v"] for e in grid.get("energy", [{"v": .5}])])
    et = np.array([e["t"] for e in grid.get("energy", [{"t": 0.0}])])
    emax = float(ev.max()) if len(ev) else 1.0

    def e_at(t):
        j = int(np.argmin(np.abs(et - t)))
        return float(ev[j])

    # assign shots to segments
    segs = []
    last = -1
    order = list(range(len(shots)))
    for k in range(len(cuts) - 1):
        t0, t1 = cuts[k], min(cuts[k + 1], dur)
        seg_dur = t1 - t0
        if seg_dur < 0.15:
            continue
        is_down = any(abs(t0 - d) < 0.09 for d in downs)
        e = e_at(t0)
        is_drop = e >= 0.85 * emax
        if is_down or is_drop:
            pick = order[0]  # strongest motion
            if pick == last and len(order) > 1:
                pick = order[1]
        else:
            # round-robin through ranked shots, avoid immediate repeat
            cands = [x for x in order if x != last]
            pick = cands[(k) % len(cands)]
        last = pick
        segs.append({"t0": t0, "t1": t1, "dur": seg_dur, "shot": shots[pick],
                     "downbeat": is_down, "drop": is_drop, "energy": e})

    # Build ffmpeg filter graph with xfade chain.
    # Each segment: trim+setpts+scale/pad+fps, plus punch (crop-zoom) and
    # white-flash-in on downbeats, speedup at drops.
    ain = ffprobe_info(audio_path)
    audio_dur = float(ain["format"]["duration"])
    total = min(dur, audio_dur)
    # clamp last segment
    if segs and segs[-1]["t1"] > total:
        segs[-1]["t1"] = total
        segs[-1]["dur"] = total - segs[-1]["t0"]
    segs = [s for s in segs if s["dur"] >= 0.15]
    med = float(np.median(ev)) if len(ev) else 0.5

    # xfade overlaps consume time: extend coverage with extra round-robin
    # segments (new shots, no immediate repeats) so video length == audio.
    vlen, _ = _chain_len(segs, med)
    guard = 0
    while vlen < total and guard < 200:
        prev = segs[-1]
        cands = [x for x in order if x != last]
        pick = cands[len(segs) % len(cands)]
        last = pick
        segs.append({"t0": total, "t1": total + prev["dur"], "dur": prev["dur"],
                     "shot": shots[pick], "downbeat": False, "drop": False,
                     "energy": prev["energy"]})
        vlen, _ = _chain_len(segs, med)
        guard += 1
    log(f"  segments: {len(segs)} (video chain {vlen:.2f}s for {total:.2f}s audio)")

    # unique input files
    used_clips = []
    for s in segs:
        if s["shot"]["clip"] not in used_clips:
            used_clips.append(s["shot"]["clip"])

    fc_parts = []
    # Per-segment chains (linear comma-chains, one labeled output each).
    for si, s in enumerate(segs):
        ci = used_clips.index(s["shot"]["clip"])
        sh = s["shot"]
        sh_len = max(0.4, sh["end"] - sh["start"])
        ss = sh["start"]
        # speed ramp at drops: 1.4x (take a longer window, play it faster)
        speed = 1.4 if s["drop"] else 1.0
        need = s["dur"] * speed + 0.1
        chain = [f"[{ci}:v]trim=start={ss:.3f}:duration={need:.3f},setpts=PTS-STARTPTS"]
        if sh_len < s["dur"]:
            chain.append("loop=loop=-1:size=500")
        if speed != 1.0:
            chain.append(f"setpts={1.0/speed:.4f}*PTS")
        chain.append(f"trim=duration={s['dur']:.4f},setpts=PTS-STARTPTS")
        chain.append(f"scale={TW}:{TH}:force_original_aspect_ratio=increase,crop={TW}:{TH},setsar=1,fps={fps}")
        if s["downbeat"]:
            # zoom-punch: static 8% punch-in + white flash-in (fade from white)
            chain.append(f"scale={int(TW*1.08)}:{int(TH*1.08)},crop={TW}:{TH}")
            chain.append("fade=t=in:st=0:d=0.08:color=white")
        fc_parts.append(",".join(chain) + f"[v{si}]")

    # xfade chain: calm (low-energy) gets 0.6s, energetic 0.15s
    med = float(np.median(ev)) if len(ev) else 0.5
    cur = "[v0]"
    # xfade chain (same durations as _chain_len so the math matches)
    cur = "[v0]"
    tcur = segs[0]["dur"]
    for k in range(1, len(segs)):
        xd = _xd_for(segs[k - 1]["energy"], segs[k]["energy"],
                     segs[k - 1]["dur"], segs[k]["dur"], tcur, med)
        off = tcur - xd
        out = f"[x{k}]" if k < len(segs) - 1 else "[vout]"
        fc_parts.append(f"{cur}[v{k}]xfade=transition=fade:duration={xd:.3f}:offset={off:.3f}{out}")
        cur = out
        tcur = tcur + segs[k]["dur"] - xd

    fc = ";".join(fc_parts)
    preset = os.environ.get("MASHUP_X264_PRESET", "veryfast")
    cmd = ["ffmpeg", "-y"]
    for c in used_clips:
        cmd += ["-i", c]
    cmd += ["-i", audio_path,
            "-filter_complex", fc,
            "-map", "[vout]", "-map", f"{len(used_clips)}:a",
            "-c:v", "libx264", "-preset", preset, "-pix_fmt", "yuv420p",
            "-r", str(fps), "-c:a", "aac", "-b:a", "160k",
            "-t", f"{total:.3f}", "-shortest", out_mp4]
    log(f"  render: {len(segs)} segments, {TW}x{TH}@{fps} -> {out_mp4}")
    r = _run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"video render failed:\n{r.stderr[-4000:]}")
    # verify
    info = ffprobe_info(out_mp4)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    vw, vh = int(v["width"]), int(v["height"])
    assert (vw, vh) == (TW, TH), f"size {vw}x{vh} != {TW}x{TH}"
    assert a is not None, "missing audio stream"
    vd = float(info["format"]["duration"])
    assert abs(vd - total) < 0.6, f"duration {vd:.2f} != audio {total:.2f}"
    log(f"  verified: {vw}x{vh} {v.get('avg_frame_rate')} audio={a['codec_name']} dur={vd:.2f}s")
    return out_mp4
