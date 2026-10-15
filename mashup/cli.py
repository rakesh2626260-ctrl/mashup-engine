"""CLI: mashup make|video|all."""
import argparse
import glob
import json
import os
import statistics
import subprocess
import sys


def _log(msg):
    print(msg, flush=True)


def _songs_in(d):
    exts = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus")
    if os.path.isfile(d):
        return [d]
    files = []
    for e in exts:
        files += glob.glob(os.path.join(d, f"*{e}"))
    return sorted(files)


def cmd_make(args):
    from .analysis import analyze_song
    from .plan import build_plan, save_plan, load_plan
    from .build import render_plan, SR
    from .mastering import master
    from .sync import sync_report, write_grid

    songs = _songs_in(args.songs)
    if len(songs) < 2:
        sys.exit(f"error: need 2+ songs, found {len(songs)} in {args.songs}")
    os.makedirs(args.out, exist_ok=True)
    cache_dir = args.cache_dir or os.path.join(args.out, ".mashup_cache")

    if args.plan:
        _log(f"load plan {args.plan}")
        plan = load_plan(args.plan)
        # still need analyses for rendering (downbeat grids come from plan though)
        analyses = [analyze_song(s, cache_dir) for s in songs]
    else:
        _log(f"analyze {len(songs)} songs...")
        analyses = []
        for s in songs:
            a = analyze_song(s, cache_dir)
            analyses.append(a)
            _log(f"  {os.path.basename(s)}: {a['tempo']:.1f} BPM {a['key']} ({a['camelot']}) "
                 f"E={a['mean_energy']:.4f} beats={len(a['beats'])} downbeats={len(a['downbeats'])}")
        bpm = None if (args.bpm in (None, "auto")) else float(args.bpm)
        plan = build_plan(analyses, master_bpm=bpm, phrase_bars=args.phrase_bars)
        save_plan(plan, os.path.join(args.out, "plan.json"))
        _log(f"master: {plan['master_bpm']:.1f} BPM, key {plan['master_key']} ({plan['master_camelot']})")
        for s in plan["songs"]:
            _log(f"  {os.path.basename(s['path'])}: rate={s['rate']:.3f} (eff {s['effective_bpm']:.1f}) "
                 f"shift={s['pitch_shift']:+d} -> {s['transposed_camelot']}")

    use_stems = bool(args.stems) and not bool(args.no_stems)
    stems_dir = None
    if use_stems:
        from .stems import separate_songs
        stems_dir = os.path.join(args.out, "separated")
        paths = [s["path"] for s in plan["songs"]]
        separate_songs(paths, stems_dir, log=_log)

    _log("render arrangement...")
    tmp = os.path.join(args.out, "_mix_raw.wav")
    mix = render_plan(plan, analyses, tmp, use_stems=use_stems, stems_dir=stems_dir, log=_log)

    _log("sync check...")
    rows = sync_report(plan, analyses, mix, SR, log=_log)
    bad = [r for r in rows if not r["ok"]]
    _log("SYNC REPORT " + json.dumps([{"t": round(r["at_time"], 2), "ms": round(r["ms"], 2),
                                       "ok": r["ok"]} for r in rows]))
    _log("master...")
    master(tmp, os.path.join(args.out, "mashup.wav"), os.path.join(args.out, "mashup.mp3"), log=_log)
    try:
        os.remove(tmp)
    except OSError:
        pass
    grid_path = os.path.join(args.out, "mashup_grid.json")
    write_grid(plan, mix, SR, analyses, grid_path)
    _log(f"wrote {grid_path}")
    if bad:
        _log(f"WARNING: {len(bad)} transition(s) over 20 ms off")
        sys.exit(2)


def cmd_video(args):
    from .video import build_video
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    build_video(args.clips, args.audio, args.grid, args.out,
                vertical=bool(args.vertical), fps=int(args.fps), log=_log)


def cmd_all(args):
    mk = argparse.Namespace(songs=args.songs, bpm=args.bpm, stems=args.stems,
                            no_stems=args.no_stems, out=args.out, plan=args.plan,
                            cache_dir=args.cache_dir, phrase_bars=args.phrase_bars)
    cmd_make(mk)
    vd = argparse.Namespace(clips=args.clips, audio=os.path.join(args.out, "mashup.wav"),
                            grid=os.path.join(args.out, "mashup_grid.json"),
                            out=os.path.join(args.out, "mashup.mp4"),
                            vertical=bool(args.vertical), fps=int(args.fps))
    cmd_video(vd)


def build_parser():
    p = argparse.ArgumentParser(prog="mashup", description="mashup-engine: songs -> mashup -> video")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_make(sp):
        sp.add_argument("songs", help="folder of songs (or single file)")
        sp.add_argument("--bpm", default="auto", help="'auto' (median) or a number")
        g = sp.add_mutually_exclusive_group()
        g.add_argument("--stems", dest="stems", action="store_true", help="Demucs stems mode")
        g.add_argument("--no-stems", dest="no_stems", action="store_true", help="fast full-mix mode (default)")
        sp.set_defaults(stems=False, no_stems=False)
        sp.add_argument("--out", default="out/", help="output dir")
        sp.add_argument("--plan", default=None, help="plan.json to load (hand-edited arrangement)")
        sp.add_argument("--cache-dir", default=None)
        sp.add_argument("--phrase-bars", type=int, default=8)

    m = sub.add_parser("make", help="make mashup audio")
    add_make(m)
    m.set_defaults(fn=cmd_make)

    v = sub.add_parser("video", help="edit video to mashup")
    v.add_argument("clips", help="folder of clips or single video")
    v.add_argument("--audio", required=True)
    v.add_argument("--grid", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--vertical", action="store_true")
    v.add_argument("--fps", type=int, default=30)
    v.set_defaults(fn=cmd_video)

    a = sub.add_parser("all", help="make + video")
    a.add_argument("songs")
    a.add_argument("clips")
    a.add_argument("--bpm", default="auto")
    g = a.add_mutually_exclusive_group()
    g.add_argument("--stems", dest="stems", action="store_true")
    g.add_argument("--no-stems", dest="no_stems", action="store_true")
    a.set_defaults(stems=False, no_stems=False)
    a.add_argument("--out", default="out/")
    a.add_argument("--plan", default=None)
    a.add_argument("--cache-dir", default=None)
    a.add_argument("--phrase-bars", type=int, default=8)
    a.add_argument("--vertical", action="store_true")
    a.add_argument("--fps", type=int, default=30)
    a.set_defaults(fn=cmd_all)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
