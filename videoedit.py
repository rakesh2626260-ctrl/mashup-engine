"""Backward-compat shim: python videoedit.py clips/ --audio ... --grid ... --out ..."""
import argparse
from mashup.cli import cmd_video, _log
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser(prog="videoedit")
    p.add_argument("clips")
    p.add_argument("--audio", required=True)
    p.add_argument("--grid", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--vertical", action="store_true")
    p.add_argument("--fps", type=int, default=30)
    a = p.parse_args()
    cmd_video(a)


if __name__ == "__main__":
    main()
