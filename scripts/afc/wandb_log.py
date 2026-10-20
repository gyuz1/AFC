"""Console log of a run in the W&B "Logs" tab layout: a timestamp line, then
the original line (metric dicts, messages, the training progress bar).

    python scripts/afc/wandb_log.py global_all             # last 40 lines, then follow
    python scripts/afc/wandb_log.py block_all --last 200
    python scripts/afc/wandb_log.py global_all --no-follow

Reads checkpoints/llava-v1.5-7b-afc-<run>/train.log. New lines are stamped
with the time they appear; for the backlog the time is reconstructed from the
progress bar's elapsed time (exact to a few seconds). The progress bar is
printed at most every --bar-every seconds, like W&B. Ctrl+C stops the view
only, never the training.
"""

import argparse
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BAR = re.compile(r"(\d+)/(\d+) \[([\d:]+)<")
GREY, RESET = "\033[90m", "\033[0m"
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")   # tqdm cursor moves


def seconds(hms: str) -> int:
    s = 0
    for part in hms.split(":"):
        s = s * 60 + int(part)
    return s


def segments(text: str):
    """Lines split on \\n and on tqdm's \\r redraws."""
    for seg in re.split(r"[\r\n]+", ANSI.sub("", text)):
        seg = seg.rstrip()
        if seg.strip():
            yield seg


class View:
    def __init__(self, bar_every: float):
        self.total = 0           # main progress bar length (largest total)
        self.start = None        # wall time of training step 0
        self.elapsed = 0
        self.bar_every = bar_every
        self.last_bar_t = None
        self.pending_bar = None

    def stamp(self, live: bool) -> datetime:
        if live or self.start is None:
            return datetime.now()
        return self.start + timedelta(seconds=self.elapsed)

    def feed(self, seg: str, live: bool):
        """Lines to print for one segment."""
        m = BAR.search(seg)
        if m and int(m.group(2)) >= self.total:
            self.total = int(m.group(2))
            self.elapsed = seconds(m.group(3))
            t = self.stamp(live)
            self.pending_bar = (t, seg.strip())
            if (self.last_bar_t is None
                    or (t - self.last_bar_t).total_seconds() >= self.bar_every):
                self.last_bar_t = t
                self.pending_bar = None
                return [(t, " " + seg.strip())]
            return []
        if m:                      # evaluation / loading bars: skip
            return []
        return [(self.stamp(live), seg)]


def session_starts(segs, text, log):
    """Wall time of step 0 of every training session in the log (a resume
    starts a new session whose tqdm elapsed restarts at 0).

    The last session is anchored on the file mtime (= its last bar redraw);
    earlier ones on the W&B run directories of the same run id
    (wandb/run-YYYYMMDD_HHMMSS-<id>, one per session).
    """
    total = max((int(b.group(2)) for b in BAR.finditer(text)), default=0)
    ends, prev = [], -1                 # last elapsed of each session
    for s in segs:
        m = BAR.search(s)
        if m and int(m.group(2)) == total:
            el = seconds(m.group(3))
            if el < prev:
                ends.append(prev)
            prev = el
    if prev < 0:
        return []
    ends.append(prev)
    mtime = datetime.fromtimestamp(os.path.getmtime(log))
    starts = [None] * len(ends)
    starts[-1] = mtime - timedelta(seconds=ends[-1])
    rid = re.search(r"wandb\.ai/\S+/runs/([a-z0-9]+)", text)
    dirs = sorted(REPO.glob(f"wandb/run-*-{rid.group(1)}")) if rid else []
    stamps = [datetime.strptime(d.name.split("-")[1], "%Y%m%d_%H%M%S")
              for d in dirs]
    for i in range(len(ends) - 1):
        if len(stamps) == len(ends):
            starts[i] = stamps[i]
        else:                           # no W&B dirs: assume no gap
            starts[i] = starts[i + 1] - timedelta(seconds=ends[i])
    return starts


def show(lines):
    for t, s in lines:
        sys.stdout.write(f"{GREY}{t:%Y-%m-%d %H:%M:%S}{RESET}\n{s}\n")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--last", type=int, default=40, help="backlog lines")
    ap.add_argument("--no-follow", action="store_true")
    ap.add_argument("--bar-every", type=float, default=60.0)
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()
    log = REPO / f"checkpoints/llava-v1.5-7b-afc-{args.run}/train.log"
    if not log.exists():
        sys.exit(f"no log: {log}")

    with open(log, errors="replace") as f:
        text = f.read()
        pos = f.tell()
    segs = list(segments(text))
    starts = session_starts(segs, text, log)
    view = View(args.bar_every)
    backlog, session, prev = [], 0, -1
    view.start = starts[0] if starts else None
    for s in segs:
        m = BAR.search(s)
        if m and int(m.group(2)) >= view.total:
            el = seconds(m.group(3))
            if el < prev:                       # resumed: tqdm restarts at 0
                session += 1
                view.start = starts[min(session, len(starts) - 1)]
                view.last_bar_t = None
            prev = el
        backlog += view.feed(s, live=False)
    if view.pending_bar:
        backlog.append((view.pending_bar[0], " " + view.pending_bar[1]))
        view.pending_bar = None
    show(backlog[-args.last:])
    if args.no_follow:
        return
    view.last_bar_t = None                      # live stamps use the clock

    buf = ""
    try:
        while True:
            time.sleep(args.interval)
            with open(log, errors="replace") as f:
                f.seek(pos)
                new = f.read()
                pos = f.tell()
            if not new:
                continue
            buf += new
            # keep the unfinished tail (no \r or \n yet) for the next read
            cut = max(buf.rfind("\n"), buf.rfind("\r"))
            done, buf = buf[:cut + 1], buf[cut + 1:]
            out = []
            for s in segments(done):
                out += view.feed(s, live=True)
            show(out)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
