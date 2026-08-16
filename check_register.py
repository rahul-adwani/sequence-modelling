"""Check the two typography rules that are rules rather than preferences.

    python check_register.py

**No em-dash, en-dash or unicode minus, anywhere.** This is a house style rule
for the articles, and it is enforced here rather than by eye because an em-dash
that reaches a published draft costs a rewrite. Hyphen-minus is the only dash.

**Nothing outside ASCII in a section module.** This one is not style. Sections
print their transcript to stdout as well as writing it to `logs/`, and on
Windows a redirected stdout uses the system code page rather than UTF-8. One
Greek sigma in a `kv` label is then the difference between a run that reports
its claims and a run that dies with a UnicodeEncodeError partway through
section 7, which is precisely how the character this script now catches was
found. Figure and diagram scripts are exempt: their strings go into PNGs, where
a proper multiplication sign is the right character.

Exit code is 0 only if nothing was found, so the batch files can gate on it.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Written as escapes rather than as the characters themselves, so that this
# file passes its own check. A linter that trips on its own source is a linter
# people turn off.
DASHES = {
    chr(0x2014): 'em-dash',
    chr(0x2013): 'en-dash',
    chr(0x2015): 'horizontal bar',
    chr(0x2212): 'unicode minus',
    chr(0x2011): 'non-breaking hyphen',
}

# The rejected Part 3 draft, kept only until the rewrite replaces it. It is full
# of em-dashes, which is one of the reasons it was rejected.
EXEMPT = {"POST_3_inference_stack.md"}

# Strings here end up in a PNG, not in a log, so typography is allowed.
ASCII_EXEMPT = {"make_figures.py", "make_diagrams.py"}

SKIP_DIRS = {"logs", "figures", "__pycache__", ".pytest_cache", ".git"}


def files() -> list[Path]:
    out = []
    for path in ROOT.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix in (".py", ".md", ".bat") and path.name not in EXEMPT:
            out.append(path)
    return sorted(out)


def main() -> int:
    problems: list[str] = []
    for path in files():
        rel = path.relative_to(ROOT)
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.split("\n"), start=1):
            for ch, name in DASHES.items():
                if ch in line:
                    problems.append(f"{rel}:{line_no}  {name}")
            if (path.suffix == ".py" and path.name not in ASCII_EXEMPT
                    and any(ord(c) > 127 for c in line)):
                shown = sorted({hex(ord(c)) for c in line if ord(c) > 127})
                problems.append(
                    f"{rel}:{line_no}  non-ascii in a module that prints to a "
                    f"log: {' '.join(shown)}")

    if problems:
        print(f"register check FAILED, {len(problems)} problems:")
        for p in problems:
            print("  " + p)
        return 1

    print(f"register check ok across {len(files())} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
