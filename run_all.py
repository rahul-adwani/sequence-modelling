"""Run every section, write the logs, and report the claim ledger.

    python run_all.py                 everything that exists
    python run_all.py --article 2     just the inference-stack sections
    python run_all.py --only s02 s13  named sections

Exit code is 0 only if every claim in every section passed. That is the point of
the script: the articles quote these numbers, so "the repo is green" has to mean
something a build server could check.

Sections not yet written are reported as pending and skipped, rather than
crashing the run. A half-built repo should still be able to tell you what it has.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time

from foundations import ARTICLE_TITLES, SECTIONS, sections_for
from foundations.harness import LOG_DIR, RULE, environment, write_claims


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--article", type=int, choices=(2, 3, 4), default=None,
                    help="restrict to one article's sections")
    ap.add_argument("--only", nargs="+", metavar="SID", default=None,
                    help="run only these section ids, e.g. s02 s13")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress section output; print the summary only")
    args = ap.parse_args()

    wanted = sections_for(args.article)
    if args.only:
        keep = set(args.only)
        wanted = [s for s in wanted if s[0] in keep]
    if not wanted:
        print("no sections matched", file=sys.stderr)
        return 2

    sections, pending, results = [], [], []
    real_stdout = sys.stdout
    t_start = time.perf_counter()

    for sid, module, article, title in wanted:
        try:
            mod = importlib.import_module(f"foundations.{module}")
        except ModuleNotFoundError:
            pending.append((sid, article, title))
            continue

        t0 = time.perf_counter()
        if args.quiet:
            # Sections write their log file regardless; only the console is muted.
            import io
            sys.stdout = io.StringIO()
        try:
            sec = mod.run()
        finally:
            sys.stdout = real_stdout
        dt = time.perf_counter() - t0

        sections.append(sec)
        ok = sum(c.ok for c in sec.claims)
        results.append((sid, article, title, ok, len(sec.claims), dt))

    payload = write_claims(sections)

    # ------------------------------------------------------------------ summary
    print(RULE)
    print("SUMMARY")
    print(RULE)
    last_article = None
    for sid, article, title, ok, total, dt in results:
        if article != last_article:
            print(f"\nArticle {article} - {ARTICLE_TITLES[article]}")
            last_article = article
        mark = " " if ok == total else "!"
        dots = "." * max(2, 52 - len(title))
        print(f"  {mark} {sid}  {title} {dots} {ok:>3}/{total:<3} {dt:6.1f}s")

    if pending:
        print("\nPending (not yet written)")
        for sid, article, title in pending:
            print(f"    {sid}  {title}  [article {article}]")

    # This run decides the exit code. The ledger is the union across runs, so
    # quoting its totals here would report Article 2's claims as though this
    # invocation had checked them.
    ran = [c for s in sections for c in s.claims]
    failed = [c for c in ran if not c.ok]
    print()
    print(RULE)
    print(f"claims {len(ran) - len(failed)}/{len(ran)} passed across "
          f"{len(results)} sections in {time.perf_counter() - t_start:.1f}s")
    env = environment()
    print(f"environment: python {env['python']}, numpy {env['numpy']}, "
          f"{env['platform']}")
    print(f"wrote {LOG_DIR / 'claims.json'} and {len(sections)} section logs")
    if payload["total"] != len(ran):
        carried = payload["total"] - len(ran)
        print(f"ledger now holds {payload['passed']}/{payload['total']} across "
              f"{len(payload['sections'])} sections, {carried} carried over "
              f"from earlier runs of sections not run just now")

    if failed:
        print()
        print(f"FAILED CLAIMS ({len(failed)}):")
        for c in failed:
            print(f"  [{c.section}] {c.text}")
            if c.detail:
                print(f"           {c.detail}")
        print(RULE)
        return 1

    print(RULE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
