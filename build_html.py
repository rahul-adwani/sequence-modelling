"""Build a single self-contained HTML file from an article's markdown.

    python build_html.py <article>.md

Every image is inlined as a base64 data URI, so the result is one file with no
external requests: it renders on a phone, in an email, or from a USB stick, and
it keeps working when the repository is not around it.

Mobile is the primary target rather than an afterthought. Body text is set large
enough to read at arm's length, images run edge to edge, and the two things that
usually break long technical articles on a small screen (wide tables and code
blocks) each scroll inside their own container so the page itself never scrolls
sideways.
"""
from __future__ import annotations

import base64
import mimetypes
import re
import sys
from pathlib import Path

try:
    import markdown
except ImportError:  # pragma: no cover
    print("pip install markdown", file=sys.stderr)
    sys.exit(1)

CSS = """
:root{
  --paper:#f4f5f7; --paper-2:#e9ecf0; --ink:#15181d; --ink-soft:#4b535d;
  --ink-faint:#79828d; --rule:#d6dbe1;
  --accent:#1a4fa0; --accent-bg:#e5ecf8;
  --warn:#a33a2a; --warn-bg:#f7eae7;
  --code-bg:#1b1f26; --code-ink:#d2d8de; --code-ok:#6fbf8b;
  --serif:Georgia,'Iowan Old Style','Palatino Linotype',Palatino,serif;
  --sans:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif;
  --mono:ui-monospace,'SF Mono',SFMono-Regular,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark){
  :root:where(:not([data-theme="light"])){
    --paper:#14171c; --paper-2:#1c2027; --ink:#e7eaee; --ink-soft:#a9b2bc;
    --ink-faint:#7c858f; --rule:#2b313a;
    --accent:#7aa9e8; --accent-bg:#172433;
    --warn:#e0836f; --warn-bg:#2e1e1a;
    --code-bg:#0e1116; --code-ink:#c9cfd7;
  }
}
:root[data-theme="dark"]{
  --paper:#14171c; --paper-2:#1c2027; --ink:#e7eaee; --ink-soft:#a9b2bc;
  --ink-faint:#7c858f; --rule:#2b313a;
  --accent:#7aa9e8; --accent-bg:#172433;
  --warn:#e0836f; --warn-bg:#2e1e1a;
  --code-bg:#0e1116; --code-ink:#c9cfd7;
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
     font-family:var(--serif);font-size:18px;line-height:1.68;
     -webkit-text-size-adjust:100%;-webkit-font-smoothing:antialiased}
.wrap{max-width:40rem;margin:0 auto;padding:2.25rem 1.15rem 5rem}
@media(min-width:48rem){ body{font-size:19px} .wrap{padding:3.5rem 1.5rem 6rem} }

h1{font-size:clamp(1.85rem,7vw,2.7rem);line-height:1.12;margin:0 0 .65rem;
   font-weight:600;letter-spacing:-.02em;text-wrap:balance}
h1+h3{font-family:var(--sans);font-size:1.02rem;font-weight:400;
      color:var(--ink-soft);margin:0 0 1.6rem;letter-spacing:0}
h2{font-size:clamp(1.35rem,5vw,1.7rem);line-height:1.22;margin:3.2rem 0 1rem;
   font-weight:600;letter-spacing:-.012em;text-wrap:balance;
   padding-top:1.4rem;border-top:2px solid var(--rule)}
h3{font-family:var(--sans);font-size:1.02rem;font-weight:650;
   margin:2.1rem 0 .7rem;letter-spacing:-.004em}
p{margin:0 0 1.15rem;text-wrap:pretty}
strong{font-weight:650}
a{color:var(--accent);text-decoration-thickness:1px;text-underline-offset:2px}
a:focus-visible{outline:2px solid var(--accent);outline-offset:3px}
hr{border:none;border-top:1px solid var(--rule);margin:2.5rem 0}
ul,ol{margin:0 0 1.2rem;padding-left:1.35rem}
li{margin-bottom:.5rem}

blockquote{margin:1.6rem 0;padding:.9rem 1.1rem;background:var(--accent-bg);
           border-left:3px solid var(--accent);font-style:normal}
blockquote p{margin:0;font-weight:600}

img{max-width:100%;height:auto;display:block;margin:1.8rem auto;
    background:#fff;border:1px solid var(--rule);border-radius:3px}

code{font-family:var(--mono);font-size:.86em;background:var(--paper-2);
     padding:.1em .34em;border-radius:2px;overflow-wrap:break-word}
pre{background:var(--code-bg);color:var(--code-ink);font-size:.76rem;
    line-height:1.6;padding:.95rem 1rem;border-radius:4px;overflow-x:auto;
    margin:1.6rem 0;-webkit-overflow-scrolling:touch}
pre code{background:none;padding:0;font-size:inherit;color:inherit;
         white-space:pre}

.tablewrap{overflow-x:auto;margin:1.6rem 0;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-family:var(--sans);
      font-size:.83rem;font-variant-numeric:tabular-nums}
th{text-align:left;font-size:.68rem;letter-spacing:.06em;text-transform:uppercase;
   color:var(--ink-faint);border-bottom:1.5px solid var(--rule);
   padding:.5rem .7rem .5rem 0;white-space:nowrap}
td{padding:.5rem .7rem .5rem 0;border-bottom:1px solid var(--rule);
   vertical-align:top}
tbody tr:last-child td{border-bottom:none}
"""


def inline_images(html: str, base: Path) -> tuple[str, int]:
    """Replace every local <img src> with a base64 data URI."""
    n = 0

    def sub(m: re.Match) -> str:
        nonlocal n
        src = m.group(1)
        if src.startswith(("http:", "https:", "data:")):
            return m.group(0)
        path = (base / src).resolve()
        if not path.exists():
            print(f"  missing image: {src}", file=sys.stderr)
            return m.group(0)
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        n += 1
        return m.group(0).replace(src, f"data:{mime};base64,{b64}")

    return re.sub(r'<img[^>]+src="([^"]+)"', sub, html), n


def build(md_path: Path) -> Path:
    text = md_path.read_text(encoding="utf-8")
    title = next((ln[2:].strip() for ln in text.split("\n")
                  if ln.startswith("# ")), md_path.stem)

    body = markdown.markdown(
        text, extensions=["tables", "fenced_code", "sane_lists", "attr_list"])
    # Tables need their own scroll container or the page scrolls sideways on a
    # phone, which is the single most common way a long article breaks there.
    body = body.replace("<table>", '<div class="tablewrap"><table>')
    body = body.replace("</table>", "</table></div>")
    body, n_img = inline_images(body, md_path.parent)

    out = md_path.with_suffix(".html")
    out.write_text(
        f"<title>{title}</title>\n<style>{CSS}</style>\n"
        f'<div class="wrap">\n{body}\n</div>\n', encoding="utf-8")
    size = out.stat().st_size / 1024
    print(f"  {out.name}: {size:,.0f} KB, {n_img} images inlined")
    return out


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    print("building self-contained HTML")
    for a in args:
        build(Path(a).resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
