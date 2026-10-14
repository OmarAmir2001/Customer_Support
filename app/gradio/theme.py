"""Look and feel, in one place.

Bilingual is the constraint that drives everything here. Arabic is right-to-left, and
an answer about the CS handbook will be Arabic while the surrounding UI chrome is
English. So:

* answer and transcript panels get `direction: rtl` support via a CSS class that is
  applied per-element rather than to the page — mixing an RTL page with LTR controls
  is worse than mixing scripts inside one panel;
* the font stack puts a Naskh-capable face first, because the default sans in most
  browsers renders Arabic diacritics poorly at small sizes.
"""

from __future__ import annotations

import gradio as gr

THEME = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[
        gr.themes.GoogleFont("Inter"),
        gr.themes.GoogleFont("Noto Naskh Arabic"),
        "system-ui",
        "sans-serif",
    ],
)

CSS = """
/* Answer and transcript panels: let the browser pick direction per block, so an
   Arabic answer reads right-to-left inside an otherwise left-to-right page. */
.answer-panel, .answer-panel * { unicode-bidi: plaintext; }
.answer-panel { font-size: 1.02rem; line-height: 1.85; }

/* Status banners. Colour carries meaning here, so each also carries a word — a
   colour-only signal is unreadable to a chunk of your users. */
.banner { padding: 0.75rem 1rem; border-radius: 8px; border-left: 4px solid; margin: 0.25rem 0; }
.banner-ok      { border-color: #0d9488; background: rgba(13, 148, 136, 0.08); }
.banner-warn    { border-color: #d97706; background: rgba(217, 119, 6, 0.08); }
.banner-danger  { border-color: #dc2626; background: rgba(220, 38, 38, 0.08); }
.banner-info    { border-color: #475569; background: rgba(71, 85, 105, 0.08); }

/* Citations: visibly data, not prose. */
.citation { display:inline-block; padding:0.2rem 0.6rem; margin:0.15rem 0.2rem 0.15rem 0;
            border-radius:999px; background:rgba(13,148,136,0.12);
            border:1px solid rgba(13,148,136,0.35); font-size:0.85rem; }

.gate-table { font-size: 0.9rem; }
footer { display: none !important; }
"""


def banner(kind: str, title: str, body: str = "") -> str:
    """One status banner as HTML. `kind` is ok | warn | danger | info."""
    body_html = f"<div style='margin-top:0.35rem;opacity:0.9'>{body}</div>" if body else ""
    return f"<div class='banner banner-{kind}'><strong>{title}</strong>{body_html}</div>"
