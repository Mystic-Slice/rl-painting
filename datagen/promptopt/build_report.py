#!/usr/bin/env python
"""
datagen/promptopt/build_report.py

Turn one or more compare.py manifests into a single simple, self-contained HTML
page: seed-vs-optimized paintings side by side for every combo, with judge scores.
Multiple manifests (one per generator model) become toggle-able tabs.

    python datagen/promptopt/build_report.py \
        --manifest "GPT-5.6-sol=datagen/promptopt/compare/<stamp>/manifest.json" \
        --out datagen/promptopt/compare/report.html
"""
import argparse
import base64
import html
import io
import json
from pathlib import Path

from PIL import Image

CRITERIA = ["subject_recognisability", "painterly_looseness", "composition",
            "colour_harmony", "overall_aesthetic"]


def img_data_uri(png_path, max_px=380, quality=80):
    """Downscale a PNG to a compact JPEG data URI (keeps the page self-contained/small)."""
    try:
        im = Image.open(png_path).convert("RGB")
    except Exception:
        return None
    im.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def panel(variant_result, label):
    """One image panel (seed or optimized)."""
    if variant_result.get("ok") and variant_result.get("_uri"):
        score = variant_result.get("score")
        sc = ("%.3f" % score) if isinstance(score, (int, float)) else "n/a"
        return ('<figure class="panel"><img loading="lazy" src="%s" alt="%s render">'
                '<figcaption><span class="vlabel">%s</span>'
                '<span class="score">%s</span></figcaption></figure>'
                % (variant_result["_uri"], label, label, sc))
    # failed render / generation
    stage = html.escape(str(variant_result.get("stage", "failed")))
    err = html.escape(str(variant_result.get("error", ""))[:160])
    return ('<figure class="panel panel-fail"><div class="failbox">'
            '<span class="failx">render failed</span>'
            '<span class="failstage">%s</span><span class="failerr">%s</span></div>'
            '<figcaption><span class="vlabel">%s</span>'
            '<span class="score">&mdash;</span></figcaption></figure>'
            % (stage, err, label))


def delta_chip(seed_r, opt_r):
    s, o = seed_r.get("score"), opt_r.get("score")
    if not (isinstance(s, (int, float)) and isinstance(o, (int, float))):
        return '<span class="delta delta-na">no Δ</span>'
    d = o - s
    cls = "delta-up" if d > 0.001 else ("delta-down" if d < -0.001 else "delta-flat")
    arrow = "▲" if d > 0.001 else ("▼" if d < -0.001 else "=")
    return '<span class="delta %s">%s %+.3f</span>' % (cls, arrow, d)


def model_section(label, manifest, section_id, active):
    entries = manifest["entries"]
    # attach data URIs
    for e in entries:
        for v in ("seed", "opt"):
            r = e.get(v) or {}
            png = r.get("png")
            r["_uri"] = img_data_uri(png) if png else None
            e[v] = r

    # aggregates
    def mean_scores(v):
        xs = [e[v]["score"] for e in entries if isinstance(e[v].get("score"), (int, float))]
        return (sum(xs) / len(xs)) if xs else None
    seed_mean, opt_mean = mean_scores("seed"), mean_scores("opt")
    wins = sum(1 for e in entries
               if isinstance(e["seed"].get("score"), (int, float))
               and isinstance(e["opt"].get("score"), (int, float))
               and e["opt"]["score"] > e["seed"]["score"] + 0.001)
    losses = sum(1 for e in entries
                 if isinstance(e["seed"].get("score"), (int, float))
                 and isinstance(e["opt"].get("score"), (int, float))
                 and e["opt"]["score"] < e["seed"]["score"] - 0.001)
    seed_fail = sum(1 for e in entries if not e["seed"].get("ok"))
    opt_fail = sum(1 for e in entries if not e["opt"].get("ok"))

    # per-criterion aggregate (0-10)
    crit_rows = []
    for c in CRITERIA:
        sv = [e["seed"]["scores"][c] for e in entries
              if e["seed"].get("ok") and c in (e["seed"].get("scores") or {})]
        ov = [e["opt"]["scores"][c] for e in entries
              if e["opt"].get("ok") and c in (e["opt"].get("scores") or {})]
        sm = sum(sv) / len(sv) if sv else None
        om = sum(ov) / len(ov) if ov else None
        crit_rows.append((c, sm, om))

    def fmt(x, p="%.3f"):
        return (p % x) if isinstance(x, (int, float)) else "n/a"

    dmean = (opt_mean - seed_mean) if (seed_mean is not None and opt_mean is not None) else None
    pct = (100.0 * dmean / seed_mean) if (dmean is not None and seed_mean) else None

    # summary block
    parts = ['<section class="model" id="%s" data-active="%s">' % (section_id, str(active).lower())]
    parts.append('<div class="summary">')
    parts.append('<div class="bignum"><span class="k">seed mean</span>'
                 '<span class="v">%s</span></div>' % fmt(seed_mean))
    parts.append('<div class="arrow2">&rarr;</div>')
    parts.append('<div class="bignum bignum-opt"><span class="k">optimized mean</span>'
                 '<span class="v">%s</span></div>' % fmt(opt_mean))
    if dmean is not None:
        parts.append('<div class="bignum bignum-delta"><span class="k">change</span>'
                     '<span class="v">%+.3f%s</span></div>'
                     % (dmean, ("  (%+.0f%%)" % pct) if pct is not None else ""))
    parts.append('<div class="counts">improved <b>%d</b> / regressed <b>%d</b> of %d'
                 '%s%s</div>'
                 % (wins, losses, len(entries),
                    ("  · seed render-fails %d" % seed_fail) if seed_fail else "",
                    ("  · opt render-fails %d" % opt_fail) if opt_fail else ""))
    parts.append('</div>')  # summary

    # per-criterion table
    parts.append('<table class="crit"><thead><tr><th>criterion (0–10)</th>'
                 '<th>seed</th><th>optimized</th><th>Δ</th></tr></thead><tbody>')
    for c, sm, om in crit_rows:
        d = (om - sm) if (sm is not None and om is not None) else None
        dcls = "delta-up" if (d is not None and d > 0.02) else \
               ("delta-down" if (d is not None and d < -0.02) else "delta-flat")
        parts.append('<tr><td>%s</td><td class="num">%s</td><td class="num">%s</td>'
                     '<td class="num %s">%s</td></tr>'
                     % (c.replace("_", " "), fmt(sm, "%.2f"), fmt(om, "%.2f"),
                        dcls, ("%+.2f" % d) if d is not None else "n/a"))
    parts.append('</tbody></table>')

    # cards grouped by animal
    parts.append('<div class="grid">')
    last_animal = None
    for e in entries:
        if e["animal"] != last_animal:
            parts.append('<h2 class="animal">%s</h2>' % html.escape(e["animal"]))
            last_animal = e["animal"]
        parts.append('<article class="card">')
        parts.append('<div class="prompt">%s</div>' % html.escape(e["prompt"]))
        parts.append('<div class="cid">%s</div>' % html.escape(e["id"]))
        parts.append('<div class="diptych">%s%s</div>'
                     % (panel(e["seed"], "seed"), panel(e["opt"], "optimized")))
        parts.append('<div class="cardfoot">%s</div>' % delta_chip(e["seed"], e["opt"]))
        parts.append('</article>')
    parts.append('</div>')  # grid
    parts.append('</section>')
    return "".join(parts), (label, seed_mean, opt_mean, dmean, pct)


CSS = """
:root{--bg:#eceae7;--surface:#f7f6f4;--ink:#20222a;--muted:#6c6f78;
--line:#d7d5d0;--accent:#3f5c78;--up:#2f7d64;--down:#b0553f;--fail:#9a6b60;}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){
--bg:#17181c;--surface:#232429;--ink:#e8e7e3;--muted:#9a9ca4;
--line:#33343a;--accent:#82a6c9;--up:#5cbf98;--down:#d68a6f;--fail:#c39a90;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
.wrap{max-width:1200px;margin:0 auto;padding:28px 20px 80px;}
h1{font-size:1.5rem;margin:0 0 4px;font-weight:650;}
.sub{color:var(--muted);margin:0 0 20px;font-size:.92rem;}
.tabs{display:flex;gap:8px;margin:0 0 22px;flex-wrap:wrap;}
.tab{border:1px solid var(--line);background:var(--surface);color:var(--ink);
padding:7px 16px;border-radius:999px;cursor:pointer;font-size:.9rem;font-weight:550;}
.tab[aria-selected=true]{background:var(--accent);color:#fff;border-color:var(--accent);}
.model[data-active=false]{display:none;}
.summary{display:flex;align-items:center;gap:22px;flex-wrap:wrap;
background:var(--surface);border:1px solid var(--line);border-radius:12px;
padding:18px 22px;margin:0 0 18px;}
.bignum{display:flex;flex-direction:column;}
.bignum .k{font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);}
.bignum .v{font-size:1.7rem;font-weight:680;font-variant-numeric:tabular-nums;}
.bignum-opt .v{color:var(--accent);}
.bignum-delta .v{color:var(--up);}
.arrow2{font-size:1.5rem;color:var(--muted);}
.counts{color:var(--muted);font-size:.9rem;margin-left:auto;}
.counts b{color:var(--ink);}
table.crit{width:100%;border-collapse:collapse;margin:0 0 28px;font-size:.9rem;
background:var(--surface);border:1px solid var(--line);border-radius:12px;overflow:hidden;}
.crit th,.crit td{padding:9px 14px;text-align:left;border-bottom:1px solid var(--line);}
.crit thead th{font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);font-weight:600;}
.crit tr:last-child td{border-bottom:none;}
.crit .num{text-align:right;font-variant-numeric:tabular-nums;}
.animal{grid-column:1/-1;margin:22px 0 2px;font-size:1.05rem;font-weight:640;
text-transform:capitalize;padding-bottom:6px;border-bottom:2px solid var(--line);}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:18px;}
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;
padding:12px 12px 10px;display:flex;flex-direction:column;}
.prompt{font-size:.92rem;font-weight:550;line-height:1.35;}
.cid{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
font-size:.72rem;color:var(--muted);margin:2px 0 9px;}
.diptych{display:grid;grid-template-columns:1fr 1fr;gap:8px;}
.panel{margin:0;}
.panel img{width:100%;aspect-ratio:1/1;object-fit:cover;border-radius:7px;
background:#fff;display:block;border:1px solid var(--line);}
.panel figcaption{display:flex;justify-content:space-between;align-items:baseline;
margin-top:5px;font-size:.8rem;}
.vlabel{color:var(--muted);text-transform:uppercase;letter-spacing:.04em;font-size:.68rem;}
.score{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
font-weight:600;font-variant-numeric:tabular-nums;}
.panel-fail .failbox{width:100%;aspect-ratio:1/1;border-radius:7px;border:1px dashed var(--fail);
background:color-mix(in srgb,var(--fail) 8%,transparent);display:flex;flex-direction:column;
align-items:center;justify-content:center;gap:3px;padding:8px;text-align:center;}
.failx{color:var(--fail);font-weight:600;font-size:.8rem;}
.failstage{color:var(--muted);font-size:.72rem;}
.failerr{color:var(--muted);font-size:.62rem;line-height:1.25;overflow:hidden;max-height:3.6em;}
.cardfoot{margin-top:9px;display:flex;justify-content:flex-end;}
.delta{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
font-size:.82rem;font-weight:650;padding:3px 9px;border-radius:6px;font-variant-numeric:tabular-nums;}
.delta-up{color:#fff;background:var(--up);}
.delta-down{color:#fff;background:var(--down);}
.delta-flat,.delta-na{color:var(--muted);background:color-mix(in srgb,var(--muted) 14%,transparent);}
"""

JS = """
document.querySelectorAll('.tab').forEach(function(t){
  t.addEventListener('click',function(){
    document.querySelectorAll('.tab').forEach(function(x){x.setAttribute('aria-selected','false');});
    document.querySelectorAll('.model').forEach(function(m){m.setAttribute('data-active','false');});
    t.setAttribute('aria-selected','true');
    var m=document.getElementById(t.dataset.target); if(m) m.setAttribute('data-active','true');
  });
});
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="append", required=True,
                    help='LABEL=path/to/manifest.json  (repeatable, one per model)')
    ap.add_argument("--title", default="Prompt Comparison")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "compare" / "report.html"))
    args = ap.parse_args()

    models = []
    for spec in args.manifest:
        label, path = spec.split("=", 1)
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        models.append((label, manifest))

    sections, tab_meta = [], []
    for i, (label, manifest) in enumerate(models):
        sid = "m%d" % i
        html_sec, meta = model_section(label, manifest, sid, active=(i == 0))
        sections.append((sid, label, html_sec))
        tab_meta.append(meta)

    tabs = "".join(
        '<button class="tab" data-target="%s" aria-selected="%s">%s</button>'
        % (sid, "true" if i == 0 else "false", html.escape(label))
        for i, (sid, label, _) in enumerate(sections))

    gen = models[0][1].get("generator", "?")
    judge = models[0][1].get("judge", "?")
    sub = ("Seed vs. GEPA-optimized system prompt · judged by %s (composite of 5 criteria, 0–1) · "
           "%d combos" % (html.escape(judge), models[0][1].get("n", 0)))

    body = ('<div class="wrap"><h1>%s</h1><p class="sub">%s</p>%s%s</div>'
            % (html.escape(args.title), sub,
               ('<div class="tabs">%s</div>' % tabs) if len(sections) > 1 else "",
               "".join(s for _, _, s in sections)))

    # Publish-ready content: <title>+<style>+markup+<script>, WITHOUT
    # <!doctype>/<html>/<head>/<body> (the Artifact publisher supplies those; browsers
    # also render this file fine standalone).
    doc = ("<title>%s</title><style>%s</style>%s<script>%s</script>"
           % (html.escape(args.title), CSS, body, JS))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    size_mb = len(doc.encode("utf-8")) / 1e6
    print("wrote %s  (%.2f MB)" % (out, size_mb))
    for (label, sm, om, dm, pct) in tab_meta:
        print("  %-16s seed %.3f -> opt %.3f  (%+.3f%s)"
              % (label, sm or 0, om or 0, dm or 0,
                 (" %+.0f%%" % pct) if pct is not None else ""))


if __name__ == "__main__":
    main()
