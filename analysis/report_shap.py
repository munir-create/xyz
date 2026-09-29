"""
SHAP report: report/shap/index.html (published as an Artifact), report/shap/SHAP_Figures.pdf and
report/shap/SHAP_NOTES.md, built from results/shap/shap_results.json and the figures.

Run:  python analysis/report_shap.py
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess

import numpy as np
import pandas as pd

import design as dz
from build_report import CSS, SUMMARY_CSS, table

RES = os.path.join(dz.ROOT, "results", "shap")
REP = os.path.join(dz.ROOT, "report", "shap")
os.makedirs(os.path.join(REP, "figures"), exist_ok=True)
KEYS = ("7d", "28d", "paired")
TITLE = {"7d": "7-day model", "28d": "28-day model", "paired": "Paired 7/28-day model"}


def f2(x): return f"{x:.2f}"
def f3(x): return f"{x:.3f}"


def load():
    with open(os.path.join(RES, "shap_results.json")) as f:
        return json.load(f)


def imp_table(R):
    rows = []
    for k in KEYS:
        feats = R[k]["features"]
        for j, fn in enumerate(feats):
            m = R[k]["ln"]["mean_abs"][j]
            rows.append({"model": TITLE[k], "feature": fn,
                         "ln": "0 (not in model)" if m < 1e-12 else f"{m:.3f} [{R[k]['ln']['mean_abs_ci'][0][j]:.3f}–{R[k]['ln']['mean_abs_ci'][1][j]:.3f}]",
                         "fac": "–" if m < 1e-12 else f"×{np.exp(m):.2f}",
                         "MPa": "0" if m < 1e-12 else f"{R[k]['MPa']['mean_abs'][j]:.2f} [{R[k]['MPa']['mean_abs_ci'][0][j]:.2f}–{R[k]['MPa']['mean_abs_ci'][1][j]:.2f}]",
                         "first": f"{100 * R[k]['ln']['rank_first_share'][j]:.0f} %"})
    return pd.DataFrame(rows)


def inter_table(R):
    rows = []
    for k in KEYS:
        feats = R[k]["features"]
        Mx = np.array(R[k]["ln"]["mean_abs_inter"])
        for a in range(len(feats)):
            for b in range(a + 1, len(feats)):
                if Mx[a, b] > 1e-10:
                    rows.append({"model": TITLE[k], "pair": f"{feats[a]} × {feats[b]}", "half": f"{Mx[a, b]:.3f}", "full": f"{2 * Mx[a, b]:.3f}",
                                 "fac": f"×{np.exp(2 * Mx[a, b]):.2f}"})
    return pd.DataFrame(rows)


def absent_inputs(R):
    """Inputs with zero importance in each separate model, e.g. 'A/B in the 7-day model'."""
    out = []
    for k in ("7d", "28d"):
        z = [fn for fn, m in zip(R[k]["features"], R[k]["ln"]["mean_abs"]) if m < 1e-12]
        if z:
            out.append(f"{' and '.join(z)} in the {TITLE[k]}")
    return "; ".join(out)


def captions(R):
    add = [TITLE[k] for k in ("7d", "28d") if R[k].get("additive")]
    add_txt = (f"; the {' and the '.join(add)} {'is' if len(add) == 1 else 'are'} additive, so its points fall on single curves" if add else "")
    absent = absent_inputs(R)
    return [
        ("Z01_mean_abs_SHAP_importance", "Mean |SHAP| feature importance",
         "Average absolute SHAP value of each input over the 30 mixtures (60 mixture × age rows for the paired model), on the ln-strength scale. "
         "Whiskers: 95 % interval over 1,000 case-bootstrap refits stratified by carbonation level. The intervals are conditional on the selected "
         "terms, so an input the model does not contain has zero importance by construction" + (f" ({absent})." if absent else ".")),
        ("Z02_SHAP_summary_beeswarm", "SHAP summary (beeswarm)",
         "Each dot is one mixture: its horizontal position is the SHAP value of that input, the change in predicted ln strength (top axis: the "
         "equivalent multiplicative factor on strength) relative to the average prediction. Colour gives the input's own value (blue low, red high); "
         "carbonation is categorical and is shown by level. Inputs are ordered by mean |SHAP|."),
        ("Z03_SHAP_dependence_7d_vs_28d", "SHAP dependence plots, 7-day and 28-day models on one scale",
         "SHAP value of each input against its value, for every mixture. Vertical spread at a given input value comes from interactions with the "
         f"other inputs{add_txt}. Carbonation (right) is plotted by level, coloured by SS, with the level mean as a bar."),
        ("Z04_SHAP_interaction_matrices", "SHAP interaction matrices",
         "Mean absolute SHAP interaction values. The diagonal is the main effect of each input; each off-diagonal cell holds one half of the pair's "
         "interaction (the full pair effect is twice the cell). A zero cell means the model has no term linking the two inputs."),
        ("Z05_SHAP_interaction_dependence", "SHAP interaction dependence plots",
         "Full interaction effect (both halves) for the key pairs. a: carbonation raises strength at low SS and lowers it at high SS in the 7-day "
         "model; b: RCF helps more when carbonated; c: RCF helps more at low SS; d: in the paired model the 28-day advantage is largest at low SS."),
        ("Z06_SHAP_waterfalls_selected_mixes", "Waterfall decompositions (MPa)",
         "Exact Shapley decomposition of each model's median prediction in MPa for four mixtures: the highest-strength mix (25), a weak NaOH-only mix "
         "(29), a carbonated high-RCF NaOH-only mix (30) and a carbonated silicate-rich mix (27). E[f] is the mean prediction over the 30 mixtures."),
        ("Z07_SHAP_paired_model_age", "Paired model: curing age",
         "a: SHAP value of curing age against SS, showing that the 7 → 28-day contribution is largest in NaOH-only mixes; b: SHAP value of SS "
         "coloured by age, showing the flatter SS response at 28 days; c: SHAP value of RCF by carbonation level."),
        ("Z08a_mean_abs_SHAP_importance_MPa", "Supplementary: importance on the MPa scale", "As Z01, with SHAP values of the median prediction in MPa."),
        ("Z08b_SHAP_summary_beeswarm_MPa", "Supplementary: summary on the MPa scale",
         "As Z02, with SHAP values in MPa. On this scale the log-linear models are no longer additive"
         + (f", so even the {' and the '.join(add)} show{'s' if len(add) == 1 else ''} small interactions." if add else ".")),
    ]


def ranked(R, k):
    """'SS (0.45), RCF (0.21), ...' for the inputs a model contains, by mean |SHAP|."""
    pairs = sorted(((fn, m) for fn, m in zip(R[k]["features"], R[k]["ln"]["mean_abs"]) if m > 1e-12), key=lambda x: -x[1])
    return pairs


def main():
    R = load()
    for fn in glob.glob(os.path.join(RES, "figures", "*.png")):
        shutil.copy(fn, os.path.join(REP, "figures", os.path.basename(fn)))
    it = imp_table(R)
    ix = inter_table(R)
    it_tab = table(it, ["model", "feature", "ln", "fac", "MPa", "first"],
                   ["Model", "Input", "Mean |SHAP|, ln [95 % boot]", "Typical factor", "Mean |SHAP|, MPa [95 % boot]", "Ranked first in refits"],
                   num=["ln", "fac", "MPa", "first"], cls="small")
    ix_tab = table(ix, ["model", "pair", "half", "full", "fac"], ["Model", "Pair", "Mean |interaction|, per half", "Full pair effect", "Typical factor"],
                   num=["half", "full", "fac"], cls="small")
    cc = {k: R[k]["crosscheck"] for k in KEYS}
    if all(c.get("available") for c in cc.values()):
        ccmax = max(max(c["max_abs_diff_values"], c["max_abs_diff_interactions"]) for c in cc.values())
        xcheck = (f"The values agree with <code>shap.ExactExplainer</code> ({cc['7d']['shap_version']}) to within {ccmax:.1e} "
                  f"for values and interaction values.")
    else:
        xcheck = "The optional cross-check against the <code>shap</code> package was not run (package not installed)."
    add = [k for k in ("7d", "28d") if R[k].get("additive")]
    analytic = "".join(f" For the additive {TITLE[k]} they also equal the closed-form contributions (difference "
                       f"{R[k]['analytic_check_max_diff']:.1e})." for k in add)
    imp = {k: dict(zip(R[k]["features"], R[k]["ln"]["mean_abs"])) for k in KEYS}
    impM = {k: dict(zip(R[k]["features"], R[k]["MPa"]["mean_abs"])) for k in KEYS}
    first = {k: dict(zip(R[k]["features"], R[k]["ln"]["rank_first_share"])) for k in KEYS}
    CAPTIONS = captions(R)
    with open(os.path.join(dz.ROOT, "results", "separate", "separate_results.json")) as f:
        R_sep = json.load(f)
    pred = {a: R_sep[a]["fit"]["predR2"] for a in ("7", "28")}
    nc28 = pd.DataFrame(R_sep["28"]["nested_cv"])
    ncv28 = f"{nc28[(nc28.cv == 'leave-one-mixture-out') & (nc28.procedure == 'AICc (primary)')].predR2.iloc[0]:.2f}".replace("-", "−")

    def pairs_txt(k):
        """Non-zero interaction pairs of a model, largest first, as 'SS × Carbonation (0.123)' (full pair effect)."""
        Mx = np.array(R[k]["ln"]["mean_abs_inter"]); f = R[k]["features"]
        pr = sorted(((f"{f[a]} × {f[b]}", 2 * Mx[a, b]) for a in range(len(f)) for b in range(a + 1, len(f)) if Mx[a, b] > 1e-10),
                    key=lambda x: -x[1])
        return pr

    def model_point(k):
        rk = ranked(R, k)
        top, rest = rk[0], rk[1:]
        lead = "contributes most" if first[k][top[0]] >= 0.5 else "is marginally first, but no input clearly dominates"
        s = (f"{top[0]} {lead} (mean |SHAP| {top[1]:.2f} on the ln scale, about {impM[k][top[0]]:.1f} MPa; ranked first in "
             f"{100 * first[k][top[0]]:.0f} % of refits)")
        if rest:
            lst = [f"{fn} ({m:.2f})" for fn, m in rest]
            s += ", followed by " + (", ".join(lst[:-1]) + " and " + lst[-1] if len(lst) > 1 else lst[0])
        s += "."
        pr = pairs_txt(k)
        if pr:
            s += (" Largest interaction: " if len(pr) == 1 else " Largest interactions: ") + "; ".join(f"{p} ({v:.3f} for the pair)" for p, v in pr[:3]) + "."
        else:
            s += " The model has no interaction terms, so every interaction value is zero."
        z = [fn for fn, m in zip(R[k]["features"], R[k]["ln"]["mean_abs"]) if m < 1e-12]
        if z:
            s += f" {' and '.join(z)} {'is' if len(z) == 1 else 'are'} not in the model."
        return s

    Ip = np.array(R["paired"]["ln"]["mean_abs_inter"]); fp_ = R["paired"]["features"]
    age_pairs = [(fp_[b], 2 * Ip[fp_.index("Curing age"), b]) for b in range(len(fp_))
                 if fp_[b] != "Curing age" and Ip[fp_.index("Curing age"), b] > 1e-10]
    rkp = ranked(R, "paired")
    figs_html = "\n".join(
        f'<figure class="plate" id="{fn}"><img src="figures/{fn}.png" alt="{t}" loading="lazy"><figcaption><b>{fn.split("_")[0]}. {t}.</b> {c}</figcaption></figure>'
        for fn, t, c in CAPTIONS)
    key_points = f"""
<ul class="kp">
  <li><b>7-day model.</b> {model_point('7d')}</li>
  <li><b>28-day model.</b> {model_point('28d')}</li>
  <li><b>Paired model.</b> Inputs by mean |SHAP|: {', '.join(f'{fn} ({m:.2f})' for fn, m in rkp)}. Curing age interacts only with {' and '.join(f'{fn} ({v:.3f} for the pair)' for fn, v in age_pairs)}: the 28-day advantage is largest in NaOH-only mixes.</li>
</ul>"""
    method = f"""
<p>SHAP values here are exact interventional Shapley values of each fitted model, computed over all coalitions of its inputs (16 for four inputs, 32 with curing age) with the 30 real mixtures as the background set (60 mixture × age rows for the paired model). No sampling approximation is involved. The inputs are the actual model inputs: RCF (%), SS (%), A/B and carbonation. Carbonation is one categorical input with four levels (NC, 0.5 h, 1 h, 5 h). When it is absent from a coalition its level is taken from a background mixture, so its level indicators and its carbonated × RCF and carbonated × SS terms always move together. It is never treated as a number. The paired model adds curing age (7 or 28 d) as a fifth input. SHAP interaction values follow Lundberg et al. (2020).</p>
<p>Values are computed on the ln-strength scale, where the models were fitted. On that scale the contributions add up exactly and the interaction values correspond one-to-one to the models' interaction terms. Axes also show the multiplicative factor exp(SHAP). The waterfalls and the supplementary figures use the MPa scale (median prediction). Checks: every decomposition satisfies Shapley efficiency to machine precision. {xcheck}{analytic} Importance intervals come from 1,000 case-bootstrap refits stratified by carbonation level ({R['paired']['n_boot']} usable for the paired model).</p>
<p class="note">SHAP describes how each fitted model uses its inputs, not causal effects. An input a model does not contain gets exactly zero. The values depend on the background set (here the design itself), and the 28-day model predicts new mixtures much less precisely than the 7-day model (predicted R² of the fixed terms {pred['28']:.2f} vs {pred['7']:.2f}; {ncv28} for the 28-day model when its selection is repeated in each fold), so its SHAP pattern is less certain.</p>"""

    body = f"""
<header class="top">
  <p class="eyebrow">Compressive strength · 7-day, 28-day and paired models · exact Shapley values</p>
  <h1>SHAP Explanations</h1>
  <p class="lede">How each finalised model uses its inputs: which inputs matter most, how the contribution of each input changes with its value, which inputs interact, and how individual predictions are built up. Carbonation is treated as one categorical input with four levels throughout.</p>
  <p class="meta">Analysis: <code>analysis/run_shap.py</code> · figures: <code>analysis/make_figures_shap.py</code> · built {R['meta']['date']}</p>
  <nav class="toc"><a href="#points">Key points</a><a href="#figures">Figures</a><a href="#tables">Tables</a><a href="#method">Method</a></nav>
</header>
<main>
<section id="points"><h2>Key points</h2>{key_points}</section>
<section id="figures"><h2>Figures</h2>{figs_html}</section>
<section id="tables"><h2>Tables</h2><h3>Mean |SHAP| importance</h3>{it_tab}<h3>Non-zero interactions</h3>{ix_tab}
<p class="note">Per-mixture SHAP values: <code>results/shap/tables/shap_values_*.csv</code>.</p></section>
<section id="method"><h2>Method</h2>{method}</section>
</main>"""
    extra = """
.kp { max-width: 80ch; padding-left: 20px; } .kp li { margin: 8px 0; }
figure.plate figcaption { color: #3d4249; font-size: 13px; line-height: 1.5; padding: 8px 4px 2px; }
"""
    page = f"""<title>SHAP Explanations</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@75..100,500..800&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{CSS}{extra}</style>
<div class="wrap">{body}</div>
"""
    with open(os.path.join(REP, "index.html"), "w") as f:
        f.write(page)
    print("wrote report/shap/index.html")

    # PDF: one figure per page with caption
    fig = "../../results/shap/figures"
    pages = "".join(f'<div class="pg"><h2>{fn.split("_")[0]}. {t}</h2><img src="{fig}/{fn}.png"><p class="cap">{c}</p></div>' for fn, t, c in CAPTIONS)
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>SHAP Explanations</title><style>{SUMMARY_CSS}
@page {{ size: A4 landscape; margin: 12mm; }} .pg {{ break-after: page; }} .pg img {{ width: 100%; max-height: 150mm; object-fit: contain; }}
.cap {{ font-size: 9pt; color: #3d4249; }} .kp li {{ margin: 4px 0; }}</style></head><body>
<div class="pg"><h1>SHAP explanations of the strength models</h1><p class="sub">7-day, 28-day and paired 7/28-day models · exact Shapley values · carbonation as a 4-level categorical input · {R['meta']['date']}</p>
{key_points}{method}{it_tab}{ix_tab}</div>{pages}</body></html>"""
    p = os.path.join(REP, "summary.html")
    with open(p, "w") as f:
        f.write(doc)
    g = glob.glob("/opt/pw-browsers/chromium*/chrome-linux/chrome")
    if g:
        out = os.path.join(REP, "SHAP_Figures.pdf")
        subprocess.run([g[0], "--headless", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={out}", "file://" + p],
                       check=False, capture_output=True, timeout=180)
        print("wrote report/shap/SHAP_Figures.pdf" if os.path.exists(out) else "PDF not written")

    import html as _h
    import re
    plain = lambda s: _h.unescape(re.sub(r"<[^>]+>", "", s))
    md = ["# SHAP explanations of the strength models", "", "*Generated by `analysis/report_shap.py`.*", "", "## Key points", ""]
    md += ["* " + plain(li) for li in re.findall(r"<li>(.*?)</li>", key_points, flags=re.S)]
    md += ["", "## Method", ""] + [plain(p_) + "\n" for p_ in re.findall(r"<p[^>]*>(.*?)</p>", method, flags=re.S)]
    md += ["## Mean |SHAP| importance", "", "| Model | Input | Mean abs SHAP, ln [95 % boot] | Typical factor | Mean abs SHAP, MPa | Ranked first |", "|---|---|---|---|---|---|"]
    md += [f"| {r.model} | {r.feature} | {r.ln} | {r.fac} | {r.MPa} | {r.first} |" for r in it.itertuples()]
    md += ["", "## Non-zero interactions (ln scale)", "", "| Model | Pair | Per half | Full pair | Factor |", "|---|---|---|---|---|"]
    md += [f"| {r.model} | {r.pair} | {r.half} | {r.full} | {r.fac} |" for r in ix.itertuples()]
    md += ["", "## Figures", ""] + [f"* `results/shap/figures/{fn}.png|pdf` — **{t}.** {c}" for fn, t, c in CAPTIONS]
    with open(os.path.join(REP, "SHAP_NOTES.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("wrote report/shap/SHAP_NOTES.md")


if __name__ == "__main__":
    main()
