# Model handoff — revised-data (v3) bivariate strength model

Finalized in commit `f4d6dfb` (branch `claude/youthful-planck-4kcc4y`). Every number below is copied from `results/v3/v3_results.json` and `results/v3/tables/`; nothing was refitted or re-selected for this note. Do not re-run `analysis/run_v3.py` just to predict: it repeats selection and resampling and overwrites `results/v3/`.

## 1  Dataset

* **File:** `data/strength_7d_28d_v3_noiseless.csv` — 30 rows, one per mixture; programmatic transcription of `data/source/30_mixes_strength_results_noiseless.pdf`.
* **Responses:** `f7_mean`, `f28_mean` (MPa, mean of the specimens): 3.08–22.20 MPa at 7 d, 7.64–36.04 MPa at 28 d. `f7_sd`, `f28_sd`, `f7_n`, `f28_n` are not used by the model.
* **Replicate batches** (`replicate_group`, same nominal composition): R1 = mixes 1, 12, 19 · R2 = 10, 14 · R3 = 6, 17, 28 · R4 = 8, 23 → 24 distinct design points. Single-specimen means: mix 11 (28 d), mix 29 (7 and 28 d), mix 30 (28 d).

## 2  Predictors

| Predictor (source label) | Column | Observed range | Coded form in the model |
|---|---|---|---|
| RCF, % | `RCF_pct` | 10–50 | A = (RCF − 30) / 20 |
| A/B | `AB` | 0.419–0.480 | C = (A/B − 0.45) / 0.03 |
| SS, % ¹ | `SS_pct` | 0–75 | D = (SS − 37.5) / 37.5 |
| Carbonation of the RCF | `carbonation` (`carb_h`) | NC (8 mixes), 0.5 h (5), 1 h (9), 5 h (8) | indicators [0.5 h], [1 h], [5 h] (all 0 for NC); K = 1 for any carbonated level |

¹ Read as the sodium-silicate share of an SS + NaOH-solution activator (assumption carried over from the earlier report in `reference/`; the source table does not define it). Carbonation × mixture-variable slopes use K because the 0.5 h level has only two distinct compositions (R1, R2). A/B 0.419 is mix 7 (coded C = −1.03).

## 3  Retained terms

| Age | Main effects | Quadratic | Interactions |
|---|---|---|---|
| 7 d | RCF (A), SS (D), carbonation ([0.5 h], [1 h], [5 h]) | SS² (D²) | RCF·SS (A·D), K·RCF (K·A), K·SS (K·D) |
| 28 d | RCF (A), A/B (C), SS (D), carbonation ([0.5 h], [1 h], [5 h]) | RCF² (A²), (A/B)² (C²) | none |

Not retained: A/B, RCF², (A/B)² at 7 d; SS² and every interaction at 28 d; RCF·A/B, A/B·SS, K·A/B at both ages. Term codes (`analysis/design.py`, `LEVEL_TERMS`): `A C D Carb AC AD CD A2 C2 D2 KA KC KD`. Stored final lists (`final_terms`): 7 d `["A", "D", "Carb", "AD", "D2", "KA", "KD"]`, 28 d `["A", "C", "D", "Carb", "A2", "C2"]`.

## 4  Equations (exact coefficients)

Response: ln(strength, MPa). Predicted strength = exp(·) = median of a new batch (the lognormal mean is ×1.009 higher at 7 d, ×1.022 at 28 d).

**Coded form, as estimated** (`results/v3/tables/V06_final_coefficients_coded.csv`; SE and p from the bivariate REML fit):

| Term | 7 d coefficient | SE | p | 28 d coefficient | SE | p |
|---|---|---|---|---|---|---|
| Intercept | 2.7048112488721143 | 0.0639 | < 0.001 | 3.0227077565505045 | 0.1052 | < 0.001 |
| A (RCF) | 0.0640282665394202 | 0.0548 | 0.26 | 0.14769427074370423 | 0.0530 | 0.011 |
| C (A/B) | — |  |  | −0.13993696381022 | 0.0501 | 0.011 |
| D (SS) | 0.7273707630409376 | 0.0544 | < 0.001 | 0.23337401032856803 | 0.0519 | < 0.001 |
| [0.5 h] | 0.07168877176779731 | 0.0873 | 0.42 | 0.2954402087660526 | 0.1385 | 0.044 |
| [1 h] | −0.08840319985916328 | 0.0680 | 0.21 | −0.20097606436734783 | 0.1061 | 0.072 |
| [5 h] | 0.0653496434079538 | 0.0696 | 0.36 | −0.21055948515821665 | 0.1065 | 0.061 |
| A·D | −0.09934102857540501 | 0.0362 | 0.013 | — |  |  |
| A² | — |  |  | −0.2341036144512385 | 0.0854 | 0.013 |
| C² | — |  |  | 0.2515094042043904 | 0.1001 | 0.021 |
| D² | −0.6411526356603168 | 0.0594 | < 0.001 | — |  |  |
| K·A | 0.19868634197002513 | 0.0655 | 0.007 | — |  |  |
| K·D | −0.2846847504086578 | 0.0663 | < 0.001 | — |  |  |

**Actual units** (`results/v3/tables/V08_equations_actual_units.csv`) as a dependency-free function. It reproduces `V10_predictions_all_mixes.csv` to < 2 × 10⁻¹¹ relative error. Keep every digit of the A/B coefficients; they nearly cancel.

```python
import math

def strength(RCF, SS, AB, carb, age):
    """Median predicted strength (MPa). RCF, SS in %; AB as a ratio; carb in {"NC", "0.5 h", "1 h", "5 h"}; age 7 or 28."""
    c05, c1, c5 = carb == "0.5 h", carb == "1 h", carb == "5 h"
    K = carb != "NC"
    if age == 7:
        ln = (1.0912339074986843 + 0.008168464755740509 * RCF + 0.057565002059324896 * SS
              - 0.00013245470476707947 * RCF * SS - 0.0004559307631362696 * SS ** 2
              + 0.058344009221407755 * c05 - 0.10174796240555262 * c1 + 0.052004880861564184 * c5
              + 0.009934317098501356 * K * RCF - 0.00759159334423091 * K * SS)
    else:
        ln = (60.72972961075752 + 0.04250025570364845 * RCF + 0.00622330694208123 * SS
              - 256.17396966472506 * AB - 0.0005852590361182138 * RCF ** 2 + 279.4548935604297 * AB ** 2
              + 0.2954402087658765 * c05 - 0.2009760643673695 * c1 - 0.21055948515899411 * c5)
    return math.exp(ln)
```

## 5  How it was fitted

* **Term selection, per age:** OLS on ln strength, exhaustive AICc over all 716 strong-hierarchy sub-models of `A C D Carb AC AD CD A2 C2 D2 KA KC KD`; the AICc-best model at each age is final (`bivariate.AgeSelector`).
* **Joint (paired) estimation:** one bivariate model with a block design — 7-d columns are zero on each mixture's 28-d row and vice versa — and an unstructured 2 × 2 covariance for each mixture's (ln f7, ln f28) pair. REML for the covariance (BFGS, then Nelder–Mead), GLS for the coefficients, Wald t/F tests with Satterthwaite df (`analysis/bivariate.py` → `analysis/lmm.py`, `PairedLMM(..., struct="UN")`).
* **Covariance (ln scale):** σ7 = 0.13694042318315142, σ28 = 0.2096622949131737, ρ = 0.43826552037197897 (var7 0.018752679501580594, var28 0.043958277908258625, cov 0.012583148025265934).
* **Scale:** Box–Cox λ = 0.04 (95 % CI −0.26 to 0.34) for both ages jointly; λ = 1 is outside the CI at each age (7 d −0.09 to 0.53; 28 d −0.88 to 0.45).
* Joint-likelihood AICc −0.878. Python 3.11, NumPy 2.4.6, SciPy 1.17.1, pandas 3.0.6; resampling seed 20261002.

## 6  Performance and robustness

| | 7 d | 28 d |
|---|---|---|
| R² (ln) / adjusted | 0.961 / 0.944 | 0.738 / 0.639 |
| Predicted R² (ln), terms fixed: leave one mixture out / leave one design point out | 0.923 / 0.929 | 0.487 / 0.386 |
| Predicted R² (ln), selection repeated in every fold: same two schemes | 0.919 / 0.928 | 0.315 / −0.230 |
| RMSE fitted / leave-one-mixture-out predicted (MPa) | 1.36 / 1.88 | 3.81 / 5.06 |
| Residual SD (ln) vs replicate SD (ln) | 0.137 vs 0.148 | 0.210 vs 0.097 |
| Lack of fit vs replicates | F(14, 6) = 0.79, p = 0.66 | F(15, 6) = 6.06, p = 0.018 |
| Same terms re-selected with one mixture left out | 30 / 30 | 27 / 30 (without mix 7 or 13: + K·SS; without mix 29: A/B + SS only) |

* **Versus the shared-effects paired model** (one set of mixture effects for both ages plus selected age interactions): ΔAICc = 4.80 in favour of this model; nested leave-one-mixture-out predicted R² 0.315 vs −0.063 at 28 d, 0.919 vs 0.911 at 7 d.
* **Residual-bootstrap selection frequency** (2,000 resamples): 7 d — SS, SS², RCF 100%, carbonation 88%, K·SS 87%, K·RCF 72%, RCF·SS 71%; 28 d — SS 100%, RCF 89%, A/B 86%, carbonation 74%, RCF² 68%, (A/B)² 54%.
* **Fragile at 28 d:** the carbonation effect rests on R1 and R2 (the only 0.5 h compositions) and R3 (5 h). Without R1, 0.5 h/NC p = 0.28; without R2, p = 0.063; without R3, 5 h/NC = ×0.93 (p = 0.57) and the (A/B)² p = 0.14. A/B 0.42 vs 0.45 stays significant in every exclusion (×1.41–1.50, p ≤ 0.013).
* **Effects that differ between 7 and 28 d** (term entered at both ages, Wald test of equality): SS p < 0.001, SS² p < 0.001, carbonation p = 0.005, A/B p = 0.019, RCF² p = 0.029. Not significant: (A/B)² 0.054, K·RCF 0.079, K·SS 0.11, RCF·SS 0.15, RCF 0.21.
* **Diagnostics:** largest |studentized residual| 2.07 (7 d, mix 14) and 2.58 (28 d, mix 25), Bonferroni p ≥ 0.54; largest Cook's D 0.18 (7 d, mix 13), 0.24 (28 d, mix 7); Shapiro–Wilk p 0.84 / 0.38.

## 7  Reproducing predictions

* **Needed:** `data/strength_7d_28d_v3_noiseless.csv`; `analysis/bivariate.py`, `design.py`, `lmm.py`, `model_tools.py`, `selection.py`; `results/v3/v3_results.json` (term lists); numpy, scipy, pandas (`requirements.txt`). Or only the `strength()` function above (no intervals).
* **Stored outputs to check against:** `results/v3/tables/V06_final_coefficients_coded.csv` (coefficients), `V08_equations_actual_units.csv` (actual-unit coefficients), `V10_predictions_all_mixes.csv` (median, CI, PI and leave-one-out prediction for every mixture and age).

```python
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, "analysis")                    # run from the repo root
import design as dz, bivariate as bv

R = json.load(open("results/v3/v3_results.json"))
t7, t28 = R["final_terms"]["7"], R["final_terms"]["28"]
d = dz.load("data/strength_7d_28d_v3_noiseless.csv")
m, _, _ = bv.fit(d, t7, t28)                      # deterministic REML refit = stored model (V06)

# predictions at the observed mixtures (= V10): median, ci_lo/ci_hi (mean), pi_lo/pi_hi (one batch), ln_fit, se, df
pred7 = bv.predict(m, t7, t28, d[["RCF_pct", "SS_pct", "AB", "carbonation"]].assign(age=7))
pred28 = bv.predict(m, t7, t28, d[["RCF_pct", "SS_pct", "AB", "carbonation"]].assign(age=28))

REF = {"RCF_pct": 30.0, "SS_pct": 37.5, "AB": 0.45, "carbonation": "NC"}

def oat(var, values, age, **fixed):
    """Vary one predictor; hold the others at REF (override any of them with fixed=...)."""
    pts = pd.DataFrame([{**REF, **fixed, var: v, "age": age} for v in values])
    return pd.concat([pts, bv.predict(m, t7, t28, pts)], axis=1)

ss_7d_nc = oat("SS_pct", np.arange(0, 75.01, 2.5), age=7)                                   # SS profile, NC
ss_7d_1h = oat("SS_pct", np.arange(0, 75.01, 2.5), age=7, carbonation="1 h")                # same, carbonated
rcf_28d = oat("RCF_pct", np.arange(10, 50.01, 2.5), age=28)
ab_28d = oat("AB", np.arange(0.42, 0.4801, 0.005), age=28)
carb_7d = oat("carbonation", ["NC", "0.5 h", "1 h", "5 h"], age=7, SS_pct=0.0, RCF_pct=50.0)
```

Check values (median MPa): RCF 30, SS 37.5, A/B 0.45 — NC 14.9515 (7 d) / 20.5469 (28 d), 1 h 13.6865 / 16.8059; mix 1 12.4813 / 18.9339.

## 8  One-factor-at-a-time predictions

* Use `oat()` above: vary one predictor across its tested range (RCF 10–50 %, SS 0–75 %, A/B 0.42–0.48, the four carbonation levels) and hold the others at `REF` (RCF 30 %, SS 37.5 %, A/B 0.45, NC — the reference of report Figure 4) or at values passed as `fixed=...`. The dependency-free alternative is `strength()` in a loop (medians only).
* At 7 d the RCF, SS and carbonation profiles depend on the fixed values (RCF·SS, K·RCF, K·SS): always state them, and compare profiles at e.g. NC vs 1 h or SS 0 / 37.5 / 75 %. At 28 d there are no interactions, so changing the fixed values only multiplies a profile by a constant. A/B is flat at 7 d.
* Raw mixtures differ in several factors at once, so do not read raw points against a one-factor curve as if they were its trend. Compare each observed mean with its own prediction (`pred7` / `pred28`, i.e. V10), or call `oat()` with `fixed=` set to that mixture's other values.
* The report figures were drawn the same way: `analysis/make_figures_v3.py` (`Model.curve()`; Figures G03, G04 and G06).
