# Paired 7/28-day compressive-strength model (RCF · SS · A/B · carbonation)

Reproducible analysis of the corrected 7- and 28-day compressive strengths of 30 mixtures. One linear mixed
model covers both curing ages: curing age is a two-level factor, carbonation a four-level factor (NC, 0.5, 1, 5 h),
and the paired 7/28-day results of each mixture share a random mixture effect with age-specific residual variance.

**Data update.** The analysis now uses the updated corrected results
(`data/source/30_mixes_7_and_28_days_strength_results.pdf`), which change the 28-day strength of 11 mixes and the
7-day strength of mix 12 16.20 → 14.70 MPa (31 entries in all, `results/tables/T00b_changes_vs_previous_corrected_data.csv`).
The whole protocol was re-run unchanged. It selects the same model as before.
Previous and current results side by side: `results/tables/T24_comparison_with_previous_data.csv` and the "What changed" section of the report.

## Headline results

* Final model (exhaustive AICc search over 53,105 hierarchical models, both covariance structures):
  ln f = SS + SS² + RCF + carbonation (4 levels) + RCF·SS + K·RCF + K·SS **+ Age × (1 + SS + SS²)**,
  where K = carbonated RCF. A/B has no detectable effect.
* **In the selected model, curing age interacts with SS only.** f28/f7 = 2.88 (95 % CI 2.40–3.45) at SS 0 %, minimum 1.36 near SS 54 %,
  1.53 (1.20–1.95) at SS 75 %.
* Carbonated RCF raises strength in NaOH-rich mixes and lowers it in silicate-rich mixes; RCF raises strength more when carbonated
  and when SS is low. The three carbonation durations are not distinguishable (p = 0.098).
* 7-day strength of a left-out mixture: predicted R² 0.91, RMSE 2.1 MPa (replicate scatter 2.2 MPa).
  28-day: predicted R² -0.06, RMSE 7.6 MPa. 28-day results scatter more than the mixture variables explain.

Full write-up: [`report/MODEL_REPORT.md`](report/MODEL_REPORT.md) · printable summary: [`report/Model_Summary.pdf`](report/Model_Summary.pdf) ·
interactive report: [`report/index.html`](report/index.html) (open in a browser; it loads D3 from cdnjs).

## Layout

| Path | Content |
|---|---|
| `data/strength_7d_28d_corrected.csv` | corrected data set (controlling), one row per mixture |
| `data/strength_long.csv` | long format, one row per mixture and age |
| `data/previous_corrected_v1.csv` | previous corrected data set (superseded; for the change log and the comparison only) |
| `data/previous_7d_run_sheet.csv` | run sheet used by the earlier 7-day model (for the change log only) |
| `data/source/` | corrected data as supplied (PDF); `strength_7d_28d_corrected.csv` is transcribed from `30_mixes_7_and_28_days_strength_results.pdf`; `superseded/` holds the previous sheet |
| `reference/previous_7day_model/` | earlier 7-day model report (HTML, PDF), used as a methodological reference |
| `analysis/prepare_data.py` | long format, change logs (T00, T00b), replicate error |
| `analysis/compare_previous.py` | previous vs current results of the same protocol (T24); `results/previous_v1/` holds the previous key results |
| `analysis/run_analysis.py` | the protocol: scale, covariance, coding, exhaustive selection, validation, resampling, robustness, contrasts |
| `analysis/lmm.py` | bivariate REML/GLS mixed model with Satterthwaite df |
| `analysis/selection.py` | hierarchy bookkeeping, exhaustive AICc search (exact CS/UN factorisation), backward elimination |
| `analysis/design.py`, `analysis/model_tools.py` | factor coding, design matrices, predictions, contrasts, actual-unit equations |
| `analysis/make_figures.py` | figures F01–F11 (PNG 300 dpi + PDF) |
| `analysis/build_report.py`, `analysis/report_text.py` | reports; every number is read from `results/model_results.json` |
| `results/model_results.json` | all results |
| `results/tables/` | T00–T24 CSV tables |
| `results/figures/` | F01–F11 |
| `report/` | `index.html`, `Model_Summary.pdf`, `MODEL_REPORT.md` |

## Reproduce

```bash
pip install -r requirements.txt
python analysis/prepare_data.py
python analysis/run_analysis.py          # about 10 min on 4 cores (2,000 resamples per scheme); --quick for a test run
python analysis/compare_previous.py      # previous vs current data (T24)
python analysis/make_figures.py
python analysis/build_report.py          # needs Chromium for the PDF
```

Seed 20260928. Tested with Python 3.11, NumPy 2.4.6, pandas 3.0.6.

## Separate 7-day and 28-day models

`analysis/run_separate.py` builds two independent models, one per curing age, each from its own 30 results with the same protocol
(Box–Cox scale, the same 16-parameter candidate set, exhaustive AICc over 716 hierarchical models, identical validation and resampling).
Outputs: `results/separate/` (JSON, tables `S7_*`, `S28_*`, figures `H01–H10`) and `report/separate/`
(`index.html`, `Separate_Models_Summary.pdf`, `SEPARATE_MODELS.md`). Run `python analysis/run_separate.py`,
`python analysis/make_figures_separate.py`, `python analysis/report_separate.py`.

## SHAP explanations

`analysis/run_shap.py` computes exact interventional Shapley (SHAP) values and SHAP interaction values for the separate
7-day and 28-day models and the paired model, over all coalitions of the actual model inputs (RCF, SS, A/B, carbonation,
plus curing age in the paired model), with the 30 mixtures as background. Carbonation is one categorical input with
four levels (NC, 0.5 h, 1 h, 5 h). Importance intervals come from 1,000 stratified case-bootstrap refits.
Outputs: `results/shap/` (JSON, tables, figures `Z01–Z08b`) and `report/shap/` (`index.html`, `SHAP_Figures.pdf`,
`SHAP_NOTES.md`). Run `python analysis/run_shap.py`, `python analysis/make_figures_shap.py`,
`python analysis/report_shap.py`. The optional cross-check against the `shap` package (0.51.0) runs only when it is installed.

## Relation to the earlier 7-day model

The earlier 7-day model (numeric carbonation, 7-day data only) was used as a methodological and presentation reference.
The new model was selected independently from the combined data. It arrives at the same pattern of 7-day terms
(RCF, SS, SS², carbonation, RCF·SS, carbonation·RCF, carbonation·SS) with categorical carbonation, and adds the SS-dependent age effect.
The corrected data set differs from the earlier run sheet in 36 entries (`results/tables/T00_changes_vs_previous_7d_data.csv`).
