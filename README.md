# Explanation Multiplicity: Do Post-hoc Explanations Survive Retraining?

> **Starter code for a DSC 291 course project** (Interpretable Machine Learning, UC San Diego, Fall 2026).
> We built this after presenting Lipton, *The Mythos of Model Interpretability* (2016), and Rudin, *Stop Explaining Black Box Machine Learning Models for High Stakes Decisions* (2019), in class on Nov 19, 2026.
> It gives you a research question, a consolidated and checked reading list ([RESEARCH.md](RESEARCH.md)), and a baseline that runs end to end in about 30 seconds, so a project can start from experiments instead of setup. Fork it, change the question, make it yours.

## Quickstart
Python 3.10 to 3.12.
```bash
git clone https://github.com/NishchayMahor/dsc291-explanation-multiplicity.git
cd dsc291-explanation-multiplicity
pip install -r requirements.txt
python src/run_baseline.py
```
Data is downloaded on the first run (no account needed).

| File | What it is |
|---|---|
| `README.md` | Question, motivation, hypotheses, method, plan (this page) |
| `RESEARCH.md` | Consolidated literature: what is known, the gap, code to reuse, first experiments |
| `NOTES.md` | Open design choices and what has and has not been verified |
| `references.bib` | BibTeX for every paper cited |
| `src/run_baseline.py` | The runnable baseline |

## Research question
When a black box is retrained (bootstrap, seed) or swapped for an equally accurate one, how much do its SHAP attributions change for the same person, compared with how much a sparse model's own reasons (coefficients that fire, the rule on the decision path) change? And how often does the explanation change while the decision does not?

## Motivation
- **Lipton (2016)** says post-hoc explanations can mislead and may not reflect how the model decides. **Rudin (2019)** goes further: explanations of black boxes "must be wrong" somewhere, or they would be the model. Both are mostly argued, not measured.
- Our rule-list vs SHAP check (talk slide 15): a rule list and XGBoost agree on 84.8% of COMPAS defendants, and SHAP's top-2 features include one the rule list never uses in 11.6% of cases. Example: a 57-year-old with 12 priors; the rule says "more than 3 priors", SHAP says age (-1.10) cancels priors (+1.19).
- Our retraining test (talk slide 13): retraining XGBoost flips 45.9% of people at least once (sparse logistic 24.1%, depth-3 tree 59.5%). If predictions move that much, the attributions should move too. A person given a SHAP explanation may get a different one from the same pipeline next week.
- The comparison that matters for Rudin's argument: is a sparse model's *reason* more stable than a black box's *explanation*, for people whose decision stays the same?

## Related work (titles/IDs checked 2026-11-10; venues and years checked 2026-11-17 against arXiv, Crossref, PMLR, NeurIPS, JMLR, Nature and conference pages. Details in RESEARCH.md)
1. Lipton, *The Mythos of Model Interpretability*, ICML 2016 Workshop on Human Interpretability (WHI); journal versions in ACM Queue 2018 and Communications of the ACM 2018. https://arxiv.org/abs/1606.03490
2. Rudin, *Stop Explaining Black Box ML Models for High Stakes Decisions...*, Nature Machine Intelligence 2019. https://arxiv.org/abs/1811.10154
3. Lundberg, Lee, *A Unified Approach to Interpreting Model Predictions* (SHAP), NeurIPS 2017. https://arxiv.org/abs/1705.07874
4. Krishna et al., *The Disagreement Problem in Explainable ML*, TMLR 2024. https://arxiv.org/abs/2202.01602 (disagreement across explainers; we study disagreement across retrains)
5. Brunet, Anderson, Zemel, *Implications of Model Indeterminacy for Explanations of Automated Decisions*, NeurIPS 2022. https://doi.org/10.52202/068431-0567
6. Laberge et al., *Partial Order in Chaos: Consensus on Feature Attributions in the Rashomon Set*, JMLR 2023. https://arxiv.org/abs/2110.13369
7. Fisher, Rudin, Dominici, *All Models are Wrong, but Many are Useful* (model class reliance), JMLR 2019. https://arxiv.org/abs/1801.01489
8. Dong, Rudin, *Variable Importance Clouds* (arXiv title; published as *Exploring the cloud of variable importance for the set of all good models*), Nature MI 2020. https://arxiv.org/abs/1901.03209
9. Slack et al., *Fooling LIME and SHAP*, AIES 2020. https://arxiv.org/abs/1911.02508
10. Adebayo et al., *Sanity Checks for Saliency Maps*, NeurIPS 2018. https://arxiv.org/abs/1810.03292
11. Ley et al., *Consistent Explanations in the Face of Model Indeterminacy via Ensembling*, ICML 2023 Workshop on Interpretable ML in Healthcare (IMLH). https://arxiv.org/abs/2306.06193
12. Hwang et al., *Explanation Multiplicity in SHAP: Characterization and Assessment*, NeurIPS 2026 Evaluations and Datasets Track (per arXiv v3 comment and paper footer). https://arxiv.org/abs/2601.12654 (**closest prior art; read and summarized in RESEARCH.md**)
13. Marx, Calmon, Ustun, *Predictive Multiplicity in Classification*, ICML 2020. https://arxiv.org/abs/1909.06677
14. Black, Leino, Fredrikson, *Selective Ensembles for Consistent Predictions*, ICLR 2022. https://arxiv.org/abs/2111.08230
15. D'Amour et al., *Underspecification Presents Challenges for Credibility in Modern ML*, JMLR 2022 (arXiv 2020). https://arxiv.org/abs/2011.03395

Gap: refs 5, 6, 11, 12 study explanation instability for black boxes; refs 7-8 study importance over the set of good models globally. Our angle is the head-to-head with **sparse models' native reasons** at the individual level, conditioned on the decision not changing, which is the direct test of Rudin's argument. Ref 12 was checked (2026-11-17): it does **not** include sparse/interpretable models' own reasons, bootstrap retraining, or decision-unchanged conditioning, and it uses Kernel/Permutation SHAP rather than exact TreeSHAP. See RESEARCH.md for the positioning and for newer related work (Müller et al. 2023, Donnelly et al. 2023, Xin et al. 2022).

## Hypotheses (falsifiable)
- **H1.** Among people whose XGBoost decision does not flip across retrains, mean top-2 Jaccard of SHAP features is below 0.8. *False if* >= 0.9 (explanations are stable when decisions are).
- **H2.** Sparse L1 per-person reasons (top-2 coef*x) are more stable than XGBoost SHAP top-2 (Jaccard higher by >= 0.1). *False if* the difference is < 0.03 or reversed.
- **H3.** Depth-3 tree decision paths are *less* stable than sparse logistic reasons (mirrors our retraining test). *False if* tree path Jaccard >= L1 Jaccard.
- **H4.** Explanation instability concentrates on people near the decision boundary (|p - 0.5| small). *False if* no association (|rho| < 0.1).

## Method
1. Retrain sources: (a) bootstrap + seed retrains, fixed hyperparameters (our retraining-test setup); (b) members of a 1-pt Rashomon set of XGBoost models (varying depth, learning rate, subsample).
2. Explanations: TreeSHAP and KernelSHAP/LIME for XGBoost; exact contributions coef*x for sparse logistic; decision-path features for trees and rule lists.
3. Metrics per person, averaged over model pairs: top-k Jaccard (k = 1, 2, 3), Spearman rank correlation of |attribution|, sign agreement on the top feature; all computed overall and for decision-unchanged people.
4. Repeat on COMPAS, Adult, German, FICO; 10 outer splits; bootstrap CIs.

## Datasets (open)
- COMPAS: https://github.com/propublica/compas-analysis (local copy in `../../experiments/data/`)
- Adult: https://archive.ics.uci.edu/dataset/2/adult
- German credit: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
- FICO HELOC: https://community.fico.com/s/explainable-machine-learning-challenge (registration required)

## Evaluation metrics
Top-k Jaccard; Spearman rho of |attributions|; top-feature sign agreement; prediction flip rate; all split by decision-changed vs unchanged and by distance to the boundary.

## Baselines
XGBoost + TreeSHAP (main black box); sparse L1 logistic (C = 0.05, as in our talk); depth-3 CART; Rudin's 3-rule list (fixed, so its reasons never change: a zero-instability reference); ensembled SHAP (Ley et al.) as the "fix" baseline.

## Current baseline (`src/run_baseline.py`)
20 bootstrap retrains each of XGBoost, a sparse L1 model and a depth-3 tree on COMPAS. Prints top-2/top-3 Jaccard, Spearman rank correlation and flip rate, and the same metrics for people whose decision did not change. Works without `shap` (XGBoost's built-in TreeSHAP gives identical values; checked).
```bash
python src/run_baseline.py --retrains 20
```
**First run (2026-11-17, one split, 600 test people), people whose decision did not change:**

| Model and explanation | Top-2 Jaccard | Spearman |
|---|---|---|
| XGBoost + TreeSHAP | 0.822 | 0.842 |
| Sparse L1 (coef x feature) | 0.685 | 0.886 |
| Depth-3 tree (decision path) | 0.650 (path Jaccard) | n/a |

At top-2, the sparse model's own reasons look *less* stable than SHAP, which goes against H2, while Spearman points the other way. Before reading anything into it, check whether it comes from L1 coefficients dropping to zero across bootstraps (zeros are excluded from top-k) or from ties. That is experiment 1 in RESEARCH.md.

## Expected results and what would falsify them
We expect SHAP top-2 sets to change for a sizeable minority of decision-unchanged people (Jaccard around 0.7-0.85), sparse L1 reasons to be more stable, and tree paths least stable. If SHAP is as stable as sparse reasons (H2 false), Rudin's "explanations are unreliable" argument loses its empirical leg on tabular data, and the real problem is predictive, not explanatory, instability.

## Risks and limitations
- SHAP on correlated features splits credit arbitrarily; instability may reflect feature correlation, not model change. Control with grouped features.
- coef*x and SHAP are not the same object; compare rank-based metrics, not magnitudes.
- Novelty risk from Hwang et al. (2026): read and summarized in RESEARCH.md. They cover SHAP-vs-SHAP rerun stability for black boxes; the sparse-model comparison is not in their paper. Sparse models also have Rashomon sets (Xin et al. 2022), so a sparse model's reasons are not guaranteed stable either.
- 7 COMPAS features make top-k metrics coarse; use richer feature sets on Adult/FICO.

## 6-week plan
| Week | Work |
|---|---|
| 1 | Run baseline; read Hwang et al. 2026, Brunet et al. 2022, Laberge et al. 2023; finalize metrics |
| 2 | Rashomon-set XGBoost retrain source; KernelSHAP/LIME |
| 3 | Adult, German, FICO |
| 4 | Boundary analysis (H4); feature-correlation control |
| 5 | Ensembled-SHAP baseline; CIs over outer splits |
| 6 | Write-up, figures |

## Path to a workshop paper
Fits XAI or trustworthy-ML workshops at NeurIPS / ICML / FAccT (check current calls). Pitch: "the reason you were given changes when the model is retrained, even when your outcome doesn't; sparse models' reasons change less." Needs H1-H2 on 3+ datasets with CIs and a clear delta over Hwang et al.
