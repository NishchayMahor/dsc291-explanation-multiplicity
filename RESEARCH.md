# Research notes: do post-hoc explanations survive retraining?

Last checked 2026-11-17. Anything not marked verified came from an abstract or a paper's own claim and was not re-checked.

## The problem
Retrain a black box on a fresh bootstrap sample or with a new seed, or swap it for another model that is just as accurate, and the same person can get a different SHAP explanation. A sparse model gives its own reasons (the coefficients that fire, the rule on the decision path), and those can change after retraining too. The question is which changes more for the same person, especially when that person's decision stays the same. If SHAP reasons move a lot while sparse reasons hold steady, that supports Lipton's warning and Rudin's argument with data. If they move about the same amount, the problem is retraining instability in general, not post-hoc explanation as such.

## What is known (venues verified unless marked)
- **Lipton (ICML 2016 WHI workshop; ACM Queue 2018).** Argues that post-hoc explanations can mislead and that "interpretability" covers several different goals. No measurements.
- **Rudin (Nature Machine Intelligence 2019).** Argues that in high-stakes settings we should use interpretable models, because an explanation of a black box must be unfaithful somewhere. No retraining experiments.
- **Lundberg & Lee (NeurIPS 2017).** Introduces SHAP. The XGBoost and `shap` TreeSHAP implementations we use are listed below.
- **Marx, Calmon, Ustun (ICML 2020); Black, Leino, Fredrikson (ICLR 2022); D'Amour et al. (JMLR 2022).** Models that are equally accurate can still give different predictions to the same people (predictive multiplicity, underspecification). Black et al. also show that the inconsistency carries over to feature attributions, and propose selective ensembles.
- **Brunet, Anderson, Zemel (NeurIPS 2022).** Model indeterminacy makes explanations of individual decisions indeterminate as well.
- **Laberge et al. (JMLR 2023).** Over a Rashomon set of models, only a *partial order* of feature importance can be agreed on, so some feature pairs cannot be compared. Code available.
- **Fisher, Rudin, Dominici (JMLR 2019); Dong & Rudin (Nature MI 2020); Donnelly et al. (NeurIPS 2023, "RID").** Variable importance measured over the whole set of good models, at the global (not per-person) level.
- **Ley et al. (ICML 2023 IMLH workshop).** Ensembling over seeds makes explanations more consistent. This is the "fix" baseline.
- **Müller et al. (ECML PKDD 2023).** Empirical study of the Rashomon effect on attributions across datasets, models and metrics. Finds that hyperparameter tuning and the choice of metric both matter.
- **Xin et al. (NeurIPS 2022, TreeFARMS).** Enumerates the *whole* Rashomon set of sparse decision trees and finds many near-optimal sparse trees. **This matters for us: sparse models also have many equally good versions, so their reasons are not stable for free.**
- **Meyer et al. (UAI 2023; proceedings page not checked).** Explanation stability under retraining after dataset shift; model curvature and weight decay drive it.
- **Preprints (venue unverified):** Caraker et al. 2026 (arXiv 2603.22346) report that gradient boosting concentrates SHAP credit on one of two correlated features ("first-mover bias") and that averaging over seeds fixes it. B. Thackshanaramana 2026 (arXiv 2603.15821) report that accurate models of *different* classes (tree vs linear) disagree on attributions much more than models of the same class do.

## Closest prior art: Hwang et al. (NeurIPS 2026 E&D track; arXiv 2601.12654, read in full)
- **Setup.** An "explanation query" fixes the dataset, task, model class, explainer and person. Running the pipeline again gives a draw from a *distribution* of explanations. A "dual-seed protocol" varies (i) the model seed (training plus hyperparameter selection) with the explainer fixed, and (ii) the explainer's background-sample seed with the model fixed. They use 5 runs per setting, so 10 pairs per person.
- **Explainers.** Model-agnostic Kernel SHAP and Permutation SHAP with K = 100 background rows (Random, K-Means, prediction-stratified, CTE). **Not exact TreeSHAP.**
- **Models.** Decision tree, random forest, XGBoost, MLP, FT-Transformer, all tuned by grid search. **No sparse or linear model, and no model's own reasons.**
- **Datasets.** ACS Income (Virginia 2018), German Credit, Give Me Some Credit, Diabetes (392 rows), and Arrhythmia in the appendix.
- **Metrics.** L2 distance, top-k Jaccard distance, rank-biased overlap, and per-feature sensitivity, each compared against random null models (Dirichlet for magnitudes, Mallows for rankings).
- **Findings.** L2 says "stable" (0% of pairs exceed the null baseline), but rankings are not. Top-3 Jaccard exceeds the baseline in up to 28% (German), 30% (Diabetes), 24% (ACS) and 49% (GMSC) of pairs for FT-Transformer. On GMSC the leading feature differs in 42.9% of pairs. Model seeds cause more of the disagreement on small datasets, background sampling more on large ones. Better samplers (CTE) do not remove rank churn, and K-Means is stable but drifts from the reference. High-confidence predictions are also unstable. Their advice is to report SHAP as a distribution, not a single answer. Code: https://github.com/Aguno/SHAP_Multiplicity (verified that the repo exists).

**Overlap with this project.** Both projects measure per-person top-k Jaccard of SHAP across model retrains, and both look at confidence or the decision boundary (our H4). H1 ("SHAP top-2 changes even when the decision doesn't") is largely already shown by them in spirit, though not with that exact conditioning.

**What they do not do (our gap).** (1) **They never compare against an interpretable model's own reasons.** That comparison is the actual test of Rudin's argument, and it remains open. (2) They use sampling-based SHAP, so their "model seed" results still contain explainer noise. Exact TreeSHAP removes that noise and isolates retraining. (3) They do not use bootstrap or data-resampling retrains, or a Rashomon set built from different hyperparameters. (4) They do not condition on the person's decision staying the same.

**Repositioning.** Do not pitch "SHAP is unstable under retraining"; that result belongs to Hwang et al. and Müller et al. Pitch instead: *"For the same person and the same retraining noise, are a sparse model's own reasons more stable than a black box's exact SHAP explanation, and does the answer survive when the sparse model also has a Rashomon set?"* Cite Hwang et al. as the explainer-noise side of the story, and use their metrics (top-k Jaccard plus RBO) and Mallows null baseline so the numbers are comparable.

## Open-source implementations (links checked 2026-11-17)
- SHAP: https://github.com/shap/shap · XGBoost: https://github.com/dmlc/xgboost (`Booster.predict(pred_contribs=True)` is exact TreeSHAP; `shap.TreeExplainer` calls it directly for XGBoost, verified in NOTES.md)
- Hwang et al.: https://github.com/Aguno/SHAP_Multiplicity
- Laberge et al. (partial orders): https://github.com/gablabc/Partial_Order_in_Chaos
- TreeFARMS, the Rashomon set of sparse trees: https://github.com/ubc-systopia/treeFarms
- RID: https://github.com/jdonnelly36/Rashomon_Importance_Distribution (the README says code is posted on publication; check that it is there)
- DASH / seed averaging for correlated features: https://github.com/DrakeCaraker/dash-shap (preprint)

## Datasets
COMPAS (https://github.com/propublica/compas-analysis, local copy present), Adult (UCI #2), German credit (UCI #144), and FICO HELOC (registration required). To line up directly with Hwang et al., add ACS Income through `folktables` (https://github.com/socialfoundations/folktables) and Give Me Some Credit (Kaggle). Adult and German are already in `../../experiments/data/`.

## Week 1–2 experiments
1. **Diagnose the early H2 result.** The first baseline run (NOTES.md) gives L1 coef*x top-2 Jaccard 0.685 vs XGBoost SHAP 0.822 for people whose decision did not change, while Spearman favors L1 (0.886 vs 0.842). Check four things: how often a feature's L1 coefficient switches between zero and non-zero across bootstraps (support instability), whether ties or zero-exclusion shrink the top-k sets, a sweep of C, and k = 1..3 plus RBO. Expected outcome: we learn whether "sparse reasons" are unstable because the *support* moves (a real finding, and bad news for Rudin) or because of how we measure.
2. **Separate the noise sources, as Hwang et al. do.** Retrain XGBoost with (a) seed only, (b) bootstrap, and (c) members of a 1-point Rashomon set over hyperparameters. Note that XGBoost with `subsample = colsample = 1` is deterministic given the data, so seed-only retrains may change nothing: confirm this. Then add Kernel SHAP (K = 100) on one fixed model to measure explainer noise on the same scale as exact TreeSHAP.
3. **Give sparse models a Rashomon set too.** On binarized COMPAS, use TreeFARMS to sample equally good sparse trees and compute decision-path Jaccard for people whose decision is unchanged. Compare it with XGBoost's Rashomon-set SHAP Jaccard. This is the fair test of Rudin's claim, and the part nobody has done.

## Key risks
- **Novelty.** Hwang et al. 2026 and Müller et al. 2023 already show that SHAP is unstable. The new part is only the sparse-model head-to-head, so keep the framing tight.
- **The result may go against Rudin.** The first run already hints at this. Report it either way; a clean negative result is still a result.
- **Correlated features.** SHAP splits credit between correlated features (and boosting has a first-mover bias), so measured instability may reflect correlation rather than real model change. Add a grouped-feature control.
- **Apples and oranges.** coef*x, path rules and SHAP are different objects, so compare only rank and set metrics, and say so.
- **Coarse metrics on COMPAS.** With 7 features, top-2 Jaccard has very few possible values. Use Adult, FICO or ACS for k up to 5.
- **Hwang et al.'s venue** comes only from the arXiv comment and paper footer; check the NeurIPS proceedings before citing.
