# ADR-002 — Logistic regression, not gradient boosting

**Status:** Accepted · **Date:** 2026-04

## Context

The obvious upgrade for a tabular binary classifier is gradient boosting. It would almost
certainly score higher, and it is what a reviewer expects to see.

The synthetic track has a number that makes the decision measurable rather than a matter of
taste. Because labels are *sampled* from the true conversion probability rather than
thresholded, the generator has a **Bayes-optimal ROC-AUC of 0.8898** — the score obtained by
ranking on the true probability itself. The fitted model reaches **0.8614**, about 97% of
the ranking signal that exists in the data.

The remaining headroom is 0.0284.

## Decision

Logistic regression (`StandardScaler` → `LogisticRegression`), with `C` selected by 5-fold
CV on the training split only, and the test split scored once.

## Alternatives considered

**Gradient boosting (`HistGradientBoostingClassifier`, XGBoost, LightGBM).** Would very
likely close some of the 0.0284 headroom. Rejected because closing it proves nothing here:
the generator's true log-odds is a smooth function with a saturating support penalty, a
`log1p` term, a square-root term and one interaction. A more flexible model recovering more
of a function *we wrote* is a statement about the generator, not about the method. The
ceiling is what makes this arguable at all — without it, "we could score higher" has no
answer.

**Ship both and report the comparison.** The pattern used elsewhere in this portfolio for
[reranking](https://github.com/Adityansh-Chand/enterprise-rag-knowledge-system/blob/main/docs/adr/003-reranker-reported-not-removed.md)
and for the incident detector. Genuinely tempting, and the reason it was not done is
narrower than it looks: on the *real* data track the comparison would be meaningful, and
there the finding is already about evaluation design (ADR-001) rather than model class. Two
findings competing for the same headline weakens both. This is the weakest point of this
record.

**A neural model.** Rejected as clearly disproportionate for six features and 5,000 rows.

**Keep the hand-picked logit that was here before.** Recorded because it is what this
replaced: `1.35 * engagement_score + 1.20 * spend_score + ...` over magic-number
normalisers, with a hardcoded feature-importance dict presented as learned. Not an
alternative — the thing being fixed.

## Consequences

- Coefficients are directly interpretable, and committed in readable form
  (`models/artifacts/coefficients.json`) so a reviewer can confirm they are learned rather
  than chosen. `usage_frequency` +0.99, `support_tickets` −0.80: signs a domain reader can
  check against intuition.
- Explanations are real linear attributions (`coef[i] * scaled_value[i]`) rather than a
  post-hoc approximation. With boosting this would have meant SHAP — a dependency and an
  approximation, for a model whose main audience is a sales rep asking "why this account".
- Calibration comes free and is reported (Brier 0.1373), which matters because the output
  drives segment thresholds derived from train-set score quantiles, not just a ranking.
- **The model is very likely leaving accuracy on the table on real data.** Stated rather
  than hidden: 97% of the ceiling is a claim about the synthetic track only.

## Revisit when

The real-data track becomes the served model rather than a parallel evaluation. At that
point the ceiling argument no longer applies — nobody wrote that generating process — and
model class becomes an empirical question worth measuring instead of reasoning about.
