# Model Card - Account Propensity

## What this is

A logistic regression predicting whether an account converts. The coefficients are
**fitted** by `training/train.py`; regularisation strength (`C`) was selected by
5-fold cross-validation on the training split only. The held-out
test split was scored once.

## Training data - synthetic

The model was trained on **synthetic data** produced by `training/generate_dataset.py`
(seeded, reproducible). It is **not** real CRM data. These metrics describe how well the
model recovers a generating process we wrote down; they are **not** evidence of
real-world performance, and the model has never been validated against real outcomes.

The generator deliberately uses a saturating support penalty, a log spend term, a
square-root maturity term, and a usage x renewal interaction - none of which a linear
model can represent exactly. Labels are sampled from the true probability rather than
thresholded, so some error is irreducible by construction.

## Measured performance (held-out test set, n=1250)

| Metric | Value |
|---|---|
| ROC-AUC | 0.8614 |
| PR-AUC | 0.758 |
| Accuracy | 0.7976 |
| Precision | 0.7262 |
| Recall | 0.59 |
| F1 | 0.651 |
| Brier score | 0.1373 |

Cross-validated ROC-AUC on the training split:
**0.8495 +/- 0.0135** - consistent with the
held-out figure, so the model is not overfit.

### Headroom

The generator's **Bayes-optimal ROC-AUC on this test split is 0.8898** -
that is the score achieved by ranking on the true conversion probability itself, and it is
below 1.0 because labels are sampled from that probability rather than thresholded.

The model reaches 0.8614 against that 0.8898 ceiling, so it
captures nearly all of the ranking signal that exists in the data. This number is what gives
the headline metric scale: a ROC-AUC approaching 1.0 here would indicate leakage or
degenerate labels, not a better model.

Confusion matrix at threshold 0.5: TN=761, FP=89,
FN=164, TP=236.

Base rate is 0.3204, so accuracy alone is a weak summary here;
ROC-AUC and PR-AUC are the figures to read.

## Fitted coefficients (standardised feature space)

| Feature | Coefficient |
|---|---|
| `usage_frequency` | +0.9911 |
| `support_tickets` | -0.8018 |
| `renewal_days` | -0.3541 |
| `spend` | +0.3066 |
| `account_age_days` | +0.2351 |
| `visits` | +0.1458 |

Intercept: -1.1787

## Segment thresholds

`high_propensity` at score >= 0.4438,
`medium_propensity` at score >= 0.1747.

These are the 70th and 40th percentiles of
the **training-set** score distribution, not hand-picked round numbers. Raising
`HIGH_QUANTILE` in `training/train.py` makes the top segment rarer and more precise; it is
a business calibration knob, not a modelling constant.

## Known limitations

- **`industry` is a real signal the served model cannot see.** It shifts conversion
  materially in the generator (roughly 0.23 to 0.42 across sectors) and is part of
  `datasets/production_schema.json`, but it is not in the `/score` request payload, so the
  model is not trained on it. Adding it is the clearest available improvement and would
  require a serving contract change.
- Linear decision boundary; the generator's interaction and saturation terms are
  approximated, not recovered. This gap is intentional and is why the metrics are not
  near-perfect.
- No calibration layer beyond what logistic regression provides natively.
- No drift detection, no retraining trigger, no monitoring of live score distributions.
- Trained and evaluated on a single seeded synthetic draw; no confidence intervals
  across seeds.
