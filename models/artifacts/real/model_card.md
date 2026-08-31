# Model Card - Account Propensity on REAL data

## What this is

The same pipeline as `models/artifacts/model_card.md` -- one-hot encoding, standard
scaling, logistic regression, `C` chosen by 5-fold cross-validation on the
training split only -- fitted on **real outcomes** instead of generated ones.

**Data:** REAL -- UCI Bank Marketing (CC BY 4.0)
Moro, Cortez and Rita, 'A Data-Driven Approach to Predict the Success of Bank Telemarketing', Decision Support Systems, 2014
https://archive.ics.uci.edu/dataset/222/bank+marketing

41188 real contacts from a Portuguese bank's direct marketing
campaigns, May 2008 to November 2010. The label is whether the client actually
subscribed to a term deposit.

## Two decisions that make this number lower, and honest

### `duration` is excluded

Call duration is known only after the call ends, and the call ending in a subscription is what makes it long -- it leaks the label and cannot be used at prediction time.

Including it scores **ROC-AUC 0.7931**. Excluding it scores
**0.709**. That **0.0841** gap is the
size of the leak, and it is why published results on this dataset vary so widely:
a model with `duration` looks far better and cannot be deployed, because at the
moment you must decide whether to call someone, the length of that call does not
exist yet.

### The split is chronological

The rows are date-ordered across the financial crisis, with euribor collapsing
over the period. A random split would train on 2010 to predict 2009. This one
trains on the first 30891 contacts and tests on the
last 10297 -- a genuine forward test, which is harder
and is the only version of the question worth asking.

The same pipeline under a **random** split scores
**0.7987**, against
**0.709** chronologically: an inflation of
**0.0897**.

## The finding worth stating plainly

| Variant | ROC-AUC | Deployable? |
|---|---|---|
| **no `duration`, chronological split** | **0.709** | **yes -- the headline** |
| with `duration`, chronological split | 0.7931 | no, the feature does not exist yet at decision time |
| no `duration`, random split | 0.7987 | no, the split leaks the future |
| **both mistakes together** | **0.9364** | no -- and this is the number usually published |

Choosing the split wrongly costs **0.0897**.
Leaving the leaky feature in costs **0.0841**. The
evaluation design is worth *more* here than the leakage everyone warns about, and
doing both compounds them to
**0.9364** -- an inflation of
**0.2274** describing nothing anyone could
deploy. All four rows were measured here; none is quoted from a paper.

That is the entire argument this portfolio makes, measured on somebody else's data.

### Base rates move, which is the point

Train-period positive rate is 0.0642; test
period is 0.2581 -- a fourfold shift as the
campaign changed. A random split averages that away and reports a model that was
never asked the hard question.

Note also that cross-validated ROC-AUC on the training split
(0.6768) is *below* the held-out figure
(0.709). That inversion is a property of the drift, not a mistake:
the later period is both more positive and more separable. It is reported rather
than smoothed over.

## Measured performance (held-out final 25%, n=10297)

| Metric | Value |
|---|---|
| ROC-AUC | 0.709 |
| PR-AUC | 0.4593 |
| Accuracy | 0.7516 |
| Precision | 0.5395 |
| Recall | 0.257 |
| F1 | 0.3481 |
| Brier score | 0.1733 |

Cross-validated ROC-AUC on the training split:
**0.6768 +/- 0.0142**.

Confusion matrix at threshold 0.5: TN=7056, FP=583,
FN=1975, TP=683.

Base rate in the test period is 0.2581, so **accuracy is
not a meaningful summary** -- always predicting "no" scores well above it while
finding nobody. ROC-AUC and PR-AUC are the figures to read, and PR-AUC should be
read against the 0.2581 base rate rather than against 1.0.

## How this relates to the served model

The service serves the synthetic-schema model. This one is **not** deployed: its
features are a bank's campaign schema, not the six account features in the `/score`
payload. What transfers is the method -- the pipeline, the leakage discipline, the
chronological split -- validated here against outcomes nobody designed.

Read together, the two model cards say: the approach works on real data, and the
served model demonstrates it on a schema we control.

## Known limitations

- One dataset, one domain, one period. Real, but not a general claim.
- No fairness analysis across the demographic attributes present (`age`, `job`,
  `marital`, `education`). For a real deployment that would be required, not optional.
- Linear decision boundary; no interaction terms beyond what one-hot encoding gives.
- The economic indicator columns are period-level, so the model can lean on
  macroeconomic conditions rather than anything about the individual client.
