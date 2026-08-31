# ADR-001 — Chronological split, leaky feature dropped, all four variants published

**Status:** Accepted · **Date:** 2026-06

## Context

The real-data track runs this pipeline against [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) —
41,188 real contacts from a Portuguese bank, May 2008 to November 2010, labelled with
whether the client subscribed.

Two evaluation choices are available, and each can be made the easy way or the hard way.

`duration` is the length of the call being predicted. A call ending in a subscription is a
long call, so the feature encodes the label — and it does not exist at the moment you decide
whom to phone. Every tutorial using this dataset warns about it.

The split is the less-discussed one. The rows are date-ordered across the financial crisis.
A random split trains on 2010 to predict 2009, and the train-period positive rate is 0.0642
against 0.2581 in the test period — a fourfold drift that a shuffle erases entirely.

## Decision

Take the hard option on both. Report **0.7090** as the headline. Publish all four
combinations, measured here rather than cited:

| Variant | ROC-AUC | Deployable? |
|---|---|---|
| **no `duration`, chronological split** | **0.7090** | **yes — the headline** |
| with `duration`, chronological split | 0.7931 | no — the feature doesn't exist at decision time |
| no `duration`, random split | 0.7987 | no — the split leaks the future |
| both together | 0.9364 | no — and this is the figure usually published |

Dropping `duration` costs 0.0841. Splitting chronologically costs **0.0897 — slightly more
than the leaky feature does.** Together they inflate the headline by **0.2274**.

## Alternatives considered

**Report 0.9364 and mention the caveats below the table.** This is the common presentation
and it is not dishonest by the letter — the caveats are there. Rejected because the number a
reader remembers is the headline, and a headline that is 32% higher than the deployable
figure is not a summary of the work, it is a different claim.

**Report only 0.7090 and say the others were rejected.** The safe honest option, and the
first draft. Rejected because it throws away the finding. Anyone can drop a leaky feature;
the interesting result is the *comparison* — that the split choice cost more than the
leakage everyone warns about. Without the other three rows there is no evidence for that,
only an assertion.

**Use a random split because the class balance is more stable.** Genuinely defensible on
variance grounds, and wrong here for a specific reason: the instability *is the problem
being modelled.* A model deployed in 2010 has to work on 2010, and a validation scheme that
averages 2008 and 2010 together answers a question nobody will ask it.

**Drop `duration` but keep the random split, as most published baselines do.** Rejected once
measured. That variant scores 0.7987 — closer to the honest number than the fully-leaky one,
and still 0.0897 above what the model can actually do.

## Consequences

- The headline number is visibly lower than this dataset is usually reported at, and the
  README has to explain that before it can say anything else. Accepted deliberately.
- The comparison required building all four variants as first-class code paths rather than
  one path with flags. That surfaced a real bug: the `ColumnTransformer` had its numeric
  columns hardcoded, so the leaky variant silently scored **identically** to the honest one
  (0.709 both). `build_pipeline(numeric)` now takes the columns as a parameter, and a test
  asserts the leaky variant must win — a comparison where the wrong side cannot win is not
  a comparison.
- The finding generalises further than the model does. The model is not served; its features
  are a bank campaign schema, not the six account features in `/score`. **What transfers is
  the method**, and that is stated where the number is.

## Revisit when

Nothing about this changes with more data — the choices are structural. It is worth
revisiting only if the deployment context changes such that a random split becomes the
honest one, which would mean the data stopped being temporal.
