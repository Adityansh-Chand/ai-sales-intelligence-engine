# ADR-003 — Train only on features available at decision time

**Status:** Accepted · **Date:** 2026-04

## Context

The dataset generator produces an `industry` field. It is correlated with conversion and
would improve held-out performance if included.

It is not in the `/score` request payload. The serving contract is six fields —
`visits`, `spend`, `account_age_days`, `usage_frequency`, `support_tickets`,
`renewal_days` — and a caller scoring a new account does not send an industry.

This is the same shape of problem as `duration` in the real-data track (ADR-001), arriving
from the opposite direction: there, a feature that encodes the label; here, a feature that
simply is not there when the question is asked. Both are training-serving skew, and both
inflate a metric that a deployment then fails to reproduce.

## Decision

Train on exactly the six features the API accepts. Generate `industry`, because it makes the
synthetic population more realistic, and **do not train on it**.

Disclose it as a named limitation in the model card rather than deleting it quietly.

## Alternatives considered

**Include `industry` and add it to the API payload.** The clean fix if the field were
genuinely available. Rejected because it would be inventing a data source: this service's
consumer is the customer-operations service, which sends an account identifier and account
attributes it actually holds. Adding a required field the caller cannot populate moves the
problem to the caller rather than solving it.

**Include `industry` with a default for missing values at serving time.** The pragmatic
production answer, and the most dangerous one here. Every served request would take the
default, so the model would be evaluated on a distribution of industries it will never see
and served on a constant — a metric that cannot be reproduced by the running system. If it
is always the default in serving, it is not a feature; it is a bias term with extra steps.

**Delete `industry` from the generator entirely.** Rejected as hiding the decision. The
field is realistic — real CRM data has attributes the scoring endpoint does not receive —
and keeping it makes the constraint visible in the code rather than only in prose.

**Train two models: a rich one for batch scoring, a thin one for the API.** Legitimate, and
what a larger system would do. Rejected as disproportionate: there is no batch scoring path
here, so the second model would exist only to demonstrate that it could.

## Consequences

- Reported performance is lower than this generator can support. That gap is the point: it
  is the difference between what the model scores and what the *service* scores.
- The generated `industry` column is committed and unused, which looks like an oversight
  until the model card is read. Accepted, and the model card names it first among
  limitations.
- The `/score` contract and the training feature list cannot drift apart, because they are
  the same six names in `pipeline/features.py` — and the versioned API means a change to
  either is a visible contract change rather than a quiet one.
- Combined with ADR-001 this repository now demonstrates both directions of training-serving
  skew on the same pipeline: a feature that leaks the label, and a feature that is absent at
  inference. That pairing is more useful than either alone.

## Revisit when

The caller can genuinely supply industry — for example if account enrichment adds it
upstream. At that point it becomes a feature rather than a hazard, and the API version
mechanism is how it would be introduced without breaking existing consumers.
