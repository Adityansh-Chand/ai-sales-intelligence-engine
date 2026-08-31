# Architecture Decision Records

Decisions that shaped this service, each with the alternatives that were actually
considered, the evidence that settled it, and what would make it worth revisiting.

A record is written when a choice was **contested** — when a competent engineer could
reasonably have gone the other way, and the reason it went this way is not recoverable
from reading the code. Choices with one obvious answer are not recorded.

Records are immutable once accepted. A decision that changes gets a new record that
supersedes the old one, and the old one stays, because the reasoning that turned out to be
wrong is usually the more useful half.

| # | Decision | Status |
|---|---|---|
| [001](001-evaluate-the-hard-way.md) | Chronological split, leaky feature dropped, all four variants published | Accepted |
| [002](002-logistic-regression-over-boosting.md) | Logistic regression, not gradient boosting | Accepted |
| [003](003-train-only-on-servable-features.md) | Train only on features available at decision time | Accepted |

Portfolio-wide decisions live in
[`ai-engineering-portfolio/docs/adr/`](https://github.com/Adityansh-Chand/ai-engineering-portfolio/tree/main/docs/adr).
