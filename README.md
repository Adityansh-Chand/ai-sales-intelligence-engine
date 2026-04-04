# AI Sales Intelligence Engine

Predictive ML pipeline for lead scoring and revenue intelligence.

## Architecture

```mermaid
flowchart LR

CRM --> FeatureEngineering
FeatureEngineering --> Model
Model --> PredictionAPI
PredictionAPI --> Dashboard
PredictionAPI --> CRM
```
