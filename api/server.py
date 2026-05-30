
from fastapi import FastAPI
from pydantic import BaseModel, Field

from models.explainability import explain
from models.scoring import predict, segment
from pipeline.features import build_features, features_to_dict

app = FastAPI()


class Account(BaseModel):
    account_id: str = "unknown"
    visits: int = Field(0, ge=0)
    spend: float = Field(0, ge=0)
    account_age_days: int = Field(0, ge=0)
    usage_frequency: float = Field(0, ge=0)
    support_tickets: int = Field(0, ge=0)
    renewal_days: int = Field(365, ge=0)


@app.get("/")
def health():
    return {"status": "running"}


@app.get("/health")
def health_check():
    return {"status": "running"}


@app.post("/score")
def score_account(account: Account):
    features = build_features(account.model_dump())
    score = predict(features)
    return {
        "account_id": account.account_id,
        "score": score,
        "segment": segment(score),
        "features": features_to_dict(features),
        "explanation": explain(features),
    }
