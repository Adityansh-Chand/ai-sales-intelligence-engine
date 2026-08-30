
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from monitoring.metrics import metrics
from models.explainability import explain
from models.scoring import model_metadata, predict, segment
from pipeline.accounts import account_count, known_account_ids, lookup
from pipeline.features import build_features, features_to_dict
from utils.security import request_id_middleware, require_api_key
from utils.storage import recent_events, save_event

app = FastAPI(title="AI Sales Intelligence Engine", version="1.0.0")
app.middleware("http")(request_id_middleware)


class Account(BaseModel):
    account_id: str = "unknown"
    visits: int = Field(0, ge=0)
    spend: float = Field(0, ge=0)
    account_age_days: int = Field(0, ge=0)
    usage_frequency: float = Field(0, ge=0)
    support_tickets: int = Field(0, ge=0)
    renewal_days: int = Field(365, ge=0)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    metrics.increment("http_errors_total")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "path": str(request.url.path)},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    metrics.increment("validation_errors_total")
    return JSONResponse(
        status_code=422,
        content={
            "error": "Invalid request",
            "details": exc.errors(),
            "path": str(request.url.path),
        },
    )


@app.exception_handler(Exception)
async def unexpected_exception_handler(request: Request, exc: Exception):
    metrics.increment("unhandled_errors_total")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "path": str(request.url.path)},
    )


@app.get("/")
def health():
    return {"status": "running"}


@app.get("/health")
def health_check():
    """Health plus the identity of the model actually loaded.

    Exposed so a reviewer can confirm from the API alone that a fitted artifact
    is being served, and that its training data was synthetic.
    """
    return {
        "status": "running",
        "model": model_metadata(),
        "accounts_indexed": account_count(),
        "example_account_ids": known_account_ids(),
    }


@app.get("/metrics")
def metrics_endpoint():
    return metrics.snapshot()


@app.get("/events", dependencies=[Depends(require_api_key)])
def events(limit: int = 20):
    return {"events": recent_events(limit=min(limit, 100))}


@app.get("/accounts/{account_id}/score", dependencies=[Depends(require_api_key)])
def score_known_account(account_id: str):
    """Score an account the service already holds features for.

    Exists so a caller that knows only an account id -- the customer operations
    service, for instance -- can ask for propensity without carrying feature
    vectors around. The features stay owned by the service that models them.
    """
    features, metadata = lookup(account_id)
    if features is None:
        raise HTTPException(status_code=404, detail=f"Unknown account {account_id}")

    metrics.increment("account_lookups_total")
    score = predict(features)
    return {
        **metadata,
        "score": score,
        "segment": segment(score),
        "features": features_to_dict(features),
        "explanation": explain(features),
        "data_source": "synthetic",
    }


@app.post("/score", dependencies=[Depends(require_api_key)])
def score_account(account: Account):
    metrics.increment("scores_total")
    features = build_features(account.model_dump())
    score = predict(features)
    result = {
        "account_id": account.account_id,
        "score": score,
        "segment": segment(score),
        "features": features_to_dict(features),
        "explanation": explain(features),
    }
    save_event("sales_score", result)
    return result
