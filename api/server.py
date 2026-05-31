
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from monitoring.metrics import metrics
from models.explainability import explain
from models.scoring import predict, segment
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
    return {"status": "running"}


@app.get("/metrics")
def metrics_endpoint():
    return metrics.snapshot()


@app.get("/events", dependencies=[Depends(require_api_key)])
def events(limit: int = 20):
    return {"events": recent_events(limit=min(limit, 100))}


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
