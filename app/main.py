"""API de prediccion de churn.

Levantar en desarrollo:

    uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

Endpoints:
    GET  /health   el servicio esta vivo y que version del modelo cargo
    POST /predict  probabilidad de abandono y nivel de riesgo de un cliente
    GET  /docs     documentacion Swagger generada por FastAPI
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import CustomerFeatures, HealthResponse, PredictionResponse
from src.inference.model_loader import load_model
from src.inference.predict import predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("churn_api")

# El modelo se carga una sola vez, al arrancar, y vive en memoria mientras la
# app atiende pedidos. Cargarlo en cada request seria leer de disco cada vez.
ml = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Si el modelo no carga, la app no arranca: se detecta al inicio y no con
    # el primer cliente.
    ml["model"] = load_model()
    logger.info("API lista con %s version %s", ml["model"].name, ml["model"].version)
    yield
    ml.clear()


app = FastAPI(
    title="Churn Prediction API",
    description="Estima la probabilidad de que un cliente de telecomunicaciones abandone el servicio.",
    version="2.0.0",
    lifespan=lifespan,
)


def to_dataframe(customer: CustomerFeatures) -> pd.DataFrame:
    """Del objeto validado por Pydantic a la tabla que espera el pipeline."""
    return pd.DataFrame([customer.model_dump()])


@app.get("/health", response_model=HealthResponse)
def health():
    modelo = ml["model"]
    return HealthResponse(
        status="ok",
        model_name=modelo.name,
        model_version=modelo.version,
        model_uri=modelo.uri,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerFeatures):
    request_id = str(uuid.uuid4())
    inicio = time.perf_counter()
    modelo = ml["model"]

    try:
        resultado = predict(modelo.pipeline, to_dataframe(customer))[0]
    except Exception:
        # Un error aca es del servidor (el contrato ya lo valido Pydantic).
        # Se loguea con el request_id para poder rastrearlo.
        logger.exception("request_id=%s error en la inferencia", request_id)
        raise HTTPException(
            status_code=500,
            detail={"error": "inference_failed", "request_id": request_id},
        )

    logger.info(
        "request_id=%s model_version=%s probability=%.4f risk=%s latency_ms=%.1f",
        request_id, modelo.version, resultado["probability"],
        resultado["risk_level"], (time.perf_counter() - inicio) * 1000,
    )
    return PredictionResponse(**resultado, model_version=modelo.version, request_id=request_id)
