"""Contratos de entrada y salida de la API, con Pydantic.

Pydantic valida el pedido antes de que llegue al modelo: si no cumple el
contrato, FastAPI responde 422 y no se ejecuta la inferencia.

Se declaran las 19 features que usa el pipeline, con los mismos nombres que en
el dataset. Las categoricas aceptan solo los valores que vio el modelo al
entrenar (Literal); las numericas tienen rangos del dominio (Field).
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SiNo = Literal["Yes", "No"]
ConInternet = Literal["Yes", "No", "No internet service"]

EJEMPLO_CLIENTE = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.5,
    "TotalCharges": 190.4,
}


class CustomerFeatures(BaseModel):
    """Las caracteristicas de un cliente: el payload de POST /predict."""

    # Un campo que el modelo no conoce es un error del cliente, no se ignora.
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EJEMPLO_CLIENTE]})

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1] = Field(description="1 si el cliente es mayor de 65 años")
    Partner: SiNo
    Dependents: SiNo
    tenure: int = Field(ge=0, le=72, description="Meses como cliente (el dataset llega hasta 72)")
    PhoneService: SiNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: ConInternet
    OnlineBackup: ConInternet
    DeviceProtection: ConInternet
    TechSupport: ConInternet
    StreamingTV: ConInternet
    StreamingMovies: ConInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: SiNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(gt=0, description="Cargo mensual")
    # Puede venir vacio: pasa en el dataset con los clientes recien dados de
    # alta. El pipeline lo imputa con la mediana del entrenamiento.
    TotalCharges: float | None = Field(default=None, ge=0, description="Cargo acumulado")


class PredictionResponse(BaseModel):
    """Lo que devuelve POST /predict."""

    prediction: Literal["CHURN", "NO_CHURN"]
    probability: float = Field(ge=0, le=1, description="Probabilidad estimada de abandono")
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    model_version: str = Field(description="Version del Model Registry que respondio")
    request_id: str = Field(description="Identificador unico del pedido, para trazabilidad")


class HealthResponse(BaseModel):
    """Lo que devuelve GET /health."""

    status: Literal["ok"]
    model_name: str
    model_version: str
    model_uri: str
