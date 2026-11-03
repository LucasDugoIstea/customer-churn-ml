"""Tests de la API (contract tests).

TestClient levanta la app en memoria, con su lifespan: carga el modelo del
registry temporal igual que en produccion carga el de production.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client(registered_model):
    with TestClient(app) as c:
        yield c


def test_health(client, registered_model):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_version"] == registered_model["version"]


def test_predict_valido(client, sample_customer, registered_model):
    r = client.post("/predict", json=sample_customer)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"prediction", "probability", "risk_level", "model_version", "request_id"}
    assert body["prediction"] in ["CHURN", "NO_CHURN"]
    assert isinstance(body["probability"], float)
    assert 0 <= body["probability"] <= 1
    assert body["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert body["model_version"] == registered_model["version"]


def test_predict_acepta_totalcharges_vacio(client, sample_customer):
    """Clientes recien dados de alta: TotalCharges vacio lo imputa el pipeline."""
    sample_customer["TotalCharges"] = None
    assert client.post("/predict", json=sample_customer).status_code == 200


def test_cada_pedido_tiene_su_request_id(client, sample_customer):
    a = client.post("/predict", json=sample_customer).json()["request_id"]
    b = client.post("/predict", json=sample_customer).json()["request_id"]
    assert a != b


@pytest.mark.parametrize("cambio", [
    {"tenure": -1},                     # fuera de rango
    {"MonthlyCharges": "alto"},         # tipo incorrecto
    {"MonthlyCharges": 0},              # tiene que ser mayor a 0
    {"Contract": "Weekly"},             # categoria que el modelo no conoce
    {"InternetService": None},          # obligatorio
    {"SeniorCitizen": 2},               # solo 0 o 1
    {"customerID": "1234-ABCD"},        # campo que no es parte del contrato
])
def test_entrada_invalida_devuelve_422(client, sample_customer, cambio):
    sample_customer.update(cambio)
    r = client.post("/predict", json=sample_customer)
    assert r.status_code == 422


def test_falta_un_campo_devuelve_422(client, sample_customer):
    del sample_customer["tenure"]
    assert client.post("/predict", json=sample_customer).status_code == 422


def test_body_vacio_devuelve_422(client):
    assert client.post("/predict", json={}).status_code == 422


def test_swagger_documenta_los_endpoints(client):
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert "/predict" in paths and "/health" in paths
