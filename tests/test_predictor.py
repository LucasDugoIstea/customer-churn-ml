"""Contrato de prediccion y reglas de negocio.

No evaluan si la prediccion es buena (eso lo miden las metricas), sino que
respete el contrato. Por eso se testean rangos y tipos, no valores exactos.
"""

import pytest

from src import config
from src.inference.model_loader import load_model
from src.inference.predict import get_risk_level, predict


def test_predict_contract(sample_customer, model):
    result = predict(model.pipeline, sample_customer)[0]
    assert set(result) == {"prediction", "probability", "risk_level"}
    assert result["prediction"] in ["CHURN", "NO_CHURN"]
    assert isinstance(result["probability"], float)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["risk_level"] in ["LOW", "MEDIUM", "HIGH"]


def test_predict_varios_clientes(sample_customer, model):
    import pandas as pd
    df = pd.DataFrame([sample_customer] * 3)
    assert len(predict(model.pipeline, df)) == 3


def test_predict_falla_si_falta_una_columna(sample_customer, model):
    del sample_customer["Contract"]
    with pytest.raises(KeyError):
        predict(model.pipeline, sample_customer)


@pytest.mark.parametrize("probabilidad, nivel", [
    (0.0, "LOW"),
    (config.RISK_LOW_MAX - 0.001, "LOW"),
    (config.RISK_LOW_MAX, "MEDIUM"),
    (config.RISK_MEDIUM_MAX - 0.001, "MEDIUM"),
    (config.RISK_MEDIUM_MAX, "HIGH"),
    (1.0, "HIGH"),
])
def test_niveles_de_riesgo(probabilidad, nivel):
    assert get_risk_level(probabilidad) == nivel


def test_un_churn_nunca_sale_con_riesgo_low():
    """Coherencia entre el umbral de decision y los niveles de riesgo."""
    assert config.RISK_LOW_MAX <= config.DECISION_THRESHOLD < config.RISK_MEDIUM_MAX


def test_smoke_end_to_end(sample_customer, registered_model):
    """Arranca, carga, predice. No que prediga bien: que prediga."""
    modelo = load_model(registered_model["uri"])
    result = predict(modelo.pipeline, sample_customer)[0]
    assert result["prediction"] in ["CHURN", "NO_CHURN"]
    assert 0 <= result["probability"] <= 1
