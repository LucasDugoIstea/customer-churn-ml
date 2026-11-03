"""Fixtures compartidas por toda la suite.

Se usan datos sinteticos para que los tests corran rapido y sin haber hecho
dvc pull ni entrenado nada.

El modelo de los tests se entrena con esos datos y se registra en un MLflow
temporal, con el mismo codigo del entrenamiento real. Asi se prueba la carga
por alias sin tocar el mlflow.db del proyecto.
"""

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from src import config  # noqa: E402

CATEGORIAS = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": ["Electronic check", "Mailed check",
                      "Bank transfer (automatic)", "Credit card (automatic)"],
}

CLIENTE_VALIDO = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 79.9,
    "TotalCharges": 950.0,
}


def make_raw_df(n=200, seed=0):
    """Un dataset con la misma forma que el CSV crudo, incluidos sus problemas:
    TotalCharges como texto y con celdas en blanco."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({col: rng.choice(vals, n) for col, vals in CATEGORIAS.items()})
    df["customerID"] = ["C%04d" % i for i in range(n)]
    df["SeniorCitizen"] = rng.integers(0, 2, n)
    df["tenure"] = rng.integers(0, 73, n)
    df["MonthlyCharges"] = rng.uniform(18, 120, n).round(2)
    total = (df["tenure"] * df["MonthlyCharges"]).round(2).astype(str)
    total[df["tenure"] == 0] = " "
    df["TotalCharges"] = total
    # Que el target tenga algo de relacion con las features, y las dos clases
    churn = (df["Contract"] == "Month-to-month") & (rng.random(n) < 0.6)
    df["Churn"] = np.where(churn, "Yes", "No")
    return df


@pytest.fixture(scope="session")
def raw_csv(tmp_path_factory):
    """El dataset sintetico guardado como CSV, como el de data/raw/."""
    path = tmp_path_factory.mktemp("data") / "churn.csv"
    make_raw_df().to_csv(path, index=False)
    return path


@pytest.fixture
def raw_df(raw_csv):
    from src.data.load import load_data
    return load_data(raw_csv)


@pytest.fixture
def sample_customer():
    """Un cliente valido, como lo manda un CRM."""
    return dict(CLIENTE_VALIDO)


@pytest.fixture(scope="session")
def registered_model(tmp_path_factory):
    """Entrena con los datos sinteticos, registra y promueve a production en un
    MLflow temporal. Deja apuntadas las variables de entorno para que la API y
    load_model() usen ese registry."""
    from src.data.load import load_data, split_data
    from src.training.registry import promote, register_candidate, setup_mlflow
    from src.training.train import run_experiment

    carpeta = tmp_path_factory.mktemp("mlflow")
    # Se pisan las dos variables aunque esten en el .env: los tests nunca
    # tienen que tocar el MLflow real (ni el local ni DagsHub).
    uri = "models:/%s@production" % config.REGISTERED_MODEL
    viejos = {k: os.environ.get(k) for k in ("MLFLOW_TRACKING_URI", "MODEL_URI")}
    os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///" + (carpeta / "mlflow.db").as_posix()
    os.environ["MODEL_URI"] = uri

    path = carpeta / "churn.csv"
    make_raw_df().to_csv(path, index=False)
    X_train, X_test, y_train, y_test = split_data(load_data(path))

    setup_mlflow("tests", artifacts_dir=carpeta / "artifacts")
    spec = {"run_name": "test_logistic", "type": "logistic", "params": {"max_iter": 500}}
    run_id, metricas = run_experiment(spec, X_train, X_test, y_train, y_test)
    register_candidate(run_id, metricas)
    version = promote("production", "candidate", "modelo de prueba")

    yield {"uri": uri, "version": str(version)}

    for k, v in viejos.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture(scope="session")
def model(registered_model):
    from src.inference.model_loader import load_model
    return load_model(registered_model["uri"])
