"""Configuracion del proyecto.

Las rutas y el contrato de datos (nombres de columnas) viven aca. Los valores
que son una decision del equipo (semillas, umbrales, modelos a probar) se leen
de params.yaml, asi hay un solo lugar donde cambiarlos y DVC sabe cuando
reentrenar.

Las credenciales (DagsHub) NUNCA van aca: se leen de variables de entorno o del
archivo .env, que esta en el .gitignore.
"""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

# --- Rutas -----------------------------------------------------------------
# BASE_DIR apunta a la raiz del proyecto (dos niveles arriba de este archivo)
BASE_DIR = Path(__file__).resolve().parent.parent

# Si existe un .env en la raiz, carga sus variables (credenciales de DagsHub).
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "customer_churn_historical.csv"
RAW_DATA_DVC_FILE = DATA_DIR / "raw" / "customer_churn_historical.csv.dvc"
PROCESSED_DIR = DATA_DIR / "processed"

REPORT_DIR = BASE_DIR / "reports"
METRICS_PATH = REPORT_DIR / "metrics.json"

PARAMS_PATH = BASE_DIR / "params.yaml"

with open(PARAMS_PATH, encoding="utf-8") as f:
    PARAMS = yaml.safe_load(f)

# --- Contrato de datos -----------------------------------------------------
TARGET = "Churn"
ID_COL = "customerID"

# TotalCharges viene como texto y con blancos, por eso se trata aparte
NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columnas que el dataset tiene que traer si o si para poder entrenar
REQUIRED_COLUMNS = [ID_COL, TARGET] + FEATURES

# Version del contrato de entrada/salida. Si cambian las features o el formato
# de la respuesta, se sube este numero (y el modelo nuevo es "otro modelo").
SCHEMA_VERSION = "1"

# --- Particion -------------------------------------------------------------
RANDOM_STATE = PARAMS["split"]["random_state"]
TEST_SIZE = PARAMS["split"]["test_size"]

# --- Decision y riesgo -----------------------------------------------------
DECISION_THRESHOLD = PARAMS["decision"]["threshold"]
POSITIVE_CLASS = PARAMS["decision"]["positive_class"]
RISK_LOW_MAX = PARAMS["riesgo"]["low_max"]
RISK_MEDIUM_MAX = PARAMS["riesgo"]["medium_max"]

# --- MLflow ----------------------------------------------------------------
EXPERIMENT_NAME = PARAMS["mlflow"]["experiment_name"]
REGISTERED_MODEL = PARAMS["mlflow"]["registered_model"]

# Sin MLFLOW_TRACKING_URI en el entorno se usa un MLflow local: una base SQLite
# (mlflow.db) para runs y registry, y la carpeta mlartifacts/ para los modelos.
# Con las variables de DagsHub en el .env, todo va al servidor remoto.
LOCAL_TRACKING_URI = "sqlite:///" + (BASE_DIR / "mlflow.db").as_posix()
LOCAL_ARTIFACTS_DIR = BASE_DIR / "mlartifacts"

# La API carga el modelo por alias del registry, nunca por un path fijo.
DEFAULT_MODEL_ALIAS = "production"


def tracking_uri():
    """Servidor de MLflow a usar: el del entorno, o el local por defecto."""
    return os.getenv("MLFLOW_TRACKING_URI", LOCAL_TRACKING_URI)


def model_uri():
    """Referencia del modelo que carga la API: models:/<nombre>@<alias>."""
    return os.getenv(
        "MODEL_URI", "models:/%s@%s" % (REGISTERED_MODEL, DEFAULT_MODEL_ALIAS)
    )
