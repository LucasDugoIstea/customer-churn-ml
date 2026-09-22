"""Configuracion del proyecto.

Todo lo que es una decision (semillas, rutas, umbrales, nombres de columnas)
vive aca y no disperso en el codigo. Si manana cambia el tamano del test o el
umbral de riesgo, se cambia en un solo lugar.
"""

from pathlib import Path

# --- Rutas -----------------------------------------------------------------
# BASE_DIR apunta a la raiz del proyecto (dos niveles arriba de este archivo)
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "customer_churn_historical.csv"
PROCESSED_DIR = DATA_DIR / "processed"

MODEL_DIR = BASE_DIR / "models"
MODEL_FILE = "churn_pipeline.joblib"
MODEL_PATH = MODEL_DIR / MODEL_FILE

REPORT_DIR = BASE_DIR / "reports"
METRICS_PATH = REPORT_DIR / "metrics.json"

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

# Columnas que el dataset tiene que traer si o si para poder entrenar
REQUIRED_COLUMNS = [ID_COL, TARGET] + NUMERIC_FEATURES + CATEGORICAL_FEATURES

# --- Particion -------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20

# --- Umbrales de riesgo ----------------------------------------------------
# La probabilidad que devuelve el modelo se traduce a LOW / MEDIUM / HIGH.
# Estos cortes son una decision del equipo y hay que poder justificarlos.
RISK_LOW_MAX = 0.35
RISK_MEDIUM_MAX = 0.65

# Umbral para pasar de probabilidad a etiqueta CHURN / NO_CHURN.
# Se deja configurable porque en este problema el falso negativo es mas caro
# que el falso positivo, asi que puede convenir bajarlo de 0.5.
DECISION_THRESHOLD = 0.50
