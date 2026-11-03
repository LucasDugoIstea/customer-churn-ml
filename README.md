# customer-churn-ml

Trabajo integrador de Laboratorio de Minería de Datos (ISTEA, 2do cuatrimestre 2026).

Predicción de churn para una empresa de telecomunicaciones: con los datos de un
cliente el modelo da la probabilidad de que se vaya y un nivel de riesgo
(LOW, MEDIUM o HIGH). El dataset tiene 7043 clientes.

En la entrega 2 se agregó MLflow (runs y Model Registry), una API con FastAPI y
más tests. Va sin Docker, como se habló en clase; Docker queda para el final.

## Cómo correrlo

Con Python 3.13:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
dvc pull

pytest
python -m src.training.train
python -m scripts.promote_model
uvicorn app.main:app --port 8000
```

Los tests usan datos inventados y un MLflow temporal, así que andan sin el
dataset. El entrenamiento corre los 7 modelos de params.yaml y registra el mejor
como CustomerChurnModel con alias candidate; promote_model lo pasa a production.
La API carga el modelo que tenga el alias production. También se puede entrenar
con `dvc repro`.

Los runs se ven con `mlflow ui --backend-store-uri sqlite:///mlflow.db`. Cada run
guarda parámetros, métricas, matriz de confusión y como tags el commit y el md5
del dataset.

## API

GET /health, POST /predict y la doc en /docs. Ejemplo:

```
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":2,"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes","StreamingMovies":"Yes","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":95.5,"TotalCharges":190.4}'
```

Devuelve prediction, probability, risk_level, la versión del modelo y un
request_id. Si falta un campo o viene algo raro (tenure negativo, un Contract que
no existe, etc.) devuelve 422. TotalCharges puede venir vacío porque pasa con los
clientes nuevos.

## Resultados

Split 80/20 estratificado, umbral 0.30.

| run | recall | f1 | roc_auc |
|---|---|---|---|
| baseline_dummy | 0.000 | 0.000 | 0.500 |
| logistic_C0.1 | 0.696 | 0.592 | 0.812 |
| logistic_C1 | 0.696 | 0.594 | 0.812 |
| logistic_C10 | 0.696 | 0.594 | 0.812 |
| forest_depth6 | 0.685 | 0.574 | 0.801 |
| forest_depth8 | 0.683 | 0.575 | 0.802 |
| forest_depth12 | 0.669 | 0.570 | 0.796 |

Elijo por roc_auc. Las logísticas empatan, me quedo con C=0.1 que es la más
regularizada. Los random forest dan peor cuanto más profundos.

Bajé el umbral a 0.30 porque lo que más cuesta es no detectar a un cliente que
se va. Con 0.50 el recall era 0.44 (208 falsos negativos) y con 0.30 pasa a 0.70
(113). Probé 0.25 pero sumaba muchos falsos positivos para lo poco que ganaba.

Riesgo: LOW menos de 0.30, MEDIUM hasta 0.60, HIGH arriba de eso. En el test se
fueron el 12.5% de los LOW, el 43.8% de los MEDIUM y el 68.6% de los HIGH.

## Falta

- DagsHub (remote de DVC y MLflow remoto). Por ahora todo queda en local y el dvc pull no anda todavía.
- Docker para el final.
