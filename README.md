# customer-churn-ml

Proyecto integrador de Laboratorio de Minería de Datos, ISTEA, 2do cuatrimestre 2026.

## Problema

Predicción de abandono de clientes (churn) en una empresa de telecomunicaciones.
Clasificación binaria supervisada sobre 7043 clientes históricos. La salida es la
probabilidad de que el cliente se vaya y un nivel de riesgo LOW / MEDIUM / HIGH.

## Entrega 2

Esta entrega agrega:

- MLflow: los experimentos como runs y el modelo elegido en el Model Registry
- una API con FastAPI y Pydantic para pedir predicciones
- tests con pytest, incluidos los de la API

La entrego el 03/11 sin la parte de Docker, como se habló en clase. Docker y
Docker Compose quedan para el final.

## Instalación

Con Python 3.13:

```
python -m venv .venv
source .venv/bin/activate        (en Windows: .venv\Scripts\activate)
pip install -r requirements.txt
dvc pull
```

## Cómo se corre

```
pytest
python -m src.training.train
python -m scripts.promote_model
uvicorn app.main:app --port 8000
```

1. Los tests no necesitan los datos ni un modelo entrenado, se pueden correr
   apenas se clona el repo.
2. El entrenamiento corre los 7 modelos de params.yaml y registra el mejor
   en el Model Registry con el alias candidate.
3. promote_model pasa ese modelo a production.
4. La API carga el modelo que tenga el alias production.

También se puede entrenar con `dvc repro`, que solo reentrena si cambió algo.

## MLflow

Cada modelo de params.yaml es un run del experimento churn-classification. En
cada run guardo los parámetros, las métricas, la matriz de confusión, el esquema
de las features y el pipeline completo (preprocesamiento + modelo). Como tags van
el commit de Git y el md5 del dataset que da DVC, así se puede saber con qué
código y con qué datos salió cada modelo.

El mejor modelo se registra como CustomerChurnModel con el alias candidate. Pasarlo
a production lo hago aparte con promote_model, que lee la sección promocion de
params.yaml y deja anotado el motivo. Si hay que volver atrás, se pone el número
de la versión anterior en params.yaml y se corre de nuevo.

Para ver los runs:

```
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Sin configurar nada, MLflow guarda todo en local (mlflow.db y mlartifacts/). Para
usar DagsHub hay que completar las variables MLFLOW_TRACKING_URI,
MLFLOW_TRACKING_USERNAME y MLFLOW_TRACKING_PASSWORD en un .env (ver .env.example).

## API

- `GET /health`: si el servicio está andando y qué versión del modelo cargó
- `POST /predict`: recibe los datos de un cliente y devuelve la predicción
- `GET /docs`: la documentación de Swagger

Ejemplo:

```
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":2,"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes","StreamingMovies":"Yes","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":95.5,"TotalCharges":190.4}'
```

Respuesta:

```
{"prediction":"CHURN","probability":0.9001,"risk_level":"HIGH","model_version":"1","request_id":"69c97f25-4d73-4ac9-8a6c-5b509da3fd2a"}
```

El modelo se carga una sola vez cuando arranca la API, y se carga por el alias
del registry, no desde un archivo de models/. Si el modelo no carga, la API no
arranca.

El contrato de entrada está en app/schemas.py. Pide las 19 variables con los
mismos nombres del dataset. Las categóricas solo aceptan los valores que vio el
modelo al entrenar, tenure tiene que estar entre 0 y 72 y MonthlyCharges ser mayor
a 0. TotalCharges puede venir vacío, porque pasa con los clientes nuevos, y lo
imputa el pipeline. Si algo no cumple, la API devuelve 422 y dice qué campo falló.

## Tests

```
pytest
```

Hay 40 tests separados en datos, preprocesamiento, carga del modelo, predicción y
API. Para que corran rápido y sin depender del dataset usan datos inventados con
la misma forma que el CSV, y un MLflow temporal donde entrenan y registran un
modelo chico.

Prueban cosas como que el pipeline aguante faltantes y categorías nuevas, que el
modelo cargue por alias, que la respuesta tenga los campos y tipos correctos, y
que la API devuelva 422 con datos inválidos (tenure negativo, un texto donde va un
número, un Contract que no existe, un campo de más, etc.).

## Resultados

Partición 80/20 estratificada, random_state 42, umbral 0.30.

```
run              accuracy  precision  recall  f1     roc_auc
baseline_dummy   0.736     0.000      0.000   0.000  0.500
logistic_C0.1    0.747     0.515      0.696   0.592  0.812
logistic_C1      0.749     0.518      0.696   0.594  0.812
logistic_C10     0.749     0.518      0.696   0.594  0.812
forest_depth6    0.732     0.494      0.685   0.574  0.801
forest_depth8    0.734     0.497      0.683   0.575  0.802
forest_depth12   0.734     0.497      0.669   0.570  0.796
```

Elijo por roc_auc porque mide qué tan bien ordena el modelo a los clientes sin
depender del umbral. Las tres regresiones logísticas empatan y le ganan a los
random forest, así que me quedo con la de C=0.1, que es la más regularizada. Los
random forest empeoran cuanto más profundos, parece overfitting.

## Umbral

El error que más cuesta es el falso negativo: decir que un cliente se queda cuando
en realidad se va, porque ahí se pierde sin haber intentado retenerlo. Por eso bajé
el umbral de 0.50 a 0.30. Con el modelo elegido, en el test:

```
umbral  falsos negativos  falsos positivos  recall
0.50    208               86                0.441
0.40    152               158               0.591
0.30    113               244               0.696
0.25    94                302               0.747
```

Con 0.30 detecta el 70% de los que se van en vez del 44%, y es el umbral con mejor
f1 de estos. Bajarlo a 0.25 ya no conviene: evita 19 falsos negativos pero suma 58
falsos positivos.

## Niveles de riesgo

- LOW: menos de 0.30
- MEDIUM: de 0.30 a 0.60
- HIGH: 0.60 o más

LOW termina donde empieza la predicción CHURN, así nunca sale un CHURN con riesgo
LOW. En el test se fueron el 12.5% de los LOW, el 43.8% de los MEDIUM y el 68.6%
de los HIGH (el promedio es 26.4%).

## Estructura

```
customer-churn-ml/
├── app/                main.py (API) y schemas.py (contratos)
├── data/raw/           dataset (versionado con DVC)
├── notebooks/
├── scripts/            promote_model.py
├── src/
│   ├── config.py       rutas, columnas y lectura de params.yaml
│   ├── data/           carga, validación y partición
│   ├── features/       ColumnTransformer y Pipeline
│   ├── training/       entrenamiento y Model Registry
│   ├── evaluation/     métricas
│   └── inference/      carga del modelo y predicción
├── tests/
├── reports/            metrics.json
├── dvc.yaml
├── params.yaml
└── requirements.txt
```

## Pendientes

- Configurar DagsHub (el remote de DVC y MLflow remoto). Por ahora los runs están
  en local y el dvc pull no funciona.
- Docker y Docker Compose, para el final.
- Al entrenar la regresión logística aparece un aviso de SciPy
  (Unknown solver options: iprint). No cambia los resultados.
