# customer-churn-ml

Proyecto integrador de Laboratorio de Minería de Datos, ISTEA, 2do cuatrimestre 2026.

## Problema

Predicción de abandono de clientes (churn) en una empresa de telecomunicaciones.
Clasificación binaria supervisada sobre 7043 clientes históricos. La salida es la
probabilidad de que el cliente se vaya y un nivel de riesgo LOW / MEDIUM / HIGH.

## Instalación

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Para traer los datos hace falta DVC:

```
dvc pull
```

## Uso

Entrenar:

```
python -m src.training.train
```

Entrena los tres modelos, los compara y guarda el mejor en models/.

Salida:

```
INFO:__main__:Training started
INFO:src.data.load:Rows loaded: 7043
INFO:src.data.load:TotalCharges con faltantes: 26
INFO:src.data.load:Train: 5634 filas | Test: 1409 filas
INFO:__main__:Modelo candidato: logistic (roc_auc = 0.8120)
INFO:__main__:Model saved: models/churn_pipeline.joblib
INFO:__main__:Training finished
```

También se puede correr como pipeline de DVC, que solo reejecuta si cambió algo:

```
dvc repro
```

Tests:

```
pytest tests/ -q
```

## Estructura

```
customer-churn-ml/
├── data/
│   ├── raw/            dataset (versionado con DVC)
│   ├── processed/
│   └── reference/
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── config.py       rutas, semillas, columnas, umbrales
│   ├── data/           carga, validación y partición
│   ├── features/       ColumnTransformer y Pipeline
│   ├── training/       entrenamiento
│   ├── evaluation/     métricas
│   └── inference/      usar el modelo entrenado
├── app/                API (entrega 2)
├── monitoring/         observabilidad (entrega final)
├── models/             artefacto serializado
├── reports/            metrics.json
├── tests/
├── dvc.yaml
├── params.yaml
└── requirements.txt
```

## Qué se versiona

Código, configuración y metadatos en Git. Los datasets con DVC: en Git queda
el archivo .dvc con el md5 y el tamaño, el contenido va al remote.

El artefacto del modelo queda fuera de Git y lo produce el pipeline.

## Reproducibilidad

- random_state 42 en la partición y en los modelos
- partición estratificada 80/20
- versiones fijadas en requirements.txt
- el preprocesamiento vive dentro del Pipeline, así entrenamiento e inferencia
  aplican exactamente las mismas transformaciones
- parámetros declarados en params.yaml

Clonando el repo, haciendo dvc pull e instalando el requirements, el
entrenamiento da los mismos números.

## Resultados

Partición 80/20 estratificada, random_state 42.

```
modelo     accuracy  precision  recall  f1     roc_auc
baseline   0.736     0.000      0.000   0.000  0.500
logistic   0.794     0.663      0.449   0.535  0.812
forest     0.787     0.673      0.376   0.483  0.803
```

Elegí la regresión logística porque tiene el roc_auc más alto.

Algo que me llamó la atención: el baseline saca 0.736 de accuracy sin aprender
nada, solo diciendo siempre que el cliente se queda. Por eso no alcanza con
mirar accuracy.

## Algunas cosas del dataset

TotalCharges viene como texto porque tiene 26 celdas vacías. Lo convierto con
errors="coerce" así quedan como NaN y los imputa el pipeline.

customerID no lo uso como predictor, es un identificador.

Las versiones del requirements no son las mismas de la clase 3 porque con
Python 3.13 no andaban (no hay wheel y pip trata de compilar).

## Limitaciones y pendientes

- El remote de DVC en DagsHub no está configurado todavía, así que el dvc pull
  no funciona hasta que se agregue.
- Falta correr el notebook de EDA y escribir las conclusiones.
- El umbral de decisión quedó en 0.50, que es el valor por defecto. En el
  notebook está el análisis de cómo cambia el recall al moverlo, pero falta
  decidirlo y justificarlo.
- MLflow y el Model Registry todavía no están (se ven después del 22/09).
