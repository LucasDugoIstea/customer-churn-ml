# customer-churn-ml

Proyecto integrador de Laboratorio de Minería de Datos, ISTEA, 2do cuatrimestre 2026.

Predicción de abandono de clientes (churn) en una empresa de telecomunicaciones.
El dataset histórico tiene 7043 clientes y la idea es estimar la probabilidad de
que cada uno se vaya.

Esta es la primera entrega: pasar el notebook a un proyecto Python que se pueda
volver a correr.

## Instalación

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

Para entrenar:

```
python -m src.training.train
```

Entrena los tres modelos, los compara y guarda el mejor en models/.

Salida:

```
Training started
Rows loaded: 7043
Train: 5634 filas | Test: 1409 filas
Modelo candidato: logistic (roc_auc = 0.8120)
Model saved: models/churn_pipeline.joblib
```

Para ver qué pasa si se mueve el umbral de decisión:

```
python -m src.evaluation.threshold
```

Los tests:

```
pytest tests/ -q
```

## Carpetas

- data/raw: el dataset
- notebooks: el EDA
- src/config.py: rutas, semillas, nombres de columnas, umbrales
- src/data: carga y partición
- src/features: el preprocesamiento
- src/training: entrenamiento
- src/evaluation: métricas
- src/inference: usar el modelo ya entrenado
- models: el artefacto
- tests

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

## Falta

- correr el notebook y escribir las conclusiones
- decidir el umbral
- DVC y DagsHub
- MLflow
