# Customer Churn — Proyecto Integrador

Laboratorio de Minería de Datos · ISTEA · Segundo cuatrimestre 2026

Sistema de predicción de abandono de clientes para una empresa de
telecomunicaciones. El objetivo del proyecto no es solamente entrenar un
modelo, sino llevarlo desde un notebook hasta un servicio que se pueda
reproducir, desplegar y operar.

---

## Estado actual

**Entrega 1 (22/09) — proyecto reproducible.**

| Pieza | Estado |
|---|---|
| Estructura de proyecto | ✅ |
| Carga y validación de datos | ✅ |
| Partición reproducible y estratificada | ✅ |
| Pipeline de preprocesamiento (scikit-learn) | ✅ |
| Baseline + modelo lineal + modelo de árboles | ✅ |
| Métricas y matriz de confusión | ✅ |
| Artefacto serializado con joblib | ✅ |
| Entrenamiento ejecutable por consola | ✅ |
| Tests básicos | ✅ |
| EDA | ⏳ notebook listo, falta correrlo y escribir conclusiones |
| DVC + DagsHub | ⏳ pendiente |
| MLflow + Model Registry | ⏳ pendiente (se ve en clase después del 22/09) |

---

## Cómo ejecutarlo

```bash
# 1. Crear el entorno
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / Mac

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Entrenar
python -m src.training.train

# 4. Ver cómo cambia el resultado según el umbral de decisión
python -m src.evaluation.threshold

# 5. Correr los tests
pytest tests/ -q
```

Salida esperada del entrenamiento:

```
Training started
Rows loaded: 7043
TotalCharges con faltantes: 26
Train: 5634 filas | Test: 1409 filas
...
Modelo candidato: logistic (roc_auc = 0.8120)
Model saved: models/churn_pipeline.joblib
```

---

## Estructura

```
customer-churn-ml/
├── data/
│   ├── raw/                  # dataset histórico (va a DVC, no a Git)
│   └── processed/
├── notebooks/
│   └── 01_eda.ipynb          # exploración
├── src/
│   ├── config.py             # constantes, rutas, semillas, umbrales
│   ├── data/load.py          # carga, validación y partición
│   ├── features/build.py     # ColumnTransformer y Pipeline
│   ├── training/train.py     # entrenamiento y comparación de modelos
│   ├── evaluation/
│   │   ├── metrics.py        # cálculo y guardado de métricas
│   │   └── threshold.py      # análisis del umbral de decisión
│   └── inference/predict.py  # carga del artefacto y predicción
├── models/                   # artefactos serializados
├── reports/                  # metrics.json
├── tests/
├── params.yaml
├── requirements.txt
└── README.md
```

Cada módulo de `src/` tiene una sola responsabilidad: si una función carga
datos, no entrena; si una función predice, no decide cómo se entrenó.

---

## Decisiones técnicas

**`TotalCharges` se convierte con `errors="coerce"`.** La columna viene como
texto porque tiene 26 celdas en blanco. Esos valores quedan como `NaN` y los
imputa el pipeline. No se edita el CSV: la limpieza manual arregla un archivo,
el pipeline arregla el proceso.

**`customerID` no se usa como predictor.** Es un identificador. El README del
dataset lo pide explícitamente.

**La partición usa `stratify`.** Las clases están desbalanceadas (~73 % de
clientes que no se van), así que sin estratificar el test podría quedar con una
proporción distinta a la del train.

**El preprocesamiento vive dentro del `Pipeline`.** Se serializa el pipeline
entero, no solo el estimador. Si se guardara solo el modelo, en inferencia
habría que reconstruir a mano la imputación, el encoding y el escalado, y
cualquier diferencia equivaldría a predecir con otro modelo.

**`handle_unknown="ignore"` en el OneHotEncoder.** Una categoría que no estaba
en el entrenamiento no debería tirar abajo el servicio. Hay un test que cubre
justamente ese caso.

**Las versiones del `requirements.txt` son más altas que las de la clase 3.**
Las que figuran en la presentación (pandas 2.2.2, numpy 1.26.4, scikit-learn
1.5.1) no publican wheel para Python 3.13 y pip intenta compilarlas desde
fuente. Se subió cada paquete a la primera versión que sí lo publica.

---

## Resultados

Partición 80/20 estratificada, `random_state=42`.

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| baseline (DummyClassifier) | 0,736 | 0,000 | 0,000 | 0,000 | 0,500 |
| logistic (LogisticRegression) | **0,794** | 0,663 | **0,449** | **0,535** | **0,812** |
| forest (RandomForestClassifier) | 0,787 | **0,673** | 0,376 | 0,483 | 0,803 |

**Modelo candidato: regresión logística**, seleccionado por ROC-AUC.

### Por qué accuracy no alcanza

El baseline predice siempre "no se va" y aun así llega a **73,6 % de
accuracy**. No aprendió nada: simplemente hay más clientes que se quedan. Esa
sola fila explica por qué la consigna no acepta accuracy como métrica única.

### El costo del falso negativo

Un falso negativo es decir que un cliente se queda cuando en realidad se iba:
el cliente se va **y ni siquiera se intentó retenerlo**. Un falso positivo, en
cambio, es ofrecerle una promoción a alguien que se iba a quedar igual: cuesta
el descuento, pero no pierde al cliente.

Si retener sale más barato que perder, conviene un modelo con **más recall**,
aunque baje la precisión.

Con el umbral por defecto de 0,50 el recall es 0,449: de 372 clientes que se
iban, **se detectan 167 y se escapan 205**. Moviendo el umbral:

| Umbral | Precision | Recall | F1 | Falsos neg. | Falsos pos. |
|---|---|---|---|---|---|
| 0,30 | 0,518 | 0,696 | **0,594** | 113 | 241 |
| 0,35 | 0,551 | 0,642 | 0,593 | 133 | 195 |
| 0,40 | 0,580 | 0,597 | 0,588 | 150 | 161 |
| 0,50 | 0,663 | 0,449 | 0,535 | 205 | 85 |

Bajando a **0,30** se recuperan 92 clientes que antes se escapaban, a cambio de
156 falsas alarmas más. El F1 también es más alto ahí.

> ⏳ **Pendiente:** fijar el umbral definitivo en `config.py` y justificarlo.
> La tabla da el dato; la decisión depende de cuánto cuesta cada error.

---

## Lo que falta

- [ ] Correr el notebook de EDA y escribir las conclusiones
- [ ] Decidir y justificar el umbral de decisión
- [ ] `dvc init` + trackear el dataset + remote en DagsHub
- [ ] MLflow: registrar los runs y subir el candidato al Model Registry
- [ ] Tag `entrega-1` sobre el commit presentado
