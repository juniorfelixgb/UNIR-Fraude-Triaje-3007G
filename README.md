# Sistema de Triaje Inteligente para Detección de Fraude en Reclamaciones de Automóvil

Proyecto de maestría UNIR — Grupo 3007G. Sistema de detección de fraude en reclamaciones de seguros de vehículos de motor en República Dominicana, utilizando un modelo híbrido que combina análisis tabular (XGBoost) con procesamiento de lenguaje natural (DistilBERT) y un meta-learner de stacking.

## Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    DATOS DE ENTRADA                         │
│         Datos Tabulares  +  Descripción del Siniestro       │
│              ↓                        ↓                     │
│     ┌──────────────┐       ┌────────────────────┐          │
│     │   XGBoost    │       │ DistilBERT → NN    │          │
│     │  (Tabular)   │       │    (Texto NLP)     │          │
│     └──────┬───────┘       └────────┬───────────┘          │
│            │    P(fraude|tab)       │  P(fraude|texto)      │
│            └──────────┬────────────┘                        │
│                       ↓                                     │
│              ┌──────────────┐                               │
│              │ Meta-Learner │                               │
│              │  (Stacking)  │                               │
│              └──────┬───────┘                               │
│                     ↓                                       │
│            Probabilidad Final de Fraude                     │
└─────────────────────────────────────────────────────────────┘
```

## Estructura del Proyecto

```
UNIR-Fraude-Triaje-3007G/
├── ml/                          # Machine Learning
│   ├── notebooks/
│   │   ├── 01_GenerarDataset.ipynb    # Generación de dataset sintético (5K registros)
│   │   ├── 02_AnalizarDataset.ipynb   # Análisis lingüístico psicológico
│   │   ├── 03_ModeloHibrido.ipynb     # Pipeline completo: XGBoost + DistilBERT + Meta-Learner
│   │   └── 04_MVP_Triaje.ipynb        # Experimentación y MVP
│   ├── data/                          # Datasets (.xlsx)
│   ├── artifacts/                     # Modelos exportados (.json, .h5, .pkl)
│   └── visualizations/               # Gráficas generadas (.png)
│
├── api/                         # Backend (FastAPI) — en desarrollo
├── client/                      # Frontend (React) — en desarrollo
├── main.py                      # Entry point MLflow
├── pyproject.toml               # Dependencias del proyecto (uv)
└── .env                         # Variables de entorno (no se sube a git)
```

## Stack Tecnológico

| Componente | Tecnología |
|---|---|
| Datos Sintéticos | Python, Faker, OpenAI GPT-4o-mini |
| ML Tabular | XGBoost |
| NLP | DistilBERT (PyTorch), TensorFlow/Keras |
| Ensemble | Meta-Learner Keras (Stacking) |
| Tracking | MLflow |
| API | FastAPI |
| Frontend | React + TypeScript |
| Package Manager | uv |

## Configuración para Colaboradores

### Requisitos

- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (gestor de paquetes)

### Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/UNIR-Fraude-Triaje-3007G.git
cd UNIR-Fraude-Triaje-3007G

# 2. Instalar uv (si no lo tienes)
# Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Instalar dependencias de ML
uv sync --extra ml

# 4. Crear tu archivo .env con tu API key de OpenAI
cp .env.example .env
# Editar .env y colocar tu OPENAI_API_KEY
```

### Variables de Entorno

Crear un archivo `.env` en la raíz del proyecto:

```env
OPENAI_API_KEY=sk-tu-clave-aqui
```

### Ejecución de Notebooks

Los notebooks están diseñados para ejecutarse en orden:

1. **01_GenerarDataset** — Genera 5,000 registros sintéticos con patrones de fraude y texto generado por GPT-4o-mini
2. **02_AnalizarDataset** — Valida los marcadores lingüísticos psicológicos (superlativos, distanciamiento emocional, detalles sensoriales)
3. **03_ModeloHibrido** — Entrena el pipeline completo: XGBoost + DistilBERT NN + Meta-Learner, incluye UI interactiva con ipywidgets

Seleccionar el kernel **"UNIR Fraude (uv)"** en VS Code al abrir los notebooks.

### MLflow

```bash
# Levantar el servidor de tracking
uvx mlflow ui
# Abrir http://localhost:5000
```

### API (en desarrollo)

```bash
uv sync --extra api
uv run uvicorn api.main:app --reload
```

## Dataset Sintético

- **5,000 registros** con tasa de fraude del 10% y ruido del 15%
- **25 features**: demográficas, financieras, de póliza, vehiculares, texto narrativo
- **Marcadores de fraude**: montos altos (80-98% de cobertura), reportes tardíos, talleres sospechosos, pólizas recientes
- **Texto generado** con GPT-4o-mini usando marcadores lingüísticos de engaño comprobados (distanciamiento emocional, superlativos, vaguedad)

## Modelo Híbrido

- **XGBoost** (100 árboles, depth=6): análisis de datos estructurados
- **DistilBERT** multilingual + Red Neuronal (768→128→32→1): análisis semántico de texto
- **Meta-Learner** (2→8→1): combina predicciones de ambos modelos
- **Split estratificado** en 4 conjuntos (train/val/meta/test) para evitar data leakage

## Métricas

- F1-Score, AUC-ROC, Precision, Recall
- Confusion Matrix, Curva ROC
- Comparación XGBoost solo vs. Modelo Híbrido
