# 🚗 Sistema de Triaje Inteligente para Detección de Fraudes en Reclamaciones de Vehículos

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Keras](https://img.shields.io/badge/Keras-3.0+-red.svg)](https://keras.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-latest-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Sistema de Detección de Fraude Híbrido que combina Machine Learning tradicional y Deep Learning para identificar reclamaciones fraudulentas en seguros de automóviles.

## ✨ Características Principales

- **🤖 Modelo Híbrido**: XGBoost + DistilBERT + Meta-Learner
- **📊 Análisis Tabular**: Variables estructuradas con XGBoost
- **📝 Procesamiento de Texto**: Análisis de descripciones con DistilBERT
- **🎯 Meta-Learning**: Combinación inteligente de predicciones
- **🚀 API REST**: Servicio web listo para producción
- **🧪 Testing**: Scripts de validación incluidos

## 📁 Estructura del Proyecto

```
UNIR-Fraude-Triaje-3007G/
├── 📁 models/                           # Modelos entrenados y preprocesadores
│   ├── xgb_model.json                  # Modelo XGBoost
│   ├── nn_text_model.keras             # Modelo de texto (Keras 3 - .keras format)
│   ├── meta_nn_model.keras             # Meta-learner (Keras 3 - .keras format)
│   ├── scaler.pkl                      # Escalador StandardScaler
│   ├── label_encoders.pkl              # Codificadores categóricos
│   └── X_tab_features.pkl              # Lista de características tabulares
├── 📁 data/                             # Datasets
│   └── dataset_reclamos_ia_ruidoso_extremo.xlsx
├── 📁 notebooks/                        # Notebooks Jupyter
│   ├── Sistema_Triaje_Inteligente_Reclamaciones_Automovil.ipynb (PRINCIPAL)
│   ├── AnalizarDataSetSintetico.ipynb
│   ├── GenerandoDatasetSinteticoV1.ipynb
│   └── MVP_Triaje_Fraude.ipynb
├── 📁 scripts/                          # Scripts legacy (opcionales)
│   ├── regenerate_models_simple.py
│   └── regenerate_models.py
├── 📁 docs/                             # Documentación
│   └── dashboard_marcadores_fraude_completo.png
├── 🌐 api_server.py                    # API REST con Flask
├── 🧪 test_api.py                      # Script de pruebas
├── 🔧 check_project.py                 # Validador del proyecto
├── ⚙️ config.py                        # Configuración de paths
├── 📋 Fraude_Detection_API.postman_collection.json
├── 📦 requirements.txt                 # Dependencias
├── 📖 README.md                        # Este archivo
└── 🔒 .env                             # Variables de entorno (opcional)
```

## 🚀 Guía Rápida de Inicio

### 1️⃣ Verificar Instalación del Proyecto
```bash
python check_project.py
```

### 2️⃣ Generar Modelos (Ejecutar Notebook)
**El notebook genera automáticamente todos los modelos necesarios:**

```bash
# Abrir el notebook en Jupyter
jupyter notebook notebooks/Sistema_Triaje_Inteligente_Reclamaciones_Automovil.ipynb

# Ejecutar todas las celdas (Kernel -> Run All)
# Esto entrenará los modelos y los guardará en formato .keras en la carpeta models/
```

> ✅ **Importante**: El notebook solo necesita ejecutarse UNA VEZ. Después, la API cargará los modelos automáticamente.

### 3️⃣ Iniciar la API
```bash
# En una terminal
python api_server.py

# La API estará disponible en http://localhost:5002
```

### 4️⃣ Probar la API (en otra terminal)
```bash
python test_api.py
```

---

## 📋 Workflow Completo (End-to-End)

```
1. PREPARACIÓN
   └─ Verificar Python 3.11+ instalado
   └─ pip install -r requirements.txt

2. ENTRENAMIENTO (Una sola vez)
   └─ jupyter notebook notebooks/Sistema_Triaje_Inteligente_Reclamaciones_Automovil.ipynb
   └─ Kernel → Run All
   └─ Esperar a que terminen todas las celdas (incluida la exportación)
   └─ Verificar que se crearon: models/nn_text_model.keras y meta_nn_model.keras

3. SERVICIO EN PRODUCCIÓN
   ├─ Terminal 1: python api_server.py
   └─ API lista en http://localhost:5002

4. PRUEBAS
   ├─ Terminal 2: python test_api.py
   └─ Verificar respuestas en terminal 1

5. MANTENIMIENTO (Opcional)
   └─ Para reentrenar con nuevos datos, ejecutar el notebook nuevamente
```

---

## 🧪 Testing con Postman

**Importar colección:**
1. Abrir Postman
2. Click en "Import"
3. Seleccionar: `Fraude_Detection_API.postman_collection.json`
4. Ejecutar requests pre-configurados

**Requests disponibles:**
- ✅ **Legítimo**: Reclamación normal de accidente
- ⚠️ **Fraudulento**: Patrón sospechoso de fraude

---

## 🔧 Uso de la API

### Endpoint Principal
```
POST http://localhost:5002/predict
Content-Type: application/json
```

### Ejemplo de Solicitud (JSON)
```json
{
  "Customer_Age": 35,
  "Gender": "M",
  "Insured_MaritalStatus": "Casado",
  "Insured_Occupation": "Empleado",
  "Insured_Zip": 28001,
  "Insured_Inception_Date": "2018-01-01",
  "Policy_Start_Date": "2023-01-01",
  "Last_Purchase_History_Date": "2023-01-01",
  "Coverage_description": "Completa",
  "Coverage_Amount": 50000,
  "Premium_Amount": 1200,
  "Beneficiary_Type_Description": "Asegurado",
  "Claim_History_Count_This_Policy": 0,
  "Claim_Frequency_Last_12_Month": 0,
  "Vehicle_Make": "Toyota",
  "Vehicle_Model": "Corolla",
  "Model_Year": 2020,
  "Incident_Date": "2023-06-15",
  "Date_Reported": "2023-06-16",
  "Claim_Amount": 3000,
  "LossType_Description": "Pérdida Parcial",
  "Branch_Description": "Centro",
  "WorkShop_Name": "Taller Autorizado A",
  "Claim_Description": "Tuve un accidente en la autopista cuando un vehículo me cerró el paso."
}
```

### Ejemplo de Respuesta (JSON)
```json
{
  "fraud_probability": 0.1866,
  "xgb_probability": 0.1523,
  "nlp_probability": 0.2210,
  "prediction": "Legítimo"
}
```

### Con curl
```bash
curl -X POST http://localhost:5002/predict \
  -H "Content-Type: application/json" \
  -d @payload.json
```

---

## 🔧 Solución de Problemas

### ❌ Error: "Archivos de modelo no encontrados"
**Causa**: El notebook no ha sido ejecutado completamente.

**Solución**:
```bash
# Ejecutar notebook completo en Jupyter
jupyter notebook notebooks/Sistema_Triaje_Inteligente_Reclamaciones_Automovil.ipynb
# Seleccionar: Kernel -> Run All
# Esperar a que finalice (incluyendo la exportación de modelos)
```

### ❌ Error: "File format not supported" (Keras 3)
**Causa**: Modelos en formato antiguo (SavedModel) en lugar de `.keras`.

**Solución**:
```bash
# Eliminar modelos antiguos
rm -rf models/nn_text_model models/meta_nn_model
rm -f models/*.h5 models/*.json

# Regenerar ejecutando el notebook
jupyter notebook notebooks/Sistema_Triaje_Inteligente_Reclamaciones_Automovil.ipynb
# Kernel -> Run All
```

### ❌ Error: "ModuleNotFoundError" o dependencias faltantes
**Solución**:
```bash
# Reinstalar dependencias
pip install -r requirements.txt --upgrade

# Si hay conflicto con Keras 3
pip install --upgrade tensorflow keras
```

### ❌ Error: "Puerto 5002 ocupado"
**Solución**:
```bash
# Matar proceso en puerto 5002
lsof -ti:5002 | xargs kill -9

# O cambiar puerto en api_server.py (línea ~200)
# app.run(host='0.0.0.0', port=5003)  # Cambiar puerto
```


---

## 📊 Arquitectura del Modelo (Stacking Ensemble)

```
┌─────────────────────────────────────────────────────────────────┐
│                      DATOS DE ENTRADA                           │
├─────────────────────────────────────────────────────────────────┤
│  • Variables Tabulares (20+ features)                           │
│  • Descripción de Siniestro (Texto libre)                       │
└──────────────┬──────────────────────────────────────────────────┘
               │
        ┌──────┴──────┐
        │             │
┌───────▼────────┐  ┌─▼──────────────┐
│  🤖 XGBoost    │  │ 🧠 DistilBERT  │
│ (Tabular)      │  │ + Keras NN     │
│ 100 trees      │  │ (Text)         │
│ max_depth=6    │  │ 768→128→32→1   │
└───────┬────────┘  └─┬──────────────┘
        │             │
        │ P(fraud)    │ P(fraud)
        │             │
        └──────┬──────┘
               │
        ┌──────▼────────────┐
        │ Stack: [p1, p2]   │
        │ Meta-Learner (NN).│
        │ 2→8→1             │
        └──────┬────────────┘
               │
        ┌──────▼─────────────────┐
        │ 📊 Final Prediction    │
        │ P(Fraude) + Details    │
        └────────────────────────┘
```

**Ventajas del Enfoque Híbrido:**
- ✅ XGBoost captura patrones tabular-numéricos
- ✅ DistilBERT extrae contexto semántico del texto
- ✅ Meta-Learner combina ambas perspectivas optimalmente
- ✅ Reduce riesgo de overfitting con validación estratificada

---

## 📈 Métricas de Rendimiento
- **F1-Score**: >~90%
- **AUC-ROC**: ~99%

---

## 🛠️ Tecnologías Utilizadas

| Componente | Descripción | Versión |
|-----------|-----------|---------|
| **Python** | Lenguaje principal | 3.11+ |
| **Keras** | Framework de Deep Learning | 3.0+ |
| **TensorFlow** | Backend para Keras | 2.15+ |
| **XGBoost** | Gradient Boosting | Latest |
| **Transformers** | DistilBERT para embeddings | Latest |
| **PyTorch** | Procesamiento de texto | Latest |
| **Flask** | API REST | Latest |
| **scikit-learn** | Preprocesamiento | Latest |
| **pandas** | Manipulación de datos | Latest |
| **NumPy** | Computación numérica | Latest |

---

## 📝 Desarrollo y Contribución

### Configuración para Desarrollo
```bash
# Instalar dependencias de desarrollo
pip install -r requirements-dev.txt


---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver archivo `LICENSE` para más detalles.

---

## 👥 Autores y Créditos

- **Ariel Bonifacio Mejia** - Desarrollo principal
- **Juan Rubén Marrero Vizcaíno** - Desarrollo principal
- **Junior Felix Gervacio Burgos** - Desarrollo principal
- **Jack Edwards Zenozain Flores** - Desarrollo principal
- **Universidad Internacional de La Rioja (UNIR)** - Institución académica
- **Seminario de Innovación en Inteligencia Artificial** - Contexto del proyecto

---
**Última actualización**: Abril 2026
