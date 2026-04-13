# Configuración de rutas del proyecto
# Este archivo define las rutas estándar para mantener consistencia

# Rutas de modelos
MODEL_DIR = "models"
XGB_MODEL_PATH = f"{MODEL_DIR}/xgb_model.json"
NN_TEXT_MODEL_PATH = f"{MODEL_DIR}/nn_text_model"
META_NN_MODEL_PATH = f"{MODEL_DIR}/meta_nn_model"
SCALER_PATH = f"{MODEL_DIR}/scaler.pkl"
LABEL_ENCODERS_PATH = f"{MODEL_DIR}/label_encoders.pkl"
X_TAB_FEATURES_PATH = f"{MODEL_DIR}/X_tab_features.pkl"

# Rutas de datos
DATA_DIR = "data"
DATASET_PATH = f"{DATA_DIR}/dataset_reclamos_ia_ruidoso_extremo.xlsx"

# Rutas de scripts
SCRIPTS_DIR = "scripts"
REGENERATE_SCRIPT = f"{SCRIPTS_DIR}/regenerate_models_simple.py"

# Configuración de API
API_HOST = "0.0.0.0"
API_PORT = 5002
API_ENDPOINT = "/predict"