from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import xgboost as xgb
from tensorflow.keras.models import load_model
import joblib
import pickle
from transformers import AutoTokenizer, AutoModel
import torch
import gc
import os

# Wrapper para LabelEncoder que maneja valores desconocidos
class RobustLabelEncoder:
    def __init__(self):
        self.value_to_code = {}
        self.default_code = 0
    
    def transform(self, y):
        # Convertir a string y mapear usando el diccionario (maneja desconocidos)
        y_str = np.array([str(v) for v in y])
        return np.array([self.value_to_code.get(v, self.default_code) for v in y_str])

# Verificar que los archivos de modelos existen
required_files = ['models/xgb_model.json', 'models/scaler.pkl', 'models/label_encoders.pkl', 'models/X_tab_features.pkl']

# Verificar archivos de modelos Keras (.keras o carpetas SavedModel)
nn_text_patterns = ['models/nn_text_model.keras', 'models/nn_text_model']
meta_nn_patterns = ['models/meta_nn_model.keras', 'models/meta_nn_model']

nn_text_found = any(os.path.exists(p) for p in nn_text_patterns)
meta_nn_found = any(os.path.exists(p) for p in meta_nn_patterns)

missing_files = [f for f in required_files if not os.path.exists(f)]
if not nn_text_found:
    missing_files.append('models/nn_text_model (o .keras)')
if not meta_nn_found:
    missing_files.append('models/meta_nn_model (o .keras)')

if missing_files:
    print(f"Error: Los siguientes archivos de modelo no se encontraron: {missing_files}")
    print("Ejecuta el notebook completo primero para generar los modelos.")
    exit(1)

# Cargar modelos y preprocesadores
print("Cargando modelos...")
xgb_model_loaded = xgb.XGBClassifier()
xgb_model_loaded.load_model('models/xgb_model.json')

# Cargar modelo de texto: Intentar primero .keras, luego SavedModel
try:
    if os.path.exists('models/nn_text_model.keras'):
        from tensorflow.keras.models import load_model as keras_load_model
        nn_text_model_loaded = keras_load_model('models/nn_text_model.keras', compile=False)
    else:
        # Cargar SavedModel usando TFSMLayer
        import tensorflow as tf
        from tensorflow.keras.layers import TFSMLayer
        from tensorflow.keras.models import Model
        tfsm_layer = TFSMLayer('models/nn_text_model', call_endpoint='serving_default')
        input_tensor = tf.keras.Input(shape=(768,), name='text_embedding_input')
        output_tensor = tfsm_layer(input_tensor)
        nn_text_model_loaded = Model(inputs=input_tensor, outputs=output_tensor)
    # Recompilar con configuración estándar
    nn_text_model_loaded.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
except Exception as e:
    print(f"Error cargando nn_text_model: {e}")
    print("Intenta recrear los modelos ejecutando el notebook completo.")
    exit(1)

# Cargar meta-learner: Intentar primero .keras, luego SavedModel
try:
    if os.path.exists('models/meta_nn_model.keras'):
        from tensorflow.keras.models import load_model as keras_load_model
        meta_nn_loaded = keras_load_model('models/meta_nn_model.keras', compile=False)
    else:
        # Cargar SavedModel usando TFSMLayer
        from tensorflow.keras.layers import TFSMLayer
        from tensorflow.keras.models import Model
        import tensorflow as tf
        tfsm_layer = TFSMLayer('models/meta_nn_model', call_endpoint='serving_default')
        input_tensor = tf.keras.Input(shape=(2,), name='meta_input')
        output_tensor = tfsm_layer(input_tensor)
        meta_nn_loaded = Model(inputs=input_tensor, outputs=output_tensor)
    # Recompilar con configuración estándar
    meta_nn_loaded.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
except Exception as e:
    print(f"Error cargando meta_nn_model: {e}")
    print("Intenta recrear los modelos ejecutando el notebook completo.")
    exit(1)

scaler_loaded = joblib.load('models/scaler.pkl')
with open('models/label_encoders.pkl', 'rb') as f:
    label_encoders_loaded = pickle.load(f)
with open('models/X_tab_features.pkl', 'rb') as f:
    X_tab_features_loaded = pickle.load(f)

# Cargar DistilBERT para extracción de embeddings
tokenizer_loaded = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
distilbert_model_loaded = AutoModel.from_pretrained("distilbert-base-multilingual-cased")
distilbert_model_loaded.eval()

# Detectar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device("mps")
distilbert_model_loaded.to(device)

def extract_embeddings_api(text_list, batch_size=16):
    all_embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i+batch_size]
        inputs = tokenizer_loaded(batch, padding=True, truncation=True, max_length=120, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = distilbert_model_loaded(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_embeddings)
        del inputs, outputs, cls_embeddings
        try:
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            elif device.type == 'mps':
                if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
        except:
            pass
        gc.collect()
    return np.vstack(all_embeddings)

# Crear aplicación Flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extraer datos del JSON
        input_data = {
            'Customer_Age': data.get('Customer_Age', 0),
            'Gender': data.get('Gender', 'M'),
            'Insured_MaritalStatus': data.get('Insured_MaritalStatus', 'Soltero'),
            'Insured_Occupation': data.get('Insured_Occupation', 'Empleado'),
            'Insured_Zip': data.get('Insured_Zip', 0),
            'Insured_Inception_Date': data.get('Insured_Inception_Date', '2020-01-01'),
            'Policy_Start_Date': data.get('Policy_Start_Date', '2023-01-01'),
            'Last_Purchase_History_Date': data.get('Last_Purchase_History_Date', '2023-01-01'),
            'Coverage_description': data.get('Coverage_description', 'Básica'),
            'Coverage_Amount': data.get('Coverage_Amount', 10000),
            'Premium_Amount': data.get('Premium_Amount', 500),
            'Beneficiary_Type_Description': data.get('Beneficiary_Type_Description', 'Asegurado'),
            'Claim_History_Count_This_Policy': data.get('Claim_History_Count_This_Policy', 0),
            'Claim_Frequency_Last_12_Month': data.get('Claim_Frequency_Last_12_Month', 0),
            'Vehicle_Make': data.get('Vehicle_Make', 'Toyota'),
            'Vehicle_Model': data.get('Vehicle_Model', 'Corolla'),
            'Model_Year': data.get('Model_Year', 2020),
            'Incident_Date': data.get('Incident_Date', '2023-06-01'),
            'Date_Reported': data.get('Date_Reported', '2023-06-02'),
            'Claim_Amount': data.get('Claim_Amount', 5000),
            'LossType_Description': data.get('LossType_Description', 'Pérdida Parcial'),
            'Branch_Description': data.get('Branch_Description', 'Centro'),
            'WorkShop_Name': data.get('WorkShop_Name', 'Taller A'),
            'Claim_Description': data.get('Claim_Description', 'Accidente en carretera.')
        }

        # Preprocesar
        input_df = pd.DataFrame([input_data])
        date_cols = ['Incident_Date', 'Date_Reported', 'Policy_Start_Date']
        for col in date_cols:
            input_df[col] = pd.to_datetime(input_df[col], errors='coerce')

        input_df['Report_Delay'] = (input_df['Date_Reported'] - input_df['Incident_Date']).dt.days.fillna(0)
        input_df['Days_Since_Policy'] = (input_df['Incident_Date'] - input_df['Policy_Start_Date']).dt.days.fillna(0)
        input_df['Car_Age'] = input_df['Incident_Date'].dt.year - input_df['Model_Year']

        cols_to_scale = ['Claim_Amount', 'Coverage_Amount', 'Premium_Amount']
        input_df[cols_to_scale] = scaler_loaded.transform(input_df[cols_to_scale])

        for col, le in label_encoders_loaded.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col].astype(str))

        X_tab_new = input_df[X_tab_features_loaded].values
        texto_crudo = [input_df['Claim_Description'].iloc[0]]

        # Predicciones
        p1 = xgb_model_loaded.predict_proba(X_tab_new)[:, 1]
        emb_nuevo = extract_embeddings_api(texto_crudo)
        p2 = nn_text_model_loaded.predict(emb_nuevo, verbose=0).flatten()
        meta_in = np.column_stack((p1, p2))
        prob_stack = meta_nn_loaded.predict(meta_in, verbose=0).flatten()

        result = {
            'fraud_probability': float(prob_stack[0]),
            'xgb_probability': float(p1[0]),
            'nlp_probability': float(p2[0]),
            'prediction': 'Fraude' if prob_stack[0] > 0.5 else 'Legítimo'
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("API iniciada en http://localhost:5002")
    print("Endpoint: POST /predict")
    app.run(debug=True, host='0.0.0.0', port=5002)