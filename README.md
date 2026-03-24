# UNIR-Fraude-Triaje-3007G

## Entregable MVP

El entregable final es el notebook [Sistema_Triaje_Inteligente_Reclamaciones_Automovil.ipynb](Sistema_Triaje_Inteligente_Reclamaciones_Automovil.ipynb), que implementa un modelo híbrido de detección de fraude (Stacking):

- XGBoost (tabular)
- DistilBERT (extracción de embeddings, sin fine-tuning) + red densa (Keras)
- Meta-learner (Keras) para ensamblar probabilidades

## Ejecución (Google Colab)

1. Abrir [Sistema_Triaje_Inteligente_Reclamaciones_Automovil.ipynb](Sistema_Triaje_Inteligente_Reclamaciones_Automovil.ipynb) en Colab.
2. Subir/ubicar el dataset Excel en Google Drive.
3. Ajustar `file_path` en la celda **1.1) Carga de datos** al path real de Drive.
4. Ejecutar el notebook de inicio a fin.

El MVP **no incluye UI compleja** ni despliegue a producción; cierra con una demo de inferencia en forma de tabla y una sección de próximos pasos (DevOps/MLOps).
