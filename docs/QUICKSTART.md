# 🚀 Guía de Inicio Rápido

## Primeros Pasos

1. **Verificar configuración**:
   ```bash
   python check_project.py
   ```

2. **Si los modelos no existen, regenerarlos**:
   ```bash
   python scripts/regenerate_models_simple.py
   ```

3. **Iniciar la API**:
   ```bash
   python api_server.py
   ```

4. **Probar la API**:
   ```bash
   python test_api.py
   ```

## Uso con Postman

1. Importar `Fraude_Detection_API.postman_collection.json`
2. Ejecutar los requests de ejemplo
3. Usar el endpoint `POST http://localhost:5002/predict`

## Desarrollo

- **Notebooks**: Ver `notebooks/` para análisis y desarrollo
- **Scripts**: Utilidades en `scripts/`
- **Modelos**: Archivos en `models/` (no commitear)
- **Datos**: Datasets en `data/`

## Solución de Problemas

- **Modelos corruptos**: `python scripts/regenerate_models_simple.py`
- **Dependencias**: `pip install -r requirements.txt`
- **Puerto ocupado**: Cambiar en `config.py` o matar proceso