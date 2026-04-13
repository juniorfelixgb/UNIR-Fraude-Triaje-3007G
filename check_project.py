#!/usr/bin/env python3
"""
Script de verificación del proyecto
Verifica que todos los archivos y dependencias estén en orden
"""

import os
import sys
import importlib

def check_structure():
    """Verifica la estructura de carpetas y archivos"""
    required_dirs = ['models', 'data', 'notebooks', 'scripts', 'docs']
    required_files = [
        'config.py', 'api_server.py', 'test_api.py', 'requirements.txt',
        'README.md', 'Fraude_Detection_API.postman_collection.json'
    ]

    print("🔍 Verificando estructura del proyecto...")

    # Verificar carpetas
    for dir_name in required_dirs:
        if os.path.isdir(dir_name):
            print(f"✅ Carpeta {dir_name}/ existe")
        else:
            print(f"❌ Carpeta {dir_name}/ faltante")

    # Verificar archivos
    for file_name in required_files:
        if os.path.isfile(file_name):
            print(f"✅ Archivo {file_name} existe")
        else:
            print(f"❌ Archivo {file_name} faltante")

def check_models():
    """Verifica que los modelos estén presentes"""
    model_files = [
        'models/xgb_model.json',
        'models/nn_text_model',
        'models/meta_nn_model',
        'models/scaler.pkl',
        'models/label_encoders.pkl',
        'models/X_tab_features.pkl'
    ]

    print("\n🤖 Verificando modelos...")
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"✅ {model_file} encontrado")
        else:
            print(f"❌ {model_file} faltante")

def check_dependencies():
    """Verifica dependencias críticas"""
    critical_deps = [
        'flask', 'tensorflow', 'xgboost', 'transformers',
        'pandas', 'numpy', 'sklearn', 'joblib'
    ]

    print("\n📦 Verificando dependencias...")
    missing_deps = []

    for dep in critical_deps:
        try:
            importlib.import_module(dep)
            print(f"✅ {dep} disponible")
        except ImportError:
            print(f"❌ {dep} faltante")
            missing_deps.append(dep)

    if missing_deps:
        print(f"\n⚠️  Dependencias faltantes: {', '.join(missing_deps)}")
        print("Ejecuta: pip install -r requirements.txt")
        return False

    return True

def main():
    print("🚗 Verificación del Sistema de Detección de Fraudes\n")

    check_structure()
    check_models()
    deps_ok = check_dependencies()

    print("\n" + "="*50)

    if deps_ok:
        print("✅ Proyecto configurado correctamente!")
        print("\nPara iniciar:")
        print("1. python scripts/regenerate_models_simple.py  # Si necesitas regenerar modelos")
        print("2. python api_server.py                      # Iniciar API")
        print("3. python test_api.py                         # Probar API")
    else:
        print("❌ Hay problemas de configuración")
        sys.exit(1)

if __name__ == "__main__":
    main()