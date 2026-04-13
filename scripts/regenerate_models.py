#!/usr/bin/env python3
"""
Script para regenerar los modelos en caso de problemas de compatibilidad.
Ejecuta el notebook completo para recrear los modelos con el formato correcto.
"""

import subprocess
import sys
import os

def main():
    print("🔄 Regenerando modelos del sistema de detección de fraude...")

    # Verificar que estamos en el directorio correcto
    if not os.path.exists('Sistema_Triaje_Inteligente_Reclamaciones_Automovil.ipynb'):
        print("❌ Error: No se encuentra el notebook principal.")
        print("Asegúrate de ejecutar este script desde el directorio del proyecto.")
        sys.exit(1)

    print("📓 Ejecutando notebook completo para regenerar modelos...")
    print("Esto puede tomar varios minutos...")

    try:
        # Ejecutar el notebook usando jupyter nbconvert con ruta completa
        result = subprocess.run([
            '/usr/local/bin/jupyter', 'nbconvert', '--to', 'notebook', '--execute',
            '--inplace', 'Sistema_Triaje_Inteligente_Reclamaciones_Automovil.ipynb'
        ], capture_output=True, text=True, timeout=1800)  # 30 minutos timeout

        if result.returncode == 0:
            print("✅ Modelos regenerados exitosamente!")
            print("Ahora puedes ejecutar: python api_server.py")
        else:
            print("❌ Error al ejecutar el notebook:")
            print("STDOUT:", result.stdout[-1000:])  # Últimos 1000 caracteres
            print("STDERR:", result.stderr[-1000:])  # Últimos 1000 caracteres
            sys.exit(1)

    except subprocess.TimeoutExpired:
        print("⏰ Timeout: El proceso tomó demasiado tiempo.")
        print("Los modelos pueden haber sido parcialmente generados.")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ Error: Jupyter no está instalado o no está en el PATH.")
        print("Instala Jupyter con: pip install jupyter")
        sys.exit(1)

if __name__ == "__main__":
    main()