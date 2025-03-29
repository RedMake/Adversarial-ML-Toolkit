#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para instalar dependencias para el proyecto de Adversarial ML.
- Detecta si ya existe un entorno virtual activo y ofrece usarlo
- Instala dependencias solo si no están ya instaladas
- Proporciona instrucciones para activar el entorno si lo desea o no
"""

import importlib
import subprocess
import sys
import os
import pkg_resources
import platform
import venv
from pathlib import Path

def is_venv_active():
    """Comprobar si hay un entorno virtual activo."""
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

def get_active_venv_path():
    """Obtener la ruta del entorno virtual activo."""
    if is_venv_active():
        return sys.prefix
    return None

def create_venv(venv_path):
    """Crear un entorno virtual en la ruta especificada."""
    print(f"⏳ Creando entorno virtual en '{venv_path}'...")
    venv.create(venv_path, with_pip=True)
    print(f"✅ Entorno virtual creado en '{venv_path}'")
    
    # Determinar comandos de activación según el sistema operativo
    if platform.system() == "Windows":
        activate_cmd = f"{venv_path}\\Scripts\\activate"
        python_path = f"{venv_path}\\Scripts\\python"
    else:  # Unix/Linux/MacOS
        activate_cmd = f"source {venv_path}/bin/activate"
        python_path = f"{venv_path}/bin/python"
    
    print("\nPara activar el entorno virtual, ejecuta:")
    print(f"  {activate_cmd}")
    
    return python_path

def get_python_executable():
    """Obtener el intérprete de Python actual."""
    return sys.executable

def is_package_installed(package_name, python_exe=None):
    """Comprobar si un paquete ya está instalado."""
    if python_exe is None:
        python_exe = get_python_executable()
        
    try:
        # Usar el intérprete especificado para verificar la instalación
        result = subprocess.run(
            [python_exe, "-c", f"import pkg_resources; pkg_resources.get_distribution('{package_name}')"],
            capture_output=True
        )
        return result.returncode == 0
    except:
        return False

def install_package(package_name, python_exe=None):
    """Instalar un paquete usando pip."""
    if python_exe is None:
        python_exe = get_python_executable()
        
    subprocess.check_call([python_exe, "-m", "pip", "install", package_name])
    print(f"✅ Instalado: {package_name}")

def check_and_install(packages, python_exe=None):
    """Comprobar e instalar paquetes si no están presentes."""
    for package in packages:
        if is_package_installed(package, python_exe):
            print(f"✓ Ya instalado: {package}")
        else:
            print(f"⏳ Instalando: {package}...")
            install_package(package, python_exe)

def ask_yes_no(question):
    """Preguntar al usuario una pregunta de sí/no."""
    while True:
        response = input(f"{question} (s/n): ").lower()
        if response in ["s", "si", "sí", "y", "yes"]:
            return True
        elif response in ["n", "no"]:
            return False
        else:
            print("Por favor, responde con 's' (sí) o 'n' (no).")

def main():
    """Función principal para verificar e instalar dependencias."""
    print("="*70)
    print("Configuración del entorno para Adversarial ML Toolkit")
    print("="*70)
    
    python_executable = get_python_executable()
    venv_path = None
    
    # Verificar si ya hay un entorno virtual activo
    active_venv = get_active_venv_path()
    if active_venv:
        print(f"\n🔍 Detectado entorno virtual activo en: {active_venv}")
        use_active_venv = ask_yes_no("¿Deseas usar este entorno virtual para instalar las dependencias?")
        
        if use_active_venv:
            print(f"\n✅ Usando el entorno virtual activo en: {active_venv}")
            python_executable = get_python_executable()
        else:
            # Preguntar si se desea crear otro entorno virtual
            use_venv = ask_yes_no("\n¿Deseas crear un nuevo entorno virtual para este proyecto?")
            if use_venv:
                # Solicitar la ruta para el entorno virtual o usar el valor predeterminado
                default_venv_path = "venv"
                venv_input = input(f"\nIntroduce la ruta para el entorno virtual (presiona Enter para usar '{default_venv_path}'): ")
                venv_path = venv_input if venv_input.strip() else default_venv_path
                
                # Verificar si ya existe
                if os.path.exists(venv_path):
                    replace_venv = ask_yes_no(f"El directorio '{venv_path}' ya existe. ¿Deseas reemplazarlo?")
                    if not replace_venv:
                        print("Operación cancelada.")
                        return
                
                # Crear entorno virtual
                python_executable = create_venv(venv_path)
                
                # Instalar paquete pkg_resources en el nuevo entorno
                subprocess.check_call([python_executable, "-m", "pip", "install", "--upgrade", "setuptools"])
                
                print("\n⚠️ IMPORTANTE: Debes activar el entorno virtual antes de continuar.")
                print("  Cierra este script, activa el entorno virtual y luego ejecútalo de nuevo.")
                
                if not ask_yes_no("\n¿Deseas continuar con la instalación de dependencias ahora?"):
                    return  # Salir para que el usuario active el entorno
    else:
        # No hay entorno virtual activo, preguntar si se desea crear uno
        use_venv = ask_yes_no("\n¿Deseas crear un entorno virtual para este proyecto?")
        
        if use_venv:
            # Solicitar la ruta para el entorno virtual o usar el valor predeterminado
            default_venv_path = "venv"
            venv_input = input(f"\nIntroduce la ruta para el entorno virtual (presiona Enter para usar '{default_venv_path}'): ")
            venv_path = venv_input if venv_input.strip() else default_venv_path
            
            # Verificar si ya existe
            if os.path.exists(venv_path):
                replace_venv = ask_yes_no(f"El directorio '{venv_path}' ya existe. ¿Deseas reemplazarlo?")
                if not replace_venv:
                    print("Operación cancelada.")
                    return
            
            # Crear entorno virtual
            python_executable = create_venv(venv_path)
            
            # Instalar paquete pkg_resources en el nuevo entorno
            subprocess.check_call([python_executable, "-m", "pip", "install", "--upgrade", "setuptools"])
            
            print("\n⚠️ IMPORTANTE: Debes activar el entorno virtual antes de continuar.")
            print("  Cierra este script, activa el entorno virtual y luego ejecútalo de nuevo.")
            
            if not ask_yes_no("\n¿Deseas continuar con la instalación de dependencias ahora?"):
                return  # Salir para que el usuario active el entorno
    
    print("\n" + "="*70)
    print("Verificando e instalando dependencias")
    print("="*70)
    
    # Lista de dependencias requeridas
    required_packages = [
        "torch",
        "torchvision",
        "numpy",
        "matplotlib", 
        "scikit-learn",
        "tqdm",
        "pillow"
    ]
    
    # Verificar e instalar las dependencias
    check_and_install(required_packages, python_executable)
    
    # Verificar que todo se instaló correctamente
    all_installed = True
    for pkg in required_packages:
        if not is_package_installed(pkg, python_executable):
            all_installed = False
            break
    
    if all_installed:
        print("\n✅ Todas las dependencias están instaladas correctamente.")
        
        if venv_path:  # Solo mostrar instrucciones si se creó un nuevo entorno
            print("\nRecuerda activar el entorno virtual antes de ejecutar el proyecto:")
            if platform.system() == "Windows":
                print(f"  {venv_path}\\Scripts\\activate")
            else:
                print(f"  source {venv_path}/bin/activate")
        elif active_venv and use_active_venv:
            print(f"\nUsando el entorno virtual activo en: {active_venv}")
        
        print("\nPuedes ejecutar los ejemplos con:")
        print("  python demo_script.py --mode basico")
    else:
        print("\n❌ Hubo problemas al instalar algunas dependencias.")
        print("   Intenta instalarlas manualmente utilizando:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()