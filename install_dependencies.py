#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para instalar dependencias para el proyecto de Adversarial ML.
- Detecta si ya existe un entorno virtual activo y ofrece usarlo
- Instala dependencias solo si no están ya instaladas
- Proporciona instrucciones para activar el entorno si lo desea o no
- Ofrece la opción de instalar PyTorch con soporte CUDA
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

def check_cuda_availability():
    """Verificar disponibilidad de CUDA."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        # Si torch no está instalado, no podemos verificar CUDA todavía
        return False

def check_existing_pytorch():
    """Verificar si ya hay una versión de PyTorch instalada y si tiene CUDA."""
    try:
        import torch
        has_pytorch = True
        has_cuda = torch.cuda.is_available()
        version = torch.__version__
        return has_pytorch, has_cuda, version
    except ImportError:
        return False, False, None

def uninstall_pytorch(python_exe=None):
    """Desinstalar PyTorch para evitar conflictos."""
    if python_exe is None:
        python_exe = get_python_executable()
    
    packages = ["torch", "torchvision", "torchaudio"]
    
    print("⏳ Desinstalando versiones anteriores de PyTorch...")
    try:
        for package in packages:
            subprocess.check_call([python_exe, "-m", "pip", "uninstall", "-y", package])
        print("✅ Versiones anteriores de PyTorch desinstaladas correctamente.")
        return True
    except subprocess.CalledProcessError:
        print("⚠️  No se pudieron desinstalar todas las versiones anteriores.")
        return False

def install_pytorch_with_cuda(python_exe=None):
    """Instalar PyTorch con soporte para CUDA."""
    if python_exe is None:
        python_exe = get_python_executable()
    
    # Comando para instalar PyTorch con CUDA 11.8
    cuda_install_cmd = "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    
    print("⏳ Instalando PyTorch con soporte para CUDA 11.8...")
    try:
        subprocess.check_call([python_exe, "-m", "pip", "install", *cuda_install_cmd.split()])
        print("✅ PyTorch con soporte CUDA instalado correctamente.")
        return True
    except subprocess.CalledProcessError:
        print("❌ Error al instalar PyTorch con CUDA. Intentando con la versión estándar...")
        return False

def install_pytorch_standard(python_exe=None):
    """Instalar PyTorch sin soporte para CUDA específico."""
    if python_exe is None:
        python_exe = get_python_executable()
    
    packages = ["torch", "torchvision", "torchaudio"]
    
    print("⏳ Instalando PyTorch versión estándar...")
    try:
        subprocess.check_call([python_exe, "-m", "pip", "install", *packages])
        print("✅ PyTorch instalado correctamente.")
        return True
    except subprocess.CalledProcessError:
        print("❌ Error al instalar PyTorch.")
        return False

def check_and_install(packages, python_exe=None, skip_pytorch=False):
    """Comprobar e instalar paquetes si no están presentes."""
    for package in packages:
        # Saltarse PyTorch si se manejará por separado
        if skip_pytorch and package in ["torch", "torchvision", "torchaudio"]:
            continue
            
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
    
    # Verificar la instalación existente de PyTorch
    has_pytorch, has_cuda, version = check_existing_pytorch()
    if has_pytorch:
        print(f"\n🔍 Detectada instalación existente de PyTorch versión {version}")
        print(f"   Soporte CUDA: {'✅ Disponible' if has_cuda else '❌ No disponible'}")
        
        # Preguntar qué hacer con la instalación existente
        print("\nOpciones disponibles:")
        print("1. Mantener la instalación actual de PyTorch")
        print("2. Reinstalar PyTorch" + (" con soporte CUDA" if has_cuda else ""))
        print("3. Reinstalar PyTorch sin soporte específico para CUDA")
        print("4. Desinstalar PyTorch completamente")
        
        while True:
            choice = input("\nSelecciona una opción (1-4): ")
            
            if choice == "1":
                print("Manteniendo la instalación existente de PyTorch.")
                pytorch_installed = True
                cuda_option = has_cuda  # Mantener estado actual
                break
            elif choice == "2":
                if has_cuda:
                    print("Reinstalando PyTorch con soporte CUDA...")
                else:
                    print("Instalando PyTorch con soporte CUDA...")
                uninstall_pytorch(python_executable)
                cuda_option = True
                break
            elif choice == "3":
                print("Reinstalando PyTorch sin soporte específico para CUDA...")
                uninstall_pytorch(python_executable)
                cuda_option = False
                break
            elif choice == "4":
                print("Desinstalando PyTorch completamente...")
                uninstall_pytorch(python_executable)
                pytorch_installed = False
                cuda_option = False
                if ask_yes_no("¿Deseas continuar con la instalación de las demás dependencias?"):
                    break
                else:
                    print("Operación cancelada.")
                    return
            else:
                print("Opción no válida. Por favor, selecciona una opción del 1 al 4.")
    else:
        print("\nNo se detectó una instalación existente de PyTorch.")
        cuda_option = ask_yes_no("¿Deseas instalar PyTorch con soporte para CUDA? (recomendado si tienes una GPU NVIDIA)")
        if not cuda_option:
            print("Se instalará PyTorch sin soporte específico para CUDA.")
    
    # Lista de dependencias requeridas, separando PyTorch del resto
    pytorch_packages = ["torch", "torchvision", "torchaudio"]
    other_packages = [
        "numpy",
        "matplotlib", 
        "scikit-learn",
        "tqdm",
        "pillow"
    ]
    
    # Solo instalar PyTorch si es necesario
    if not has_pytorch or (has_pytorch and not pytorch_installed):
        # Instalar según la preferencia de CUDA
        if cuda_option:
            # Instalar PyTorch con CUDA
            pytorch_installed = install_pytorch_with_cuda(python_executable)
            if not pytorch_installed:
                # Si falla, intentar con la versión estándar
                pytorch_installed = install_pytorch_standard(python_executable)
        else:
            # Instalar PyTorch sin soporte específico para CUDA
            pytorch_installed = install_pytorch_standard(python_executable)
    else:
        pytorch_installed = True  # Ya está instalado y decidimos mantenerlo
    
    # Verificar el estado actual de PyTorch
    print("\nVerificando la instalación de PyTorch...")
    try:
        # Importar torch desde el intérprete instalado para verificar CUDA
        result = subprocess.run(
            [python_executable, "-c", "import torch; print(f'Versión PyTorch: {torch.__version__}'); print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'Versión CUDA (si disponible): {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"],
            capture_output=True,
            text=True
        )
        print(result.stdout.strip())
        
        # Verificar si hay problemas de compatibilidad
        if "cuda" in result.stdout.lower() and "disponible: false" in result.stdout.lower():
            print("\n⚠️ ADVERTENCIA: PyTorch está instalado con soporte CUDA, pero CUDA no está disponible.")
            print("   Posibles causas:")
            print("   - Drivers NVIDIA no instalados o desactualizados")
            print("   - Versión de CUDA incompatible con tu hardware")
            print("   - Problemas con la instalación de PyTorch")
            print("\n   Sugerencia: Verifica la instalación de los drivers NVIDIA y asegúrate")
            print("   de que son compatibles con CUDA 11.8")
    except:
        print("No se pudo verificar la instalación de PyTorch correctamente.")
    
    # Verificar e instalar las demás dependencias
    check_and_install(other_packages, python_executable)
    
    # Verificar que todo se instaló correctamente
    all_installed = True
    for pkg in pytorch_packages + other_packages:
        if not is_package_installed(pkg, python_executable):
            all_installed = False
            print(f"❌ {pkg} no está instalado correctamente.")
    
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
        if cuda_option:
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            print("   pip install numpy matplotlib scikit-learn tqdm pillow")
        else:
            print("   pip install torch torchvision torchaudio")
            print("   pip install numpy matplotlib scikit-learn tqdm pillow")

if __name__ == "__main__":
    main()