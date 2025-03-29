#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para el proyecto Adversarial ML Toolkit.
Este script guía al usuario a través de la configuración inicial
y ejecución del proyecto.
"""

import os
import sys
import subprocess
import platform
import importlib.util
import pkg_resources
from pathlib import Path

def clear_screen():
    """Limpiar la pantalla de la terminal."""
    os.system('cls' if platform.system() == 'Windows' else 'clear')

def print_header():
    """Imprimir encabezado del proyecto."""
    clear_screen()
    print("="*70)
    print("         ADVERSARIAL MACHINE LEARNING TOOLKIT")
    print("="*70)
    print("Un conjunto de herramientas para experimentar con ataques")
    print("adversariales en modelos de aprendizaje automático")
    print("="*70)
    print()

def check_venv():
    """Verificar si se está ejecutando dentro de un entorno virtual."""
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

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

def setup_environment():
    """Configurar el entorno para el proyecto."""
    if not check_venv():
        print("⚠️  No se detectó un entorno virtual activo.")
        if ask_yes_no("¿Deseas configurar el entorno ahora?"):
            subprocess.run([sys.executable, "install_dependencies.py"])
            print("\nPor favor, activa el entorno virtual y ejecuta este script nuevamente.")
            return False
    else:
        print("✅ Entorno virtual detectado.")
        
    return True

def is_package_installed(package_name):
    """Comprobar si un paquete ya está instalado usando pkg_resources."""
    try:
        pkg_resources.get_distribution(package_name)
        return True
    except pkg_resources.DistributionNotFound:
        return False
    except:
        # En caso de otro error, intentamos un método alternativo
        try:
            spec = importlib.util.find_spec(package_name)
            return spec is not None
        except:
            return False

def check_dependencies():
    """Verificar si las dependencias están instaladas."""
    required_packages = [
        "torch", "torchvision", "numpy", "matplotlib", 
        "scikit-learn", "tqdm", "pillow"
    ]
    
    missing_packages = []
    
    print("Verificando dependencias instaladas...")
    for package in required_packages:
        if is_package_installed(package):
            print(f"✓ {package}: Instalado")
        else:
            print(f"✗ {package}: No encontrado")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Faltan algunas dependencias: {', '.join(missing_packages)}")
        if ask_yes_no("¿Deseas instalar las dependencias faltantes ahora?"):
            subprocess.run([sys.executable, "install_dependencies.py"])
            return False
        else:
            return False
    else:
        print("\n✅ Todas las dependencias están instaladas correctamente.")
        return True

def run_demo():
    """Ejecutar la demostración del proyecto."""
    print("\n" + "="*70)
    print("OPCIONES DE DEMOSTRACIÓN")
    print("="*70)
    print("1. Demostración básica (FGSM, ejemplo sencillo)")
    print("2. Demostración avanzada (FGSM, PGD, transferencia)")
    print("3. Demostración completa (todos los ataques y defensas)")
    print("4. Salir")
    
    while True:
        choice = input("\nSelecciona una opción (1-4): ")
        
        if choice == "1":
            subprocess.run([sys.executable, "demo_script.py", "--mode", "basico"])
            break
        elif choice == "2":
            subprocess.run([sys.executable, "demo_script.py", "--mode", "avanzado"])
            break
        elif choice == "3":
            subprocess.run([sys.executable, "demo_script.py", "--mode", "completo"])
            break
        elif choice == "4":
            print("Saliendo del programa...")
            return
        else:
            print("Opción no válida. Por favor, selecciona una opción del 1 al 4.")

def show_examples():
    """Mostrar ejemplos de uso del proyecto."""
    print("\n" + "="*70)
    print("EJEMPLOS DE USO")
    print("="*70)
    
    print("""
# Ejemplo básico:
from adversarial_ml_toolkit import AdversarialMLToolkit

# Crear toolkit
toolkit = AdversarialMLToolkit()

# Cargar modelo y datos
toolkit.load_simple_model()
toolkit.load_mnist_dataset()

# Entrenar modelo
toolkit.train_model(epochs=2)

# Demostrar ataques adversariales
toolkit.demonstrate_fgsm_attack(epsilon=0.1)

# Ejemplo avanzado:
from adversarial_utils import AdversarialUtils
import torch
import torch.nn as nn

# Evaluar robustez contra FGSM
robustness_metrics = AdversarialUtils.evaluate_model_robustness(
    model,
    test_loader,
    AdversarialUtils.fgsm_attack,
    device,
    epsilon=0.1
)
""")
    
    input("\nPresiona Enter para continuar...")

def main_menu():
    """Mostrar el menú principal."""
    while True:
        print_header()
        print("MENÚ PRINCIPAL")
        print("-" * 70)
        print("1. Ejecutar demostración")
        print("2. Ver ejemplos de uso")
        print("3. Configurar entorno")
        print("4. Salir")
        
        choice = input("\nSelecciona una opción (1-4): ")
        
        if choice == "1":
            if check_dependencies():
                run_demo()
            input("\nPresiona Enter para volver al menú principal...")
        elif choice == "2":
            show_examples()
        elif choice == "3":
            setup_environment()
            input("\nPresiona Enter para volver al menú principal...")
        elif choice == "4":
            print("Gracias por usar Adversarial ML Toolkit. ¡Hasta pronto!")
            break
        else:
            print("Opción no válida. Por favor, selecciona una opción del 1 al 4.")
            input("\nPresiona Enter para continuar...")

def main():
    """Función principal."""
    print_header()
    
    # Verificar si se está ejecutando en un entorno virtual
    env_ready = setup_environment()
    
    if env_ready:
        # Verificar dependencias
        deps_ready = check_dependencies()
        
        if deps_ready:
            print("\n✅ Entorno listo para ejecutar el proyecto.")
            
            if ask_yes_no("\n¿Deseas continuar al menú principal?"):
                main_menu()
            else:
                print("\nGracias por usar Adversarial ML Toolkit. ¡Hasta pronto!")
        else:
            print("\nPor favor, instala las dependencias necesarias antes de continuar.")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()