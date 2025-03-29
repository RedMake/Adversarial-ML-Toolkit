#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de demostración para Ataques Adversariales en Modelos de Machine Learning
Este script implementa conceptos y fórmulas del proyecto de investigación 
sobre Adversarial Machine Learning.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os

# Importar nuestras implementaciones
from adversarial_ml_toolkit import AdversarialMLToolkit
from adversarial_utils import AdversarialUtils
from adversarial_examples import ejemplo_basico, ejemplo_avanzado

def main():
    """Función principal para ejecutar las demostraciones"""
    
    parser = argparse.ArgumentParser(description='Demostración de Ataques Adversariales')
    parser.add_argument('--mode', type=str, default='basico', 
                        choices=['basico', 'avanzado', 'completo'],
                        help='Modo de demostración (basico, avanzado, completo)')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='Epsilon para ataques adversariales (default: 0.1)')
    parser.add_argument('--modelo', type=str, default='cnn',
                        choices=['cnn', 'mlp', 'resnet'],
                        help='Tipo de modelo a utilizar')
    parser.add_argument('--defensa', action='store_true',
                        help='Incluir demostraciones de defensas')
    parser.add_argument('--guardar', action='store_true',
                        help='Guardar resultados y gráficos')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("DEMOSTRACIÓN DE ATAQUES ADVERSARIALES EN MACHINE LEARNING")
    print("=" * 80)
    print(f"Modo: {args.mode}")
    print(f"Epsilon: {args.epsilon}")
    print(f"Modelo: {args.modelo}")
    print(f"Incluir defensas: {args.defensa}")
    print("=" * 80)
    
    # Verificar disponibilidad de GPU
    # Se asume que si hay GPU, se usará la primera GPU disponible que tenga CUDA, 
    # de lo contrario se usará la CPU. Para instalar CUDA, ver: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    if torch.cuda.is_available():
        print("GPU disponible. Usando GPU para entrenamiento.")
        
    else:
        print("GPU no disponible. Usando CPU para entrenamiento.")
        
    # Crear directorio para resultados si es necesario
    if args.guardar:
        os.makedirs("resultados", exist_ok=True)
    
    # Ejecutar el modo seleccionado
    if args.mode == 'basico' or args.mode == 'completo':
        print("\n\n" + "=" * 40)
        print("EJECUTANDO DEMOSTRACIÓN BÁSICA")
        print("=" * 40)
        ejemplo_basico()
    
    if args.mode == 'avanzado' or args.mode == 'completo':
        print("\n\n" + "=" * 40)
        print("EJECUTANDO DEMOSTRACIÓN AVANZADA")
        print("=" * 40)
        ejemplo_avanzado()
    
    # Mensaje final
    print("\n\n" + "=" * 80)
    print("DEMOSTRACIÓN COMPLETADA")
    print("=" * 80)

if __name__ == "__main__":
    main()