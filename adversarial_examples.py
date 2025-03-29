import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from adversarial_ml_toolkit import AdversarialMLToolkit
from adversarial_utils import AdversarialUtils

def ejemplo_basico():
    """Ejemplo básico de entrenamiento y ataque a un modelo MNIST"""
    print("Iniciando ejemplo básico de ataques adversariales...")
    
    # Crear toolkit
    toolkit = AdversarialMLToolkit()
    
    # Cargar modelo y datos
    toolkit.load_simple_model()
    toolkit.load_mnist_dataset(batch_size=100)
    
    # Entrenar modelo (reducido para demostración)
    print("Entrenando modelo (versión reducida)...")
    toolkit.train_model(epochs=1)
    
    # Evaluar modelo
    toolkit.evaluate_model()
    
    # Demostrar ataques
    print("\nDemostrando ataque FGSM...")
    toolkit.demonstrate_fgsm_attack(epsilon=0.1)
    
    print("\nDemostrando ataque PGD...")
    toolkit.demonstrate_pgd_attack(epsilon=0.1, alpha=0.01, iterations=10)
    
    print("Ejemplo básico completado!")

def ejemplo_avanzado():
    """Ejemplo más avanzado usando las utilidades para investigación"""
    print("Iniciando ejemplo avanzado de ataques y defensas adversariales...")
    
    # Configurar dispositivo
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Cargar conjunto de datos MNIST
    transform = transforms.Compose([transforms.ToTensor()])
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    
    # Crear un modelo simple
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(784, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 10)
    ).to(device)
    
    # Entrenar brevemente para demostración
    print("Entrenando modelo básico (versión muy reducida)...")
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(1):  # Solo 1 época para demostración
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 50 == 0:
                print(f"Entrenamiento: Batch {batch_idx}/{len(test_loader)}")
                if batch_idx >= 100:  # Limitar para demostración
                    break
    
    # Evaluar brevemente
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"Precisión del modelo: {accuracy:.2f}%")
    
    # Demostrar ataque FGSM
    print("\nDemostrando ataque FGSM con las utilidades...")
    
    # Obtener un lote para visualización
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)
    
    # Seleccionar algunas imágenes para visualización
    images_subset = images[:5]
    labels_subset = labels[:5]
    
    # Visualizar ataque FGSM
    print("Generando y visualizando ejemplos adversariales con FGSM...")
    AdversarialUtils.visualize_adversarial_examples(
        model, 
        images_subset, 
        labels_subset, 
        AdversarialUtils.fgsm_attack, 
        device, 
        epsilon=0.2
    )
    
    # Evaluar robustez
    print("\nEvaluando robustez del modelo contra FGSM...")
    fgsm_results = AdversarialUtils.evaluate_model_robustness(
        model,
        test_loader,
        AdversarialUtils.fgsm_attack,
        device,
        epsilon=0.1
    )
    
    print(f"Precisión en datos limpios: {fgsm_results['clean_accuracy']:.2f}%")
    print(f"Precisión en datos adversariales: {fgsm_results['adversarial_accuracy']:.2f}%")
    print(f"Tasa de éxito del ataque: {fgsm_results['attack_success_rate']:.2f}%")
    
    # Demostrar defensa de feature squeezing
    print("\nDemostrando defensa con feature squeezing...")
    
    # Función para aplicar feature squeezing
    def feature_squeeze(x, bit_depth=3):
        return AdversarialUtils.feature_squeezing(x, bit_depth=bit_depth)
    
    # Visualizar la defensa
    AdversarialUtils.visualize_defense_comparison(
        model,
        images_subset,
        labels_subset,
        AdversarialUtils.fgsm_attack,
        lambda x: feature_squeeze(x, bit_depth=3),
        device,
        epsilon=0.2
    )
    
    # Demostrar entrenamiento adversarial brevemente
    print("\nDemostrando entrenamiento adversarial (versión muy reducida)...")
    
    # Crear un nuevo modelo para entrenamiento adversarial
    adv_model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(784, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 10)
    ).to(device)
    
    # Configurar optimizador
    adv_optimizer = torch.optim.Adam(adv_model.parameters())
    
    # Entrenar con ejemplos adversariales (solo unas pocas iteraciones para demostración)
    small_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    
    # Crear un pequeño conjunto de datos para entrenamiento y evaluación
    limited_data = []
    for i, (data, target) in enumerate(small_loader):
        limited_data.append((data, target))
        if i >= 4:  # 5 batches
            break
    
    # Función para convertir lista de batches a DataLoader
    def create_loader_from_batches(batches):
        all_data = []
        all_targets = []
        for data, target in batches:
            all_data.append(data)
            all_targets.append(target)
        
        all_data = torch.cat(all_data)
        all_targets = torch.cat(all_targets)
        
        return DataLoader(
            torch.utils.data.TensorDataset(all_data, all_targets),
            batch_size=64,
            shuffle=False
        )
    
    # Crear cargadores a partir de los batches limitados
    train_loader = create_loader_from_batches(limited_data)
    eval_loader = train_loader  # Usar los mismos datos para evaluar
    
    # Entrenar con adversarial training
    history = AdversarialUtils.adversarial_training(
        adv_model,
        train_loader,
        adv_optimizer,
        criterion,
        device,
        AdversarialUtils.fgsm_attack,
        epochs=2,  # Solo 2 épocas para demostración
        alpha=0.5,  # 50% normal, 50% adversarial
        eval_loader=eval_loader,  # Usar los mismos datos para evaluación
        epsilon=0.1
    )
    
    # Graficar el historial de entrenamiento
    AdversarialUtils.plot_training_history(history)
    
    # Finalizar
    print("\nEjemplo avanzado completado!")