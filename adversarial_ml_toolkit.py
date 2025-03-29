import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class AdversarialMLToolkit:
    """
    Toolkit para demostrar ataques adversariales en modelos de aprendizaje automático
    basado en los conceptos del proyecto de investigación.
    """
    
    def __init__(self):
        """
            <summary>
            Inicializa el toolkit de machine learning adversarial.
            </summary>
            
            <remarks>
            Configura automáticamente el dispositivo (GPU/CPU) y prepara el entorno para los ataques.
            </remarks>
        """
        # Configuración del dispositivo (GPU si está disponible, CPU en caso contrario)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Modelo base que será atacado (por defecto, una CNN simple)
        self.model = None
        
        # Conjunto de datos
        self.data_loader = None
        
        print(f"AdversarialMLToolkit inicializado en: {self.device}")
    
    def load_simple_model(self):
        """
        <summary>
        Carga un modelo CNN simple para clasificación de imágenes MNIST.
        </summary>
        
        <returns>
        None
        </returns>
        
        <example>
        <code>
        toolkit = AdversarialMLToolkitEnhanced()
        toolkit.load_simple_model()
        </code>
        </example>
        """
        class SimpleNet(nn.Module):
            def __init__(self):
                super(SimpleNet, self).__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, 1)
                self.conv2 = nn.Conv2d(32, 64, 3, 1)
                # Cambiamos dropout2d por dropout normal si la entrada es 2D
                self.dropout1 = nn.Dropout(0.25)  # Cambiado de Dropout2d a Dropout
                self.dropout2 = nn.Dropout(0.5)   # Cambiado de Dropout2d a Dropout
                self.fc1 = nn.Linear(9216, 128)
                self.fc2 = nn.Linear(128, 10)

            def forward(self, x):
                x = self.conv1(x)
                x = nn.functional.relu(x)
                x = self.conv2(x)
                x = nn.functional.relu(x)
                x = nn.functional.max_pool2d(x, 2)
                # Aplicamos dropout - aseguramos que la entrada tenga la dimensión correcta
                x = self.dropout1(x)  # Estamos aplicando dropout a una representación convolucional
                x = torch.flatten(x, 1)
                x = self.fc1(x)
                x = nn.functional.relu(x)
                x = self.dropout2(x)  # Aquí ya estamos con representación flattened, por lo que dropout regular es apropiado
                x = self.fc2(x)
                return nn.functional.log_softmax(x, dim=1)
        
        self.model = SimpleNet().to(self.device)
        print("Modelo simple cargado.")
        
    def load_mnist_dataset(self, batch_size=128):
        """Carga el conjunto de datos MNIST para pruebas"""
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        train_dataset = torchvision.datasets.MNIST(
            root='./data', 
            train=True, 
            download=True, 
            transform=transform
        )
        
        test_dataset = torchvision.datasets.MNIST(
            root='./data', 
            train=False, 
            download=True, 
            transform=transform
        )
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
        
        print(f"Conjuntos de datos MNIST cargados (tamaño de batch: {batch_size})")
    
    def train_model(self, epochs=5):
        """Entrena el modelo con el conjunto de datos cargado"""
        if self.model is None or self.train_loader is None:
            print("Error: Primero debe cargar un modelo y un conjunto de datos.")
            return
        
        optimizer = optim.Adam(self.model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(self.train_loader)}, "
                          f"Loss: {running_loss/100:.4f}")
                    running_loss = 0.0
        
        print("Entrenamiento finalizado.")
    
    def evaluate_model(self):
        """Evalúa el modelo en el conjunto de prueba"""
        if self.model is None or self.test_loader is None:
            print("Error: Primero debe cargar un modelo y un conjunto de datos.")
            return
        
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Precisión en el conjunto de prueba: {accuracy:.2f}%")
        return accuracy
    
    def fgsm_attack(self, image, epsilon, target, criterion):
        """
        <summary>
        Implementación del Fast Gradient Sign Method (FGSM)
        </summary>
        
        <param name="image">La imagen de entrada a perturbar</param>
        <param name="epsilon">La magnitud de la perturbación</param>
        <param name="target">La etiqueta objetivo</param>
        <param name="criterion">La función de pérdida a utilizar</param>
        
        <returns>
        torch.Tensor: La imagen perturbada
        </returns>
        
        <remarks>
        Fórmula: x' = x + ε⋅sign(∇x J(θ,x,y))
        
        Donde:
        - x' es la imagen perturbada
        - x es la imagen original
        - ε es la magnitud de la perturbación
        - J(θ,x,y) es la función de pérdida
        - ∇x es el gradiente con respecto a x
        </remarks>
        """
        image.requires_grad = True
        
        output = self.model(image)
        loss = criterion(output, target)
        
        # Obtener gradientes
        self.model.zero_grad()
        loss.backward()
        
        # Aplicar FGSM
        perturbed_image = image + epsilon * torch.sign(image.grad.data)
        
        # Asegurar que los valores estén en [0,1]
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        
        return perturbed_image
    
    def pgd_attack(self, image, epsilon, alpha, iterations, target, criterion):
        """
        <summary>
        Implementación del Projected Gradient Descent (PGD)
        </summary>
        
        <param name="image">La imagen de entrada a perturbar</param>
        <param name="epsilon">La magnitud máxima de la perturbación</param>
        <param name="alpha">El tamaño del paso en cada iteración</param>
        <param name="iterations">El número de iteraciones</param>
        <param name="target">La etiqueta objetivo</param>
        <param name="criterion">La función de pérdida a utilizar</param>
        
        <returns>
        torch.Tensor: La imagen perturbada
        </returns>
        
        <remarks>
        Fórmula: x^(t+1) = Π_(x+S)(x^t + α⋅sign(∇x J(θ,x^t,y)))
        
        Donde:
        - x^t es la imagen en la iteración t
        - Π_(x+S) es la proyección al espacio de perturbaciones válidas
        - α es el tamaño del paso
        </remarks>
        """
        perturbed_image = image.clone().detach()
        
        for i in range(iterations):
            perturbed_image.requires_grad = True
            
            output = self.model(perturbed_image)
            loss = criterion(output, target)
            
            # Obtener gradientes
            self.model.zero_grad()
            loss.backward()
            
            # Aplicar PGD
            with torch.no_grad():
                adv_step = alpha * torch.sign(perturbed_image.grad.data)
                perturbed_image = perturbed_image + adv_step
                
                # Proyectar de vuelta a la región válida (restricción de epsilon)
                delta = torch.clamp(perturbed_image - image, -epsilon, epsilon)
                perturbed_image = torch.clamp(image + delta, 0, 1)
        
        return perturbed_image
    
    def demonstrate_fgsm_attack(self, epsilon=0.1):
        """Demuestra un ataque FGSM en una imagen del conjunto de prueba"""
        if self.model is None or self.test_loader is None:
            print("Error: Primero debe cargar un modelo y un conjunto de datos.")
            return
        
        # Obtener una imagen para atacar
        data_iter = iter(self.test_loader)
        images, labels = next(data_iter)
        
        # Seleccionar una imagen
        img_idx = 0
        image = images[img_idx:img_idx+1].to(self.device)
        label = labels[img_idx:img_idx+1].to(self.device)
        
        # Clasificación original
        self.model.eval()
        with torch.no_grad():
            output = self.model(image)
            _, predicted = torch.max(output, 1)
            
        print(f"Imagen original - Etiqueta real: {label.item()}, Predicción: {predicted.item()}")
        
        # Generar ataque FGSM
        criterion = nn.CrossEntropyLoss()
        perturbed_image = self.fgsm_attack(image, epsilon, label, criterion)
        
        # Clasificación de la imagen perturbada
        with torch.no_grad():
            output = self.model(perturbed_image)
            _, predicted_adv = torch.max(output, 1)
            
        print(f"Imagen atacada (FGSM) - Etiqueta real: {label.item()}, Predicción: {predicted_adv.item()}")
        
        # Visualizar original vs perturbada
        self._visualize_attack(image, perturbed_image, label, predicted, predicted_adv, f"FGSM (ε={epsilon})")
    
    def demonstrate_pgd_attack(self, epsilon=0.1, alpha=0.01, iterations=40):
        """Demuestra un ataque PGD en una imagen del conjunto de prueba"""
        if self.model is None or self.test_loader is None:
            print("Error: Primero debe cargar un modelo y un conjunto de datos.")
            return
        
        # Obtener una imagen para atacar
        data_iter = iter(self.test_loader)
        images, labels = next(data_iter)
        
        # Seleccionar una imagen
        img_idx = 0
        image = images[img_idx:img_idx+1].to(self.device)
        label = labels[img_idx:img_idx+1].to(self.device)
        
        # Clasificación original
        self.model.eval()
        with torch.no_grad():
            output = self.model(image)
            _, predicted = torch.max(output, 1)
            
        print(f"Imagen original - Etiqueta real: {label.item()}, Predicción: {predicted.item()}")
        
        # Generar ataque PGD
        criterion = nn.CrossEntropyLoss()
        perturbed_image = self.pgd_attack(image, epsilon, alpha, iterations, label, criterion)
        
        # Clasificación de la imagen perturbada
        with torch.no_grad():
            output = self.model(perturbed_image)
            _, predicted_adv = torch.max(output, 1)
            
        print(f"Imagen atacada (PGD) - Etiqueta real: {label.item()}, Predicción: {predicted_adv.item()}")
        
        # Visualizar original vs perturbada
        self._visualize_attack(image, perturbed_image, label, predicted, predicted_adv, 
                              f"PGD (ε={epsilon}, α={alpha}, iter={iterations})")
    
    def _visualize_attack(self, original, perturbed, true_label, orig_pred, adv_pred, attack_name):
        """Visualiza la imagen original, perturbada y la diferencia"""
        # Convertir tensores a arrays de numpy
        # Usar detach() para tensores que puedan tener requires_grad=True
        original_np = original.detach().cpu().squeeze().numpy()
        perturbed_np = perturbed.detach().cpu().squeeze().numpy()
        difference = np.abs(perturbed_np - original_np)
        
        # Crear figura
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        # Mostrar imágenes
        axs[0].imshow(original_np, cmap='gray')
        axs[0].set_title(f"Original (Pred: {orig_pred.item()}, Real: {true_label.item()})")
        
        axs[1].imshow(perturbed_np, cmap='gray')
        axs[1].set_title(f"Perturbada (Pred: {adv_pred.item()}, Real: {true_label.item()})")
        
        axs[2].imshow(difference, cmap='viridis')
        axs[2].set_title(f"Diferencia - {attack_name}")
        
        # Añadir colorbar para la diferencia
        plt.colorbar(axs[2].imshow(difference, cmap='viridis'), ax=axs[2])
        
        plt.tight_layout()
        plt.show()
    
    def demonstrate_adversarial_training(self, epsilon=0.1, epochs=3):
        """
        <summary>
        Demuestra el entrenamiento adversarial como método de defensa
        </summary>
        
        <param name="epsilon">La magnitud de la perturbación para los ejemplos adversariales</param>
        <param name="epochs">El número de épocas de entrenamiento</param>
        
        <returns>
        None
        </returns>
        
        <remarks>
        Fórmula: min_θ E_(x,y)~D [max_δ∈S L(θ, x+δ, y)]
        
        Esta implementación utiliza tanto ejemplos originales como adversariales
        durante el entrenamiento para mejorar la robustez del modelo.
        </remarks>
        
        <example>
        <code>
        toolkit = AdversarialMLToolkitEnhanced()
        toolkit.load_simple_model()
        toolkit.load_mnist_dataset()
        toolkit.demonstrate_adversarial_training(epsilon=0.1, epochs=3)
        </code>
        </example>
        """
        if self.model is None or self.train_loader is None:
            print("Error: Primero debe cargar un modelo y un conjunto de datos.")
            return
        
        # Evaluar antes del entrenamiento adversarial
        print("Evaluación antes del entrenamiento adversarial:")
        regular_accuracy = self.evaluate_model()
        
        # Función para evaluar la robustez adversarial
        def eval_adv_robustness(epsilon=0.1):
            self.model.eval()
            correct = 0
            total = 0
            criterion = nn.CrossEntropyLoss()
            
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Generar ejemplos adversariales
                perturbed_data = self.fgsm_attack(data.clone(), epsilon, target, criterion)
                
                # Evaluar en ejemplos adversariales
                with torch.no_grad():
                    outputs = self.model(perturbed_data)
                    _, predicted = torch.max(outputs.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            
            adv_accuracy = 100 * correct / total
            print(f"Robustez adversarial (ε={epsilon}): {adv_accuracy:.2f}%")
            return adv_accuracy
        
        # Evaluar robustez adversarial inicial
        print("Evaluación de robustez adversarial antes del entrenamiento:")
        initial_adv_accuracy = eval_adv_robustness(epsilon)
        
        # Entrenamiento adversarial
        print(f"\nIniciando entrenamiento adversarial (ε={epsilon})...")
        optimizer = optim.Adam(self.model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Generar ejemplos adversariales para el entrenamiento
                perturbed_data = self.fgsm_attack(data.clone(), epsilon, target, criterion)
                
                # Entrenar con ejemplos originales y adversariales
                optimizer.zero_grad()
                
                # Pérdida con datos originales
                output = self.model(data)
                loss_natural = criterion(output, target)
                
                # Pérdida con datos perturbados
                output_adv = self.model(perturbed_data)
                loss_adv = criterion(output_adv, target)
                
                # Combinar pérdidas (entrenamiento adversarial)
                loss = 0.5 * loss_natural + 0.5 * loss_adv
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(self.train_loader)}, "
                          f"Loss: {running_loss/100:.4f}")
                    running_loss = 0.0
        
        print("Entrenamiento adversarial finalizado.")
        
        # Evaluar después del entrenamiento adversarial
        print("\nEvaluación después del entrenamiento adversarial:")
        robust_accuracy = self.evaluate_model()
        
        # Evaluar robustez adversarial final
        print("Evaluación de robustez adversarial después del entrenamiento:")
        final_adv_accuracy = eval_adv_robustness(epsilon)
        
        # Mostrar resultados
        print("\nResumen de resultados:")
        print(f"Precisión estándar antes:  {regular_accuracy:.2f}%")
        print(f"Precisión estándar después: {robust_accuracy:.2f}%")
        print(f"Robustez adversarial antes:  {initial_adv_accuracy:.2f}%")
        print(f"Robustez adversarial después: {final_adv_accuracy:.2f}%")
        print(f"Mejora en robustez: {final_adv_accuracy - initial_adv_accuracy:.2f}%")
        
        # Mostrar comparación
        labels = ['Precisión estándar', 'Robustez adversarial']
        before = [regular_accuracy, initial_adv_accuracy]
        after = [robust_accuracy, final_adv_accuracy]
        
        x = np.arange(len(labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, before, width, label='Antes del entrenamiento adversarial')
        rects2 = ax.bar(x + width/2, after, width, label='Después del entrenamiento adversarial')
        
        ax.set_ylabel('Precisión (%)')
        ax.set_title('Impacto del entrenamiento adversarial')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        
        # Añadir etiquetas a las barras
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}%',
                            xy=(rect.get_x() + rect.get_width()/2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        autolabel(rects1)
        autolabel(rects2)
        
        plt.tight_layout()
        plt.show()

    def analyze_attack_transferability(self, epsilon=0.1):
        """
        Analiza la transferibilidad de ataques adversariales entre dos modelos
        
        Estudia si los ejemplos adversariales generados para un modelo
        son efectivos contra otro modelo diferente.
        """
        if self.model is None or self.test_loader is None:
            print("Error: Primero debe cargar un modelo y un conjunto de datos.")
            return
        
        # Crear un segundo modelo para estudiar transferibilidad
        class ComplexNet(nn.Module):
            def __init__(self):
                super(ComplexNet, self).__init__()
                self.conv1 = nn.Conv2d(1, 64, 3, 1)
                self.conv2 = nn.Conv2d(64, 128, 3, 1)
                self.dropout1 = nn.Dropout2d(0.25)
                self.dropout2 = nn.Dropout2d(0.5)
                self.fc1 = nn.Linear(9216, 256)
                self.fc2 = nn.Linear(256, 10)

            def forward(self, x):
                x = self.conv1(x)
                x = nn.functional.relu(x)
                x = self.conv2(x)
                x = nn.functional.relu(x)
                x = nn.functional.max_pool2d(x, 2)
                x = self.dropout1(x)
                x = torch.flatten(x, 1)
                x = self.fc1(x)
                x = nn.functional.relu(x)
                x = self.dropout2(x)
                x = self.fc2(x)
                return nn.functional.log_softmax(x, dim=1)
        
        # Guardar el modelo original y crear un segundo modelo
        model_source = self.model  # Modelo fuente (atacado directamente)
        model_target = ComplexNet().to(self.device)  # Modelo objetivo (para evaluar transferibilidad)
        
        # Entrenar el segundo modelo
        print("Entrenando el segundo modelo (objetivo)...")
        optimizer = optim.Adam(model_target.parameters())
        criterion = nn.CrossEntropyLoss()
        
        model_target.train()
        for epoch in range(3):  # Entrenar brevemente para demostración
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model_target(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                if batch_idx % 200 == 0:
                    print(f"Epoch {epoch+1}/3, Batch {batch_idx}/{len(self.train_loader)}")
        
        print("Segundo modelo entrenado.")
        
        # Evaluar ambos modelos en el conjunto de prueba limpio
        print("\nEvaluando modelos en datos limpios:")
        
        def evaluate(model, name):
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in self.test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    outputs = model(data)
                    _, predicted = torch.max(outputs.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            
            accuracy = 100 * correct / total
            print(f"Precisión del modelo {name}: {accuracy:.2f}%")
            return accuracy
        
        acc_source_clean = evaluate(model_source, "fuente")
        acc_target_clean = evaluate(model_target, "objetivo")
        
        # Generar ejemplos adversariales usando el modelo fuente
        print("\nGenerando ejemplos adversariales para transferibilidad...")
        
        def generate_adv_examples(batch_size=100):
            # Obtener un lote de imágenes para atacar
            data_iter = iter(self.test_loader)
            images, labels = next(data_iter)
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Generar ataques FGSM usando el modelo fuente
            self.model = model_source  # Asegurar que estamos usando el modelo fuente
            criterion = nn.CrossEntropyLoss()
            perturbed_images = self.fgsm_attack(images.clone(), epsilon, labels, criterion)
            
            return images, perturbed_images, labels
        
        # Evaluar la transferibilidad
        print(f"\nEvaluando transferibilidad con ε={epsilon}:")
        
        clean_images, adv_images, true_labels = generate_adv_examples()
        
        # Evaluar ejemplos adversariales en ambos modelos
        def evaluate_adv(model, name, clean_imgs, adv_imgs, labels):
            model.eval()
            
            # Evaluar en imágenes limpias
            with torch.no_grad():
                outputs = model(clean_imgs)
                _, preds_clean = torch.max(outputs, 1)
                acc_clean = 100 * (preds_clean == labels).sum().item() / labels.size(0)
            
            # Evaluar en imágenes adversariales
            with torch.no_grad():
                outputs = model(adv_imgs)
                _, preds_adv = torch.max(outputs, 1)
                acc_adv = 100 * (preds_adv == labels).sum().item() / labels.size(0)
            
            # Calcular tasa de éxito del ataque
            attack_success = 100 - acc_adv
            
            print(f"Modelo {name}:")
            print(f"  - Precisión en imágenes limpias: {acc_clean:.2f}%")
            print(f"  - Precisión en imágenes adversariales: {acc_adv:.2f}%")
            print(f"  - Tasa de éxito del ataque: {attack_success:.2f}%")
            
            return acc_clean, acc_adv, attack_success
        
        # Evaluar en modelo fuente (atacado directamente)
        _, _, attack_source = evaluate_adv(model_source, "fuente", clean_images, adv_images, true_labels)
        
        # Evaluar en modelo objetivo (para medir transferibilidad)
        _, _, attack_target = evaluate_adv(model_target, "objetivo", clean_images, adv_images, true_labels)
        
        # Transferibilidad: qué tan efectivo es el ataque en el modelo objetivo comparado con el modelo fuente
        transferability = (attack_target / attack_source) * 100 if attack_source > 0 else 0
        print(f"\nTasa de transferibilidad: {transferability:.2f}%")
        
        # Visualizar un ejemplo de transferibilidad
        idx = 0  # Índice de la imagen a visualizar
        
        # Predicciones del modelo fuente
        model_source.eval()
        with torch.no_grad():
            outputs = model_source(clean_images[idx:idx+1])
            _, pred_clean_source = torch.max(outputs, 1)
            
            outputs = model_source(adv_images[idx:idx+1])
            _, pred_adv_source = torch.max(outputs, 1)
        
        # Predicciones del modelo objetivo
        model_target.eval()
        with torch.no_grad():
            outputs = model_target(clean_images[idx:idx+1])
            _, pred_clean_target = torch.max(outputs, 1)
            
            outputs = model_target(adv_images[idx:idx+1])
            _, pred_adv_target = torch.max(outputs, 1)
        
        # Visualizar
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Imagen original
        clean_np = clean_images[idx].detach().cpu().squeeze().numpy()
        axs[0, 0].imshow(clean_np, cmap='gray')
        axs[0, 0].set_title(f"Original - Real: {true_labels[idx].item()}")
        axs[0, 0].set_xlabel(f"Pred Modelo Fuente: {pred_clean_source.item()}")
        axs[0, 0].set_ylabel(f"Pred Modelo Objetivo: {pred_clean_target.item()}")
        
        # Imagen perturbada
        adv_np = adv_images[idx].detach().cpu().squeeze().numpy()
        axs[0, 1].imshow(adv_np, cmap='gray')
        axs[0, 1].set_title(f"Adversarial - Real: {true_labels[idx].item()}")
        axs[0, 1].set_xlabel(f"Pred Modelo Fuente: {pred_adv_source.item()}")
        axs[0, 1].set_ylabel(f"Pred Modelo Objetivo: {pred_adv_target.item()}")
        
        # Diferencia (perturbación)
        diff = np.abs(adv_np - clean_np)
        im = axs[1, 0].imshow(diff, cmap='viridis')
        axs[1, 0].set_title(f"Perturbación (ε={epsilon})")
        plt.colorbar(im, ax=axs[1, 0])
        
        # Gráfico de barras de transferibilidad
        bars = ['Modelo Fuente', 'Modelo Objetivo', 'Transferibilidad']
        values = [attack_source, attack_target, transferability]
        axs[1, 1].bar(bars, values)
        axs[1, 1].set_title("Tasa de éxito del ataque y transferibilidad")
        axs[1, 1].set_ylabel("Porcentaje (%)")
        axs[1, 1].set_ylim(0, 100)
        
        # Añadir valores a las barras
        for i, v in enumerate(values):
            axs[1, 1].text(i, v + 2, f"{v:.1f}%", ha='center')
        
        plt.tight_layout()
        plt.show()
        
        # Restaurar el modelo original
        self.model = model_source

# Ejemplo de uso
if __name__ == "__main__":
    # Crear toolkit
    toolkit = AdversarialMLToolkit()
    
    # Cargar modelo y datos
    toolkit.load_simple_model()
    toolkit.load_mnist_dataset()
    
    # Entrenar modelo
    toolkit.train_model(epochs=2)
    
    # Evaluar modelo
    toolkit.evaluate_model()
    
    # Demostrar ataques adversariales
    toolkit.demonstrate_fgsm_attack(epsilon=0.1)
    toolkit.demonstrate_pgd_attack(epsilon=0.1, alpha=0.01, iterations=40)
    
    # Demostrar entrenamiento adversarial (defensa)
    toolkit.demonstrate_adversarial_training(epsilon=0.1, epochs=2)
    
    # Analizar transferibilidad de ataques
    toolkit.analyze_attack_transferability(epsilon=0.1)