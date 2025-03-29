# Implementación Teórica de Fórmulas del Proyecto de Investigación
# Notebook explicativo para las fórmulas matemáticas de Ataques Adversariales

# %% [markdown]
# # Notebook: Implementación Teórica de Ataques Adversariales
# 
# Este notebook implementa y explica las fórmulas matemáticas presentadas en el proyecto de investigación sobre Adversarial Machine Learning.

# %% [markdown]
# ## 1. Importación de bibliotecas

# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Para reproducibilidad
np.random.seed(0)
torch.manual_seed(0)

# Verificar si hay GPU disponible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# %% [markdown]
# ## 2. Definición Matemática de Ataques Adversariales
# 
# ### Conceptos Básicos
# 
# En el proyecto se define un ataque adversarial como un ejemplo $x'$ que maximiza el error del modelo mientras minimiza la distancia al ejemplo original $x$:
# 
# $$x^\prime=\arg{\min_{x^\prime}}{d(x,x^\prime)}\mathrm{\ sujeto\ a\ }f_\theta(x^\prime)\neq f_\theta(x)$$
# 
# donde:
# - $x$ es el ejemplo original
# - $x'$ es el ejemplo adversarial
# - $d(x,x')$ es una función de distancia
# - $f_\theta$ es el modelo con parámetros $\theta$
# 
# Vamos a implementar esta definición con un modelo simple:

# %%
# Definir un modelo simple para demostración
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = x.view(-1, 784)  # Aplanar la imagen
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Cargar datos MNIST para ejemplos
transform = transforms.Compose([transforms.ToTensor()])
mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
test_loader = DataLoader(mnist_test, batch_size=1, shuffle=False)

# Obtener una imagen de ejemplo
for data, target in test_loader:
    original_image = data.to(device)
    true_label = target.to(device)
    break

# Crear y entrenar el modelo (simplificado para el notebook)
model = SimpleModel().to(device)
model.eval()  # Modo evaluación para simplificar

# %% [markdown]
# ## 3. Implementación del Fast Gradient Sign Method (FGSM)
# 
# ### Fórmula Matemática
# En el proyecto se describe FGSM con la siguiente fórmula:
# 
# $$x^\prime=x+\varepsilon\cdot\mathrm{sign}(\nabla_xJ(\theta,x,y))$$
# 
# donde:
# - $J(\theta,x,y)$ es la función de pérdida
# - $\nabla_x$ es el gradiente con respecto a $x$
# - $\varepsilon$ es la magnitud de la perturbación
# - $\mathrm{sign}$ es la función signo

# %%
def fgsm_attack(model, x, y, epsilon, loss_fn):
    """
    Implementación del Fast Gradient Sign Method (FGSM)
    
    Args:
        model: Modelo a atacar
        x: Imagen original
        y: Etiqueta verdadera
        epsilon: Magnitud de la perturbación
        loss_fn: Función de pérdida
        
    Returns:
        Imagen perturbada
    """
    # Requerir gradientes para x
    x.requires_grad = True
    
    # Calcular la pérdida
    output = model(x)
    loss = loss_fn(output, y)
    
    # Calcular gradientes
    model.zero_grad()
    loss.backward()
    
    # Implementar la fórmula FGSM
    perturbed_image = x + epsilon * torch.sign(x.grad.data)
    
    # Asegurar que los valores estén en [0, 1]
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    return perturbed_image

# %% [markdown]
# ## 4. Implementación del Projected Gradient Descent (PGD)
# 
# ### Fórmula Matemática
# En el proyecto se describe PGD con la siguiente fórmula:
# 
# $$x^{t+1}=\prod_{x+S}(x^t+\alpha\cdot\mathrm{sign}(\nabla_xJ(\theta,x^t,y)))$$
# 
# donde:
# - $\prod_{x+S}$ representa la proyección al conjunto válido de perturbaciones
# - $x^t$ es la imagen en la iteración $t$
# - $\alpha$ es el tamaño del paso

# %%
def pgd_attack(model, x, y, epsilon, alpha, num_iter, loss_fn):
    """
    Implementación del Projected Gradient Descent (PGD)
    
    Args:
        model: Modelo a atacar
        x: Imagen original
        y: Etiqueta verdadera
        epsilon: Límite máximo de la perturbación
        alpha: Tamaño del paso
        num_iter: Número de iteraciones
        loss_fn: Función de pérdida
        
    Returns:
        Imagen perturbada
    """
    # Crear copia para no modificar la original
    perturbed_image = x.clone().detach().requires_grad_(True)
    
    for i in range(num_iter):
        # Forward pass
        output = model(perturbed_image)
        loss = loss_fn(output, y)
        
        # Calcular gradientes
        model.zero_grad()
        loss.backward()
        
        # Actualizar con PGD
        with torch.no_grad():
            # Aplicar paso de gradiente
            perturbation = alpha * torch.sign(perturbed_image.grad.data)
            perturbed_image.data += perturbation
            
            # Proyectar de vuelta a la región válida
            perturbation = torch.clamp(perturbed_image.data - x.data, -epsilon, epsilon)
            perturbed_image.data = x.data + perturbation
            
            # Mantener en rango [0, 1]
            perturbed_image.data = torch.clamp(perturbed_image.data, 0, 1)
        
        # Reiniciar gradientes para la siguiente iteración
        if i < num_iter - 1:
            perturbed_image.grad.zero_()
    
    return perturbed_image.detach()

# %% [markdown]
# ## 5. Implementación del ataque Carlini & Wagner (C&W)
# 
# ### Fórmula Matemática
# En el proyecto se describe C&W con la siguiente fórmula:
# 
# $$\min_\delta||\delta||_p+c\cdot f(x+\delta)$$
# 
# donde:
# - $\delta$ es la perturbación
# - $||\delta||_p$ es la norma p de la perturbación
# - $c$ es un parámetro de balance
# - $f(x+\delta)$ es una función que mide cuán efectivo es el ataque

# %%
def carlini_wagner_attack(model, x, y, target_class=None, max_iterations=100, c=1.0, learning_rate=0.01):
    """
    Versión simplificada del ataque Carlini & Wagner (L2)
    
    Args:
        model: Modelo a atacar
        x: Imagen original
        y: Etiqueta verdadera
        target_class: Clase objetivo (None para ataque no dirigido)
        max_iterations: Número máximo de iteraciones
        c: Parámetro de balance
        learning_rate: Tasa de aprendizaje
        
    Returns:
        Imagen perturbada
    """
    # Inicializar perturbación
    delta = torch.zeros_like(x, requires_grad=True)
    optimizer = optim.Adam([delta], lr=learning_rate)
    
    # Número de clases (para MNIST es 10)
    num_classes = 10
    
    for i in range(max_iterations):
        # Imagen perturbada
        perturbed_image = torch.clamp(x + delta, 0, 1)
        
        # Predicción del modelo
        outputs = model(perturbed_image)
        
        # Función f para C&W
        if target_class is not None:
            # Ataque dirigido
            target = torch.tensor([target_class], device=device)
            f_loss = F.cross_entropy(outputs, target)
        else:
            # Ataque no dirigido (maximizar pérdida para clase verdadera)
            f_loss = -F.cross_entropy(outputs, y)
        
        # Norma L2 de la perturbación
        l2_norm = torch.norm(delta)
        
        # Pérdida total: norma L2 + c * función f
        loss = l2_norm + c * f_loss
        
        # Actualizar perturbación
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Imagen final perturbada
    perturbed_image = torch.clamp(x + delta.detach(), 0, 1)
    
    return perturbed_image

# %% [markdown]
# ## 6. Implementación del Entrenamiento Adversarial (Defensa)
# 
# ### Fórmula Matemática
# En el proyecto se describe el entrenamiento adversarial con la siguiente fórmula:
# 
# $$\min_\theta{E_{(x,y)\sim D}}[\max_{\delta\in S}{L}(\theta,x+\delta,y)]$$
# 
# donde:
# - $D$ es la distribución de datos
# - $S$ es el conjunto de perturbaciones permitidas
# - $L$ es la función de pérdida

# %%
def adversarial_training_step(model, x, y, loss_fn, optimizer, epsilon=0.1):
    """
    Un paso de entrenamiento adversarial
    
    Args:
        model: Modelo a entrenar
        x: Datos de entrada
        y: Etiquetas verdaderas
        loss_fn: Función de pérdida
        optimizer: Optimizador
        epsilon: Magnitud de la perturbación
        
    Returns:
        Pérdida total
    """
    # Calcular gradiente para generar ejemplos adversariales
    x.requires_grad = True
    output = model(x)
    loss = loss_fn(output, y)
    model.zero_grad()
    loss.backward()
    
    # Generar ejemplos adversariales con FGSM
    x_adv = x + epsilon * torch.sign(x.grad.data)
    x_adv = torch.clamp(x_adv, 0, 1)
    
    # Reiniciar gradientes
    model.zero_grad()
    x.requires_grad = False
    
    # Entrenamiento con ejemplos originales
    output = model(x)
    loss_natural = loss_fn(output, y)
    
    # Entrenamiento con ejemplos adversariales
    output_adv = model(x_adv)
    loss_adv = loss_fn(output_adv, y)
    
    # Pérdida combinada
    loss = 0.5 * loss_natural + 0.5 * loss_adv
    
    # Actualizar modelo
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

# %% [markdown]
# ## 7. Visualización de Ataques Adversariales

# %%
def visualize_adversarial_examples(model, original_image, true_label):
    """
    Visualiza ejemplos adversariales generados con diferentes métodos
    
    Args:
        model: Modelo a atacar
        original_image: Imagen original
        true_label: Etiqueta verdadera
    """
    # Función de pérdida
    loss_fn = nn.CrossEntropyLoss()
    
    # Generar ejemplos adversariales
    fgsm_image = fgsm_attack(model, original_image.clone(), true_label, epsilon=0.1, loss_fn=loss_fn)
    pgd_image = pgd_attack(model, original_image.clone(), true_label, epsilon=0.1, alpha=0.01, 
                           num_iter=20, loss_fn=loss_fn)
    cw_image = carlini_wagner_attack(model, original_image.clone(), true_label, max_iterations=50)
    
    # Predicciones
    with torch.no_grad():
        original_pred = model(original_image).argmax(dim=1).item()
        fgsm_pred = model(fgsm_image).argmax(dim=1).item()
        pgd_pred = model(pgd_image).argmax(dim=1).item()
        cw_pred = model(cw_image).argmax(dim=1).item()
    
    # Convertir a numpy para visualización
    original_np = original_image.detach().squeeze().cpu().numpy()
    fgsm_np = fgsm_image.detach().squeeze().cpu().numpy()
    pgd_np = pgd_image.detach().squeeze().cpu().numpy()
    cw_np = cw_image.detach().squeeze().cpu().numpy()
    
    # Calcular perturbaciones
    fgsm_diff = np.abs(fgsm_np - original_np)
    pgd_diff = np.abs(pgd_np - original_np)
    cw_diff = np.abs(cw_np - original_np)
    
    # Visualizar
    fig, axs = plt.subplots(3, 4, figsize=(16, 12))
    
    # Imágenes
    axs[0, 0].imshow(original_np, cmap='gray')
    axs[0, 0].set_title(f'Original\nPred: {original_pred}, Real: {true_label.item()}')
    
    axs[0, 1].imshow(fgsm_np, cmap='gray')
    axs[0, 1].set_title(f'FGSM\nPred: {fgsm_pred}, Real: {true_label.item()}')
    
    axs[0, 2].imshow(pgd_np, cmap='gray')
    axs[0, 2].set_title(f'PGD\nPred: {pgd_pred}, Real: {true_label.item()}')
    
    axs[0, 3].imshow(cw_np, cmap='gray')
    axs[0, 3].set_title(f'C&W\nPred: {cw_pred}, Real: {true_label.item()}')
    
    # Perturbaciones
    axs[1, 0].axis('off')  # Vacío
    
    axs[1, 1].imshow(fgsm_diff, cmap='viridis')
    axs[1, 1].set_title('FGSM Perturbación')
    
    axs[1, 2].imshow(pgd_diff, cmap='viridis')
    axs[1, 2].set_title('PGD Perturbación')
    
    axs[1, 3].imshow(cw_diff, cmap='viridis')
    axs[1, 3].set_title('C&W Perturbación')
    
    # Histogramas de perturbación
    axs[2, 0].axis('off')  # Vacío
    
    axs[2, 1].hist(fgsm_diff.flatten(), bins=100)
    axs[2, 1].set_title('FGSM Histograma')
    
    axs[2, 2].hist(pgd_diff.flatten(), bins=100)
    axs[2, 2].set_title('PGD Histograma')
    
    axs[2, 3].hist(cw_diff.flatten(), bins=100)
    axs[2, 3].set_title('C&W Histograma')
    
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 8. Demostración de Fórmulas en Ataques y Defensas
# 
# Aquí implementaremos un ejemplo completo para demostrar las fórmulas matemáticas mencionadas en el proyecto de investigación.

# %%
def demonstrate_formulas():
    """
    Función para demostrar las fórmulas matemáticas del proyecto
    """
    print("Cargando y preparando modelo y datos...")
    
    # Cargar un modelo preentrenado para demostración
    model = SimpleModel().to(device)
    
    # Configurar optimizador y función de pérdida
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    # Cargar datos de entrenamiento para entrenar brevemente
    train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Entrenar brevemente (simplificado para el notebook)
    print("Entrenando modelo brevemente...")
    model.train()
    for epoch in range(1):  # Solo 1 época para demostración
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}: Loss {loss.item():.4f}")
            
            if batch_idx > 200:  # Limitar para demostración
                break
    
    # Obtener una imagen de prueba
    model.eval()
    for data, target in test_loader:
        test_image = data.to(device)
        test_label = target.to(device)
        break
    
    # Demostrar ataques adversariales
    print("\nDemostrando ataques adversariales basados en las fórmulas matemáticas...")
    visualize_adversarial_examples(model, test_image, test_label)
    
    # Demostrar entrenamiento adversarial
    print("\nDemostrando entrenamiento adversarial...")
    model.train()
    adv_loss = adversarial_training_step(model, test_image, test_label, loss_fn, optimizer)
    print(f"Pérdida después de entrenamiento adversarial: {adv_loss:.4f}")
    
    # Evaluar robustez antes y después del entrenamiento adversarial
    print("\nComparando robustez antes y después del entrenamiento adversarial...")
    
    # Generar ejemplos adversariales
    epsilon = 0.1
    fgsm_image_before = fgsm_attack(model, test_image.clone(), test_label, epsilon, loss_fn)
    
    # Entrenamiento adversarial (simplificado)
    for _ in range(5):  # Unos pocos pasos de entrenamiento adversarial
        adversarial_training_step(model, test_image, test_label, loss_fn, optimizer, epsilon)
    
    # Generar ejemplos adversariales después del entrenamiento
    fgsm_image_after = fgsm_attack(model, test_image.clone(), test_label, epsilon, loss_fn)
    
    # Verificar predicciones
    with torch.no_grad():
        original_output = model(test_image)
        fgsm_output_before = model(fgsm_image_before)
        fgsm_output_after = model(fgsm_image_after)
        
        original_pred = original_output.argmax(dim=1).item()
        fgsm_pred_before = fgsm_output_before.argmax(dim=1).item()
        fgsm_pred_after = fgsm_output_after.argmax(dim=1).item()
    
    print(f"Etiqueta verdadera: {test_label.item()}")
    print(f"Predicción original: {original_pred}")
    print(f"Predicción FGSM antes del entrenamiento adversarial: {fgsm_pred_before}")
    print(f"Predicción FGSM después del entrenamiento adversarial: {fgsm_pred_after}")

# %%
# Ejecutar la demostración
demonstrate_formulas()

# %% [markdown]
# ## 9. Conclusiones y Relación con el Proyecto de Investigación
# 
# En este notebook hemos implementado las fórmulas matemáticas descritas en el proyecto de investigación sobre Adversarial Machine Learning:
# 
# 1. Implementamos la definición formal de ataques adversariales como un problema de optimización.
# 2. Desarrollamos el ataque FGSM según la fórmula $x^\prime=x+\varepsilon\cdot\mathrm{sign}(\nabla_xJ(\theta,x,y))$.
# 3. Implementamos PGD según la fórmula $x^{t+1}=\prod_{x+S}(x^t+\alpha\cdot\mathrm{sign}(\nabla_xJ(\theta,x^t,y)))$.
# 4. Desarrollamos una versión simplificada del ataque C&W según la fórmula $\min_\delta||\delta||_p+c\cdot f(x+\delta)$.
# 5. Implementamos el entrenamiento adversarial según la fórmula $\min_\theta{E_{(x,y)\sim D}}[\max_{\delta\in S}{L}(\theta,x+\delta,y)]$.

# Estas implementaciones demuestran cómo los conceptos teóricos y las fórmulas matemáticas del proyecto se pueden llevar a la práctica. 
#   Los resultados visuales muestran la efectividad de los ataques adversariales y la mejora en robustez que se puede lograr mediante el entrenamiento adversarial.

# La seguridad en modelos de machine learning, como se discute en el proyecto, 
#   es un campo en constante evolución que requiere comprender 
#   tanto los fundamentos matemáticos como las implementaciones prácticas de estos métodos de ataque y defensa.

# FGSM = x' = x + ε·sign(∇x J(θ,x,y))
# PGD = x^(t+1) = Π_(x+S)(x^t + α·sign(∇x J(θ,x^t,y)))
# C&W = min_δ ||δ||_p + c·f(x+δ)
# Entrenamiento Adversarial = min_θ E_(x,y)~D [max_δ∈S L(θ,x+δ,y)]