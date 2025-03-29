# Herramientas de Análisis de Ataques Adversariales

Este repositorio contiene una implementación práctica de los conceptos y fórmulas presentados en el proyecto de investigación sobre Adversarial Machine Learning (Proyecto Universitario). El código permite experimentar con diferentes tipos de ataques adversariales contra modelos de aprendizaje automático y evaluar estrategias de defensa.

## Contenido del Repositorio

- **adversarial_ml_toolkit.py**: Toolkit principal con implementaciones de ataques y defensas
- **adversarial_utils.py**: Funciones de utilidad para ataques adversariales y visualización
- **adversarial_examples.py**: Ejemplos de uso de las herramientas
- **demo_script.py**: Script principal para ejecutar demostraciones
- **teoria_implementacion.py**: Notebook explicativo con implementaciones de las fórmulas matemáticas

## Características Principales

El proyecto implementa los siguientes ataques y defensas mencionados en la investigación:

### Ataques Implementados
- **Fast Gradient Sign Method (FGSM)** - Implementa la fórmula: 
  ```
  x' = x + ε·sign(∇x J(θ,x,y))
  ```
- **Projected Gradient Descent (PGD)** - Implementa la fórmula:
  ```
  x^(t+1) = Π_(x+S)(x^t + α·sign(∇x J(θ,x^t,y)))
  ```
- **Carlini & Wagner (C&W)** - Implementa la fórmula:
  ```
  min_δ ||δ||_p + c·f(x+δ)
  ```
- **Análisis de Transferibilidad** - Evalúa cómo los ejemplos adversariales generados para un modelo pueden engañar a otros modelos

### Defensas Implementadas
- **Entrenamiento Adversarial** - Implementa la fórmula:
  ```
  min_θ E_(x,y)~D [max_δ∈S L(θ,x+δ,y)]
  ```
- **Feature Squeezing** - Reduce la profundidad de bits para eliminar perturbaciones
- **Destilación Defensiva** - Utiliza transferencia de conocimiento para generar modelos más robustos

## Requisitos

- Python 3.7+ o mayor
- PyTorch 1.8+ o mayor
- NumPy
- Matplotlib
- torchvision

Para instalar las dependencias:
```
python install_dependencies.py
```

## Uso

### Ejecución del Script de Demostración

Con menu: 
```
python run_project.py
```
Sin menu:
```
python demo_script.py --mode [basico|avanzado|completo] --epsilon 0.1 --modelo cnn --defensa
```

Opciones disponibles:
- `--mode`: Modo de demostración (basico, avanzado, completo)
- `--epsilon`: Magnitud de la perturbación para ataques adversariales
- `--modelo`: Tipo de modelo a utilizar (cnn, mlp, resnet)
- `--defensa`: Incluir demostraciones de defensas
- `--guardar`: Guardar resultados y gráficos

### Ejemplos de Uso

Para ejecutar un ejemplo básico que muestre ataques FGSM:

```python
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
```

Para evaluar la robustez de un modelo personalizado:

```python
from adversarial_utils import AdversarialUtils
import torch
import torch.nn as nn

# Definir modelo
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Evaluar robustez contra FGSM
robustness_metrics = AdversarialUtils.evaluate_model_robustness(
    model,
    test_loader,
    AdversarialUtils.fgsm_attack,
    device,
    epsilon=0.1
)

print(f"Precisión en datos limpios: {robustness_metrics['clean_accuracy']:.2f}%")
print(f"Precisión en datos adversariales: {robustness_metrics['adversarial_accuracy']:.2f}%")
```

## Relación con el Proyecto de Investigación

Este código implementa directamente las fórmulas matemáticas y conceptos presentados en el proyecto de investigación sobre Adversarial Machine Learning, haciendo énfasis en:

1. **Clasificación de Ataques Adversariales** según:
   - El conocimiento del atacante (caja blanca, caja negra)
   - El objetivo del ataque (dirigido, no dirigido)
   - El momento del ataque (evasión, envenenamiento)

2. **Métodos de Ejecución** implementados:
   - Métodos basados en gradientes (FGSM, PGD)
   - Métodos de optimización (C&W)
   - Ataques de transferencia

3. **Estrategias de Defensa**:
   - Entrenamiento adversarial
   - Defensas arquitectónicas (destilación)
   - Preprocesamiento (feature squeezing)

## Referencias para la Implementación

Este proyecto se ha desarrollado basándose en conceptos teóricos y código de implementaciones existentes en el campo de Adversarial Machine Learning. A continuación se detallan las principales referencias utilizadas:

## Referencias académicas

1. Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). "Explaining and Harnessing Adversarial Examples." International Conference on Learning Representations (ICLR). 
   - *Referencia principal para la implementación del ataque FGSM*

2. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). "Towards Deep Learning Models Resistant to Adversarial Attacks." International Conference on Learning Representations (ICLR).
   - *Base para la implementación del ataque PGD y entrenamiento adversarial*

3. Carlini, N., & Wagner, D. (2017). "Towards Evaluating the Robustness of Neural Networks." IEEE Symposium on Security and Privacy (SP).
   - *Referencia para la implementación del ataque C&W*

4. Papernot, N., McDaniel, P., Goodfellow, I., Jha, S., Celik, Z. B., & Swami, A. (2017). "Practical Black-Box Attacks against Machine Learning." Proceedings of the 2017 ACM Asia Conference on Computer and Communications Security.
   - *Conceptos para la implementación de ataques de transferibilidad*

5. Xu, W., Evans, D., & Qi, Y. (2018). "Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks." Network and Distributed Systems Security Symposium (NDSS).
   - *Base para la implementación de feature squeezing como defensa*

## Proyectos de código abierto y tutoriales

1. [PyTorch Tutorials - Adversarial Example Generation](https://pytorch.org/tutorials/beginner/fgsm_tutorial.html)
   - Tutorial oficial de PyTorch para la generación de ejemplos adversariales mediante FGSM

2. [CleverHans](https://github.com/cleverhans-lab/cleverhans) - Biblioteca de Python para benchmarking de vulnerabilidades a ataques adversariales
   - Referencia para buenas prácticas en la implementación de diversos ataques

3. [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox) - Biblioteca para seguridad ML desarrollada por IBM
   - Inspiración para la estructura de módulos y componentes

4. [AdverTorch](https://github.com/BorealisAI/advertorch) - Caja de herramientas para seguridad adversarial en PyTorch
   - Referencia para implementaciones eficientes en PyTorch

5. [Foolbox](https://github.com/bethgelab/foolbox) - Biblioteca de Python para crear ejemplos adversariales
   - Inspiración para el diseño de las interfaces de ataque

## Recursos educativos

1. Stanford CS230 Deep Learning - [Adversarial Examples and Adversarial Training](https://cs230.stanford.edu/files/C2M3.pdf)
   - Material didáctico sobre conceptos e implementación

2. OpenAI - [Attacking machine learning with adversarial examples](https://openai.com/index/attacking-machine-learning-with-adversarial-examples/?utm_source=chatgpt.com)
   - Este artículo de OpenAI explora cómo funcionan los ejemplos adversariales en diferentes medios y las dificultades para proteger los sistemas contra ellos. ​

3. Medium - [Adversarial Attacks Explained (And How to Defend ML Models Against Them)](https://machinelearningmastery.com/how-to-attack-and-defense-deep-learning-models/)
   - En este texto de Medium, se explica de manera sencilla qué son los ataques adversariales y cómo pueden engañar a los modelos de aprendizaje automático. 
4. ScienceDirect - [Adversarial Attacks and Defenses in Deep Learning](https://www.sciencedirect.com/science/article/pii/S209580991930503X?utm_source=chatgpt.com)
   - Finalmente en el articulo de ScienceDirect introduce las bases teóricas, algoritmos y aplicaciones de las técnicas de ataque adversarial. 
   
## Notas sobre la implementación

- Las implementaciones matemáticas siguen directamente las fórmulas presentadas en las publicaciones académicas citadas.
- La estructura del código se ha diseñado para fines educativos y de investigación, priorizando la claridad sobre la eficiencia en algunos casos.
- Las visualizaciones se inspiran en técnicas comunes en la literatura para mostrar perturbaciones adversariales y su efecto en los modelos.


## Estructura del proyecto

```
    └── adversarial_examples.py
    └── adversarial_ml_toolkit.py
    └── adversarial_utils.py
    └── demo_script.py
    └── Guia-Demostraciones.md
    └── install_dependencies.py
    └── LICENSE.md
    └── README.md
    └── run_project.py
    └── teoria_implementacion.py
```

## Contribuir

Las contribuciones son bienvenidas. Para contribuir:

1. Haz un fork del repositorio
2. Crea una nueva rama (`git checkout -b feature/nueva-caracteristica`)
3. Realiza tus cambios
4. Envía un pull request

## Licencia

Este proyecto está licenciado bajo la licencia MIT - ver el archivo LICENSE para más detalles.
