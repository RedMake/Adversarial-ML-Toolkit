# Guía de Demostraciones - Adversarial ML Toolkit

Este documento explica los procesos y resultados que se muestran durante las demostraciones básicas y avanzadas del toolkit de Adversarial Machine Learning.

## Demostración Básica

La demostración básica muestra los fundamentos de los ataques adversariales FGSM y PGD en un modelo simple entrenado con el dataset MNIST.

### 1. Inicialización y Entrenamiento

```
Iniciando ejemplo básico de ataques adversariales...
Entrenando modelo (versión reducida)...
```

**¿Qué significa?** El sistema está creando un modelo de red neuronal simple y lo está entrenando con el conjunto de datos MNIST (imágenes de dígitos escritos a mano). El entrenamiento es una versión reducida (1 época) para ahorrar tiempo en la demostración.

### 2. Evaluación del Modelo

```
Evaluando modelo...
Precisión en el conjunto de prueba: 92.45%
```

**¿Qué significa?** Después del entrenamiento, se evalúa el modelo con datos de prueba para medir su precisión. Este valor (por ejemplo, 92.45%) indica el porcentaje de imágenes que el modelo clasifica correctamente en condiciones normales (sin ataques).

### 3. Demostración de Ataque FGSM

```
Demostrando ataque FGSM...
Imagen original - Etiqueta real: 7, Predicción: 7
Imagen atacada (FGSM) - Etiqueta real: 7, Predicción: 9
```

**¿Qué significa?** 
- Se selecciona una imagen (en este caso un dígito "7") y se muestra que el modelo la clasifica correctamente.
- Se aplica el ataque FGSM (Fast Gradient Sign Method) a la imagen.
- El modelo ahora clasifica incorrectamente la imagen perturbada como un "9".
- Se visualiza la imagen original, la imagen perturbada y la diferencia entre ambas.

**Fórmula FGSM:** `x' = x + ε·sign(∇x J(θ,x,y))`
- `x` es la imagen original
- `ε` (epsilon) controla la magnitud de la perturbación
- `∇x J(θ,x,y)` es el gradiente de la función de pérdida con respecto a la imagen

### 4. Demostración de Ataque PGD

```
Demostrando ataque PGD...
Imagen original - Etiqueta real: 7, Predicción: 7
Imagen atacada (PGD) - Etiqueta real: 7, Predicción: 9
```

**¿Qué significa?** 
- Similar al FGSM, pero usando el ataque PGD (Projected Gradient Descent).
- PGD es una versión iterativa y más potente de FGSM.
- Las visualizaciones muestran la imagen original, la imagen perturbada y la diferencia.

**Fórmula PGD:** `x^(t+1) = Π_(x+S)(x^t + α·sign(∇x J(θ,x^t,y)))`
- El ataque se aplica iterativamente
- `α` (alpha) es el tamaño del paso en cada iteración
- `Π_(x+S)` representa la proyección para mantener la perturbación dentro de límites válidos

## Demostración Avanzada

La demostración avanzada muestra ataques y defensas más complejos, y realiza análisis de robustez.

### 1. Entrenamiento del Modelo

```
Entrenando modelo básico (versión muy reducida)...
Entrenamiento: Batch 0/100
Entrenamiento: Batch 50/100
Precisión del modelo: 87.68%
```

**¿Qué significa?** Similar a la demostración básica, se entrena un modelo simple pero con una cantidad aún más limitada de datos para agilizar la demostración.

### 2. Visualización de Ataques FGSM

```
Demostrando ataque FGSM con las utilidades...
Generando y visualizando ejemplos adversariales con FGSM...
```

**¿Qué significa?** Se seleccionan varias imágenes del conjunto de prueba y se les aplica el ataque FGSM. La visualización muestra:
- Fila superior: Imágenes originales con su etiqueta real y la predicción del modelo
- Fila central: Imágenes perturbadas con su nueva predicción (generalmente incorrecta)
- Fila inferior: Mapa de calor que muestra dónde se aplicaron las perturbaciones (zonas más brillantes)

### 3. Evaluación de Robustez

```
Evaluando robustez del modelo contra FGSM...
Precisión en datos limpios: 87.68%
Precisión en datos adversariales: 28.94%
Tasa de éxito del ataque: 58.74%
```

**¿Qué significa?**
- **Precisión en datos limpios:** Porcentaje de imágenes correctamente clasificadas sin ataques
- **Precisión en datos adversariales:** Porcentaje de imágenes que siguen siendo correctamente clasificadas después del ataque
- **Tasa de éxito del ataque:** Diferencia entre ambas precisiones, indica el porcentaje de imágenes que el ataque logró engañar al modelo

### 4. Defensa con Feature Squeezing

```
Demostrando defensa con feature squeezing...
```

**¿Qué significa?** Feature squeezing es una técnica de defensa que reduce la profundidad de bits de la imagen (por ejemplo, de 8 bits a 3 bits). La visualización muestra:
- Fila superior: Imágenes originales
- Fila 2: Imágenes atacadas (perturbadas)
- Fila 3: Imágenes defendidas (después de aplicar feature squeezing a las imágenes atacadas)
- Fila inferior: Diferencia entre imágenes originales y defendidas

Feature squeezing funciona porque elimina las perturbaciones pequeñas que el ojo humano no puede ver pero que engañan al modelo.

### 5. Entrenamiento Adversarial

```
Demostrando entrenamiento adversarial (versión muy reducida)...
Época 1/2, Batch 0/5, Pérdida: 2.3425
Época 1/2, Pérdida: 2.3105, Precisión limpia: 34.38%, Precisión adversarial: 4.69%
```

**¿Qué significa?** El entrenamiento adversarial es una técnica de defensa donde el modelo se entrena con ejemplos adversariales (perturbados). Se muestra:
- **Época:** Iteración completa sobre el conjunto de datos
- **Pérdida:** El valor de la función de pérdida (menor es mejor)
- **Precisión limpia:** Precisión en datos normales
- **Precisión adversarial:** Precisión en datos perturbados

**Fórmula del entrenamiento adversarial:** `min_θ E_(x,y)~D [max_δ∈S L(θ,x+δ,y)]`
- Busca minimizar la pérdida en el peor caso (ejemplos adversariales)

### 6. Gráficas de Entrenamiento

Al final del entrenamiento adversarial, se muestran dos gráficas:
- **Pérdida de entrenamiento:** Muestra cómo disminuye la pérdida durante el entrenamiento
- **Precisión durante el entrenamiento:** Muestra la precisión en datos limpios y adversariales durante el entrenamiento

## Demostración Completa

La demostración completa combina los elementos básicos y avanzados, y añade:

### 1. Análisis de Transferibilidad

```
Analizando transferibilidad de ataques...
```

**¿Qué significa?** Se estudia si los ejemplos adversariales generados para engañar a un modelo (modelo fuente) también pueden engañar a otro modelo diferente (modelo objetivo). Se muestra:
- **Tasa de transferibilidad:** Porcentaje de éxito que tiene un ataque generado para un modelo cuando se aplica a otro modelo
- **Gráficos de barras:** Comparación de efectividad entre modelo fuente y objetivo

### 2. Defensa por Destilación

```
Demostrando defensa por destilación...
```

**¿Qué significa?** La destilación defensiva es una técnica donde un modelo "estudiante" se entrena para imitar a un modelo "maestro", pero con salidas suavizadas por un parámetro de temperatura. Esto hace al modelo estudiante más robusto contra ataques.

## Interpretación de Visualizaciones

### Imágenes Perturbadas
- A simple vista, las imágenes perturbadas pueden parecer idénticas a las originales
- El modelo las clasifica incorrectamente, aunque para el ojo humano sigan siendo claramente el mismo dígito
- Esto demuestra la vulnerabilidad de los modelos de ML a pequeñas perturbaciones imperceptibles

### Mapas de Diferencia
- Los mapas de calor muestran dónde se concentran las perturbaciones
- Colores más brillantes indican zonas con mayor perturbación
- El ataque FGSM suele mostrar un patrón de "ruido" uniforme
- El ataque PGD suele mostrar perturbaciones más concentradas en áreas específicas

### Gráficas de Robustez
- Una gran caída en la precisión bajo ataque indica un modelo muy vulnerable
- Un modelo robusto mantendría una precisión similar en datos limpios y adversariales
- El entrenamiento adversarial busca reducir esta brecha

## Fórmulas Clave Implementadas

1. **FGSM (Fast Gradient Sign Method):**
   ```
   x' = x + ε·sign(∇x J(θ,x,y))
   ```

2. **PGD (Projected Gradient Descent):**
   ```
   x^(t+1) = Π_(x+S)(x^t + α·sign(∇x J(θ,x^t,y)))
   ```

3. **C&W (Carlini & Wagner):**
   ```
   min_δ ||δ||_p + c·f(x+δ)
   ```

4. **Entrenamiento Adversarial:**
   ```
   min_θ E_(x,y)~D [max_δ∈S L(θ,x+δ,y)]
   ```

## Conclusiones de las Demostraciones

- Los modelos de ML, incluso con alta precisión, son vulnerables a pequeñas perturbaciones
- Ataques como FGSM pueden reducir drásticamente la precisión con cambios imperceptibles
- Técnicas de defensa como feature squeezing y entrenamiento adversarial pueden mejorar la robustez
- Un modelo robusto requiere equilibrio entre precisión en datos limpios y adversariales
- La transferibilidad de ataques entre modelos representa un riesgo de seguridad importante
