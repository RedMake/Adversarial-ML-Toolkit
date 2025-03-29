import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional, Union


class AdversarialUtils:
    """
    Utilidades para generar y analizar ataques adversariales.
    Implementa los algoritmos mencionados en el proyecto de investigación.
    """
    
    @staticmethod
    def fgsm_attack(model, x, y, epsilon, loss_fn, targeted=False):
        """
        Implementa el ataque Fast Gradient Sign Method (FGSM).
        
        Fórmula: x' = x + ε⋅sign(∇x J(θ,x,y))
        
        Argumentos:
            model: El modelo a atacar
            x: Las imágenes de entrada
            y: Las etiquetas objetivo
            epsilon: La magnitud de la perturbación
            loss_fn: La función de pérdida
            targeted: Si True, intenta causar una clasificación específica; si False, cualquier error
        
        Retorna:
            Las imágenes perturbadas
        """
        # Creamos una copia del tensor y aseguramos que requiera gradiente
        x_clone = x.detach().clone()
        x_clone.requires_grad = True
        
        # Forward pass
        outputs = model(x_clone)
        
        # Calcular pérdida
        if targeted:
            # Para ataques dirigidos, queremos minimizar la pérdida hacia la clase objetivo
            loss = -loss_fn(outputs, y)
        else:
            # Para ataques no dirigidos, queremos maximizar la pérdida
            loss = loss_fn(outputs, y)
        
        # Calcular gradientes
        loss.backward()
        
        # Extraer gradiente de x
        grad = x_clone.grad.data
        
        # Crear perturbación utilizando el signo del gradiente
        if targeted:
            perturbed_x = x - epsilon * torch.sign(grad)
        else:
            perturbed_x = x + epsilon * torch.sign(grad)
        
        # Asegurar que los valores estén en el rango válido [0, 1]
        perturbed_x = torch.clamp(perturbed_x, 0, 1)
        
        return perturbed_x
    
    @staticmethod
    def pgd_attack(model, x, y, epsilon, alpha, num_iter, loss_fn, targeted=False, random_start=True):
        """
        Implementa el ataque Projected Gradient Descent (PGD).
        
        Fórmula: x^(t+1) = Π_(x+S)(x^t + α⋅sign(∇x J(θ,x^t,y)))
        
        Argumentos:
            model: El modelo a atacar
            x: Las imágenes de entrada
            y: Las etiquetas objetivo
            epsilon: El límite máximo de la perturbación
            alpha: El tamaño del paso
            num_iter: El número de iteraciones
            loss_fn: La función de pérdida
            targeted: Si True, intenta causar una clasificación específica; si False, cualquier error
            random_start: Si True, agrega una perturbación aleatoria inicial
            
        Retorna:
            Las imágenes perturbadas
        """
        # Crear una copia para no modificar el original
        perturbed_x = x.clone().detach()
        
        # Inicialización aleatoria (opcional)
        if random_start:
            perturbed_x = perturbed_x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
            perturbed_x = torch.clamp(perturbed_x, 0, 1)
        
        for i in range(num_iter):
            perturbed_x.requires_grad = True
            
            # Forward pass
            outputs = model(perturbed_x)
            
            # Calcular pérdida
            if targeted:
                # Para ataques dirigidos, queremos minimizar la pérdida hacia la clase objetivo
                loss = -loss_fn(outputs, y)
            else:
                # Para ataques no dirigidos, queremos maximizar la pérdida
                loss = loss_fn(outputs, y)
            
            # Gradientes
            loss.backward()
            
            # Actualizar con PGD
            with torch.no_grad():
                if targeted:
                    perturbed_x = perturbed_x - alpha * torch.sign(perturbed_x.grad)
                else:
                    perturbed_x = perturbed_x + alpha * torch.sign(perturbed_x.grad)
                
                # Proyectar de vuelta a la región válida (restricción de epsilon)
                delta = torch.clamp(perturbed_x - x, -epsilon, epsilon)
                perturbed_x = torch.clamp(x + delta, 0, 1)
            
            # Limpiar gradientes para la siguiente iteración
            perturbed_x.grad.zero_()
        
        return perturbed_x
    
    @staticmethod
    def carlini_wagner_l2(model, x, y, targeted=False, max_iter=100, confidence=0, learning_rate=0.01, initial_const=0.01, max_const=10):
        """
        Implementa el ataque C&W L2.
        
        Fórmula: min_δ ||δ||_2 + c⋅f(x+δ)
        
        Esta es una versión simplificada del ataque C&W que no incluye la optimización completa.
        
        Argumentos:
            model: El modelo a atacar
            x: Las imágenes de entrada
            y: Las etiquetas objetivo
            targeted: Si True, intenta causar una clasificación específica; si False, cualquier error
            max_iter: Número máximo de iteraciones
            confidence: Parámetro de confianza para el margen
            learning_rate: Tasa de aprendizaje para la optimización
            initial_const: Constante inicial para el balance
            max_const: Constante máxima para la búsqueda binaria
            
        Retorna:
            Las imágenes perturbadas
        """
        # Número de clases (asumiendo la última capa lineal con num_classes salidas)
        num_classes = model(x).shape[1]
        batch_size = x.shape[0]
        
        # Creamos una variable optimizable (la perturbación)
        delta = torch.zeros_like(x, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=learning_rate)
        
        # Constante para balancear términos
        const = torch.ones(batch_size, device=x.device) * initial_const
        
        # Mejor solución encontrada
        best_dist = torch.ones(batch_size, device=x.device) * 1e10
        best_attack = x.clone()
        
        for _ in range(max_iter):
            # Limpiar gradientes
            optimizer.zero_grad()
            
            # Imagen perturbada
            perturbed_x = torch.clamp(x + delta, 0, 1)
            
            # Calcular la distancia L2
            dist = torch.norm((perturbed_x - x).view(batch_size, -1), dim=1)
            
            # Forward pass
            outputs = model(perturbed_x)
            
            # Calcular la función f(x+δ)
            target_logits = outputs.gather(1, y.unsqueeze(1)).squeeze()
            other_logits = outputs.clone()
            other_logits.scatter_(1, y.unsqueeze(1), -float('inf'))
            other_logits = torch.max(other_logits, dim=1)[0]
            
            if targeted:
                # Si es un ataque dirigido, queremos que la clase objetivo tenga mayor confianza
                f_loss = torch.clamp(other_logits - target_logits + confidence, min=0)
            else:
                # Si es un ataque no dirigido, queremos que cualquier otra clase tenga mayor confianza
                f_loss = torch.clamp(target_logits - other_logits + confidence, min=0)
            
            # Pérdida total: distancia L2 + constante * función f
            loss = dist + const * f_loss
            
            # Backward pass
            loss.mean().backward()
            
            # Actualizar delta
            optimizer.step()
            
            # Actualizar la mejor solución encontrada
            is_adv = (outputs.argmax(1) != y) if not targeted else (outputs.argmax(1) == y)
            is_better = dist < best_dist
            is_both = is_adv & is_better
            
            best_dist = torch.where(is_both, dist, best_dist)
            best_attack = torch.where(is_both.unsqueeze(1).unsqueeze(2).unsqueeze(3),
                                       perturbed_x, best_attack)
        
        return best_attack
    
    @staticmethod
    def evaluate_model_robustness(model, data_loader, attack_fn, device, **attack_params):
        """
        Evalúa la robustez de un modelo frente a un ataque adversarial.
        
        Argumentos:
            model: El modelo a evaluar
            data_loader: El cargador de datos para evaluación
            attack_fn: La función de ataque a utilizar
            device: El dispositivo donde realizar la evaluación
            attack_params: Parámetros adicionales para la función de ataque
            
        Retorna:
            Un diccionario con las métricas de robustez
        """
        model.eval()
        
        clean_correct = 0
        adv_correct = 0
        total = 0
        
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            total += target.size(0)
            
            # Evaluación en datos limpios
            with torch.no_grad():
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                clean_correct += pred.eq(target.view_as(pred)).sum().item()
            
            # Generar ejemplos adversariales
            loss_fn = torch.nn.CrossEntropyLoss()
            perturbed_data = attack_fn(model, data, target, loss_fn=loss_fn, **attack_params)
            
            # Evaluación en datos adversariales
            with torch.no_grad():
                output = model(perturbed_data)
                pred = output.argmax(dim=1, keepdim=True)
                adv_correct += pred.eq(target.view_as(pred)).sum().item()
        
        clean_accuracy = 100. * clean_correct / total
        adv_accuracy = 100. * adv_correct / total
        
        return {
            'clean_accuracy': clean_accuracy,
            'adversarial_accuracy': adv_accuracy,
            'attack_success_rate': clean_accuracy - adv_accuracy,
            'total_samples': total
        }
        
    @staticmethod
    def visualize_adversarial_examples(model, data, target, attack_fn, device, **attack_params):
        """
        Visualiza ejemplos adversariales y las predicciones del modelo.
        
        Argumentos:
            model: El modelo a evaluar
            data: Los datos de entrada
            target: Las etiquetas verdaderas
            attack_fn: La función de ataque a utilizar
            device: El dispositivo donde realizar la evaluación
            attack_params: Parámetros adicionales para la función de ataque
            
        Retorna:
            None (muestra un gráfico)
        """
        model.eval()
        
        # Asegurar que los datos estén en el dispositivo correcto
        data, target = data.to(device), target.to(device)
        
        # Predicción en datos limpios
        with torch.no_grad():
            output = model(data)
            pred_clean = output.argmax(dim=1)
        
        # Generar ejemplos adversariales
        loss_fn = torch.nn.CrossEntropyLoss()
        perturbed_data = attack_fn(model, data, target, loss_fn=loss_fn, **attack_params)
        
        # Predicción en datos adversariales
        with torch.no_grad():
            output = model(perturbed_data)
            pred_adv = output.argmax(dim=1)
        
        # Calcular perturbación
        perturbation = torch.abs(perturbed_data - data)
        
        # Visualizar
        num_examples = min(5, data.size(0))
        fig, axes = plt.subplots(3, num_examples, figsize=(num_examples * 3, 9))
        
        for i in range(num_examples):
            # Imagen original
            if data.size(1) == 1:  # Imágenes en escala de grises (como MNIST)
                axes[0, i].imshow(data[i, 0].detach().cpu().numpy(), cmap='gray')
            else:  # Imágenes a color
                axes[0, i].imshow(np.transpose(data[i].detach().cpu().numpy(), (1, 2, 0)))
            
            axes[0, i].set_title(f"Original\nReal: {target[i].item()}\nPred: {pred_clean[i].item()}")
            axes[0, i].axis('off')
            
            # Imagen perturbada
            if data.size(1) == 1:
                axes[1, i].imshow(perturbed_data[i, 0].detach().cpu().numpy(), cmap='gray')
            else:
                axes[1, i].imshow(np.transpose(perturbed_data[i].detach().cpu().numpy(), (1, 2, 0)))
            
            axes[1, i].set_title(f"Adversarial\nReal: {target[i].item()}\nPred: {pred_adv[i].item()}")
            axes[1, i].axis('off')
            
            # Perturbación
            if data.size(1) == 1:
                axes[2, i].imshow(perturbation[i, 0].detach().cpu().numpy(), cmap='viridis')
            else:
                # Para imágenes a color, mostramos la magnitud de la perturbación
                perturbation_mag = torch.norm(perturbation[i], dim=0).detach().cpu().numpy()
                axes[2, i].imshow(perturbation_mag, cmap='viridis')
            
            axes[2, i].set_title("Perturbación")
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def adversarial_training(model, train_loader, optimizer, loss_fn, device, attack_fn, epochs=1, 
                            alpha=0.5, eval_loader=None, **attack_params):
        """
        Realiza entrenamiento adversarial para mejorar la robustez del modelo.
        
        Fórmula: min_θ E_(x,y)~D [max_δ∈S L(θ, x+δ, y)]
        
        Argumentos:
            model: El modelo a entrenar
            train_loader: El cargador de datos de entrenamiento
            optimizer: El optimizador
            loss_fn: La función de pérdida
            device: El dispositivo donde realizar el entrenamiento
            attack_fn: La función de ataque a utilizar
            epochs: Número de épocas
            alpha: Peso para la pérdida adversarial (0.5 significa 50% normal, 50% adversarial)
            eval_loader: Cargador de datos para evaluación (opcional)
            attack_params: Parámetros adicionales para la función de ataque
            
        Retorna:
            Historial de entrenamiento
        """
        history = {
            'train_loss': [],
            'clean_accuracy': [],
            'adversarial_accuracy': []
        }
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                # Generar ejemplos adversariales
                # No pasamos data.clone() sino directamente data
                perturbed_data = attack_fn(model, data, target, loss_fn=loss_fn, **attack_params)
                
                # Entrenamiento con ejemplos limpios y adversariales
                optimizer.zero_grad()
                
                # Pérdida con datos originales
                output_clean = model(data)
                loss_clean = loss_fn(output_clean, target)
                
                # Pérdida con datos perturbados
                output_adv = model(perturbed_data)
                loss_adv = loss_fn(output_adv, target)
                
                # Combinar pérdidas (entrenamiento adversarial)
                loss = (1 - alpha) * loss_clean + alpha * loss_adv
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    print(f"Época {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                        f"Pérdida: {loss.item():.4f}")
            
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # Evaluar si se proporciona un cargador de evaluación
            if eval_loader is not None:
                robustness_metrics = AdversarialUtils.evaluate_model_robustness(
                    model, eval_loader, attack_fn, device, **attack_params
                )
                
                history['clean_accuracy'].append(robustness_metrics['clean_accuracy'])
                history['adversarial_accuracy'].append(robustness_metrics['adversarial_accuracy'])
                
                print(f"Época {epoch+1}/{epochs}, Pérdida: {avg_train_loss:.4f}, "
                    f"Precisión limpia: {robustness_metrics['clean_accuracy']:.2f}%, "
                    f"Precisión adversarial: {robustness_metrics['adversarial_accuracy']:.2f}%")
            else:
                # Si no hay eval_loader, calcular precisión sobre los datos de entrenamiento
                # para al menos tener algo que mostrar en la gráfica
                model.eval()
                correct_clean = 0
                correct_adv = 0
                total = 0
                
                with torch.no_grad():
                    for data, target in train_loader:
                        data, target = data.to(device), target.to(device)
                        total += target.size(0)
                        
                        # Precisión limpia
                        output = model(data)
                        pred = output.argmax(dim=1)
                        correct_clean += (pred == target).sum().item()
                        
                        # Precisión adversarial (usando el mismo ataque)
                        perturbed_data = attack_fn(model, data, target, loss_fn=loss_fn, **attack_params)
                        output_adv = model(perturbed_data)
                        pred_adv = output_adv.argmax(dim=1)
                        correct_adv += (pred_adv == target).sum().item()
                
                clean_acc = 100.0 * correct_clean / total
                adv_acc = 100.0 * correct_adv / total
                
                history['clean_accuracy'].append(clean_acc)
                history['adversarial_accuracy'].append(adv_acc)
                
                print(f"Época {epoch+1}/{epochs}, Pérdida: {avg_train_loss:.4f}, "
                    f"Precisión limpia (train): {clean_acc:.2f}%, "
                    f"Precisión adversarial (train): {adv_acc:.2f}%")
            
            # Volver a modo entrenamiento para la siguiente época
            model.train()
        
        return history
    
    @staticmethod
    def plot_training_history(history):
        """
        Grafica el historial de entrenamiento adversarial.
        
        Argumentos:
            history: Historial de entrenamiento obtenido de adversarial_training
            
        Retorna:
            None (muestra un gráfico)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Graficar pérdida
        epochs = range(1, len(history['train_loss']) + 1)
        ax1.plot(epochs, history['train_loss'])
        ax1.set_title('Pérdida de entrenamiento')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Pérdida')
        
        # Graficar precisión
        if 'clean_accuracy' in history and 'adversarial_accuracy' in history and len(history['clean_accuracy']) > 0:
            ax2.plot(epochs, history['clean_accuracy'], label='Limpia')
            ax2.plot(epochs, history['adversarial_accuracy'], label='Adversarial')
            ax2.set_title('Precisión durante el entrenamiento')
            ax2.set_xlabel('Época')
            ax2.set_ylabel('Precisión (%)')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No hay datos de precisión disponibles', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax2.transAxes)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def feature_squeezing(x, bit_depth=5):
        """
        Implementa la defensa de feature squeezing reduciendo la profundidad de bits.
        
        Argumentos:
            x: La imagen de entrada (tensor)
            bit_depth: La profundidad de bits a la que reducir
            
        Retorna:
            La imagen con profundidad de bits reducida
        """
        max_val = 2**bit_depth - 1
        x_int = torch.round(x * max_val)
        x_squeezed = x_int / max_val
        return x_squeezed
    
    @staticmethod
    def defensive_distillation(teacher_model, student_model, data_loader, device, temperature=10, epochs=5):
        """
        Implementa la destilación defensiva.
        
        Argumentos:
            teacher_model: Modelo maestro ya entrenado
            student_model: Modelo estudiante por entrenar
            data_loader: Cargador de datos
            device: Dispositivo
            temperature: Temperatura para suavizar las distribuciones
            epochs: Número de épocas para entrenar el estudiante
            
        Retorna:
            El modelo estudiante entrenado
        """
        teacher_model.eval()
        student_model.train()
        
        optimizer = torch.optim.Adam(student_model.parameters())
        
        for epoch in range(epochs):
            running_loss = 0.0
            
            for data, _ in data_loader:
                data = data.to(device)
                
                # Obtener salidas suavizadas del teacher
                with torch.no_grad():
                    teacher_outputs = teacher_model(data)
                    # Aplicar temperatura
                    teacher_outputs = F.log_softmax(teacher_outputs / temperature, dim=1)
                
                # Entrenar el estudiante para imitar al teacher
                optimizer.zero_grad()
                student_outputs = student_model(data)
                # Aplicar la misma temperatura
                student_outputs = F.log_softmax(student_outputs / temperature, dim=1)
                
                # Pérdida KL divergence
                loss = F.kl_div(student_outputs, teacher_outputs, reduction='batchmean')
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            print(f"Época {epoch+1}/{epochs}, Pérdida: {running_loss/len(data_loader):.4f}")
        
        return student_model
    
    @staticmethod
    def visualize_defense_comparison(model, data, target, attack_fn, defense_fn, device, **attack_params):
        """
        Visualiza la efectividad de una defensa contra un ataque.
        
        Argumentos:
            model: El modelo a evaluar
            data: Los datos de entrada
            target: Las etiquetas verdaderas
            attack_fn: La función de ataque a utilizar
            defense_fn: La función de defensa a evaluar
            device: El dispositivo donde realizar la evaluación
            attack_params: Parámetros adicionales para la función de ataque
            
        Retorna:
            None (muestra un gráfico)
        """
        model.eval()
        
        # Asegurar que los datos estén en el dispositivo correcto
        data, target = data.to(device), target.to(device)
        
        # Predicción en datos limpios
        with torch.no_grad():
            output = model(data)
            pred_clean = output.argmax(dim=1)
        
        # Generar ejemplos adversariales
        loss_fn = torch.nn.CrossEntropyLoss()
        perturbed_data = attack_fn(model, data, target, loss_fn=loss_fn, **attack_params)
        
        # Predicción en datos adversariales
        with torch.no_grad():
            output = model(perturbed_data)
            pred_adv = output.argmax(dim=1)
        
        # Aplicar defensa
        defended_data = defense_fn(perturbed_data)
        
        # Predicción en datos defendidos
        with torch.no_grad():
            output = model(defended_data)
            pred_def = output.argmax(dim=1)
        
        # Visualizar
        num_examples = min(5, data.size(0))
        fig, axes = plt.subplots(4, num_examples, figsize=(num_examples * 3, 12))
        
        for i in range(num_examples):
            # Imagen original
            if data.size(1) == 1:  # Imágenes en escala de grises
                axes[0, i].imshow(data[i, 0].detach().cpu().numpy(), cmap='gray')
            else:  # Imágenes a color
                axes[0, i].imshow(np.transpose(data[i].detach().cpu().numpy(), (1, 2, 0)))
            
            axes[0, i].set_title(f"Original\nPred: {pred_clean[i].item()}")
            axes[0, i].axis('off')
            
            # Imagen perturbada
            if data.size(1) == 1:
                axes[1, i].imshow(perturbed_data[i, 0].detach().cpu().numpy(), cmap='gray')
            else:
                axes[1, i].imshow(np.transpose(perturbed_data[i].detach().cpu().numpy(), (1, 2, 0)))
            
            axes[1, i].set_title(f"Adversarial\nPred: {pred_adv[i].item()}")
            axes[1, i].axis('off')
            
            # Imagen defendida
            if data.size(1) == 1:
                axes[2, i].imshow(defended_data[i, 0].detach().cpu().numpy(), cmap='gray')
            else:
                axes[2, i].imshow(np.transpose(defended_data[i].detach().cpu().numpy(), (1, 2, 0)))
            
            axes[2, i].set_title(f"Defendida\nPred: {pred_def[i].item()}")
            axes[2, i].axis('off')
            
            # Diferencia entre original y defendida
            diff = torch.abs(defended_data[i] - data[i])
            if data.size(1) == 1:
                axes[3, i].imshow(diff[0].detach().cpu().numpy(), cmap='viridis')
            else:
                diff_mag = torch.norm(diff, dim=0).detach().cpu().numpy()
                axes[3, i].imshow(diff_mag, cmap='viridis')
            
            axes[3, i].set_title("Diferencia con original")
            axes[3, i].axis('off')
        
        plt.tight_layout()
        plt.show()