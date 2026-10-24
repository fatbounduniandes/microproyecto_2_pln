# Anexo Técnico: Reducción del Consumo de Memoria VRAM durante el Finetuning Completo

## 1. Justificación y Desafío de Memoria
Durante el *full finetuning* de **SmolLM2-135M** y **SmolLM2-360M**, se actualizan la totalidad de los parámetros. La memoria VRAM consumida en GPU no solo almacena los pesos, sino tres factores críticos adicionales:
1. **Gradientes de los parámetros:** Mismo tamaño que los pesos ($4 \text{ bytes}$ por parámetro en FP32).
2. **Estados del optimizador (AdamW):** Mantiene dos momentos por parámetro (primer y segundo momento), consumiendo el doble de memoria que los pesos maestros ($8 \text{ bytes}$ por parámetro).
3. **Activaciones intermedias:** Almacenadas en cada capa durante el *forward pass* para calcular la regla de la cadena en el *backward pass*. Crecen con la longitud de secuencia, el batch size y la profundidad de la red.

Para entrenar modelos como SmolLM2-360M en GPUs estándar (como NVIDIA T4 de 16 GB), se debe implementar una tríada de optimización ortogonal.

---

## 2. Tríada de Técnicas de Optimización

| Técnica | Mecanismo de Ahorro | Impacto en VRAM | Costo / Compensación |
| :--- | :--- | :--- | :--- |
| **Precisión Mixta (AMP: bf16 / fp16)** | Realiza cálculos forward/backward en 16 bits; mantiene copia maestra en 32 bits. | Reduce a la mitad la memoria de activaciones y tensores de cálculo. | Ligero riesgo de underflow numérico si no se calibra adecuadamente. |
| **Acumulación de Gradientes** | Procesa micro-batches pequeños y acumula gradientes en múltiples pasos antes de actualizar pesos. | Permite desacoplar el batch efectivo del tamaño en memoria instantánea. | Mayor tiempo de ejecución por paso de optimización (más pasadas hacia adelante/atrás). |
| **Gradient Checkpointing** | Guarda activaciones solo en capas clave y recalcula el resto durante el backward. | Reduce drásticamente la memoria de activaciones ($\sim O(\sqrt{L})$ en lugar de $O(L)$). | Aumento del cómputo computacional en backward pass ($\approx 20\% - 30\%$). |

---

## 3. Especificación Técnica por Componente

### 3.1 Precisión Mixta Automática (AMP)
* **bf16 (Bfloat16):**
  * Asignación: 8 bits de exponente (igual que FP32) y 7 bits de mantisa.
  * Ventaja: Mismo rango dinámico que FP32, previniendo el desbordamiento o colapso a cero (*underflow*).
  * Regla: No requiere escalador de pérdida (`GradScaler`). Recomendado para arquitecturas NVIDIA Ampere o superiores (A100, RTX 30xx/40xx, T4 con soporte simulado según versión).
* **fp16 (Float16):**
  * Asignación: 5 bits de exponente y 10 bits de mantisa. Mayor precisión local pero rango dinámico estrecho.
  * Regla: Requiere obligatoriamente escalado de pérdida (`torch.cuda.amp.GradScaler`) para evitar que gradientes pequeños colapsen a cero.

### 3.2 Acumulación de Gradientes
* **Fórmula de Batch Efectivo:**
  $$\text{Batch Efectivo} = \text{micro\_batch} \times \text{grad\_accum}$$
* **Configuración Estándar para SmolLM2-360M:**
  * $\text{micro\_batch} = 2$
  * $\text{grad\_accum} = 8$
  * $\text{Batch Efectivo} = 2 \times 8 = 16$
* **Regla de Normalización:**
  La pérdida devuelta por el modelo debe dividirse estrictamente entre `grad_accum` antes de ejecutar `backward()`:
  $$\mathcal{L}_{\text{normalizada}} = \frac{\mathcal{L}_{\text{original}}}{\text{grad\_accum}}$$
  Esto garantiza que la magnitud de los gradientes acumulados equivalga exactamente al promedio de un batch de tamaño 16.

### 3.3 Gradient Checkpointing
* **Activación en Hugging Face Transformers:**
  ```python
  model.gradient_checkpointing_enable()
  model.config.use_cache = False  # Obligatorio durante el entrenamiento
  model.train()
  ```
* **Regla Crítica del KV-Cache:**
  El parámetro `use_cache` almacena llaves y valores para inferencia autorregresiva rápida. Durante el entrenamiento es redundante e incompatible con el recalculo de gradientes de Gradient Checkpointing, por lo que debe desactivarse de forma forzada (`model.config.use_cache = False`).

---

## 4. Patrón de Implementación del Bucle de Entrenamiento (Template para el Agente)

El agente de IA debe implementar el bucle de finetuning integrando las tres técnicas bajo la siguiente arquitectura de código:

```python
import torch
import torch.nn.utils as nn_utils

# 1. Parámetros de optimización de hardware
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_BF16 = torch.cuda.is_bf16_supported()
AMP_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16
USE_SCALER = not USE_BF16

MICRO_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8  # Batch efectivo = 16
CLIP_GRAD_NORM = 1.0

# 2. Configurar modelo
model.gradient_checkpointing_enable()
model.config.use_cache = False
model.to(DEVICE)
model.train()

# 3. Inicializar Scaler y Optimizador
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
scaler = torch.cuda.amp.GradScaler(enabled=USE_SCALER)

optimizer.zero_grad(set_to_none=True)

# 4. Bucle de entrenamiento por micro-batches
for step, batch in enumerate(train_dataloader):
    # Enviar datos a GPU
    input_ids = batch['input_ids'].to(DEVICE)
    labels = batch['labels'].to(DEVICE)
    
    # Forward Pass bajo Autocast
    with torch.autocast(device_type='cuda', dtype=AMP_DTYPE):
        outputs = model(input_ids=input_ids, labels=labels)
        # Normalización obligatoria por pasos de acumulación
        loss = outputs.loss / GRAD_ACCUM_STEPS
        
    # Backward Pass escalado
    scaler.scale(loss).backward()
    
    # Actualización periódica según grad_accum
    if (step + 1) % GRAD_ACCUM_STEPS == 0:
        # Desescalar gradientes antes del clipping
        scaler.unscale_(optimizer)
        nn_utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
        
        # Paso de optimización y actualización de escala
        scaler.step(optimizer)
        scaler.update()
        
        # Reiniciar gradientes liberando memoria
        optimizer.zero_grad(set_to_none=True)
```

---

## 5. Protocolo de Liberación de Memoria GPU entre Sesiones
Para garantizar que SmolLM2-135M y SmolLM2-360M no colisionen en VRAM:

```python
import gc
import torch

def liberar_vram_total(model, optimizer=None):
    """Elimina referencias y purga la memoria caché de PyTorch en GPU."""
    if model is not None:
        del model
    if optimizer is not None:
        del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
```