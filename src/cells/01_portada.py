# %% [markdown]
# # Microproyecto 2: Generación de Código con Transformers
#
# **Curso:** Modelos Avanzados para el Procesamiento de Lenguaje Natural (MAPLN)
# **Equipo 4**
# **Integrantes:** Juan Sebastián Páez Cortés, Julian David Parada Galvis, Luis Carlos Prada Socha, Jean Pierre Betancourt Padilla
# **Fecha:** octubre de 2026
#
# ---
#
# ## 1. Portada e introducción
#
# ### 1.1 Resumen del proyecto
#
# En el Microproyecto 1 se modeló una secuencia de trazos de un boceto
# $(\Delta x_t, \Delta y_t, \text{pen\_state}_t)$ con una LSTM implementada desde cero.
# Este microproyecto conserva la misma intuición —modelar una secuencia token a token— pero
# cambia el dominio: ahora se modela **código fuente en Python**, usando la arquitectura
# *decoder-only* de los Transformers, la misma familia de GPT.
#
# La tarea es autorregresiva: dado un prefijo de tokens, el modelo estima
# $p(x_t \mid x_{<t})$ y se entrena minimizando la entropía cruzada. Se comparan **cinco
# configuraciones** que aíslan el efecto de dos variables: el tamaño del modelo (135M vs. 360M)
# y el tipo de entrenamiento (desde cero vs. *full finetuning* vs. pesos originales).
#
# **Pipeline de datos.** Se usa `codeparrot/codeparrot-clean` en modo `streaming=True`. Se
# conservan únicamente archivos con `1000 <= len(content) <= 2000` caracteres, se extraen
# exactamente 8.000 ejemplos y se dividen en 6.400 de entrenamiento, 800 de validación y 800 de
# prueba. Todos los modelos comparten el tokenizador de `HuggingFaceTB/SmolLM2-135M`
# (vocabulario de 49.152 tokens) y un contexto fijo de 256 tokens, con empaquetado en bloques.
#
# **Evaluación.** Tres frentes complementarios:
#
# 1. **Perplejidad global** sobre los 800 ejemplos de prueba, acumulando
#    $-\log p$ por token válido y dividiendo una sola vez por el total de tokens (nunca
#    promediando perplejidades por lote).
# 2. **Validez sintáctica** con `ast.parse` sobre una submuestra reproducible de 200 ejemplos.
# 3. **Generación cualitativa** mediante un bucle autorregresivo propio, sin `model.generate()`.
#
# ### 1.2 Las cinco configuraciones
#
# | # | Configuración | Modelo / origen | Estado de los pesos | Entrenamiento | $lr$ | Épocas |
# | :-: | :--- | :--- | :--- | :--- | :--- | :--- |
# | 1 | Desde cero | GPT decoder-only en PyTorch puro | Aleatorios | 3 épocas sobre *train* | $3\times10^{-4}$ | 3 |
# | 2 | Baseline 135M | `HuggingFaceTB/SmolLM2-135M` | Preentrenados | Ninguno (inferencia) | — | — |
# | 3 | Finetuning 135M | `HuggingFaceTB/SmolLM2-135M` | Preentrenados | Full finetuning (100 %) | $5\times10^{-5}$ | 1 |
# | 4 | Baseline 360M | `HuggingFaceTB/SmolLM2-360M` | Preentrenados | Ninguno (inferencia) | — | — |
# | 5 | Finetuning 360M | `HuggingFaceTB/SmolLM2-360M` | Preentrenados | Full finetuning (100 %) | $3\times10^{-5}$ | 1 |
#
# En todas las configuraciones el batch efectivo es 16. Queda terminantemente prohibido el uso de
# LoRA/QLoRA: el finetuning actualiza el 100 % de los parámetros.
#
# ### 1.3 Entorno de ejecución
#
# El notebook está pensado para ejecutarse de principio a fin (`Run All`) en **Google Colab Pro**
# con GPU **NVIDIA L4** (~24 GB, `bfloat16` nativo); el código degrada automáticamente a `fp16`
# con `GradScaler` en T4. El *full finetuning* de SmolLM2-360M se sostiene con la tríada del
# anexo técnico: **AMP + acumulación de gradientes (micro-batch 2 × 8) + gradient checkpointing**.
# Solo hay un modelo en VRAM a la vez: al terminar cada fase se guarda el checkpoint, se liberan
# las referencias y se purga la caché de CUDA.
#
# ### 1.4 Entregables
#
# - `checkpoint_scratch.pt` — modelo desde cero (`state_dict`).
# - `checkpoint_smollm2_135m_ft/` y `checkpoint_smollm2_360m_ft/` — modelos finetuneados.
# - Este notebook, ejecutado con todas sus salidas visibles, empaquetado junto a los checkpoints.
