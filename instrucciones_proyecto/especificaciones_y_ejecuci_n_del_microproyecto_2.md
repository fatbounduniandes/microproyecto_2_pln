# Directiva de Ejecución: Microproyecto 2 - Generación de Código con Transformers

## 1. Identificación y Objetivo del Proyecto
* **Asignatura:** Modelos Avanzados para el Procesamiento de Lenguaje Natural (MAPLN) - Maestría en Inteligencia Artificial.
* **Objetivo Principal:** Implementar, entrenar, evaluar y comparar cinco (5) configuraciones de Modelos de Lenguaje autorregresivos basados en la arquitectura Transformer (decoder-only) para la tarea de generación de código fuente en Python.
* **Entorno de Ejecución:** Jupyter Notebook ejecutable de principio a fin (`Run All`) en entorno PyTorch con aceleración GPU (p. ej., NVIDIA T4 o superior).

---

## 2. Configuraciones Experimentales Obligatorias
El agente debe configurar, entrenar y evaluar exactamente cinco configuraciones:

1. **Configuración 1 (Modelo desde cero):**
   * Arquitectura: Transformer *decoder-only* tipo GPT implementada en PyTorch puro.
   * Tokenizador: Debe usar **obligatoriamente** el tokenizador de la familia SmolLM2 (vocabulario de 49.152 tokens).
   * Tamaño de contexto: 256 tokens.
   * Entrenamiento: Desde pesos aleatorios sobre el conjunto de entrenamiento.

2. **Configuración 2 (Baseline SmolLM2-135M):**
   * Modelo: `HuggingFaceTB/SmolLM2-135M` descargado directamente de Hugging Face.
   * Estado: Pesos pre-entrenados originales, **sin finetuning** (evaluación directa en inferencia).

3. **Configuración 3 (Finetuning SmolLM2-135M):**
   * Modelo base: `HuggingFaceTB/SmolLM2-135M`.
   * Entrenamiento: **Finetuning completo** (actualización del 100% de los parámetros; terminantemente prohibido LoRA/QLoRA).

4. **Configuración 4 (Baseline SmolLM2-360M):**
   * Modelo: `HuggingFaceTB/SmolLM2-360M` descargado directamente de Hugging Face.
   * Estado: Pesos pre-entrenados originales, **sin finetuning** (evaluación directa en inferencia).

5. **Configuración 5 (Finetuning SmolLM2-360M):**
   * Modelo base: `HuggingFaceTB/SmolLM2-360M`.
   * Entrenamiento: **Finetuning completo** (actualización del 100% de los parámetros; terminantemente prohibido LoRA/QLoRA).

---

## 3. Pipeline de Datos y Restricciones Estrictas

### 3.1 Dataset Oficial
* **Fuente:** `codeparrot/codeparrot-clean` (Hugging Face).
* **Modo de Carga:** Modo `streaming=True` obligatorio para gestionar eficientemente la memoria RAM/disco.
* **Prohibición:** Prohibido usar otros datasets o fuentes externas de datos.

### 3.2 Filtro y Split de Datos
1. **Filtro de Longitud:** Conservar exclusivamente archivos cuyo campo `content` tenga entre 1.000 y 2.000 caracteres ($1000 \le \text{len(code)} \le 2000$). Descartar cualquier archivo fuera de este rango.
2. **Límite de Muestras:** Extraer exactamente $8.000$ ejemplos filtrados.
3. **Split Obligatorio (80 / 10 / 10):**
   * **Entrenamiento:** $6.400$ ejemplos.
   * **Validación:** $800$ ejemplos (monitoreo de sobreajuste y detención temprana).
   * **Prueba:** $800$ ejemplos (evaluación final cuantitativa; reservado exclusivamente para métricas finales).
4. **Tokenización y Segmentación:**
   * Utilizar `AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")` (vocabulario: 49.152).
   * Segmentar y empacar el texto en secuencias de longitud de contexto fija de **256 tokens** para todos los modelos.

---

## 4. Hiperparámetros Recomendados y Entrenamiento

| Parámetro | Modelo desde Cero | SmolLM2-135M (Finetuning) | SmolLM2-360M (Finetuning) |
| :--- | :--- | :--- | :--- |
| **Optimizador** | AdamW | AdamW | AdamW |
| **Learning Rate ($lr$)** | $\approx 3 \times 10^{-4}$ | $\approx 5 \times 10^{-5}$ | $\approx 3 \times 10^{-5}$ |
| **Épocas sugeridas** | $\approx 3$ épocas | $1$ época completa | $1$ época completa |
| **Context Length** | 256 tokens | 256 tokens | 256 tokens |
| **Batch Efectivo** | 16 | 16 | 16 |
| **Estrategia Memoria** | Estándar o AMP | AMP + Grad Accum + Grad Checkpoint | AMP + Grad Accum ($8$) + Checkpoint |

---

## 5. Implementación de Evaluación y Métricas

### 5.1 Perplejidad ($PPL$) - Métrica Cuantitativa Principal
* Calcular la perplejidad sobre la totalidad del conjunto de prueba ($800$ ejemplos) a nivel de token.
* **Fórmula de Cálculo:**
  $$PPL = \exp\left( -\frac{1}{N} \sum_{i=1}^{N} \log p(x_i \mid x_{<i}) \right)$$
  Donde $N$ es el total de tokens evaluados.
* **Instrucción de Implementación:**
  * Acumular la suma de la entropía cruzada negativa ($-\log p$) multiplicada por el número de tokens válidos de cada batch.
  * Dividir por la sumatoria acumulada de tokens válidos ($N$) y luego aplicar $\exp(\cdot)$.
  * Excluir posiciones de padding del cálculo de $N$.
  * **Regla estricta:** No promediar las perplejidades calculadas por lote.

### 5.2 Validez Sintáctica con AST - Métrica Secundaria
* Seleccionar una submuestra reproducible de $200$ ejemplos del conjunto de prueba fijando una semilla aleatoria (`seed`).
* Para cada ejemplo:
  1. Extraer un prefijo inicial como *prompt*.
  2. Generar la continuación autorregresiva usando la función propia.
  3. Descartar la última línea generada si quedó incompleta o truncada por el límite de tokens.
  4. Analizar la validez sintáctica utilizando la librería estándar de Python:
     ```python
     import ast
     try:
         ast.parse(codigo_generado)
         es_valido = True
     except SyntaxError:
         es_valido = False
     ```
  5. Reportar el porcentaje de secuencias sintácticamente válidas ($\% = \frac{\text{válidas}}{200} \times 100$).

### 5.3 Función de Generación Autorregresiva Propia (REQUISITO CRÍTICO)
* **Prohibición explícita:** Está estrictamente prohibido usar `model.generate()`, pipelines o cualquier abstracción de alto nivel.
* **Requisito:** Implementar un bucle autorregresivo desde cero:
  1. Recibir un tensor de tokens de entrada (`input_ids`) de tamaño $(B, T)$.
  2. Obtener los logits del modelo en la última posición: $\text{logits}[:, -1, :]$.
  3. Aplicar escalado por temperatura, Top-$K$ o Top-$P$ (opcional/recomendado) o muestreo directo mediante Softmax y `torch.multinomial` (o `torch.argmax`).
  4. Concatenar el nuevo token generado al tensor de contexto.
  5. Repetir hasta alcanzar la longitud máxima o el token `<|endoftext|>`.
  6. Utilizar esta misma función para generar al menos **3 ejemplos cualitativos** con cada uno de los 3 modelos entrenados.

---

## 6. Estructura Obligatoria del Jupyter Notebook
El notebook debe contener exactamente las siguientes diez (10) secciones con estos títulos precisos y en este orden cronológico:

1. **Portada e introducción:** Datos del estudiante, contexto del problema, descripción técnica de las 5 configuraciones a comparar.
2. **Librerías:** Importación controlada de librerías (`torch`, `transformers`, `datasets`, `ast`, etc.).
3. **Carga del dataset y exploración:** Carga de `codeparrot/codeparrot-clean` en streaming, inspección de estructura y visualización de distribuciones.
4. **Preprocesamiento:** Aplicación del filtro $[1000, 2000]$ caracteres, tokenización SmolLM2 y empaquetado a bloques de 256 tokens.
5. **Split train / validación / prueba:** Partición matemática exacta de los 8.000 ejemplos en 6.400 (Train), 800 (Val) y 800 (Test).
6. **Modelo desde cero:** Implementación de la arquitectura GPT Decoder-Only (Multi-Head Attention, MLP, LayerNorm, Embeddings posicionales/token), bucle de entrenamiento y curvas de pérdida.
7. **Finetuning de los modelos pre-entrenados:** Procedimiento para SmolLM2-135M y SmolLM2-360M aplicando optimizaciones de memoria GPU y guardado de checkpoints.
8. **Evaluación y comparación:** Tabla y gráficas de perplejidad y porcentaje de validez sintáctica para las 5 configuraciones; análisis de sobreajuste y curvas de pérdida.
9. **Generación de ejemplos:** Implementación del bucle autorregresivo propio y despliegue de mínimo 3 ejemplos generados por: modelo desde cero, 135M finetuned y 360M finetuned.
10. **Conclusiones:** Discusión crítica de resultados empíricos, impacto del tamaño del modelo, limitaciones sintácticas y aprendizajes.

---

## 7. Protocolo de Gestión de Checkpoints y Entregables
1. **Guardado de Checkpoints:**
   * Modelo desde cero: Guardar con `torch.save(modelo.state_dict(), 'checkpoint_scratch.pt')`.
   * Modelos finetuneados: Guardar con `model.save_pretrained('./checkpoint_smollm2_135m_ft')` y `model.save_pretrained('./checkpoint_smollm2_360m_ft')`.
2. **Liberación de Memoria GPU entre Modelos:**
   * Nunca mantener dos modelos en memoria de GPU en paralelo.
   * Al finalizar cada entrenamiento: guardar checkpoint, ejecutar `del model`, `del optimizer`, `torch.cuda.empty_cache()` y llamar a `gc.collect()`.
3. **Empaquetado y Entrega:**
   * Crear archivo `.zip` que contenga: el Jupyter Notebook con todas las celdas ejecutadas y las salidas visibles, junto con las carpetas/archivos de los 3 checkpoints entrenados.
   * Subir a Google Drive con permiso público: *"Cualquier persona que tenga el vínculo (lector)"*. Probar descarga en modo incógnito.