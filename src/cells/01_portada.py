# %% [markdown]
# # Microproyecto 2: Generación de Código con Transformers
#
# **Curso:** Modelos Avanzados para el Procesamiento de Lenguaje Natural
# **Equipo 4**
# **Integrantes:** Juan Sebastián Páez Cortés, Julian David Parada Galvis, Luis Carlos Prada Socha, Jean Pierre Betancourt Padilla
# **Fecha:** octubre de 2026
#
# ---
#
# ## 1. Portada e introducción
#
# ### Resumen
#
# En este microproyecto entrenamos modelos tipo GPT (Transformer decoder-only) para generar
# código en Python. El modelo aprende a predecir el siguiente token de un archivo de código y,
# con eso, puede continuar un fragmento que le demos como entrada.
#
# Comparamos cinco configuraciones para ver cuánto depende el resultado del tamaño del modelo
# y de si se entrena desde cero o se parte de pesos ya preentrenados:
#
# | # | Configuración | Modelo | Entrenamiento |
# | :-: | :--- | :--- | :--- |
# | 1 | Desde cero | GPT decoder-only en PyTorch | 3 épocas desde pesos aleatorios |
# | 2 | Baseline 135M | SmolLM2-135M | Sin entrenar |
# | 3 | Finetuning 135M | SmolLM2-135M | Full finetuning, 1 época |
# | 4 | Baseline 360M | SmolLM2-360M | Sin entrenar |
# | 5 | Finetuning 360M | SmolLM2-360M | Full finetuning, 1 época |
#
# ### Datos
#
# Usamos `codeparrot/codeparrot-clean` en modo streaming. Nos quedamos solo con archivos de
# entre 1.000 y 2.000 caracteres, tomamos 8.000 ejemplos y los dividimos en 6.400 de
# entrenamiento, 800 de validación y 800 de prueba como lo pide el proyecto. Todos los modelos
# usan el tokenizador de SmolLM2 (49.152 tokens) y un contexto de 256 tokens. El filtro, el split
# y el contexto son los que exige la guía del proyecto.
#
# ### Evaluación
#
# - Perplejidad sobre los 800 ejemplos de prueba.
# - Porcentaje de código sintácticamente válido con `ast` en 200 ejemplos.
# - Tres ejemplos por cada modelo entrenado, con nuestro propio bucle autorregresivo.
#
# ### Entorno
#
# El notebook corre completo en Google Colab con GPU L4. Para el modelo de 360M usamos
# precisión mixta, acumulación de gradientes y gradient checkpointing, que es la estrategia de
# memoria que recomienda el personal del curso. Entre un modelo y el siguiente liberamos la
# memoria de la GPU.
#
# Todo el pipeline es reproducible: fijamos la semilla y la revisión del dataset y del
# tokenizador, de modo que dos ejecuciones producen los mismos 8.000 archivos y el mismo split.
