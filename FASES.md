# Fases del Microproyecto 2

Avanzamos **una fase a la vez**: solo se marca `[x]` cuando Jean lo autoriza, y los `git push`
solo se hacen cuando él lo pide.

## Checklist

- [x] **Fase 1 — Repositorio y base** (no va en el notebook)
- [x] **Fase 2 — Portada y librerías** (secciones 1 y 2)
- [x] **Fase 3 — Datos y exploración** (sección 3)
- [ ] **Fase 4 — Preprocesamiento** (sección 4)
- [ ] **Fase 5 — Split train/val/test** (sección 5)
- [ ] **Fase 6 — Modelo desde cero** (sección 6)
- [ ] **Fase 7 — Finetuning 135M y 360M** (sección 7)
- [ ] **Fase 8 — Evaluación y comparación** (sección 8)
- [ ] **Fase 9 — Generación de ejemplos** (sección 9)
- [ ] **Fase 10 — Conclusiones** (sección 10)
- [ ] **Fase 11 — Smoke test en Colab T4**
- [ ] **Fase 12 — Run final en L4 y entrega**

---

## Detalle de cada fase

### Fase 1 — Repositorio y base (no va en el notebook)
- Git, `.gitignore`, `.gitattributes`, `AGENTS.md`.
- `requirements-colab.txt`, `build_notebook.py` y las celdas fuente.
- **Lista cuando:** `python3 build_notebook.py --check` responde `OK`.

### Fase 2 — Portada y librerías
- Portada con datos del equipo, resumen y tabla de las 5 configuraciones.
- Instalación, imports, GPU/precisión mixta, semilla y constantes.
- **Lista cuando:** en Colab se ve el nombre de la GPU, el `AMP` y `Configuración lista`.

### Fase 3 — Datos y exploración
- Carga de `codeparrot/codeparrot-clean` en streaming.
- Inspección de columnas y gráfica de longitudes.
- **Lista cuando:** se imprimen ejemplos reales y la distribución de longitudes.
- **Resultados:** 11 columnas; 2.000 archivos medidos en 1,6 s; **399 de 2.000 (19,9 %)** caen
  en el rango 1.000–2.000 caracteres. Proyectando, hacen falta ~40.000 archivos para juntar los
  8.000 del enunciado.

### Fase 4 — Preprocesamiento
- Filtro de 1.000 a 2.000 caracteres.
- Tokenización con SmolLM2 y empaquetado en bloques de 256.
- **Lista cuando:** hay 8.000 ejemplos filtrados y los bloques tienen forma `(N, 256)`.

### Fase 5 — Split train/validación/prueba
- 6.400 / 800 / 800 sin solapamiento.
- **Lista cuando:** los tres tamaños cuadran y se verifica que no haya ejemplos repetidos.

### Fase 6 — Modelo desde cero
- GPT decoder-only en PyTorch puro (atención multi-cabeza, MLP, LayerNorm, embeddings).
- Entrenamiento y curvas de pérdida.
- **Lista cuando:** la pérdida baja y el checkpoint se guarda en Drive.

### Fase 7 — Finetuning de 135M y 360M
- Full finetuning con AMP + acumulación de gradientes + gradient checkpointing.
- **Lista cuando:** ambos checkpoints existen y no hubo OOM.

### Fase 8 — Evaluación y comparación
- Perplejidad global de las 5 configuraciones.
- Validez sintáctica con `ast` en 200 ejemplos.
- Tabla y gráficas comparativas.
- **Lista cuando:** la tabla queda completa con las 5 filas y ambas métricas.

### Fase 9 — Generación de ejemplos
- Bucle autorregresivo propio (sin `model.generate()`).
- 3 ejemplos por cada uno de los 3 modelos entrenados.
- **Lista cuando:** se ven los 9 ejemplos generados.

### Fase 10 — Conclusiones
- Análisis de resultados, tamaño del modelo, límites sintácticos y aprendizajes.
- **Lista cuando:** el notebook cierra con la discusión y no quedan celdas pendientes.

### Fase 11 — Smoke test en Colab T4
- Corrida completa con menos datos para detectar errores rápido.
- **Lista cuando:** `Run All` termina sin excepciones en T4.

### Fase 12 — Run final en L4 y entrega
- `Run All` en L4 con los hiperparámetros definitivos.
- `.zip` con notebook ejecutado + 3 checkpoints, subido a Drive con permiso de lector.
- **Lista cuando:** la descarga funciona en modo incógnito.
