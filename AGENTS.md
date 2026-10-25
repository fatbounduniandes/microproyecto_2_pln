# AGENTS.md — Bitácora de decisiones del Microproyecto 2

Este archivo es la **fuente única de verdad sobre las decisiones de ingeniería** del proyecto.
Cualquier agente o persona que modifique el repositorio debe leerlo antes de empezar y
**registrar aquí toda decisión nueva** (ver §9). No se cambia una decisión existente en
silencio: se marca como reemplazada y se agrega una nueva.

- **Curso:** Modelos Avanzados para el Procesamiento de Lenguaje Natural (MAPLN) — Maestría en IA.
- **Repositorio:** `fatbounduniandes/microproyecto_2_pln` (público desde 2026-10-24, ver D-017).
- **Rama principal:** `main`.
- **Última actualización de esta bitácora:** 2026-10-24.

---

## 1. Fuentes de verdad

| Documento | Ruta | Uso |
| :--- | :--- | :--- |
| Especificación oficial | `instrucciones_proyecto/especificaciones_y_ejecuci_n_del_microproyecto_2.md` | Requisitos obligatorios, estructura del notebook, entregables |
| Anexo técnico GPU | `instrucciones_proyecto/gu_a_t_cnica_de_optimizaci_n_de_memoria_gpu_anexo.md` | AMP, grad accumulation, gradient checkpointing, liberación de VRAM |
| Decisiones de ingeniería | `AGENTS.md` (este archivo) | Cómo y por qué se implementa |

Ante conflicto entre este archivo y la especificación, **gana la especificación**. Si una
decisión de aquí la contradice, es un error y debe corregirse.

---

## 2. Entorno de trabajo

| Rol | Dónde | Qué se hace ahí |
| :--- | :--- | :--- |
| Autoría / desarrollo | **WSL2 (Ubuntu 24.04)** sobre `/mnt/d/...` | Escribir celdas, generar el notebook, revisar diffs, git |
| Ejecución GPU | **Google Colab Pro** (runtime L4 / A100 / T4) | Entrenamiento, evaluación, generación, salida final |
| Almacenamiento de artefactos | **Google Drive** (montado en Colab) | Checkpoints y `.zip` de entrega |

Estado verificado de WSL: `python3 3.12.3`, sin `pip`, sin `torch`, sin `transformers`,
sin `nvidia-smi` (no hay GPU). **WSL no ejecuta el notebook.**

---

## 3. Decisiones (registro ADR)

Formato: `D-XXX · fecha · decisión · motivo · alternativas descartadas · consecuencias`.

### D-001 · 2026-10-24 · Entorno híbrido WSL (autoría) + Colab Pro (ejecución)
- **Decisión:** El notebook se escribe y versiona en WSL; se ejecuta únicamente en Colab Pro.
- **Motivo:** WSL no tiene GPU ni stack de PyTorch; instalar torch CPU (~2.5 GB) allí no aporta,
  porque el entregable exige ejecución con GPU y el cuello de botella real es el entrenamiento.
- **Descartado:** correr todo en WSL-CPU (horas por época, imposible para 360M); desarrollar
  directamente en Colab (pierdes historial, diffs y control de versiones ante desconexiones).
- **Consecuencia:** la iteración de código es local y rápida; toda validación de ejecución
  ocurre en Colab (primero T4, luego L4).

### D-002 · 2026-10-24 · GPU objetivo: **L4** para el run final, T4 para smoke tests
- **Decisión:** Smoke test en **T4**; corrida final en **L4**; **A100** solo como plan B.
  **No** se usa TPU (v5e/v6e) ni High-RAM.
- **Motivo (costo/beneficio):** L4 tiene ~24 GB, bf16 nativo y consume ~1/3 de unidades/hora
  frente a A100; el full-finetuning de 360M con batch efectivo 16 cabe holgado. T4 (16 GB,
  fp16) sirve para validar el pipeline pero es 2–3× más lento y no soporta bf16 real.
- **Descartado:** TPU (el stack es PyTorch + HF, no XLA; el bucle autorregresivo propio y el
  `Trainer` manual no son portables sin trabajo extra); High-RAM (irrelevante: `streaming=True`
  y bloques de 256 tokens usan pocos GB de RAM).
- **Consecuencia:** el notebook debe detectar `torch.cuda.is_bf16_supported()` y degradar a
  fp16 + `GradScaler` en T4 sin cambios de código.

### D-003 · 2026-10-24 · Sincronización vía GitHub con HTTPS + `gh`

> Reemplazada parcialmente por D-017: el repositorio pasó de privado a público.

- **Decisión:** Único mecanismo de sync: `git push` / `git pull` contra
  `https://github.com/fatbounduniandes/microproyecto_2_pln.git`.
- **Motivo:** el cliente `gh` ya está autenticado en WSL con scope `repo`; el remoto SSH
  (`git@github.com:...`) falla con `Permission denied (publickey)` porque la llave
  `~/.ssh/id_ed25519.pub` no está registrada en la cuenta de GitHub.
- **Descartado:** subir el `.ipynb` a mano entre WSL y Colab (rompe trazabilidad y provoca
  versiones divergentes); sync solo por Drive (sin historial ni diffs).
- **Consecuencia:** en Colab, la primera celda clona el repo. Si más adelante se quiere SSH:
  `gh auth refresh -h github.com -s admin:public_key && gh ssh-key add ~/.ssh/id_ed25519.pub`
  y luego `git remote set-url origin git@github.com:fatbounduniandes/microproyecto_2_pln.git`.

### D-004 · 2026-10-24 · El notebook es **autocontenido** (requisito duro)
- **Decisión:** `notebook.ipynb` no importa módulos locales, no depende de `src/`, no ejecuta
  `!git clone` de código propio ni `%run` de scripts. Todo el código vive inline en sus celdas.
- **Motivo:** requisito explícito de "ejecutable de principio a fin (`Run All`)" y de entrega
  como archivo único. Un evaluador que lo abra en Colab debe poder correrlo sin contexto extra.
- **Descartado:** paquete `src/` importado desde el notebook (obliga a clonar, rompe el `Run All`
  y hace frágil la entrega).
- **Consecuencia:** la modularidad se logra **antes** de generar el notebook (§D-006); el
  artefacto final es un único `.ipynb` con celdas en orden y con salidas visibles.

### D-005 · 2026-10-24 · Los datos se cargan con streaming **en cada** `Run All`
- **Decisión:** cada ejecución completa hace `load_dataset(..., streaming=True)` → filtro
  `1000 ≤ len(content) ≤ 2000` → tomar 8.000 → split 6.400/800/800. **Sin caché de dataset
  en Drive ni en disco.**
- **Motivo:** la especificación exige `streaming=True`; cachear el dataset y recargarlo podría
  interpretarse como incumplimiento, y el costo (~2–5 min) es aceptable frente al entrenamiento.
- **Descartado:** persistir los 8.000 ejemplos tokenizados y saltarse el streaming (más rápido,
  pero debilita el cumplimiento y la reproducibilidad del pipeline).
- **Consecuencia:** el preprocesamiento debe ser determinista (semilla fija) y tolerar
  desconexiones sin corromper el estado; ver D-007 para el reanudado.

### D-006 · 2026-10-24 · Celdas fuente en `src/cells/` + `build_notebook.py` genera el `.ipynb`
- **Decisión:** el código se escribe en archivos `.py` numerados (`src/cells/01_portada.py`, …),
  y un script sin dependencias externas los ensambla en `notebook.ipynb`. **El `.ipynb` no se
  edita a mano.**
- **Motivo:** el JSON del notebook es ilegible en diffs y propenso a corrupción; escribir celdas
  como texto permite revisar cambios, reordenar y regenerar sin tocar JSON. Garantiza que el
  notebook quede autocontenido (D-004) por construcción.
- **Descartado:** editar el `.ipynb` directamente (diffs ruidosos, riesgo de romper el JSON);
  `jupytext` (dependencia adicional innecesaria); `nbformat` (se evita cualquier dependencia).
- **Consecuencia:** única vía de cambio es `src/cells/` + `python3 build_notebook.py`. Si alguien
  edita el notebook en Colab, ese cambio se pierde en la próxima regeneración: **prohibido**.

### D-007 · 2026-10-24 · Reanudación idempotente de entrenamientos desde Drive
- **Decisión:** cada fase de entrenamiento comprueba si su checkpoint existe en Drive; si existe,
  lo carga y salta el entrenamiento (imprime un aviso claro). Si no, entrena y lo guarda.
- **Motivo:** Colab desconecta sesiones y el plan gratuito/Pro tiene límites; repetir el
  finetuning de 360M por una caída cuesta decenas de minutos.
- **Descartado:** entrenar siempre desde cero (frágil); reanudar a mitad de época con estados de
  optimizador (complejidad alta, beneficio marginal para 1 época).
- **Consecuencia:** la corrida entregada se ejecuta **una vez** de punta a punta para producir
  las salidas visibles; las re-ejecuciones posteriores serán rápidas y consistentes. El notebook
  debe dejar explícito en la salida si entrenó o cargó checkpoint.

### D-008 · 2026-10-24 · Higiene de repositorio (`.gitignore` + `.gitattributes`)
- **Decisión:** se versiona **solo** lo pequeño y reproducible: notebook, celdas fuente, script
  de build, documentación, requisitos. Se excluyen checkpoints, datasets, cachés, `.zip`,
  secretos y configuración local del agente.
- **Motivo:** GitHub rechaza archivos > 100 MB y los binarios no aportan trazabilidad; el
  entregable pesado (checkpoints) vive en Drive.
- **Consecuencia:** los checkpoints **no** se recuperan del repo; se regeneran en Colab o se
  bajan de Drive si hiciera falta.

### D-009 · 2026-10-24 · Nombres canónicos de artefactos
- **Decisión:** `checkpoint_scratch.pt`, `checkpoint_smollm2_135m_ft/`,
  `checkpoint_smollm2_360m_ft/` y `entrega_microproyecto_2.zip`, guardados en
  `/content/drive/MyDrive/microproyecto_2/`.
- **Motivo:** la especificación fija esos nombres exactos; centralizar la ruta evita
  discrepancias entre celdas.
- **Consecuencia:** una sola constante `DRIVE_DIR` en la celda de configuración; ninguna ruta
  absoluta dispersa en el código.

### D-010 · 2026-10-24 · Reproducibilidad
- **Decisión:** `SEED = 42` global (`random`, `numpy`, `torch`, `torch.cuda`), split determinista
  y submuestra de 200 ejemplos de test con su propia semilla fija.
- **Motivo:** permite comparar las 5 configuraciones sobre exactamente los mismos datos y
  regenerar la submuestra de evaluación sintáctica sin variaciones.
- **Consecuencia:** fijar semilla antes de cargar datos, antes de inicializar cada modelo y antes
  de muestrear el subconjunto de evaluación.

### D-011 · 2026-10-24 · Skills locales de OpenCode (tutoría y notebooks)
- **Decisión:** instalar en `.opencode/skills/` el set local de skills de apoyo:
  `jupyter-notebook` (`openai/skills`), `mentoring-juniors` (`github/awesome-copilot`),
  `progressive-hint-ladder`, `teach-back-evaluator` y `stuck-and-error-diagnosis-coach`
  (`GarethManning/education-agent-skills@v3.1.0`), `find-skills`
  (`vercel-labs/skills@v1.7.4`) y la skill propia `tutor-pln`. Para `find-skills`, instalar
  Node.js 22.20+ local en `~/.local/share/` y exponer `node`, `npm` y `npx` en `~/.local/bin`.
- **Motivo:** `tutor-pln` enseña los conceptos del microproyecto en modos aprendiz/docente;
  `jupyter-notebook` estandariza la creación y validación de `.ipynb`; las skills de mentoría
  refuerzan el aprendizaje socrático. Es tooling local: no altera el notebook, el pipeline ni
  el entregable.
- **Alternativas descartadas:** instalarlas globalmente en `~/.config/opencode/skills` (menos
  portable y fuera del proyecto); instalar las 165 skills de la librería de educación (infla el
  catálogo y el contexto sin aportar al alcance); editar el `.ipynb` a mano para “enseñar”
  (rompe D-004 y D-006).
- **Consecuencia:** `.opencode/` sigue sin versionarse (D-008); en otra máquina hay que
  reinstalar el set. `tutor-pln` debe respetar las reglas duras de §4 y no sustituir la
  implementación del notebook. `find-skills` requiere Node 22.20+; sin `npx`, el agente puede
  caer a `gh skill install`.

### D-012 · 2026-10-24 · Formato percent para las celdas y build determinista
- **Decisión:** `src/cells/NN_seccion.py` usa marcadores `# %%` (código) y `# %% [markdown]`
  con prefijo `# ` en cada línea de markdown. `build_notebook.py` genera ids de celda con
  `sha1(archivo:índice)`, escribe JSON con `indent=1` y **no** incluye timestamps ni rutas.
- **Motivo:** ids estables y salida byte a byte reproducible ⇒ diffs limpios en git y un modo
  `--check` fiable para detectar el notebook desactualizado.
- **Descartado:** ids aleatorios de `nbformat` (ruido en cada build); generar el `.ipynb` con
  `nbformat` (dependencia innecesaria); escribir markdown sin prefijo (rompe el resaltado del
  editor y la convención jupytext).
- **Consecuencia:** antes de cada commit se corre `python3 build_notebook.py --check`; si falla,
  el notebook no corresponde a las celdas fuente y no debe subirse.

### D-013 · 2026-10-24 · Estilo del notebook: markdown breve y código sin comentarios
- **Decisión:** el markdown va en tono de estudiante, en primera persona del plural, con
  párrafos cortos, listas y tablas simples; sin justificaciones teóricas extensas ni jerga
  académica innecesaria. Las celdas de código **no llevan comentarios ni docstrings**: la
  explicación vive en el markdown de la sección.
- **Motivo:** el entregable lo firma el equipo y debe leerse como trabajo propio, no como
  documentación generada. Además el código limpio se ve mejor y el markdown ya cumple la
  función explicativa.
- **Alternativas descartadas:** markdown denso con referencias al anexo (se siente ajeno);
  comentarios línea a línea en el código (ruido y apariencia de plantilla).
- **Consecuencia:** al escribir una celda nueva, primero el markdown explica qué se hace y
  luego el código lo hace sin comentarios. Nombres de variables autoexplicativos. Cada celda de
  código lleva justo encima un markdown de **una sola frase** que anuncia lo que hace.
  **Excepción:** Jean puede dejar comentarios cortos de una línea en el código cuando aclaran una
  decisión técnica (por ejemplo, por qué se fija `CUBLAS_WORKSPACE_CONFIG`).

### D-014 · 2026-10-24 · Los `push` se hacen solo cuando Jean lo indica
- **Decisión:** se puede trabajar y hacer commits locales, pero **ningún `git push`** se ejecuta
  sin autorización explícita de Jean en el momento.
- **Motivo:** él revisa cada fase antes de publicarla y quiere controlar qué llega al remoto.
- **Consecuencia:** al terminar una fase se reporta el estado y se espera la instrucción de
  subir. Si una fase se corrige, se puede reescribir el commit local antes de subir.

### D-015 · 2026-10-24 · El token de Hugging Face viene del entorno de Colab
- **Decisión:** no usar `userdata.get("HF_TOKEN")` ni `login()`. El notebook asume que
  `HF_TOKEN` ya existe como variable de entorno en el runtime de Colab y solo reporta si está
  definido.
- **Motivo:** Jean ya lo tiene configurado en su entorno; pedirlo otra vez agrega código y
  pasos manuales que el evaluador no necesita repetir.
- **Consecuencia:** `huggingface_hub` lo toma solo del entorno. Si no existe, las descargas
  públicas funcionan igual, con límites de tasa más estrictos.

### D-016 · 2026-10-24 · Estilo de programación súper básico
- **Decisión:** el notebook se escribe con lo mínimo indispensable de Python: `for` e `if`
  explícitos, funciones cortas sin anotaciones de tipo, rutas como texto (`str`) en vez de
  `pathlib`, sin lambdas, sin decoradores propios, sin clases auxiliares y sin comprensiones
  de lista anidadas. Se prefieren varias líneas claras a una línea ingeniosa.
- **Motivo:** el código debe parecer escrito por un estudiante y poder explicarse línea por
  línea, sin construcciones que oculten la lógica del pipeline.
- **Alternativas descartadas:** `pathlib`, helpers genéricos y estilo funcional (más elegante,
  pero opaco para sustentar); `try/except` defensivos alrededor de todo (agregan ramas que no
  aportan al alcance).
- **Consecuencia:** `drive.mount()` se llama directo, sin *fallback* local: el notebook asume
  Colab (requisito del enunciado). Las funciones tienen parámetros con valor por defecto
  simples y nombres largos en español.

### D-017 · 2026-10-24 · El repositorio pasa a público
- **Decisión:** `fatbounduniandes/microproyecto_2_pln` queda público. Se mantiene el mismo
  remoto HTTPS y el mismo flujo de `push` bajo autorización (D-014).
- **Motivo:** Colab rechazaba el clonado del repositorio privado (solo veía repositorios
  públicos) y eso bloqueaba la ejecución del notebook.
- **Consecuencia:** cualquier persona puede leer el repo; no hay secretos versionados porque el
  `HF_TOKEN` se toma del entorno (D-015) y los checkpoints viven en Drive (D-008). El entregable
  final igual va por Drive con permiso de lector.

### D-018 · 2026-10-24 · Las columnas se leen del primer ejemplo, no de `features`
- **Decisión:** para listar las columnas del dataset en streaming se usa
  `list(ejemplo.keys())` sobre el primer elemento, y ese mismo `ejemplo` se reutiliza para la
  inspección. No se usa `dataset.features`.
- **Motivo:** en la versión actual de `datasets`, `IterableDataset.features` es `None` en modo
  streaming y lanzaba `AttributeError: 'NoneType' object has no attribute 'keys'` en Colab.
- **Consecuencia:** una descarga menos (se reutiliza el ejemplo) y ninguna dependencia del
  objeto de características del dataset.

### D-019 · 2026-10-24 · `resultados/` guarda notebooks ejecutados y no se versiona
- **Decisión:** los notebooks descargados desde Colab con sus salidas se guardan en
  `resultados/` (ignorado por git) y se usan como evidencia para revisar cada fase.
- **Motivo:** permiten leer las salidas reales sin contaminar el repo con copias ejecutadas ni
  duplicar el notebook entregable, que se regenera desde `src/cells/`.
- **Consecuencia:** el notebook oficial sigue siendo `notebook.ipynb`; la copia ejecutada que se
  entregue se genera al final (Fase 12).

### D-020 · 2026-10-24 · Histograma completo y versión recortada
- **Decisión:** se dibujan **dos** histogramas: primero el completo (rango libre) y después uno
  recortado a 4.000 caracteres con un markdown que explica que solo cambia el zoom, no los datos.
- **Motivo:** Jean quiere ver la distribución real sin recortar, y al mismo tiempo una versión
  legible de la zona de interés (1.000–2.000 caracteres), que la cola larga aplasta.
- **Consecuencia:** los percentiles impresos (mínimo, mediana, máximo) dan el contexto de la cola
  que la vista recortada oculta.

### D-021 · 2026-10-24 · Exploración sobre una muestra en memoria
- **Decisión:** la celda de medición guarda los 2.000 ejemplos completos en `muestra` y deriva
  `longitudes` de ahí; todos los análisis posteriores (conteos, licencias, repos, duplicados)
  usan esa lista y no vuelven a recorrer el stream.
- **Motivo:** un solo recorrido de streaming (~2 s) y coherencia: todas las métricas se calculan
  sobre exactamente los mismos 2.000 archivos.
- **Consecuencia:** la muestra ocupa pocos MB en RAM; no se cachea en disco ni en Drive.

### D-022 · 2026-10-24 · La exploración no cambia el filtro ni el split
- **Decisión:** licencias, repos repetidos y duplicados por `hash` se reportan como hallazgos,
  pero el filtro (1.000–2.000 caracteres) y el split 6.400/800/800 se mantienen tal como los
  exige el enunciado.
- **Motivo:** se revisó la concentración de repos por si un split aleatorio dejaba el mismo
  proyecto en train y en test. Es un límite del protocolo, no algo que podamos corregir sin
  incumplirlo.
- **Consecuencia:** en la muestra de 2.000 archivos salieron **1.855 repos distintos** (máximo 5
  archivos de un mismo repo) y **0 duplicados por `hash`**, así que el riesgo de fuga es bajo y no
  hay nada que corregir en el split.

### D-023 · 2026-10-24 · El límite inferior de 1.000 casi no descarta archivos
- **Decisión:** la comparación de muestras usa el archivo **más corto de la muestra** (no uno
  `< 1000`), y se reporta cuántos archivos quedan por debajo de 1.000 caracteres.
- **Motivo:** en `codeparrot-clean` no aparecieron archivos de menos de ~1.000 caracteres; la
  búsqueda de un caso "corto" devolvía `None` y el `print` fallaba con
  `TypeError: 'NoneType' object is not subscriptable`.
- **Consecuencia:** el filtro útil es el de 2.000 caracteres. La comparación de tres tamaños sigue
  mostrando el efecto del filtro, y los casos ausentes se imprimen como "no se encontró" en vez
  de romper la celda.

### D-024 · 2026-10-24 · El empaquetado se hace por split, no sobre los 8.000 juntos
- **Decisión:** la sección 4 filtra, tokeniza y define `empacar()`, probándola con 50 archivos.
  El empaquetado definitivo de train/validación/prueba se hace en la sección 5, después de
  partir los 8.000 textos, para que ningún bloque mezcle tokens de dos splits.
- **Motivo:** si se empaca todo junto y después se cortan bloques, un bloque puede contener el
  final de un archivo de train y el inicio de uno de test, y la perplejidad quedaría contaminada.
- **Consecuencia:** se descarta el resto de tokens que no completa un bloque de 256 (`buffer`
  final); así no hay padding en ningún tensor y la PPL no necesita máscaras.

### D-025 · 2026-10-24 · Split aleatorio con semilla, no por orden del stream
- **Decisión:** antes de partir los 8.000 textos se mezclan con `SEED` y luego se cortan
  6.400 / 800 / 800. Los tokens se reordenan con el mismo índice para que texto y tokens queden
  alineados.
- **Motivo:** el stream entrega archivos en un orden que puede agrupar repos o épocas; un corte
  por posición dejaría train con unos proyectos y test con otros. Mezclar con semilla fija
  reparte el sesgo y mantiene la reproducibilidad (D-010).
- **Consecuencia:** se verifican solapamientos con `set` sobre los textos de las tres partes; el
  orden mezclado se usa también en las fases de evaluación y generación.

### D-026 · 2026-10-24 · Reproducibilidad estricta entre ejecuciones
- **Decisión:** fijar la revisión (commit SHA) del dataset y de los modelos/tokenizador, activar
  el modo determinista de PyTorch, usar un generador aleatorio local para el split y no usar
  padding. Revisiones: dataset `35a59fb…`, SmolLM2-135M `93efa2f…`, SmolLM2-360M `f8027fd…`.
- **Motivo:** Jean exige que dos ejecuciones produzcan exactamente los mismos datos, el mismo
  split y el mismo orden de entrenamiento; sin revisiones fijas, Hugging Face puede servir otra
  versión o otro orden de shards.
- **Detalle técnico:** `CUBLAS_WORKSPACE_CONFIG=:4096:8` antes de importar `torch`,
  `cudnn.deterministic=True`, `cudnn.benchmark=False`, `torch.use_deterministic_algorithms(...,
  warn_only=True)` y `random.Random(SEED)` en la mezcla.
- **Consecuencia:** los datos y el split son idénticos en cada corrida. En GPU, algunos kernels
  siguen avisando por `warn_only` (no bit a bit idéntico), pero el resultado esperado es el mismo
  y los tiempos son lo único que varía.

### D-027 · 2026-10-25 · Arquitectura del GPT desde cero
- **Decisión:** GPT-2 clásico con 6 capas, 6 cabezas y 384 dimensiones; MLP de 1.536 con GELU,
  LayerNorm pre-norm, embedding posicional aprendido de 256, bias desactivado, dropout 0,1,
  embeddings atados a `lm_head` e init normal 0,02 con residual escalado por `1/√(2·n_layer)`.
  Total: **29.599.488 parámetros**.
- **Motivo:** es la configuración de referencia de nanoGPT para contexto 256 y el enunciado pide
  explícitamente LayerNorm, MLP y embeddings posicionales. Se descartaron RoPE, RMSNorm, SwiGLU y
  GQA (lo que usa SmolLM2) porque se alejan de lo pedido y de lo que se puede sustentar.
- **Entrenamiento:** batch 16 directo (sin acumulación), AdamW con β=(0,9; 0,99), wd 0,1, clip
  1,0, lr 3e-4 con calentamiento de 100 pasos y decaimiento coseno hasta 3e-5, 3 épocas.
- **Consecuencia:** el modelo verá 8,25 M tokens (0,28 tokens por parámetro), muy por debajo de
  Chinchilla; se espera subentrenamiento y pérdidas altas. Es el argumento de la sección 10 sobre
  el valor del preentrenamiento.

### D-028 · 2026-10-25 · Épocas 10 / 3 / 3 en vez de 3 / 1 / 1
- **Decisión:** el modelo desde cero entrena **10 épocas**; los dos finetunings, **3 cada uno**.
  La tabla §4 del enunciado sugiere ≈3 / 1 / 1, así que es una desviación consciente.
- **Motivo:** con 2,75 M tokens por época el scratch queda en 0,28 tokens/parámetro por época; con
  3 épocas apenas aprende. Los finetunings también se benefician de más pasos sobre el dominio.
- **Mitigaciones:** dropout del scratch sube a 0,15, se guarda el **mejor modelo según validación**
  y en los finetunings se guardará checkpoint por época. El scheduler coseno se recalibra solo
  porque usa `PASOS_TOTALES`.
- **Consecuencia:** la corrida completa pasa de ~35–55 min a **1,5–2,3 h** en L4. Si la validación
  empeora, el checkpoint entregado es el mejor, no el último.

---

## 4. Reglas duras del proyecto (no negociables)

Se listan aquí para que ningún agente las rompa por conveniencia:

1. Exactamente **5 configuraciones**: scratch, 135M base, 135M FT, 360M base, 360M FT.
2. Tokenizador **obligatorio** `HuggingFaceTB/SmolLM2-135M` (vocab 49.152) para **todos** los modelos.
3. Contexto fijo de **256 tokens**, con empaquetado en bloques.
4. Dataset único `codeparrot/codeparrot-clean` con `streaming=True`.
5. Filtro `1000 ≤ len(content) ≤ 2000` y **exactamente 8.000** ejemplos → 6.400 / 800 / 800.
6. Full finetuning al 100% de parámetros. **Prohibido LoRA/QLoRA.**
7. **Prohibido** `model.generate()`, `pipeline()` o cualquier abstracción de generación:
   bucle autorregresivo propio con `logits[:, -1, :]`, temperatura/top-k/top-p y `multinomial`/`argmax`.
8. Perplejidad global: acumular `NLL × tokens_válidos` y dividir por el total de tokens;
   **nunca** promediar PPL por lote. Excluir padding.
9. Validez sintáctica con `ast.parse` sobre 200 ejemplos de test (submuestra con semilla).
10. 10 secciones del notebook con los **títulos exactos** y en el orden de la especificación.
11. Máximo un modelo en VRAM a la vez; `del` + `gc.collect()` + `torch.cuda.empty_cache()` entre fases.
12. Entregable: `.zip` con notebook ejecutado (salidas visibles) + 3 checkpoints, público en Drive.

---

## 5. Flujo de trabajo

```
WSL: editar src/cells/*.py
  └─> python3 build_notebook.py        # regenera notebook.ipynb
  └─> git add -A && git commit && git push
Colab: !git pull
  └─> Runtime > Run All
  └─> guarda checkpoints en Drive
  └─> descarga/arma entrega_microproyecto_2.zip
```

### Fases
| # | Fase | Entorno | Criterio de salida |
| :-- | :--- | :--- | :--- |
| 1 | Repo, `.gitignore`, bitácora | WSL | push en `main` |
| 2 | Celdas 1–5 (portada, libs, datos, prepro, split) | WSL | pipeline determinista |
| 3 | Smoke test (400 ejemplos, pocos steps) | Colab T4 | shapes, PPL y AST correctos |
| 4 | Celda 6 (scratch) + celda 7 (finetuning) | WSL | checkpoints guardados |
| 5 | Celdas 8–10 (eval, generación, conclusiones) | WSL | tabla comparativa completa |
| 6 | `Run All` final | Colab L4 | notebook con salidas + `.zip` |

### Regla de oro de Colab
- No editar el notebook en Colab.
- Antes de un run largo: `Runtime > Change runtime type > L4`, verificar
  `torch.cuda.get_device_name(0)` y `torch.cuda.is_bf16_supported()`.
- Para el run final: `Runtime > Factory reset runtime` para partir de VRAM limpia.

---

## 6. Estructura del repositorio

```
.
├── AGENTS.md                  # esta bitácora
├── FASES.md                   # checklist de fases (una a la vez)
├── README.md                  # descripción breve del proyecto
├── instrucciones_proyecto/    # enunciado y anexo (no editar)
├── src/cells/                 # celdas fuente numeradas (autoría)
├── build_notebook.py          # ensambla notebook.ipynb
├── notebook.ipynb             # ENTREGABLE generado (no editar a mano)
└── requirements-colab.txt     # versiones fijadas para Colab
```

No versionado: `checkpoint_*`, `*.pt`, `*.safetensors`, `*.zip`, `data/`, cachés de HF,
`.opencode/`, `.venv/`, secretos.

---

## 7. Convenciones de código

- Python 3.10+ (el de Colab), `torch` + `transformers` + `datasets`; nada exótico.
- Nombres en español para variables de dominio (`perdida`, `tokens_validos`) y en inglés para
  API de PyTorch (`input_ids`, `logits`).
- Constantes en mayúsculas al inicio de la celda de configuración.
- Cada celda imprime un resumen breve de lo que produjo (shapes, conteos, rutas, tiempo).
- Semillas fijas antes de cualquier operación aleatoria (D-010).
- **Código sin comentarios ni docstrings** (D-013): la explicación va en el markdown.
- **Estilo básico** (D-016): `for` y `if` explícitos, sin list comprehensions anidadas, sin
  lambdas, sin decoradores propios ni clases auxiliares; rutas como `str`, sin `pathlib`; sin
  anotaciones de tipo; nombres de variables largos y obvios.

---

## 8. Estado actual

- [x] `git init`, remoto `origin` configurado, rama `main`.
- [x] `AGENTS.md`, `.gitignore`, `.gitattributes`.
- [x] Skills locales de OpenCode para tutoría y notebooks (D-011).
- [x] `requirements-colab.txt` y `build_notebook.py` (D-012).
- [x] Fase 1 completada: repo, bitácora y `FASES.md`.
- [x] Fase 2 validada: portada y librerías corren en Colab.
- [x] Fase 3 validada en Colab: 11 columnas, 19,9 % de archivos en el rango 1.000–2.000.
- [x] Fase 3 ampliada corrida: sin nulos ni duplicados; 1.855 repos distintos.
- [ ] Fase 3 cierre: Jean redacta las conclusiones de la exploración.
- [x] Fase 4 validada en Colab: 8.000 textos, 3.460.082 tokens.
- [x] Fase 5 validada: 10.758 / 1.391 / 1.365 bloques, sin solapamientos.
- [ ] Fase 6: sección 6 escrita y probada en CPU, pendiente de Colab.
- [ ] Fase 7: finetuning de 135M y 360M.
- [ ] Fase 11: smoke test en Colab T4.
- [ ] Fase 12: `Run All` final en L4 y entrega en Drive.

---

## 9. Cómo registrar una decisión nueva

1. Agrega `### D-0NN · AAAA-MM-DD · Título corto` al final de §3 (no borres las anteriores).
2. Incluye: **Decisión**, **Motivo**, **Alternativas descartadas**, **Consecuencia**.
3. Si reemplaza una decisión previa, escribe en la vieja `> Reemplazada por D-0NN`.
4. Actualiza §8 (estado) si corresponde.
5. Commit aparte: `docs(agents): D-0NN <título>`.
