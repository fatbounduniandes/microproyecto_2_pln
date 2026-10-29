# %% [markdown]
# ## 8. Evaluación y comparación
#
# Medimos las cinco configuraciones con el mismo tokenizador: perplejidad global sobre los 1.365 bloques
# de prueba y validez sintáctica con `ast` sobre 200 ejemplos. Los modelos base se evalúan sin
# entrenar, tal como lo pide el enunciado, y cada modelo se carga solo mientras se mide.

# %% [markdown]
# Definimos los hiperparámetros de la evaluación.
#
# Evaluamos 200 ejemplos porque es lo que pide el enunciado, y los elegimos con la semilla 7
# para que la submuestra sea reproducible y distinta del split. El prompt fijo de 64 tokens
# permite generar en lotes sin padding y son unas 5 líneas de contexto. Generamos 128 tokens
# nuevos porque ahí cabe una función completa. La temperatura de 0,7 con top-k de 50 da variedad
# sin disparates. Los lotes de 8 para PPL y de 20 para generación caben en los bloques de 256
# incluso en el 360M. La PPL se mide sobre los 1.365 bloques que salen de los 800 archivos de
# prueba.

# %%
N_AST = 200
SEMILLA_AST = 7
LARGO_PROMPT = 64
NUEVOS = 128
TEMP = 0.7
TOP_K = 50
LOTE_EVAL = 8
LOTE_GEN = 20

print("Ejemplos AST:", N_AST)
print("Prompt:", LARGO_PROMPT, "| nuevos:", NUEVOS)
print("Temperatura:", TEMP, "| top-k:", TOP_K)

# %% [markdown]
# Implementamos el bucle autorregresivo propio con temperatura y top-k.

# %%
def generar_texto(modelo, inicio, nuevos, temperatura, top_k, semilla):
    modelo.eval()
    ids = inicio.clone()
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(semilla)
    es_scratch = isinstance(modelo, ModeloInicial)
    with torch.no_grad():
        pasado = None
        for paso in range(nuevos):
            if es_scratch:
                entrada = ids[:, -CTX:]
                with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                    salida = modelo(entrada)
                logits = salida[0]
            else:
                if paso == 0:
                    entrada = ids[:, -CTX:]
                    with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                        salida = modelo(entrada, use_cache=True)
                    pasado = salida.past_key_values
                    logits = salida.logits
                else:
                    with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                        salida = modelo(siguiente, past_key_values=pasado, use_cache=True)
                    pasado = salida.past_key_values
                    logits = salida.logits
            ultimos = logits[:, -1, :].float() / temperatura
            if top_k > 0:
                valores, indices = torch.topk(ultimos, top_k)
                minimo = valores[:, -1].unsqueeze(1)
                relleno = torch.tensor(float("-inf"), device=ultimos.device)
                ultimos = torch.where(ultimos < minimo, relleno, ultimos)
            probs = F.softmax(ultimos, dim=-1)
            siguiente = torch.multinomial(probs, num_samples=1, generator=gen)
            ids = torch.cat([ids, siguiente], dim=1)
            if (siguiente == tokenizer.eos_token_id).all():
                break
    return ids

# %% [markdown]
# Calculamos la perplejidad global acumulando por token.

# %%
def calcular_ppl(modelo, datos, lote_eval):
    modelo.eval()
    suma = 0.0
    total = 0
    with torch.no_grad():
        for inicio in range(0, len(datos), lote_eval):
            lote = datos[inicio:inicio + lote_eval].to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                salida = modelo(lote)
            if isinstance(salida, tuple):
                logits = salida[0]
            else:
                logits = salida.logits
            planos = logits[:, :-1, :].reshape(-1, VOCAB_SIZE)
            objetivos = lote[:, 1:].reshape(-1)
            suma += F.cross_entropy(planos, objetivos, reduction="sum").item()
            total += objetivos.numel()
    return math.exp(suma / total)

# %% [markdown]
# Medimos la validez sintáctica generando continuaciones y pasando `ast.parse`.

# %%
def evaluar_ast(modelo, textos_base, indices, lote_gen, semilla):
    validas = 0
    for inicio in range(0, len(indices), lote_gen):
        grupo = indices[inicio:inicio + lote_gen]
        prompts = []
        for j in grupo:
            ids = tokenizer.encode(textos_base[j], add_special_tokens=False)[:LARGO_PROMPT]
            while len(ids) < LARGO_PROMPT:
                ids.append(tokenizer.eos_token_id)
            prompts.append(ids)
        entrada = torch.tensor(prompts, dtype=torch.long, device=DEVICE)
        generados = generar_texto(modelo, entrada, NUEVOS, TEMP, TOP_K, semilla)
        for f in range(generados.size(0)):
            texto = tokenizer.decode(generados[f].tolist())
            lineas = texto.split("\n")
            recorte = "\n".join(lineas[:-1])
            try:
                ast.parse(recorte)
                validas += 1
            except SyntaxError:
                pass
    return validas

# %% [markdown]
# Elegimos los 200 ejemplos de prueba con su propia semilla.

# %%
rng_ast = random.Random(SEMILLA_AST)
indices_ast = rng_ast.sample(range(len(textos_test)), N_AST)

print("Ejemplos de prueba:", len(textos_test))
print("Submuestra AST:", len(indices_ast))
print("Primeros indices:", indices_ast[:5])

# %% [markdown]
# Evaluamos las cinco configuraciones cargando un modelo a la vez.

# %%
nombres = ["scratch", "135M base", "135M FT", "360M base", "360M FT"]
ppls = []
porcentajes = []
parametros_lista = []

for indice in range(len(nombres)):
    nombre = nombres[indice]
    if nombre == "scratch":
        modelo = ModeloInicial().to(DEVICE)
        ruta = os.path.join(DRIVE_DIR, "checkpoint_scratch.pt")
        modelo.load_state_dict(torch.load(ruta, map_location=DEVICE))
    elif nombre == "135M base":
        modelo = AutoModelForCausalLM.from_pretrained(SMOL_135M, revision=SMOL_135M_REVISION)
        modelo.to(DEVICE)
    elif nombre == "135M FT":
        ruta = os.path.join(DRIVE_DIR, "checkpoint_smollm2_135m_ft")
        modelo = AutoModelForCausalLM.from_pretrained(ruta)
        modelo.to(DEVICE)
    elif nombre == "360M base":
        modelo = AutoModelForCausalLM.from_pretrained(SMOL_360M, revision=SMOL_360M_REVISION)
        modelo.to(DEVICE)
    else:
        ruta = os.path.join(DRIVE_DIR, "checkpoint_smollm2_360m_ft")
        modelo = AutoModelForCausalLM.from_pretrained(ruta)
        modelo.to(DEVICE)

    if nombre != "scratch":
        modelo.config.use_cache = True

    total_parametros = 0
    for parametro in modelo.parameters():
        total_parametros += parametro.numel()
    parametros_lista.append(total_parametros)

    inicio = time.time()
    ppl = calcular_ppl(modelo, X_test, LOTE_EVAL)
    validas = evaluar_ast(modelo, textos_test, indices_ast, LOTE_GEN, SEED + indice)
    pct = validas / N_AST * 100

    ppls.append(ppl)
    porcentajes.append(pct)

    print(nombre, "| params:", total_parametros, "| PPL:", round(ppl, 2),
          "| AST:", round(pct, 1), "% | tiempo:", round(time.time() - inicio, 1), "s")

    del modelo
    liberar_vram_total()

print("VRAM en uso:", round(torch.cuda.memory_allocated() / 1024**3, 2), "GB")

# %% [markdown]
# Mostramos la tabla comparativa de las cinco configuraciones.

# %%
print("Configuracion | Parametros | PPL test | AST valido %")
for i in range(len(nombres)):
    print(nombres[i], "|", parametros_lista[i], "|", round(ppls[i], 2), "|", round(porcentajes[i], 1))

# %% [markdown]
# Graficamos las perplejidades para comparar los cinco modelos.

# %%
plt.figure(figsize=(7, 4))
plt.bar(nombres, ppls)
plt.title("Perplejidad en prueba por configuracion")
plt.ylabel("PPL")
plt.xticks(rotation=15)
plt.grid(axis="y", alpha=0.3)
plt.show()

# %% [markdown]
# Graficamos el porcentaje de código válido de cada configuración.

# %%
plt.figure(figsize=(7, 4))
plt.bar(nombres, porcentajes)
plt.title("Codigo valido por configuracion")
plt.ylabel("% valido")
plt.xticks(rotation=15)
plt.grid(axis="y", alpha=0.3)
plt.show()

# %% [markdown]
# Revisamos el sobreajuste comparando la pérdida de entrenamiento con la de validación.

# %%
ruta_historial_scratch = os.path.join(DRIVE_DIR, "historial_scratch.json")
archivo = open(ruta_historial_scratch, "r")
guardado_scratch = json.load(archivo)
archivo.close()

ruta_estado_135m = os.path.join(DRIVE_DIR, "checkpoint_smollm2_135m_ft_estado.json")
archivo = open(ruta_estado_135m, "r")
guardado_135m = json.load(archivo)
archivo.close()

ruta_estado_360m = os.path.join(DRIVE_DIR, "checkpoint_smollm2_360m_ft_estado.json")
archivo = open(ruta_estado_360m, "r")
guardado_360m = json.load(archivo)
archivo.close()

print("scratch: train", round(guardado_scratch["historial"]["train"][-1], 3),
      "| val", round(guardado_scratch["historial"]["val"][-1], 3))
print("135M FT: train", round(guardado_135m["historial"]["train"][-1], 3),
      "| val", round(guardado_135m["historial"]["val"][-1], 3))
print("360M FT: train", round(guardado_360m["historial"]["train"][-1], 3),
      "| val", round(guardado_360m["historial"]["val"][-1], 3))
