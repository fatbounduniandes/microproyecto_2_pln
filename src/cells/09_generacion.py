# %% [markdown]
# ## 9. Generación de ejemplos
#
# Generamos continuaciones con nuestra propia función autorregresiva para los tres modelos
# entrenados. Usamos los mismos 3 prompts de prueba para los tres, con semilla fija por modelo
# y por ejemplo, para que la comparación sea justa y reproducible.

# %% [markdown]
# Elegimos 3 archivos de prueba como prompts y mostramos su texto.

# %%
indices_ejemplos = [0, 50, 100]
prompts_ejemplos = []

for j in indices_ejemplos:
    ids = tokenizer.encode(textos_test[j], add_special_tokens=False)[:LARGO_PROMPT]
    while len(ids) < LARGO_PROMPT:
        ids.append(tokenizer.eos_token_id)
    prompts_ejemplos.append(ids)

for k in range(len(prompts_ejemplos)):
    print("Prompt", k + 1, ":")
    print(tokenizer.decode(prompts_ejemplos[k]))
    print()

# %% [markdown]
# Generamos 3 continuaciones con el modelo desde cero.

# %%
modelo = ModeloInicial().to(DEVICE)
ruta = os.path.join(DRIVE_DIR, "checkpoint_scratch.pt")
modelo.load_state_dict(torch.load(ruta, map_location=DEVICE))

ejemplos_scratch = []
largos_scratch = []

for p in range(len(prompts_ejemplos)):
    entrada = torch.tensor([prompts_ejemplos[p]], dtype=torch.long, device=DEVICE)
    generados = generar_texto(modelo, entrada, NUEVOS, TEMP, TOP_K, SEED + p)
    texto = tokenizer.decode(generados[0].tolist())
    ejemplos_scratch.append(texto)
    largos_scratch.append(generados.size(1) - LARGO_PROMPT)
    print("Ejemplo", p + 1, "| tokens nuevos:", generados.size(1) - LARGO_PROMPT)
    print(texto)
    print()

del modelo
liberar_vram_total()

# %% [markdown]
# Generamos las mismas 3 continuaciones con SmolLM2-135M finetuneado.

# %%
ruta = os.path.join(DRIVE_DIR, "checkpoint_smollm2_135m_ft")
modelo = AutoModelForCausalLM.from_pretrained(ruta)
modelo.to(DEVICE)
modelo.config.use_cache = True

ejemplos_135m = []
largos_135m = []

for p in range(len(prompts_ejemplos)):
    entrada = torch.tensor([prompts_ejemplos[p]], dtype=torch.long, device=DEVICE)
    generados = generar_texto(modelo, entrada, NUEVOS, TEMP, TOP_K, SEED + 10 + p)
    texto = tokenizer.decode(generados[0].tolist())
    ejemplos_135m.append(texto)
    largos_135m.append(generados.size(1) - LARGO_PROMPT)
    print("Ejemplo", p + 1, "| tokens nuevos:", generados.size(1) - LARGO_PROMPT)
    print(texto)
    print()

del modelo
liberar_vram_total()

# %% [markdown]
# Generamos las mismas 3 continuaciones con SmolLM2-360M finetuneado.

# %%
ruta = os.path.join(DRIVE_DIR, "checkpoint_smollm2_360m_ft")
modelo = AutoModelForCausalLM.from_pretrained(ruta)
modelo.to(DEVICE)
modelo.config.use_cache = True

ejemplos_360m = []
largos_360m = []

for p in range(len(prompts_ejemplos)):
    entrada = torch.tensor([prompts_ejemplos[p]], dtype=torch.long, device=DEVICE)
    generados = generar_texto(modelo, entrada, NUEVOS, TEMP, TOP_K, SEED + 20 + p)
    texto = tokenizer.decode(generados[0].tolist())
    ejemplos_360m.append(texto)
    largos_360m.append(generados.size(1) - LARGO_PROMPT)
    print("Ejemplo", p + 1, "| tokens nuevos:", generados.size(1) - LARGO_PROMPT)
    print(texto)
    print()

del modelo
liberar_vram_total()
print("VRAM en uso:", round(torch.cuda.memory_allocated() / 1024**3, 2), "GB")

# %% [markdown]
# Resumimos cuántos tokens generó cada modelo en cada ejemplo.

# %%
nombres_gen = ["scratch", "135M FT", "360M FT"]
largos_todos = [largos_scratch, largos_135m, largos_360m]

for m in range(len(nombres_gen)):
    for p in range(3):
        tope = "si" if largos_todos[m][p] == NUEVOS else "no"
        print(nombres_gen[m], "| ejemplo", p + 1, "| tokens:", largos_todos[m][p], "| llego al tope:", tope)
