# %% [markdown]
# ## 7. Finetuning de los modelos pre-entrenados
#
# Ajustamos SmolLM2-135M y SmolLM2-360M con **full finetuning**: se actualizan el 100 % de los
# parámetros, sin LoRA ni QLoRA. Para que quepan en la GPU usamos la tríada del anexo: precisión
# mixta, acumulación de gradientes y gradient checkpointing. Cada modelo se entrena 3 épocas y se
# guarda la mejor época según la pérdida de validación.

# %% [markdown]
# Definimos los hiperparámetros de cada finetuning.

# %%
EPOCAS_FT = 3
CALENTAMIENTO_FT = 50
WEIGHT_DECAY_FT = 0.01

config_135m = {
    "nombre": "SmolLM2-135M",
    "modelo": SMOL_135M,
    "revision": SMOL_135M_REVISION,
    "lr": 5e-5,
    "micro_batch": 8,
    "grad_accum": 2,
    "carpeta": "checkpoint_smollm2_135m_ft",
}

config_360m = {
    "nombre": "SmolLM2-360M",
    "modelo": SMOL_360M,
    "revision": SMOL_360M_REVISION,
    "lr": 3e-5,
    "micro_batch": 2,
    "grad_accum": 8,
    "carpeta": "checkpoint_smollm2_360m_ft",
}

print("135M: lr", config_135m["lr"], "| micro", config_135m["micro_batch"], "x", config_135m["grad_accum"])
print("360M: lr", config_360m["lr"], "| micro", config_360m["micro_batch"], "x", config_360m["grad_accum"])
print("Epocas:", EPOCAS_FT)

# %% [markdown]
# Definimos la función que ajusta un modelo y guarda la mejor época.

# %%
def ajustar_modelo(config):
    ruta_guardado = os.path.join(DRIVE_DIR, config["carpeta"])
    ruta_temporal = ruta_guardado + "_mejor"

    if os.path.exists(ruta_guardado):
        print(config["nombre"], ": ya existe el checkpoint, se omite el entrenamiento")
        return None

    modelo = AutoModelForCausalLM.from_pretrained(
        config["modelo"],
        revision=config["revision"],
    )
    modelo.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    modelo.config.use_cache = False
    modelo.to(DEVICE)

    total_parametros = 0
    for parametro in modelo.parameters():
        total_parametros += parametro.numel()

    print(config["nombre"], "- parametros:", total_parametros)

    datos = TensorDataset(X_train)
    generador = torch.Generator()
    generador.manual_seed(SEED)
    loader = DataLoader(datos, batch_size=config["micro_batch"], shuffle=True, generator=generador)

    pasos_por_epoca = len(loader) // config["grad_accum"]
    pasos_totales = pasos_por_epoca * EPOCAS_FT

    optimizador = torch.optim.AdamW(
        modelo.parameters(),
        lr=config["lr"],
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY_FT,
    )
    escalador = torch.amp.GradScaler("cuda", enabled=USE_SCALER)

    def tasa(paso):
        if paso < CALENTAMIENTO_FT:
            return (paso + 1) / CALENTAMIENTO_FT
        avance = (paso - CALENTAMIENTO_FT) / (pasos_totales - CALENTAMIENTO_FT)
        if avance > 1:
            avance = 1
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * avance))

    planificador = torch.optim.lr_scheduler.LambdaLR(optimizador, tasa)

    historial = {"train": [], "val": []}
    mejor_val = float("inf")
    mejor_epoca = 0
    inicio_modelo = time.time()

    for epoca in range(EPOCAS_FT):
        modelo.train()
        suma = 0.0
        ejemplos = 0
        paso = 0
        optimizador.zero_grad(set_to_none=True)
        inicio = time.time()

        for (lote,) in loader:
            lote = lote.to(DEVICE)

            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                salida = modelo(input_ids=lote, labels=lote)
                perdida = salida.loss / config["grad_accum"]

            escalador.scale(perdida).backward()
            suma += salida.loss.item() * lote.size(0)
            ejemplos += lote.size(0)
            paso += 1

            if paso % config["grad_accum"] == 0:
                escalador.unscale_(optimizador)
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), CLIP_GRAD_NORM)
                escalador.step(optimizador)
                escalador.update()
                optimizador.zero_grad(set_to_none=True)
                planificador.step()

        if paso % config["grad_accum"] != 0:
            escalador.unscale_(optimizador)
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), CLIP_GRAD_NORM)
            escalador.step(optimizador)
            escalador.update()
            optimizador.zero_grad(set_to_none=True)
            planificador.step()

        perdida_train = suma / ejemplos

        modelo.eval()
        suma_val = 0.0
        ejemplos_val = 0

        with torch.no_grad():
            for (lote,) in val_loader:
                lote = lote.to(DEVICE)
                with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                    salida = modelo(input_ids=lote, labels=lote)
                suma_val += salida.loss.item() * lote.size(0)
                ejemplos_val += lote.size(0)

        perdida_val = suma_val / ejemplos_val
        historial["train"].append(perdida_train)
        historial["val"].append(perdida_val)

        if perdida_val < mejor_val:
            mejor_val = perdida_val
            mejor_epoca = epoca + 1
            modelo.save_pretrained(ruta_temporal)
            print("  mejor hasta ahora, guardado en", ruta_temporal)

        print(config["nombre"],
              "- epoca", epoca + 1,
              "| train:", round(perdida_train, 3),
              "| val:", round(perdida_val, 3),
              "| lr:", round(planificador.get_last_lr()[0], 8),
              "| tiempo:", round(time.time() - inicio, 1), "s")

    segundos = time.time() - inicio_modelo
    print(config["nombre"], "- VRAM pico:", round(torch.cuda.max_memory_allocated() / 1024**3, 2), "GB")

    del optimizador
    del planificador

    if mejor_epoca != EPOCAS_FT:
        print(config["nombre"], ": se recarga la mejor epoca", mejor_epoca)
        del modelo
        liberar_vram_total()
        modelo = AutoModelForCausalLM.from_pretrained(ruta_temporal)
        modelo.to(DEVICE)

    modelo.save_pretrained(ruta_guardado)
    print(config["nombre"], ": checkpoint final en", ruta_guardado)

    if os.path.exists(ruta_temporal):
        shutil.rmtree(ruta_temporal)

    del modelo
    liberar_vram_total()

    return {
        "historial": historial,
        "mejor_val": mejor_val,
        "mejor_epoca": mejor_epoca,
        "segundos": segundos,
    }

# %% [markdown]
# Finetuneamos SmolLM2-135M.

# %%
resultado_135m = ajustar_modelo(config_135m)

# %% [markdown]
# Liberamos la memoria de la GPU antes de cargar el modelo de 360M.

# %%
liberar_vram_total()
print("VRAM en uso:", round(torch.cuda.memory_allocated() / 1024**3, 2), "GB")

# %% [markdown]
# Finetuneamos SmolLM2-360M.

# %%
resultado_360m = ajustar_modelo(config_360m)

# %% [markdown]
# Graficamos las curvas de validación de los dos finetunings.

# %%
plt.figure(figsize=(7, 4))

if resultado_135m is not None:
    plt.plot(range(1, EPOCAS_FT + 1), resultado_135m["historial"]["val"], marker="o", label="135M val")
if resultado_360m is not None:
    plt.plot(range(1, EPOCAS_FT + 1), resultado_360m["historial"]["val"], marker="o", label="360M val")

plt.title("Perdida de validacion durante el finetuning")
plt.xlabel("epoca")
plt.ylabel("cross-entropy")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# %% [markdown]
# Resumimos la mejor época y la perplejidad de cada modelo.

# %%
resultados = [("SmolLM2-135M", resultado_135m), ("SmolLM2-360M", resultado_360m)]

for nombre, resultado in resultados:
    if resultado is None:
        print(nombre, ": se cargo el checkpoint existente, sin historial de esta corrida")
    else:
        print(nombre,
              "| mejor epoca:", resultado["mejor_epoca"],
              "| val:", round(resultado["mejor_val"], 3),
              "| PPL:", round(math.exp(resultado["mejor_val"]), 2),
              "| minutos:", round(resultado["segundos"] / 60, 1))
