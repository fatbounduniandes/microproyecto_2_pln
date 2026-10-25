# %% [markdown]
# ## 6. Modelo desde cero
#
# Implementamos un Transformer decoder-only en PyTorch puro: atención multi-cabeza con máscara
# causal, bloques con LayerNorm y MLP, embeddings de token y de posición, y una cabeza de salida
# atada a los embeddings. Tiene 6 capas, 6 cabezas y 384 dimensiones (unos 29,6 M de parámetros),
# y se entrena desde pesos aleatorios.
#
# Usamos 10 épocas en vez de las 3 que sugiere la tabla del enunciado: con 2,75 M tokens por
# época el modelo queda muy subentrenado, y la curva de validación dirá si aguanta. Para compensar
# subimos el dropout y guardamos el mejor modelo según validación.

# %% [markdown]
# Definimos los hiperparámetros del modelo y la máscara causal.

# %%
N_LAYER = 6
N_HEAD = 6
N_EMBD = 384
N_MLP = 4 * N_EMBD
DROPOUT = 0.15

EPOCAS = 10
BATCH = 16
LR = 3e-4
LR_MIN = 3e-5
CALENTAMIENTO = 100
WEIGHT_DECAY = 0.1

mascara = torch.tril(torch.ones(CTX, CTX)).view(1, 1, CTX, CTX).to(DEVICE)

print("Capas:", N_LAYER, "| Cabezas:", N_HEAD, "| Dimension:", N_EMBD)
print("Dropout:", DROPOUT, "| Epocas:", EPOCAS, "| LR:", LR)

# %% [markdown]
# Implementamos la atención multi-cabeza con máscara causal.

# %%
class CabezaAtencion(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn = nn.Linear(N_EMBD, 3 * N_EMBD, bias=False)
        self.c_proy = nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(N_EMBD, dim=2)
        q = q.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2)
        k = k.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2)
        v = v.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2)

        pesos = q @ k.transpose(-2, -1)
        pesos = pesos / math.sqrt(C // N_HEAD)
        pesos = pesos.masked_fill(mascara[:, :, :T, :T] == 0, float("-inf"))
        pesos = F.softmax(pesos, dim=-1)
        pesos = self.dropout(pesos)

        salida = pesos @ v
        salida = salida.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proy(salida)

# %% [markdown]
# Armamos el bloque del Transformer con LayerNorm, atención, MLP y conexiones residuales.

# %%
class Bloque(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.atencion = CabezaAtencion()
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.mlp = nn.Sequential(
            nn.Linear(N_EMBD, N_MLP, bias=False),
            nn.GELU(),
            nn.Linear(N_MLP, N_EMBD, bias=False),
        )
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        x = x + self.dropout(self.atencion(self.ln1(x)))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x

# %% [markdown]
# Implementamos el modelo completo con embeddings de token y posición y la cabeza de salida.

# %%
class ModeloInicial(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.pos_emb = nn.Embedding(CTX, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)
        self.bloques = nn.ModuleList([Bloque() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self.inicializar)
        for nombre, parametro in self.named_parameters():
            if nombre.endswith("c_proy.weight") or nombre.endswith("mlp.2.weight"):
                nn.init.normal_(parametro, mean=0.0, std=0.02 / math.sqrt(2 * N_LAYER))

    def inicializar(self, modulo):
        if isinstance(modulo, nn.Linear):
            nn.init.normal_(modulo.weight, mean=0.0, std=0.02)
        elif isinstance(modulo, nn.Embedding):
            nn.init.normal_(modulo.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        posiciones = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(posiciones)
        x = self.dropout(x)

        for bloque in self.bloques:
            x = bloque(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        perdida = None
        if targets is not None:
            logits_planos = logits[:, :-1, :].reshape(-1, VOCAB_SIZE)
            objetivos = targets[:, 1:].reshape(-1)
            perdida = F.cross_entropy(logits_planos, objetivos)

        return logits, perdida

# %% [markdown]
# Creamos el modelo, contamos sus parámetros y probamos un forward con datos falsos.

# %%
modelo_inicial = ModeloInicial().to(DEVICE)

total_parametros = 0
for parametro in modelo_inicial.parameters():
    total_parametros += parametro.numel()

prueba = torch.randint(0, VOCAB_SIZE, (2, CTX), device=DEVICE)
logits, perdida = modelo_inicial(prueba, prueba)

print("Parametros:", total_parametros)
print("Logits:", tuple(logits.shape))
print("Perdida inicial:", round(perdida.item(), 2))

# %% [markdown]
# Preparamos los lotes de entrenamiento y de validación.

# %%
datos_train = TensorDataset(X_train)
datos_val = TensorDataset(X_val)

generador = torch.Generator()
generador.manual_seed(SEED)

train_loader = DataLoader(datos_train, batch_size=BATCH, shuffle=True, generator=generador)
val_loader = DataLoader(datos_val, batch_size=BATCH, shuffle=False)

print("Lotes de train:", len(train_loader))
print("Lotes de validacion:", len(val_loader))

# %% [markdown]
# Configuramos el optimizador con calentamiento y decaimiento coseno del learning rate.

# %%
optimizer = torch.optim.AdamW(
    modelo_inicial.parameters(),
    lr=LR,
    betas=(0.9, 0.99),
    weight_decay=WEIGHT_DECAY,
)

PASOS_POR_EPOCA = len(train_loader)
PASOS_TOTALES = PASOS_POR_EPOCA * EPOCAS


def tasa_lr(paso):
    if paso < CALENTAMIENTO:
        return (paso + 1) / CALENTAMIENTO
    avance = (paso - CALENTAMIENTO) / (PASOS_TOTALES - CALENTAMIENTO)
    if avance > 1:
        avance = 1
    return LR_MIN / LR + (1 - LR_MIN / LR) * 0.5 * (1 + math.cos(math.pi * avance))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, tasa_lr)

print("Pasos por epoca:", PASOS_POR_EPOCA)
print("Pasos totales:", PASOS_TOTALES)

# %% [markdown]
# Entrenamos y guardamos en memoria el modelo que mejor pérdida de validación obtiene.

# %%
historial = {"train": [], "val": []}
mejor_val = float("inf")
mejor_epoca = 0
mejor_estado = None

for epoca in range(EPOCAS):
    modelo_inicial.train()
    suma = 0.0
    ejemplos = 0
    inicio = time.time()

    for (lote,) in train_loader:
        lote = lote.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
            _, perdida = modelo_inicial(lote, lote)
        perdida.backward()
        torch.nn.utils.clip_grad_norm_(modelo_inicial.parameters(), CLIP_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        suma += perdida.item() * lote.size(0)
        ejemplos += lote.size(0)

    perdida_train = suma / ejemplos

    modelo_inicial.eval()
    suma_val = 0.0
    ejemplos_val = 0

    with torch.no_grad():
        for (lote,) in val_loader:
            lote = lote.to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                _, perdida = modelo_inicial(lote, lote)
            suma_val += perdida.item() * lote.size(0)
            ejemplos_val += lote.size(0)

    perdida_val = suma_val / ejemplos_val

    historial["train"].append(perdida_train)
    historial["val"].append(perdida_val)

    if perdida_val < mejor_val:
        mejor_val = perdida_val
        mejor_epoca = epoca + 1
        mejor_estado = {}
        for nombre, parametro in modelo_inicial.state_dict().items():
            mejor_estado[nombre] = parametro.detach().cpu().clone()

    print("Epoca", epoca + 1,
          "| train:", round(perdida_train, 3),
          "| val:", round(perdida_val, 3),
          "| lr:", round(scheduler.get_last_lr()[0], 7),
          "| tiempo:", round(time.time() - inicio, 1), "s")

print("Mejor epoca:", mejor_epoca, "| mejor val:", round(mejor_val, 3))

# %% [markdown]
# Graficamos las curvas de pérdida para ver si hay sobreajuste.

# %%
plt.figure(figsize=(7, 4))
plt.plot(range(1, EPOCAS + 1), historial["train"], marker="o", label="train")
plt.plot(range(1, EPOCAS + 1), historial["val"], marker="o", label="validacion")
plt.title("Perdida por epoca - modelo desde cero")
plt.xlabel("epoca")
plt.ylabel("cross-entropy")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# %% [markdown]
# Restauramos el mejor modelo, lo guardamos en Drive y liberamos la memoria de la GPU.

# %%
modelo_inicial.load_state_dict(mejor_estado)

ruta_scratch = os.path.join(DRIVE_DIR, "checkpoint_scratch.pt")
torch.save(modelo_inicial.state_dict(), ruta_scratch)

print("Checkpoint:", ruta_scratch)
print("Epoca guardada:", mejor_epoca)
print("Tamano:", round(os.path.getsize(ruta_scratch) / 1024**2, 1), "MB")

del modelo_inicial
del optimizer
del scheduler
liberar_vram_total()
