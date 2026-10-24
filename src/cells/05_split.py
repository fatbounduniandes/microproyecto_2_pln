# %% [markdown]
# ## 5. Split train / validación / prueba
#
# Partimos los 8.000 textos en 6.400 de entrenamiento, 800 de validación y 800 de prueba. Primero
# mezclamos con la semilla fija para que el corte no dependa del orden en que llegan los archivos
# del stream, y después empacamos cada parte por separado en bloques de 256 tokens.

# %% [markdown]
# Mezclamos los textos y sus tokens con la semilla fija.

# %%
rng = random.Random(SEED)
orden = list(range(N_TOTAL))
rng.shuffle(orden)

textos_mezclados = []
ids_mezclados = []

for i in orden:
    textos_mezclados.append(textos[i])
    ids_mezclados.append(ids_por_texto[i])

print("Textos mezclados:", len(textos_mezclados))
print("Semilla:", SEED)

# %% [markdown]
# Partimos en 6.400 de entrenamiento, 800 de validación y 800 de prueba.

# %%
textos_train = textos_mezclados[:N_TRAIN]
textos_val = textos_mezclados[N_TRAIN:N_TRAIN + N_VAL]
textos_test = textos_mezclados[N_TRAIN + N_VAL:]

ids_train = ids_mezclados[:N_TRAIN]
ids_val = ids_mezclados[N_TRAIN:N_TRAIN + N_VAL]
ids_test = ids_mezclados[N_TRAIN + N_VAL:]

print("Train:", len(textos_train))
print("Validacion:", len(textos_val))
print("Prueba:", len(textos_test))
print("Total:", len(textos_train) + len(textos_val) + len(textos_test))

# %% [markdown]
# Verificamos que ningún texto quede en dos partes a la vez.

# %%
set_train = set(textos_train)
set_val = set(textos_val)
set_test = set(textos_test)

print("Repetidos train-validacion:", len(set_train & set_val))
print("Repetidos train-prueba:", len(set_train & set_test))
print("Repetidos validacion-prueba:", len(set_val & set_test))
print("Textos unicos:", len(set_train | set_val | set_test))

# %% [markdown]
# Empacamos cada parte por separado en bloques de 256 tokens.

# %%
bloques_train = empacar(ids_train, CTX)
bloques_val = empacar(ids_val, CTX)
bloques_test = empacar(ids_test, CTX)

print("Bloques train:", len(bloques_train))
print("Bloques validacion:", len(bloques_val))
print("Bloques prueba:", len(bloques_test))
print("Bloques totales:", len(bloques_train) + len(bloques_val) + len(bloques_test))

# %% [markdown]
# Convertimos los bloques a tensores y revisamos sus formas.

# %%
X_train = torch.tensor(bloques_train, dtype=torch.long)
X_val = torch.tensor(bloques_val, dtype=torch.long)
X_test = torch.tensor(bloques_test, dtype=torch.long)

print("Train:", X_train.shape, X_train.dtype)
print("Validacion:", X_val.shape, X_val.dtype)
print("Prueba:", X_test.shape, X_test.dtype)
print("Tokens de entrenamiento:", X_train.numel())

# %% [markdown]
# Revisamos cuántos tokens se pierden al final de cada parte por no completar un bloque.

# %%
partes = [
    ("train", ids_train, bloques_train),
    ("validacion", ids_val, bloques_val),
    ("prueba", ids_test, bloques_test),
]

for nombre, ids, bloques in partes:
    total = 0
    for x in ids:
        total += len(x)
    usados = len(bloques) * CTX
    print(nombre, "- tokens:", total, "| usados:", usados, "| descartados:", total - usados)
