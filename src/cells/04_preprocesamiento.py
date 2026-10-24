# %% [markdown]
# ## 4. Preprocesamiento
#
# Aplicamos el filtro de longitud que pide el enunciado, tokenizamos con el tokenizador de
# SmolLM2 y empacamos los tokens en bloques fijos de 256, que es el contexto que van a usar los
# cinco modelos.

# %% [markdown]
# Cargamos el tokenizador de SmolLM2 y revisamos su vocabulario.

# %%
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizador:", TOKENIZER_ID)
print("Vocabulario:", len(tokenizer))
print("Token de fin:", tokenizer.eos_token, "-", tokenizer.eos_token_id)

# %% [markdown]
# Filtramos el stream y nos quedamos con los 8.000 primeros archivos del rango.

# %%
textos = []

inicio = time.time()
for e in dataset:
    largo = len(e["content"])
    if largo >= MIN_CHARS and largo <= MAX_CHARS:
        textos.append(e["content"])
    if len(textos) == N_TOTAL:
        break

print("Archivos filtrados:", len(textos))
print("Tiempo:", round(time.time() - inicio, 1), "segundos")

# %% [markdown]
# Tokenizamos cada archivo y le agregamos el token de fin de texto.

# %%
ids_por_texto = []
total_tokens = 0

for texto in textos:
    ids = tokenizer.encode(texto, add_special_tokens=False)
    ids.append(tokenizer.eos_token_id)
    ids_por_texto.append(ids)
    total_tokens += len(ids)

print("Archivos tokenizados:", len(ids_por_texto))
print("Tokens en total:", total_tokens)
print("Promedio por archivo:", round(total_tokens / len(ids_por_texto), 1))

# %% [markdown]
# Mostramos cómo se ve un archivo después de tokenizarlo.

# %%
primeros_ids = ids_por_texto[0][:40]
print("Primeros tokens:", primeros_ids)
print("Como texto:", tokenizer.decode(primeros_ids))

# %% [markdown]
# Empacamos los tokens en bloques fijos de 256 y probamos la función con 50 archivos.

# %%
def empacar(lista_ids, contexto):
    bloques = []
    buffer = []
    for ids in lista_ids:
        buffer.extend(ids)
        while len(buffer) >= contexto:
            bloques.append(buffer[:contexto])
            buffer = buffer[contexto:]
    return bloques


bloques_prueba = empacar(ids_por_texto[:50], CTX)

print("Bloques de prueba:", len(bloques_prueba))
print("Largo de cada bloque:", len(bloques_prueba[0]))
print()
print(tokenizer.decode(bloques_prueba[0][:80]))
