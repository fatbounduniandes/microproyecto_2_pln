# %% [markdown]
# ## 3. Carga del dataset y exploración
#
# Trabajamos con `codeparrot/codeparrot-clean`, el dataset que indica el proyecto. Lo cargamos en
# modo streaming para no descargar los 5 GB completos: los ejemplos se van pidiendo a medida que
# los necesitamos.
#
# Primero miramos la estructura del dataset, después qué tan largos son los archivos (el
# enunciado solo acepta los que tienen entre 1.000 y 2.000 caracteres) y, por último, revisamos
# licencias, repos y duplicados para saber con qué datos estamos trabajando.

# %% [markdown]
# Cargamos el dataset en streaming y miramos qué columnas trae.

# %%
dataset = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
ejemplo = next(iter(dataset))

print(dataset)
print("Columnas:", list(ejemplo.keys()))

# %% [markdown]
# Mostramos un archivo de ejemplo para ver cómo viene el campo `content`.

# %%
print("repo:", ejemplo["repo_name"])
print("archivo:", ejemplo["path"])
print("caracteres:", len(ejemplo["content"]))
print("lineas:", ejemplo["content"].count("\n"))
print()
print(ejemplo["content"][:600])

# %% [markdown]
# Medimos 2.000 archivos y guardamos la muestra completa para analizarla sin volver al stream.

# %%
N_MUESTRA = 2000
muestra = []

inicio = time.time()
for i, ejemplo in enumerate(dataset):
    if i == N_MUESTRA:
        break
    muestra.append(ejemplo)

longitudes = []
for e in muestra:
    longitudes.append(len(e["content"]))

print("Archivos medidos:", len(muestra))
print("Tiempo:", round(time.time() - inicio, 1), "segundos")

# %% [markdown]
# Contamos cuántos archivos caen en el rango que pide el enunciado y resumimos la distribución.

# %%
dentro_rango = 0
for largo in longitudes:
    if largo >= MIN_CHARS and largo <= MAX_CHARS:
        dentro_rango += 1

ordenadas = sorted(longitudes)
n = len(ordenadas)

print("En el rango:", dentro_rango, "de", n)
print("Porcentaje:", round(100 * dentro_rango / n, 1), "%")
print("Minimo:", ordenadas[0])
print("P25:", ordenadas[n // 4])
print("Mediana:", ordenadas[n // 2])
print("P75:", ordenadas[3 * n // 4])
print("P90:", ordenadas[9 * n // 10])
print("P99:", ordenadas[99 * n // 100])
print("Maximo:", ordenadas[-1])
print("Media:", round(sum(longitudes) / n, 1))

# %% [markdown]
# Graficamos la distribución completa de longitudes.

# %%
plt.figure(figsize=(8, 4))
plt.hist(longitudes, bins=60)
plt.axvline(MIN_CHARS, color="red", linestyle="--")
plt.axvline(MAX_CHARS, color="red", linestyle="--")
plt.title("Longitud de los archivos de codeparrot-clean")
plt.xlabel("caracteres")
plt.ylabel("cantidad de archivos")
plt.show()

# %% [markdown]
# La gráfica anterior muestra la distribución completa, pero la cola de archivos muy largos
# aplasta la zona que nos interesa. Por eso repetimos el histograma recortado a 4.000 caracteres:
# no estamos cambiando los datos, solo el zoom de la figura.

# %%
plt.figure(figsize=(8, 4))
plt.hist(longitudes, bins=60, range=(0, 4000))
plt.axvline(MIN_CHARS, color="red", linestyle="--")
plt.axvline(MAX_CHARS, color="red", linestyle="--")
plt.title("Longitud de los archivos (vista hasta 4.000 caracteres)")
plt.xlabel("caracteres")
plt.ylabel("cantidad de archivos")
plt.xlim(0, 4000)
plt.show()

# %% [markdown]
# Revisamos el tipo de dato y un valor de ejemplo de cada columna.

# %%
print("Columna           Tipo       Ejemplo")
for nombre in muestra[0]:
    valor = muestra[0][nombre]
    texto = str(valor).replace("\n", " ")
    print(nombre.ljust(17), type(valor).__name__.ljust(10), texto[:45])

# %% [markdown]
# Revisamos si hay archivos vacíos o con contenido nulo.

# %%
vacios = 0
solo_espacios = 0

for e in muestra:
    contenido = e["content"]
    if contenido is None or len(contenido) == 0:
        vacios += 1
    elif len(contenido.strip()) == 0:
        solo_espacios += 1

print("Vacios o nulos:", vacios)
print("Solo espacios:", solo_espacios)

# %% [markdown]
# Comparamos un archivo corto, uno del rango y uno largo para ver por qué el enunciado filtra.

# %%
corto = None
en_rango = None
largo = None

for e in muestra:
    n = len(e["content"])
    if corto is None and n < MIN_CHARS:
        corto = e
    if en_rango is None and n >= MIN_CHARS and n <= MAX_CHARS:
        en_rango = e
    if largo is None and n > MAX_CHARS:
        largo = e
    if corto is not None and en_rango is not None and largo is not None:
        break

casos = [("CORTO", corto), ("EN RANGO", en_rango), ("LARGO", largo)]

for nombre, e in casos:
    print("=====", nombre, "-", len(e["content"]), "caracteres =====")
    print(e["content"][:300])
    print()

# %% [markdown]
# Contamos cuántas veces aparece cada construcción típica de Python.

# %%
palabras = ["def ", "class ", "import ", "return ", "print("]
conteos = {}

for palabra in palabras:
    conteos[palabra] = 0

for e in muestra:
    for palabra in palabras:
        conteos[palabra] += e["content"].count(palabra)

valores = []
for palabra in palabras:
    print(palabra.strip(), conteos[palabra])
    valores.append(conteos[palabra])

plt.figure(figsize=(8, 3))
plt.barh(palabras, valores)
plt.title("Construcciones de Python en 2.000 archivos")
plt.xlabel("apariciones")
plt.show()

# %% [markdown]
# Revisamos cuáles son las licencias más comunes.

# %%
licencias = {}

for e in muestra:
    lic = str(e["license"])
    if lic in licencias:
        licencias[lic] += 1
    else:
        licencias[lic] = 1

pares = []
for lic, cantidad in licencias.items():
    pares.append((cantidad, lic))
pares.sort(reverse=True)

nombres = []
valores = []
for cantidad, lic in pares[:5]:
    print(cantidad, lic)
    nombres.append(lic[:30])
    valores.append(cantidad)

plt.figure(figsize=(8, 3))
plt.barh(nombres, valores)
plt.title("Licencias mas comunes")
plt.xlabel("archivos")
plt.show()

# %% [markdown]
# Revisamos si los archivos vienen de pocos repos y cuántos están repetidos.

# %%
repos = {}

for e in muestra:
    repo = e["repo_name"]
    if repo in repos:
        repos[repo] += 1
    else:
        repos[repo] = 1

pares = []
for repo, cantidad in repos.items():
    pares.append((cantidad, repo))
pares.sort(reverse=True)

print("Repos distintos:", len(repos))
for cantidad, repo in pares[:5]:
    print(cantidad, repo)

vistos = set()
repetidos = 0
for e in muestra:
    if e["hash"] in vistos:
        repetidos += 1
    else:
        vistos.add(e["hash"])

print("Archivos repetidos por hash:", repetidos)

# %% [markdown]
# Estimamos cuántos archivos hay que revisar para juntar los 8.000 que pide el enunciado.

# %%
acumulado = []
total = 0

for largo in longitudes:
    if largo >= MIN_CHARS and largo <= MAX_CHARS:
        total += 1
    acumulado.append(total)

esperado = round(N_TOTAL * len(longitudes) / dentro_rango)

plt.figure(figsize=(8, 4))
plt.plot(acumulado, label="muestra")
plt.plot([0, esperado], [0, N_TOTAL], linestyle="--", label="proyeccion")
plt.axhline(N_TOTAL, color="red", linestyle="--")
plt.xlim(0, esperado * 1.05)
plt.ylim(0, N_TOTAL * 1.1)
plt.title("Archivos utiles acumulados")
plt.xlabel("archivos revisados")
plt.ylabel("archivos en rango")
plt.legend()
plt.show()

print("Para 8.000 utiles hay que revisar cerca de", esperado, "archivos")
