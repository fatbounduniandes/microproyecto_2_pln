# %% [markdown]
# ## 3. Carga del dataset y exploración
#
# Trabajamos con `codeparrot/codeparrot-clean`, el dataset que indica el proyecto. Lo cargamos en
# modo streaming para no descargar los 5 GB completos: los ejemplos se van pidiendo a medida que
# los necesitamos. Primero miramos cómo viene la información y después revisamos qué tan largos
# son los archivos, porque el enunciado solo acepta los que tienen entre 1.000 y 2.000 caracteres.

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
# Medimos la longitud de 2.000 archivos para ver cómo se distribuyen.

# %%
N_MUESTRA = 2000
longitudes = []

inicio = time.time()
for i, ejemplo in enumerate(dataset):
    if i == N_MUESTRA:
        break
    longitudes.append(len(ejemplo["content"]))

print("Archivos medidos:", len(longitudes))
print("Tiempo:", round(time.time() - inicio, 1), "segundos")

# %% [markdown]
# Graficamos la distribución recortada a 4.000 caracteres y contamos cuántos archivos caen en el
# rango que pide el enunciado.

# %%
dentro_rango = 0
for largo in longitudes:
    if largo >= MIN_CHARS and largo <= MAX_CHARS:
        dentro_rango += 1

ordenadas = sorted(longitudes)

print("En el rango:", dentro_rango, "de", len(longitudes))
print("Porcentaje:", round(100 * dentro_rango / len(longitudes), 1), "%")
print("Minimo:", ordenadas[0])
print("Mediana:", ordenadas[len(ordenadas) // 2])
print("Maximo:", ordenadas[-1])

plt.figure(figsize=(8, 4))
plt.hist(longitudes, bins=60, range=(0, 4000))
plt.axvline(MIN_CHARS, color="red", linestyle="--")
plt.axvline(MAX_CHARS, color="red", linestyle="--")
plt.title("Longitud de los archivos de codeparrot-clean")
plt.xlabel("caracteres")
plt.ylabel("cantidad de archivos")
plt.xlim(0, 4000)
plt.show()
