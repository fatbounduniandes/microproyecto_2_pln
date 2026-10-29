# %% [markdown]
# ## 10. Conclusiones
#
# La perplejidad mide cuánto se sorprende el modelo con el siguiente token: si acierta, baja;
# si duda entre muchas opciones, sube. La calculamos acumulando la entropía cruzada de todos
# los tokens de prueba y dividiendo una sola vez, como pide el enunciado, porque promediar
# perplejidades por lote da un número distinto y más optimista. La validez sintáctica mide qué
# porcentaje de las generaciones pasa `ast.parse`. Ninguna de las dos dice si el código
# *funciona*: no ejecutamos nada, así que un `if` perfectamente escrito con la condición al
# revés cuenta igual que uno correcto.
#
# | Configuración | Parámetros | PPL test | AST válido | IC 95 % |
# | :--- | :--- | :--- | :--- | :--- |
# | Desde cero | 29,6 M | 15,39 | 43,0 % | 36,3 – 49,9 |
# | 135M base | 134,5 M | 5,72 | 66,0 % | 59,2 – 72,2 |
# | 135M finetuned | 134,5 M | 4,75 | 58,0 % | 51,1 – 64,6 |
# | 360M base | 361,8 M | 4,39 | 66,5 % | 59,7 – 72,7 |
# | 360M finetuned | 361,8 M | 3,85 | 60,0 % | 53,1 – 66,5 |
#
# El punto de partida pesa más que la arquitectura. Nuestro modelo desde cero baja de 10,869 a
# 2,777 en validación, un 74 % de mejora, y aun así queda en 15,39 en prueba. Para ponerlo en
# escala: adivinar al azar entre 49.152 tokens daría una perplejidad de 49.152, así que el modelo
# aprendió bastante, pero está 2,7 veces peor que un SmolLM2-135M que no vio ni un archivo
# nuestro.
#
# La razón está en el presupuesto de tokens. Diez épocas sobre 2,75 M tokens son 27,5 M tokens
# vistos, o sea 0,93 tokens por parámetro. Los números de Chinchilla hablan de unas 20 veces esa
# cantidad para un modelo bien entrenado; llegar ahí habría requerido más de 200 épocas. No es
# que la arquitectura fallara: es que le dimos el 5 % de los datos que necesita. Ese es el
# argumento más fuerte de todo el experimento sobre el valor del preentrenamiento.

# %% [markdown]
# El finetuning ayuda, pero cada vez menos. El 135M baja 17,0 % (5,72 → 4,75) y el 360M baja
# 12,3 % (4,39 → 3,85). El pequeño gana proporcionalmente más porque parte de peor base, y los
# dos ven los mismos 2,75 M tokens por época. No es que el 360M aprenda menos: es que ya estaba
# mejor y le queda menos margen en este dominio tan acotado.
#
# Tampoco fue gratis en tiempo: 15,8 minutos el 135M y 19,4 el 360M, con picos de 4,51 y 8,06
# GB de VRAM. El 360M cuesta 1,2 veces más y gana 0,9 puntos de perplejidad, así que la relación
# costo-beneficio empieza a torcerse.
#
# Más parámetros, mejor ajuste. Entre baselines el 360M es 23,3 % mejor (5,72 → 4,39), y entre
# finetuneados la diferencia es 18,9 % (4,75 → 3,85). El salto de 2,7 veces más parámetros
# compra alrededor de un 20 % de perplejidad. Es una ganancia real pero decreciente: pasar al
# siguiente tamaño ya costaría el doble de memoria y de tiempo.

# %% [markdown]
# El techo de la sintaxis, y por qué cambia la lectura. Un 58 % de código válido suena bajo
# hasta que lo comparamos con lo que da el propio corpus. Tomamos la **continuación real** de
# cada archivo de prueba, la truncamos a 128 tokens con el mismo protocolo y la medimos igual:
# acierta el **59,5 %**. Los finetuneados están justo ahí, a medio punto, lo que significa que
# escriben código tan parseable como el del dataset original.
#
# El techo no viene de Python 2: el 92 % de los archivos completos de la muestra parsea, y en la
# exploración de 2.000 archivos el 84,4 % también. Viene del truncamiento. Cortar a 128 tokens
# deja funciones sin cuerpo, paréntesis sin cerrar y bloques abiertos, y eso lo sufre cualquier
# modelo, incluso uno perfecto.
#
# Los dos baselines se pasan del techo (66,0 y 66,5 %). No es que escriban mejor: generan piezas
# más cortas y genéricas, y una continuación corta y simple parsea más fácil. Lo dejamos como
# hipótesis porque no medimos la longitud de las generaciones de los baselines; si quisiéramos
# confirmarlo, bastaría contar cuántos tokens emiten antes del `EOS`. Lo que sí es claro es que
# la métrica premia imitar el corpus, no escribir buen código.
#
# El scratch, con 43,0 % e intervalo 36,3–49,9, no toca el de ninguna otra configuración: esa
# diferencia sí es real y no es ruido de muestreo.
#
# Sobreajuste. El scratch arrancó en 10,869 y terminó en 2,777 con la validación bajando las
# diez épocas, sin un solo repunte, y la brecha final entre train y validación fue de 0,48. Con
# 2,75 M tokens y dropout 0,15 el modelo todavía tenía margen cuando se acabaron las épocas: el
# límite fue el presupuesto, no el sobreajuste.
#
# Los finetunings, en cambio, se estabilizaron de inmediato. El 135M pasó de 1,593 a 1,581
# (0,75 % de mejora) y el 360M de 1,383 a 1,376 (0,51 %), con brechas de 0,10 y 0,11 entre train
# y validación. Ahí sí se ve el techo del dominio: dos épocas sobre 6.400 archivos alcanzan y la
# tercera no aportaba. Por eso la mejor época fue la segunda en los dos y el checkpoint guardado
# es ese, no el último.

# %% [markdown]
# Lo que dicen los ejemplos. Con `def fibonacci(n):` el 360M escribió la recursiva correcta y
# siguió con una búsqueda binaria bien formada. El 135M también acertó, pero repitió `fibonacci`
# tres veces con variantes. El scratch respondió `print("DROP TABLE IF EXISTS …")`: copió cadenas
# sueltas del corpus, sin estructura ni sentido.
#
# Con `class Stack` pasa lo mismo: el 360M completa `push`, `pop`, `peek` y `size` y hasta se
# inventa un `isEmpty`; el 135M hace un `Stack` razonable con `push` y `pop`; el scratch repite
# comentarios sobre sí mismo.
#
# Y hay un detalle que explica la métrica: el 135M terminó una de sus generaciones con `EOS` a
# los 53 tokens y el 360M a los 94, mientras el scratch usó los 128 en las tres y nunca se
# detuvo. Un modelo que sabe cuándo termina un archivo produce piezas cerradas, y eso se nota
# en el AST.
#
# Las desviaciones del enunciado, y por qué. Entrenamos 10 épocas el scratch y 2 cada
# finetuning, cuando la tabla sugería 3 / 1 / 1. Lo decidimos con datos: con 3 épocas el scratch
# apenas salía de 3,3 de validación, y el 135M ya había empeorado en su tercera época en la
# corrida anterior. También usamos micro-batch 8 con acumulación 2 en el 360M en vez del 2 × 8
# del anexo: el batch efectivo sigue siendo 16, y bajar el micro-batch costaba 1.972 s por época
# frente a 583 s.
#
# Limitaciones. 6.400 archivos de 1.000–2.000 caracteres son una muestra chica y sesgada hacia
# módulos cortos; el contexto de 256 tokens impide modelar dependencias largas como imports
# usados al final del archivo; no evaluamos si el código corre, solo si parsea; con 200 ejemplos
# el margen del AST es de ±6,5 puntos, así que diferencias menores no son concluyentes; y usamos
# una sola semilla por ejemplo, así que no tenemos variabilidad por generación, solo la del
# muestreo entre ejemplos.
#
# Aprendizajes. El más importante fue comparar contra un techo: sin ese 59,5 % habríamos
# reportado un fracaso donde en realidad hay un empate con el corpus. También aprendimos que el
# prompt decide más de lo que parece —con cabeceras de licencia los tres modelos parecían
# iguales y mediocres, y al cambiar a código limpio las diferencias saltaron a la vista— y que la
# ingeniería de la corrida importa: la caché KV bajó la evaluación de 63 a 8 minutos y la
# reanudación por época nos salvó dos veces cuando Colab cerró la sesión.
#
# Si tuviéramos más cómputo, el orden sería: más datos antes que más épocas, contexto más
# largo, y una métrica funcional como `pass@k` sobre pruebas unitarias, que diría algo que la
# perplejidad y el AST no pueden decir.
