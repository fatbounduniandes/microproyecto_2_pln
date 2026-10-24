# %% [markdown]
# ## 2. Librerías
#
# Dejamos listo el entorno de ejecución antes de tocar los datos, siguiendo lo que pide la guía
# del proyecto.

# %% [markdown]
# Instalamos las versiones de las librerías de Hugging Face que se usan en el proyecto.

# %%
!pip install -q "transformers>=4.44,<5" "datasets>=2.20,<4" "accelerate>=0.33"

# %% [markdown]
# Importamos lo necesario y comprobamos las versiones cargadas; `ast` lo usaremos para la
# validación sintáctica que pide la guía.

# %%
import ast
import gc
import math
import os
import random
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import datasets
import transformers
from datasets import load_dataset
from google.colab import drive
from transformers import AutoModelForCausalLM, AutoTokenizer

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Entorno cargado")
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("numpy:", np.__version__)

# %% [markdown]
# Configuramos la GPU, la precisión mixta, las constantes que fija el enunciado (tokenizador
# SmolLM2, contexto 256 y batch efectivo 16) y la carpeta de checkpoints.

# %%
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_BF16 = DEVICE == "cuda" and torch.cuda.is_bf16_supported()

if USE_BF16:
    AMP_DTYPE = torch.bfloat16
    NOMBRE_AMP = "bfloat16"
    USE_SCALER = False
else:
    AMP_DTYPE = torch.float16
    NOMBRE_AMP = "float16"
    USE_SCALER = True

if DEVICE == "cuda":
    props = torch.cuda.get_device_properties(0)
    print("GPU:", props.name)
    print("VRAM:", round(props.total_memory / 1024**3, 1), "GB")
else:
    print("ADVERTENCIA: no hay GPU, el entrenamiento será muy lento")

print("AMP:", NOMBRE_AMP)
print("GradScaler:", USE_SCALER)

SMOL_135M = "HuggingFaceTB/SmolLM2-135M"
SMOL_360M = "HuggingFaceTB/SmolLM2-360M"
TOKENIZER_ID = SMOL_135M

CTX = 256
VOCAB_SIZE = 49152
EOS_TOKEN = "<|endoftext|>"

MIN_CHARS = 1000
MAX_CHARS = 2000
N_TOTAL = 8000
N_TRAIN = 6400
N_VAL = 800
N_TEST = 800

BATCH_EFECTIVO = 16
CLIP_GRAD_NORM = 1.0

drive.mount("/content/drive")

DRIVE_DIR = "/content/drive/MyDrive/microproyecto_2"
os.makedirs(DRIVE_DIR, exist_ok=True)

print("Checkpoints:", DRIVE_DIR)

if "HF_TOKEN" in os.environ:
    print("HF_TOKEN: definido")
else:
    print("HF_TOKEN: no definido")

# %% [markdown]
# Fijamos la semilla para que los resultados sean reproducibles y creamos la función que libera
# la GPU entre modelos, como indica el protocolo de checkpoints.

# %%
def fijar_semilla(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def liberar_vram_total(modelo=None, optimizador=None):
    if modelo is not None:
        del modelo
    if optimizador is not None:
        del optimizador
    gc.collect()
    torch.cuda.empty_cache()


fijar_semilla()
print("Semilla:", SEED)
print("Configuración lista")
