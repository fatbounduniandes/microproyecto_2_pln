# %% [markdown]
# ## 2. Librerías
#
# Dejamos listo el entorno de ejecución antes de tocar los datos.

# %% [markdown]
# Instalamos las versiones de las librerías de Hugging Face que vamos a usar.

# %%
!pip install -q "transformers>=4.44,<5" "datasets>=2.20,<4" "accelerate>=0.33"

# %% [markdown]
# Importamos lo necesario y comprobamos las versiones cargadas.

# %%
import ast
import gc
import json
import math
import os
import random
import re
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import datasets
import transformers
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Entorno cargado")
print(f"  torch        : {torch.__version__}")
print(f"  transformers : {transformers.__version__}")
print(f"  datasets     : {datasets.__version__}")
print(f"  numpy        : {np.__version__}")

# %% [markdown]
# Configuramos la GPU, la precisión mixta, las constantes del proyecto y la carpeta de checkpoints.

# %%
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_BF16 = DEVICE == "cuda" and torch.cuda.is_bf16_supported()
AMP_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16
USE_SCALER = DEVICE == "cuda" and not USE_BF16

if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    props = torch.cuda.get_device_properties(0)
    print(f"GPU        : {props.name}")
    print(f"VRAM       : {props.total_memory / 1024**3:.1f} GB")
    print(f"Capacidad  : sm_{props.major}{props.minor}")
else:
    print("ADVERTENCIA: no hay GPU, el entrenamiento será muy lento")

print(f"AMP        : {str(AMP_DTYPE).replace('torch.', '')}")
print(f"GradScaler : {USE_SCALER}")

SMOL_135M = "HuggingFaceTB/SmolLM2-135M"
SMOL_360M = "HuggingFaceTB/SmolLM2-360M"
TOKENIZER_ID = SMOL_135M

CTX = 256
VOCAB_SIZE = 49152
EOS_TOKEN = "<|endoftext|>"

MIN_CHARS, MAX_CHARS = 1000, 2000
N_TOTAL = 8000
N_TRAIN, N_VAL, N_TEST = 6400, 800, 800

BATCH_EFECTIVO = 16
CLIP_GRAD_NORM = 1.0

try:
    from google.colab import drive

    drive.mount("/content/drive")
    DRIVE_DIR = Path("/content/drive/MyDrive/microproyecto_2")
except Exception:
    DRIVE_DIR = Path("/content/microproyecto_2")

DRIVE_DIR.mkdir(parents=True, exist_ok=True)
print(f"Checkpoints: {DRIVE_DIR}")
print(f"HF_TOKEN   : {'definido' if os.environ.get('HF_TOKEN') else 'no definido'}")

# %% [markdown]
# Definimos la semilla y la función que libera la memoria de la GPU entre modelos.

# %%
def fijar_semilla(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def liberar_vram_total(model=None, optimizer=None) -> None:
    if model is not None:
        del model
    if optimizer is not None:
        del optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()


fijar_semilla()
print(f"Semilla: {SEED}")
print("Configuración lista")
