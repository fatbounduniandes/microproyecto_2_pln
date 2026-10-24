# %% [markdown]
# ## 2. Librerías
#
# Se importa únicamente lo necesario para las diez secciones: `torch` para el modelo y el
# entrenamiento, `transformers` para el tokenizador y los pesos de SmolLM2, `datasets` para el
# acceso en *streaming* a `codeparrot-clean`, `ast` para la validación sintáctica y `matplotlib`
# para las curvas y gráficas comparativas.
#
# Además se fija en un solo lugar la configuración que el resto del notebook asume:
# dispositivo, precisión mixta (bf16 en L4/A100, fp16 con `GradScaler` en T4), semilla,
# constantes del pipeline y el directorio de checkpoints en Drive.

# %%
# ─────────────────────────────────────────────────────────────────────────────
# 2.1 Dependencias
# ─────────────────────────────────────────────────────────────────────────────
# Colab ya incluye PyTorch con CUDA; solo se fijan las librerías de Hugging Face para
# garantizar reproducibilidad. Espejo exacto de requirements-colab.txt.
!pip install -q "transformers>=4.44,<5" "datasets>=2.20,<4" "accelerate>=0.33"

# %%
# ─────────────────────────────────────────────────────────────────────────────
# 2.2 Importaciones controladas
# ─────────────────────────────────────────────────────────────────────────────
import ast                  # validación sintáctica del código generado
import gc                   # liberación de memoria entre modelos
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
os.environ["TOKENIZERS_PARALLELISM"] = "false"   # evita warnings y bloqueos con dataloaders

print("Entorno cargado")
print(f"  torch        : {torch.__version__}")
print(f"  transformers : {transformers.__version__}")
print(f"  datasets     : {datasets.__version__}")
print(f"  numpy        : {np.__version__}")

# %%
# ─────────────────────────────────────────────────────────────────────────────
# 2.3 Configuración global: dispositivo, precisión, rutas y constantes
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42


def fijar_semilla(seed: int = SEED) -> None:
    """Fija todas las fuentes de aleatoriedad relevantes (reproducibilidad)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# --- Dispositivo y precisión mixta -------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_BF16 = DEVICE == "cuda" and torch.cuda.is_bf16_supported()
AMP_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16
USE_SCALER = DEVICE == "cuda" and not USE_BF16   # fp16 requiere GradScaler

if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    props = torch.cuda.get_device_properties(0)
    print(f"GPU           : {props.name}")
    print(f"VRAM total    : {props.total_memory / 1024**3:.1f} GB")
    print(f"Capacidad     : sm_{props.major}{props.minor}")
else:
    print("ADVERTENCIA: no hay GPU. El notebook funcionará, pero el entrenamiento será lento.")

print(f"Precisión AMP : {str(AMP_DTYPE).replace('torch.', '')}")
print(f"GradScaler    : {USE_SCALER}")

# --- Constantes del proyecto -------------------------------------------------
SMOL_135M = "HuggingFaceTB/SmolLM2-135M"
SMOL_360M = "HuggingFaceTB/SmolLM2-360M"
TOKENIZER_ID = SMOL_135M          # obligatorio para las cinco configuraciones

CTX = 256                          # longitud de contexto fija
VOCAB_SIZE = 49152                 # vocabulario SmolLM2
EOS_TOKEN = "<|endoftext|>"

MIN_CHARS, MAX_CHARS = 1000, 2000  # filtro sobre len(content)
N_TOTAL = 8000
N_TRAIN, N_VAL, N_TEST = 6400, 800, 800

BATCH_EFECTIVO = 16                # micro_batch * grad_accum en las tres configuraciones
CLIP_GRAD_NORM = 1.0

# --- Persistencia en Google Drive (D-009) ------------------------------------
try:
    from google.colab import drive

    drive.mount("/content/drive")
    DRIVE_DIR = Path("/content/drive/MyDrive/microproyecto_2")
except Exception:
    # Fuera de Colab (o sin autorización) se usa un directorio local equivalente.
    DRIVE_DIR = Path("/content/microproyecto_2")

DRIVE_DIR.mkdir(parents=True, exist_ok=True)
print(f"Checkpoints   : {DRIVE_DIR}")

# --- Token de Hugging Face (opcional; sube el límite de descargas) -----------
try:
    from google.colab import userdata
    from huggingface_hub import login

    _hf_token = userdata.get("HF_TOKEN")
    if _hf_token:
        login(_hf_token, add_to_git_credential=False)
        print("HF_TOKEN      : cargado desde Colab Secrets")
    else:
        print("HF_TOKEN      : no definido (se usará acceso anónimo)")
except Exception:
    print("HF_TOKEN      : no disponible (se usará acceso anónimo)")

# %%
# ─────────────────────────────────────────────────────────────────────────────
# 2.4 Utilidades de memoria
# ─────────────────────────────────────────────────────────────────────────────
def liberar_vram_total(model=None, optimizer=None) -> None:
    """Elimina referencias y purga la caché de CUDA entre fases (anexo §5)."""
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
print(f"Semilla       : {SEED}")
print("Configuración lista.")
