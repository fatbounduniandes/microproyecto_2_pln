#!/usr/bin/env python3
"""Ensambla notebook.ipynb a partir de las celdas fuente en src/cells/.

Convención de autoría (D-006): un archivo .py por sección, en orden
lexicográfico, con celdas delimitadas por marcadores tipo jupytext:

    # %%                 -> celda de código
    # %% [markdown]      -> celda markdown (cada línea lleva el prefijo "# ")

El notebook generado es el entregable y NO debe editarse a mano.

Uso:
    python3 build_notebook.py            # regenera notebook.ipynb
    python3 build_notebook.py --check    # falla si el notebook está desactualizado
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CELLS_DIR = ROOT / "src" / "cells"
SALIDA = ROOT / "notebook.ipynb"

MARCADOR = re.compile(r"^#\s*%%\s*(.*)$")

METADATA = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.10"},
    "colab": {"provenance": [], "toc_visible": True},
    "accelerator": "GPU",
}


def _desmarcar(linea: str) -> str:
    """Quita el prefijo '# ' de una línea de markdown en formato percent."""
    if linea == "#":
        return ""
    if linea.startswith("# "):
        return linea[2:]
    if linea.startswith("#"):
        return linea[1:]
    return linea


def _limpiar(texto: str) -> list[str]:
    """Normaliza un bloque: sin líneas vacías iniciales/finales."""
    lineas = texto.split("\n")
    while lineas and not lineas[0].strip():
        lineas.pop(0)
    while lineas and not lineas[-1].strip():
        lineas.pop()
    return lineas


def _markdown_a_fuente(texto: str) -> list[str]:
    lineas = [_desmarcar(l) for l in _limpiar(texto)]
    return [l + "\n" for l in lineas]


def _codigo_a_fuente(texto: str) -> list[str]:
    lineas = _limpiar(texto)
    return [l + "\n" for l in lineas]


def parsear_archivo(ruta: Path) -> list[tuple[str, str]]:
    """Devuelve [(tipo, contenido)] leyendo los marcadores '# %%' del archivo."""
    celdas: list[tuple[str, str]] = []
    tipo_actual: str | None = None
    buffer: list[str] = []

    for linea in ruta.read_text(encoding="utf-8").splitlines(keepends=True):
        encontrado = MARCADOR.match(linea.rstrip("\r\n"))
        if encontrado:
            if tipo_actual is not None:
                celdas.append((tipo_actual, "".join(buffer)))
            etiqueta = encontrado.group(1).strip().lower()
            tipo_actual = "markdown" if "markdown" in etiqueta or etiqueta == "[md]" else "code"
            buffer = []
        else:
            buffer.append(linea)

    if tipo_actual is not None:
        celdas.append((tipo_actual, "".join(buffer)))
    return celdas


def construir() -> dict:
    archivos = sorted(CELLS_DIR.glob("*.py"))
    if not archivos:
        sys.exit(f"No hay celdas fuente en {CELLS_DIR}")

    celdas: list[dict] = []
    resumen: list[str] = []

    for archivo in archivos:
        for indice, (tipo, contenido) in enumerate(parsear_archivo(archivo)):
            if not contenido.strip():
                continue
            if tipo == "markdown":
                fuente = _markdown_a_fuente(contenido)
                celda = {"cell_type": "markdown"}
            else:
                fuente = _codigo_a_fuente(contenido)
                celda = {
                    "cell_type": "code",
                    "execution_count": None,
                    "outputs": [],
                }
            if not fuente:
                continue
            celda["id"] = hashlib.sha1(f"{archivo.name}:{indice}".encode()).hexdigest()[:12]
            celda["metadata"] = {}
            celda["source"] = fuente
            celdas.append(celda)

        resumen.append(f"  {archivo.name:<28} {len(parsear_archivo(archivo))} celdas")

    return {
        "cells": celdas,
        "metadata": METADATA,
        "nbformat": 4,
        "nbformat_minor": 5,
        "_resumen": resumen,
    }


def serializar(notebook: dict) -> str:
    datos = {k: v for k, v in notebook.items() if not k.startswith("_")}
    return json.dumps(datos, ensure_ascii=False, indent=1) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="no escribe: verifica que notebook.ipynb coincida con las celdas fuente",
    )
    args = parser.parse_args()

    notebook = construir()
    texto = serializar(notebook)

    if args.check:
        if not SALIDA.exists():
            print(f"ERROR: falta {SALIDA.name}; ejecuta build_notebook.py", file=sys.stderr)
            return 1
        if SALIDA.read_text(encoding="utf-8") != texto:
            print(
                "ERROR: notebook.ipynb está desactualizado respecto a src/cells/",
                file=sys.stderr,
            )
            return 1
        print("OK: notebook.ipynb está sincronizado")
        return 0

    SALIDA.write_text(texto, encoding="utf-8")
    total = len([c for c in notebook["cells"]])
    print("Celdas fuente:")
    print("\n".join(notebook["_resumen"]))
    print(f"\nGenerado {SALIDA.name}: {total} celdas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
