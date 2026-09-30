"""Descarga el modelo de Hugging Face para usarlo luego en modo offline."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_MODEL_ID = "UMUTeam/w2v-bert-emotion-es"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Descarga el modelo local requerido por api.py/analyzer.py."
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Repositorio de Hugging Face. Por defecto: {DEFAULT_MODEL_ID}",
    )
    parser.add_argument(
        "--model-dir",
        default="models/w2v-bert-emotion-es",
        help="Carpeta destino. Debe coincidir con MODEL_DIR en produccion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = Path(args.model_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    path = snapshot_download(
        repo_id=args.model_id,
        local_dir=str(target),
        local_dir_use_symlinks=False,
    )
    model_file = Path(path) / "model.safetensors"
    if not model_file.is_file():
        raise FileNotFoundError(f"No se encontro model.safetensors en {path}")

    print(f"Modelo descargado en: {path}")
    print(f"Configura MODEL_DIR={path}")


if __name__ == "__main__":
    main()
