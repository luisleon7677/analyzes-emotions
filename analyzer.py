"""Análisis de emociones en audio por fragmentos (español, UMUTeam)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# El modelo ya está en la caché local: no consultar el Hub ni pedir token.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from safetensors.torch import load_file
from transformers import AutoConfig, AutoFeatureExtractor

from umu_model import CustomAudioClassification

# Modelo español solo-audio del mismo equipo UMUTeam.
# El multihead (audio+texto) exige transcripción/Whisper por fragmento y,
# en su propia evaluación, rinde peor que este (82% vs 88%).
MODEL_ID = "UMUTeam/w2v-bert-emotion-es"
TARGET_SR = 16_000
DEFAULT_CHUNK_SECONDS = 3.0

VALENCE_WEIGHTS = {
    "sadness": -1.0,
    "fear": -0.7,
    "disgust": -0.6,
    "anger": -0.5,
    "neutral": 0.0,
    "joy": 1.0,
}

EMOTION_ES = {
    "anger": "Enojo",
    "disgust": "Disgusto",
    "fear": "Miedo",
    "joy": "Alegre",
    "neutral": "Neutral",
    "sadness": "Triste",
}


@dataclass
class ChunkResult:
    time_sec: float
    valence_pct: float
    dominant: str
    dominant_es: str
    confidence: float
    scores: dict[str, float]


def read_waveform(path: str) -> tuple[torch.Tensor, int]:
    """Lee el audio con soundfile. torchaudio.load exige TorchCodec."""
    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T.copy())
    return waveform, int(sample_rate)


def local_model_dir() -> Path:
    """Carpeta del modelo ya descargado en la caché de Hugging Face."""
    cache = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--UMUTeam--w2v-bert-emotion-es"
    )
    ref_file = cache / "refs" / "main"
    if not ref_file.is_file():
        raise FileNotFoundError(
            "No está el modelo local "
            f"{MODEL_ID}. Se esperaba en {cache}"
        )
    snapshot = cache / "snapshots" / ref_file.read_text(encoding="utf-8").strip()
    if not (snapshot / "model.safetensors").is_file():
        raise FileNotFoundError(f"Faltan los pesos del modelo en {snapshot}")
    return snapshot


class EmotionAnalyzer:
    def __init__(self) -> None:
        print("[carga] EmotionAnalyzer: inicio", flush=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[carga] dispositivo: {self.device}", flush=True)
        model_dir = local_model_dir()
        print(f"[carga] carpeta local: {model_dir}", flush=True)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_dir, local_files_only=True
        )
        print("[carga] feature extractor listo", flush=True)
        config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        print("[carga] config lista", flush=True)
        self.model = CustomAudioClassification(config)
        print("[carga] arquitectura creada", flush=True)
        state = load_file(model_dir / "model.safetensors", device=str(self.device))
        print(f"[carga] pesos leídos: {len(state)} tensores", flush=True)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"[carga] faltan pesos: {missing}", flush=True)
            raise RuntimeError(f"Faltan pesos del modelo: {missing}")
        print(
            f"[carga] state_dict aplicado (inesperados: {len(unexpected)})",
            flush=True,
        )
        self.model.to(self.device)
        self.model.eval()
        print("[carga] modelo en eval", flush=True)
        self.id2label = {
            int(k): v.lower() for k, v in self.model.config.id2label.items()
        }
        # unexpected puede incluir buffers irrelevantes; no bloqueamos por ello
        _ = unexpected

    def load_audio(self, path: str) -> tuple[torch.Tensor, int]:
        waveform, sample_rate = read_waveform(path)

        if sample_rate != TARGET_SR:
            waveform = torchaudio.transforms.Resample(sample_rate, TARGET_SR)(waveform)
            sample_rate = TARGET_SR

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        return waveform.squeeze(0), sample_rate

    def _predict_chunk(self, chunk: torch.Tensor) -> tuple[str, float, dict[str, float]]:
        inputs = self.feature_extractor(
            chunk.numpy(),
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(
                input_features=inputs["input_features"],
                attention_mask=inputs.get("attention_mask"),
            ).logits

        probs = F.softmax(logits, dim=-1)[0]
        scores = {
            self.id2label[i]: float(probs[i].item()) for i in range(len(probs))
        }
        pred_id = int(torch.argmax(probs).item())
        dominant = self.id2label[pred_id]
        confidence = float(probs[pred_id].item())
        return dominant, confidence, scores

    @staticmethod
    def valence_percent(scores: dict[str, float]) -> float:
        raw = sum(
            scores.get(name, 0.0) * weight for name, weight in VALENCE_WEIGHTS.items()
        )
        return max(0.0, min(100.0, (raw + 1.0) * 50.0))

    def analyze(
        self,
        path: str,
        chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> list[ChunkResult]:
        waveform, sample_rate = self.load_audio(path)
        chunk_samples = max(int(chunk_seconds * sample_rate), sample_rate // 2)
        total = waveform.shape[0]
        n_chunks = max(1, (total + chunk_samples - 1) // chunk_samples)
        results: list[ChunkResult] = []

        for i in range(n_chunks):
            start = i * chunk_samples
            end = min(start + chunk_samples, total)
            chunk = waveform[start:end]

            if chunk.numel() < int(0.4 * sample_rate):
                break

            if progress_callback:
                progress_callback((i + 1) / n_chunks, f"Fragmento {i + 1}/{n_chunks}")

            dominant, confidence, scores = self._predict_chunk(chunk)
            time_sec = (start + end) / (2 * sample_rate)

            results.append(
                ChunkResult(
                    time_sec=time_sec,
                    valence_pct=self.valence_percent(scores),
                    dominant=dominant,
                    dominant_es=EMOTION_ES.get(dominant, dominant.title()),
                    confidence=confidence * 100.0,
                    scores={k: v * 100.0 for k, v in scores.items()},
                )
            )

        return results
