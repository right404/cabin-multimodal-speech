from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from cabin_speech.realtime import SpeechSegment


@dataclass(frozen=True)
class TranscriptResult:
    text: str
    audio_duration_seconds: float
    latency_seconds: float

    @property
    def real_time_factor(self) -> float:
        return self.latency_seconds / max(self.audio_duration_seconds, 1e-6)


class ASRBackend(Protocol):
    def transcribe(self, segment: SpeechSegment) -> TranscriptResult: ...


class WhisperASR:
    """PyTorch Whisper backend that accepts in-memory 16 kHz float audio."""

    def __init__(
        self,
        model_id: str = "openai/whisper-small",
        language: str = "zh",
        device: str = "auto",
        max_new_tokens: int = 128,
    ) -> None:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.language = language
        self.max_new_tokens = int(max_new_tokens)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            dtype=self.dtype,
            use_safetensors=True,
        ).to(self.device)
        self.model.eval()

    def transcribe(self, segment: SpeechSegment) -> TranscriptResult:
        if segment.sample_rate != 16_000:
            raise ValueError("WhisperASR expects 16 kHz audio")
        audio = np.asarray(segment.samples, dtype=np.float32)
        started = time.perf_counter()
        inputs = self.processor(
            audio,
            sampling_rate=segment.sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_features = inputs.input_features.to(self.device, dtype=self.dtype)
        attention_mask = inputs.attention_mask.to(self.device)
        with torch.inference_mode():
            generated_ids = self.model.generate(
                input_features,
                attention_mask=attention_mask,
                language=self.language,
                task="transcribe",
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0].strip()
        latency = time.perf_counter() - started
        return TranscriptResult(
            text=text,
            audio_duration_seconds=segment.duration_seconds,
            latency_seconds=latency,
        )


class AsyncASRWorker:
    """Single-worker ASR queue that keeps the camera loop responsive."""

    def __init__(self, backend: ASRBackend) -> None:
        self.backend = backend
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr")
        self._future: Future[TranscriptResult] | None = None
        self._pending: SpeechSegment | None = None

    @property
    def busy(self) -> bool:
        return self._future is not None

    def submit(self, segment: SpeechSegment) -> None:
        if self._future is None:
            self._future = self._executor.submit(self.backend.transcribe, segment)
        else:
            self._pending = segment

    def poll(self) -> TranscriptResult | None:
        if self._future is None or not self._future.done():
            return None
        result = self._future.result()
        self._future = None
        if self._pending is not None:
            pending = self._pending
            self._pending = None
            self._future = self._executor.submit(self.backend.transcribe, pending)
        return result

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)

    def __enter__(self) -> AsyncASRWorker:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
