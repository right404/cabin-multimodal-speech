import time

import numpy as np

from cabin_speech.asr import AsyncASRWorker, TranscriptResult
from cabin_speech.realtime import SpeechSegment


class _FakeASR:
    def transcribe(self, segment: SpeechSegment) -> TranscriptResult:
        time.sleep(0.01)
        return TranscriptResult(
            text="测试字幕",
            audio_duration_seconds=segment.duration_seconds,
            latency_seconds=0.01,
        )


def test_async_asr_worker_returns_result() -> None:
    segment = SpeechSegment(np.zeros(16_000, dtype=np.float32), 16_000)
    worker = AsyncASRWorker(_FakeASR())
    try:
        worker.submit(segment)
        result = None
        for _ in range(50):
            result = worker.poll()
            if result is not None:
                break
            time.sleep(0.005)
        assert result is not None
        assert result.text == "测试字幕"
        assert np.isclose(result.real_time_factor, 0.01)
    finally:
        worker.close()
