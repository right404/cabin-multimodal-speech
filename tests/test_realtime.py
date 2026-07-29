import numpy as np

from cabin_speech.realtime import (
    AdaptiveEnergyVAD,
    AudioState,
    SpeechSegmenter,
    live_fusion_probabilities,
    rms_dbfs,
)


def test_rms_dbfs_known_level() -> None:
    signal = np.full(1600, 0.1, dtype=np.float32)
    assert np.isclose(rms_dbfs(signal), -20.0, atol=0.05)


def test_adaptive_vad_calibrates_then_detects_speech() -> None:
    vad = AdaptiveEnergyVAD(calibration_frames=5, threshold_margin_db=8.0, hangover_frames=0)
    rng = np.random.default_rng(4)
    for _ in range(5):
        state = vad.process(rng.normal(0.0, 0.002, 800))
    assert not state.calibrated
    state = vad.process(rng.normal(0.0, 0.08, 800))
    assert state.calibrated
    assert state.is_speech
    assert state.speech_probability > 0.9


def test_live_fusion_prefers_moving_face_during_speech() -> None:
    probabilities = live_fusion_probabilities(
        0.9,
        {
            1: 0.08,
            2: 0.65,
        },
    )
    assert np.isclose(sum(probabilities.values()), 1.0)
    assert probabilities["face_2"] > probabilities["face_1"]
    assert probabilities["face_2"] > probabilities["off_camera"]


def test_speech_segmenter_emits_complete_utterance() -> None:
    segmenter = SpeechSegmenter(
        sample_rate=16_000,
        block_size=800,
        pre_roll_ms=100,
        minimum_seconds=0.2,
        maximum_seconds=5.0,
    )
    silence = np.zeros(800, dtype=np.float32)
    speech = np.full(800, 0.1, dtype=np.float32)
    segmenter.process(silence, AudioState(calibrated=True, is_speech=False))
    for _ in range(4):
        assert segmenter.process(speech, AudioState(calibrated=True, is_speech=True)) is None
    segment = segmenter.process(
        silence,
        AudioState(calibrated=True, is_speech=False),
    )
    assert segment is not None
    assert segment.duration_seconds >= 0.2
    assert np.max(segment.samples) == 0.1
