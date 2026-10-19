"""The public ASR request has one input format."""

import pytest

from examples.hf_runtime import (
    ASR_INSTRUCTION, context_chunk_frames, decode_transcript,
    prepare_asr_request,
)


def test_asr_request_matches_training_instruction():
    class Processor:
        def __call__(self, **kwargs):
            return kwargs

    audio = ("waveform", 16000)
    prepared = prepare_asr_request(Processor(), audio)
    assert prepared == {
        "text": ASR_INSTRUCTION + "<AUDIO>",
        "audio": audio, "return_tensors": "pt",
    }
    assert ASR_INSTRUCTION == (
        "Please recognize the language of this speech and transcribe it, "
        "placing a `\t` between the identified language and the transcribe."
    )


def test_context_appends_metadata_before_audio():
    class Processor:
        def __call__(self, **kwargs):
            return kwargs

    prompt = ASR_INSTRUCTION + "This is an audio about 同囚."
    prepared = prepare_asr_request(Processor(), "audio", prompt=prompt)
    assert prepared["text"] == prompt + "<AUDIO>"
    with pytest.raises(ValueError, match="base ASR instruction"):
        prepare_asr_request(Processor(), "audio", prompt="unrelated prompt")
    with pytest.raises(ValueError, match="must not contain <AUDIO>"):
        prepare_asr_request(Processor(), "audio", prompt=prompt + "<AUDIO>")
    with pytest.raises(ValueError, match="without newlines"):
        prepare_asr_request(Processor(), "audio", prompt=ASR_INSTRUCTION + "\nContext")


def test_context_supports_known_language_and_long_audio_chunks():
    torch = pytest.importorskip("torch")
    BatchFeature = pytest.importorskip("transformers").BatchFeature

    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert text == "Chinese\t"
            assert kwargs == {"add_special_tokens": False, "return_tensors": "pt"}
            return {"input_ids": torch.tensor([[9, 10]])}

    class Extractor:
        audio_config = {"chunk_frames": 3000}

        @property
        def spec(self):
            if self.audio_config["chunk_frames"] > 6000:
                raise ValueError("encoder context exceeded")
            return self.audio_config

    class Processor:
        tokenizer = Tokenizer()
        feature_extractor = Extractor()

        def __call__(self, **kwargs):
            assert kwargs["text"] == ASR_INSTRUCTION + "<AUDIO>"
            assert self.feature_extractor.audio_config["chunk_frames"] == 6000
            return BatchFeature({
                "input_ids": torch.tensor([[7, 8]]),
                "attention_mask": torch.tensor([[1, 1]]),
            })

    processor = Processor()
    original_config = processor.feature_extractor.audio_config
    inputs = prepare_asr_request(
        processor, "audio", language_hint="zh", audio_chunk_frames=6000,
    )
    assert inputs.input_ids.tolist() == [[7, 8, 9, 10]]
    assert inputs.attention_mask.tolist() == [[1, 1, 1, 1]]
    assert processor.feature_extractor.audio_config is original_config
    with pytest.raises(ValueError, match="encoder context"):
        prepare_asr_request(processor, "audio", audio_chunk_frames=6001)
    assert processor.feature_extractor.audio_config is original_config


def test_context_chunk_policy():
    assert context_chunk_frames(30) == 3000
    assert context_chunk_frames(30.01) == 6000
    with pytest.raises(ValueError, match="positive"):
        context_chunk_frames(0)
    with pytest.raises(ValueError, match="up to 60 seconds"):
        context_chunk_frames(60.01)


@pytest.mark.parametrize("raw, expected", (
    ("Chinese\t你好", "你好"),
    ("English\tHello world", "Hello world"),
    ("Chinese你好", "你好"),
    ("English hello", "hello"),
    ("Plain result", "Plain result"),
))
def test_decoder_removes_language_envelope(raw, expected):
    class Tokenizer:
        def decode(self, token_ids, skip_special_tokens):
            assert token_ids == [1, 2]
            assert skip_special_tokens
            return raw

    assert decode_transcript(Tokenizer(), [1, 2]) == expected
