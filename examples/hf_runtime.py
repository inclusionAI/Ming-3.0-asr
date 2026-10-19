"""Build the audio request used by Ming-3.0-asr-tiny."""


# Matches the released checkpoint's 25 Hz ASR input format. The
# character between the backticks is a literal tab, not a backslash and "t".
ASR_INSTRUCTION = (
    "Please recognize the language of this speech and transcribe it, "
    "placing a `\t` between the identified language and the transcribe."
)


def prepare_asr_request(
    processor, audio, *, prompt=None, language_hint=None,
    audio_chunk_frames=None, return_tensors="pt",
):
    """Prepare audio and optional context in the model's input format."""
    if prompt is None:
        prompt = ASR_INSTRUCTION
    if not isinstance(prompt, str) or not prompt.startswith(ASR_INSTRUCTION):
        raise ValueError("Prompt must start with the base ASR instruction")
    if "\n" in prompt or "\r" in prompt:
        raise ValueError("Prompt metadata must be contiguous, without newlines")
    if "<AUDIO>" in prompt:
        raise ValueError("Prompt must not contain <AUDIO>; it is appended here")
    if audio_chunk_frames is not None:
        if (isinstance(audio_chunk_frames, bool)
                or not isinstance(audio_chunk_frames, int)
                or audio_chunk_frames < 1):
            raise ValueError("audio_chunk_frames must be a positive integer")
        extractor = processor.feature_extractor
        original_config = extractor.audio_config
        extractor.audio_config = {**original_config, "chunk_frames": audio_chunk_frames}
        try:
            _ = extractor.spec
            inputs = processor(
                text=prompt + "<AUDIO>", audio=audio,
                return_tensors=return_tensors,
            )
        finally:
            extractor.audio_config = original_config
    else:
        inputs = processor(
            text=prompt + "<AUDIO>", audio=audio,
            return_tensors=return_tensors,
        )
    if language_hint is not None:
        label = {"zh": "Chinese\t", "en": "English\t"}.get(language_hint)
        if label is None:
            raise ValueError("language_hint must be 'zh', 'en', or None")
        if return_tensors != "pt":
            raise ValueError("language_hint requires return_tensors='pt'")
        import torch

        label_ids = processor.tokenizer(
            label, add_special_tokens=False, return_tensors="pt",
        )["input_ids"]
        if inputs.input_ids.shape[0] != 1:
            raise ValueError("language_hint requires exactly one audio request")
        inputs["input_ids"] = torch.cat((inputs.input_ids, label_ids), dim=1)
        if "attention_mask" in inputs:
            inputs["attention_mask"] = torch.cat(
                (inputs.attention_mask, torch.ones_like(label_ids)), dim=1,
            )
    return inputs


def context_chunk_frames(duration_seconds):
    """Use one encoder chunk for a ContextASR-Bench clip up to 60 seconds."""
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if duration_seconds > 60:
        raise ValueError("Context ASR chunk policy supports clips up to 60 seconds")
    return 6000 if duration_seconds > 30 else 3000


def decode_transcript(tokenizer, token_ids):
    """Remove the language envelope from generated text."""
    text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    if "\t" in text:
        return text.split("\t", 1)[1].strip()
    for language in ("Chinese", "English", "Canton", "Japanese", "Korean"):
        if text.startswith(language):
            return text[len(language):].strip()
    return text
