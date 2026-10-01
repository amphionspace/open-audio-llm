"""Fixed concat TS-ASR system prompt. Train and eval must use the same string."""

TS_CONCAT_SYSTEM = (
    "The clip starts with a 3-second enrollment of the target speaker, "
    "then the mixture. Transcribe only the enrolled speaker from the mixture. "
    "If the enrolled speaker is not present, output nothing."
)
