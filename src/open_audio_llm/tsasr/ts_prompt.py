"""Fixed concat TS-ASR system prompt. Train and eval must use the same string."""

TS_CONCAT_SYSTEM = (
    "The clip starts with a 3-second enrollment of the target speaker, "
    "then the mixture. Transcribe only the enrolled speaker from the mixture. "
    "If the enrolled speaker is not present, output nothing."
)


def ts_system(hotwords: str = "") -> str:
    """TS-ASR system prompt; a hotword list follows on its own line in the plain-ASR form ``Hotwords: w1,w2``.

    Without hotwords ("" or the sampler's "N/A") the prompt is exactly ``TS_CONCAT_SYSTEM``, so requests
    from clients that never send hotwords stay byte-identical.
    """
    if not hotwords or hotwords == "N/A":
        return TS_CONCAT_SYSTEM
    return f"{TS_CONCAT_SYSTEM}\nHotwords: {hotwords}"
