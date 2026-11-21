"""TS-ASR system prompt. Train and eval must use the same string."""

# The enrollment length is a request parameter (``enroll_seconds``) and the [SEP] token marks where the
# enrollment ends, so the prompt states only the task.
TS_SYSTEM = "Transcribe only the enrolled speaker. Output nothing if they do not speak."


def ts_system(hotwords: str = "") -> str:
    """TS-ASR system prompt; a hotword list follows on its own line in the plain-ASR form ``Hotwords: w1,w2``.

    Without hotwords ("" or the sampler's "N/A") the prompt is exactly ``TS_SYSTEM``.
    """
    if not hotwords or hotwords == "N/A":
        return TS_SYSTEM
    return f"{TS_SYSTEM}\nHotwords: {hotwords}"
