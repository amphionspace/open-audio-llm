"""TS-ASR system prompts. Train and eval must use the same string."""

# Since 2026-11-06. The enrollment length is a request parameter (``enroll_seconds``), not part of the
# prompt, and the [SEP] token marks where the enrollment ends, so the prompt states only the task.
TS_SYSTEM = "Transcribe only the enrolled speaker. Output nothing if they do not speak."

# The prompt every model up to the 11.06 release was trained with. It names a 3-second enrollment
# although clients now send 5 seconds; training keeps a share of it so clients that still send it
# keep working (data config ``prompts.ts_legacy_prob``).
TS_CONCAT_SYSTEM = (
    "The clip starts with a 3-second enrollment of the target speaker, "
    "then the mixture. Transcribe only the enrolled speaker from the mixture. "
    "If the enrolled speaker is not present, output nothing."
)


def ts_system(hotwords: str = "", legacy: bool = False) -> str:
    """TS-ASR system prompt; a hotword list follows on its own line in the plain-ASR form ``Hotwords: w1,w2``.

    Without hotwords ("" or the sampler's "N/A") the prompt is exactly the base prompt. ``legacy`` selects
    ``TS_CONCAT_SYSTEM``, the prompt of models trained before ``TS_SYSTEM``.
    """
    base = TS_CONCAT_SYSTEM if legacy else TS_SYSTEM
    if not hotwords or hotwords == "N/A":
        return base
    return f"{base}\nHotwords: {hotwords}"
