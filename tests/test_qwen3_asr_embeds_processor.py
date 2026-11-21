import pytest

pytest.importorskip("vllm")


class _Tokenizer:
    def encode(self, prompt, add_special_tokens=False):
        return [len(prompt), 7]


class _Info:
    def get_tokenizer(self):
        return _Tokenizer()


class _Self:
    info = _Info()


@pytest.mark.parametrize("by_keyword", [True, False])
def test_embedding_prompts_skip_the_hf_processor_on_the_vllm_018_call(by_keyword):
    """vLLM 0.18's _apply_hf_processor_text_mm calls by keyword; without raw audio only the prompt is tokenised."""
    from open_audio_llm.integrations.vllm.plugin.qwen3_asr_embeds import Qwen3ASREmbedMultiModalProcessor

    call = Qwen3ASREmbedMultiModalProcessor._call_hf_processor
    values = {"prompt": "<|audio_pad|>", "mm_data": {}, "mm_kwargs": {}, "tok_kwargs": {}}
    out = call(_Self(), **values) if by_keyword else call(_Self(), *values.values())
    assert out["input_ids"].tolist() == [[len("<|audio_pad|>"), 7]]
