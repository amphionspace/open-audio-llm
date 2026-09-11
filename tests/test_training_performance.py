from types import SimpleNamespace

import torch

from open_audio_llm.integrations.ms_swift.performance import (
    PerformanceCollator, install_performance_logging,
)


def test_collator_counts_actual_samples_and_strips_metadata():
    def collate(rows):
        assert all('_performance' not in row for row in rows)
        return {'attention_mask': torch.tensor([[1, 1, 0], [1, 1, 1]]),
                'labels': torch.tensor([[-100, 2, -100], [-100, 2, 3]])}
    rows = [{'input_ids': [1, 2], '_performance': {
        'dataset_id': 'speech', 'audio_seconds': d, 'decode_s': .01,
        'wave_augment_s': .02, 'encode_s': .03, 'prepare_s': .07,
    }} for d in (1.0, 2.0)]
    result = PerformanceCollator(collate)(rows)
    meta = result['_performance']
    assert meta['samples'] == 2 and meta['audio_seconds'] == 3
    assert meta['tokens'] == 5 and meta['tokens_capacity'] == 6
    assert meta['supervised_tokens'] == 3 and meta['sources'] == {'speech': 2}
    assert '_performance' in rows[0]


def test_metrics_do_not_reach_model_or_change_training_output(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    seen = []
    def step(model, inputs, *args, **kwargs):
        seen.append(inputs)
        return torch.tensor(2.)
    trainer = SimpleNamespace(training_step=step, get_batch_samples=lambda: 'batch',
                              add_callback=lambda c: None)
    callback = install_performance_logging(trainer)
    assert trainer.get_batch_samples() == 'batch'
    result = trainer.training_step(None, {'labels': [1], '_performance': {
        'samples': 2, 'audio_seconds': 3., 'sources': {'speech': 2}}})
    assert result.item() == 2 and seen == [{'labels': [1]}]
    assert callback.totals['samples'] == 2
    assert callback.sources == {'speech': 2}


def test_native_audio_gradient_hook_uses_real_convolution():
    import pytest
    pytest.importorskip('qwen_asr')
    from open_audio_llm.integrations.ms_swift.register_qwen3_asr import Qwen3ASRLoader
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRAudioEncoder
    Qwen3ASRLoader._import_qwen_asr()
    tower = Qwen3ASRAudioEncoder.__new__(Qwen3ASRAudioEncoder)
    torch.nn.Module.__init__(tower)
    tower.conv2d1 = torch.nn.Conv2d(1, 2, 3)
    tower.requires_grad_(False)
    tower.disable_input_require_grads()
    assert tower.get_input_embeddings() is tower.conv2d1
    tower.enable_input_require_grads()
    assert tower.conv2d1(torch.ones(1, 1, 5, 5)).requires_grad
    tower.disable_input_require_grads()
    tower.disable_input_require_grads()
    assert not tower.conv2d1(torch.ones(1, 1, 5, 5)).requires_grad
    replacement = torch.nn.Conv2d(1, 3, 3)
    tower.set_input_embeddings(replacement)
    assert tower.get_input_embeddings() is replacement
