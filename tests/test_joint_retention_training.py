import json
from types import SimpleNamespace

import pytest
import torch
from transformers import TrainingArguments

from open_audio_llm.integrations.ms_swift.retention_eval import RetentionEvaluationCallback


@pytest.mark.parametrize('freeze_audio', [False, True])
@pytest.mark.parametrize('tied_head', [False, True])
def test_differential_optimizer_covers_head_and_separates_aligner(tmp_path, freeze_audio, tied_head):
    from open_audio_llm.integrations.ms_swift import register_qwen3_asr  # noqa: F401
    from swift.model.model_arch import get_model_arch
    from swift.optimizers.multimodal import MultimodalOptimizerCallback

    model = torch.nn.Module()
    model.thinker = torch.nn.Module()
    model.thinker.audio_tower = torch.nn.Module()
    for name in ('conv2d1', 'proj1', 'proj2'):
        setattr(model.thinker.audio_tower, name, torch.nn.Linear(3, 3))
    model.thinker.model = torch.nn.Linear(3, 5)
    model.thinker.lm_head = torch.nn.Linear(3, 5)
    if tied_head:
        model.thinker.lm_head.weight = model.thinker.model.weight
    model.thinker.audio_tower.requires_grad_(not freeze_audio)
    model.model_meta = SimpleNamespace(model_arch=get_model_arch('amphion_asr_1.7b'))
    args = TrainingArguments(str(tmp_path), use_cpu=True, report_to=[], learning_rate=5e-6)
    args.vit_lr, args.aligner_lr = 1e-6, 3e-6
    optimizer = MultimodalOptimizerCallback(args, SimpleNamespace(model=model)).create_optimizer()
    grouped = [p for group in optimizer.param_groups for p in group['params']]
    expected = [p for p in model.parameters() if p.requires_grad]
    assert len(grouped) == len({id(p) for p in grouped}) == len(expected)
    assert {id(p) for p in grouped} == {id(p) for p in expected}
    rates = {id(p): group['lr'] for group in optimizer.param_groups for p in group['params']}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            expected_lr = (args.aligner_lr if '.proj' in name else args.vit_lr) if 'audio_tower' in name else args.learning_rate
            assert rates[id(parameter)] == expected_lr
    before = [p.detach().clone() for p in expected]
    sum(p.sum() for p in expected).backward()
    optimizer.step()
    assert all(not torch.equal(old, new) for old, new in zip(before, expected))


@pytest.mark.parametrize('step,invoked', [(2, True), (4, False), (6, True), (10, True)])
def test_evaluation_runs_at_first_periodic_and_final_save_on_gpu_zero(tmp_path, monkeypatch, step, invoked):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
    monkeypatch.setenv('RANK', '0')
    monkeypatch.setenv('WORLD_SIZE', '2')
    monkeypatch.setenv('TORCHELASTIC_RUN_ID', 'training')
    script = tmp_path / 'evaluate.py'
    script.write_text('import json, os, sys\nfrom pathlib import Path\n'
                      'Path(sys.argv[2], "env.json").write_text(json.dumps(dict(os.environ)))\n')
    callback = RetentionEvaluationCallback(script, 6)
    args = SimpleNamespace(output_dir=tmp_path, save_steps=2, ddp_timeout=120)
    state = SimpleNamespace(global_step=step, max_steps=10, is_world_process_zero=True)
    callback.on_save(args, state, None)
    output = tmp_path / 'retention-evaluations' / f'checkpoint-{step}' / 'env.json'
    assert output.exists() == invoked
    if invoked:
        env = json.loads(output.read_text())
        assert env['CUDA_VISIBLE_DEVICES'] == '0'
        assert all(key not in env for key in ('RANK', 'WORLD_SIZE', 'TORCHELASTIC_RUN_ID'))


@pytest.mark.parametrize('rank_zero', [True, False])
def test_evaluator_failure_reaches_every_training_rank(tmp_path, monkeypatch, rank_zero):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'barrier', lambda: None)
    def broadcast(errors, src):
        if rank_zero:
            assert 'exited 7' in errors[0]
        else:
            errors[0] = 'Evaluator exited 7'
    monkeypatch.setattr(torch.distributed, 'broadcast_object_list', broadcast)
    script = tmp_path / 'evaluate.py'
    script.write_text('raise SystemExit(7)\n')
    callback = RetentionEvaluationCallback(script, 2)
    args = SimpleNamespace(output_dir=tmp_path, save_steps=2, ddp_timeout=120)
    state = SimpleNamespace(global_step=2, max_steps=4, is_world_process_zero=rank_zero)
    with pytest.raises(RuntimeError, match='exited 7'):
        callback.on_save(args, state, None)
    assert (tmp_path / 'retention-evaluations').exists() == rank_zero
