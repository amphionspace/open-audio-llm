"""Verify fresh timestamp training, pinned weights, and real parameter updates."""
import json
from pathlib import Path

import torch
from swift.callbacks import TrainerCallback, callbacks_map
from safetensors import safe_open

def module_group(name):
    if name.startswith(('thinker.audio_tower.proj1.', 'thinker.audio_tower.proj2.')):
        return 'aligner'
    if name.startswith('thinker.audio_tower.'):
        return 'encoder'
    if name.startswith(('thinker.model.', 'thinker.lm_head.')):
        return 'llm'
    raise ValueError(name)


class SOTTrainingAudit(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        trainer = self.trainer
        self.phase = json.loads(Path(trainer.train_dataset.config['experiment_plan']).read_text())
        assert state.global_step == 0
        assert not trainer.optimizer.state
        assert args.max_steps == 1000 and args.warmup_steps == 500
        # HF restores the old save cadence from trainer_state.json after parsing args.
        state.save_steps = args.save_steps
        assert state.save_steps == 500
        assert trainer.lr_scheduler.last_epoch == 0
        assert args.world_size == 2 and args.gradient_accumulation_steps == 4
        assert not args.save_only_model and args.bf16 and not args.fp16
        objective = json.loads((Path(args.output_dir) / 'retention-objective.json').read_text())
        assert Path(objective['teacher']).resolve() == Path('/ai_sds_wuzz/MODELS/Qwen3-ASR-1.7B/Qwen/Qwen3-ASR-1___7B')
        dataset = trainer.train_dataset
        assert dataset.config['objective']['replay_kl_weight'] == 2
        assert dataset.config['replay']['window_samples'] == 10000
        sources = [source for source in dataset.sources if source['dataset_id'] == 'sot_multispeaker_zh_en']
        assert len(sources) == 62 and {source['version'] for source in sources} == {'synthetic-v2-timed-trial-v1-20260920'}
        assert sum(len(indices) for source, indices in zip(dataset.sources, dataset.source_ranges)
                   if source['dataset_id'] == 'sot_multispeaker_zh_en') == 1644206
        assert all(source['sot_timestamps'] for source in sources)
        assert all(source['sot_alignment_sha256'] == 'dd60f77acc2d164983d5906d2e8199f8f1edb7f65992768c1c89605c27567f5d' for source in sources)
        event_sources = [s for s in dataset.sources if s['dataset_id'] == self.phase['event_dataset']]
        assert len(event_sources) == 1 and event_sources[0]['version'] == self.phase['event_version']
        assert event_sources[0]['weight'] == 35
        assert sum(len(indices) for source, indices in zip(dataset.sources, dataset.source_ranges)
                   if source['dataset_id'] == self.phase['event_dataset']) == self.phase['event_records']
        meeting_ids = {'aishell4_meeting_sot', 'alimeeting_sdm_meeting_sot'}
        meeting_sources = [s for s in dataset.sources if s['dataset_id'] in meeting_ids]
        assert len(meeting_sources) == 8
        assert all(s['version'] == 'meeting-long-v1-20260920' for s in meeting_sources)
        assert sum(len(ids) for source, ids in zip(dataset.sources, dataset.source_ranges)
                   if source['dataset_id'] in meeting_ids) == 39407
        assert dataset.config['batching']['max_duration'] == 650
        actual_args = json.loads((Path(args.output_dir) / 'args.json').read_text())
        assert actual_args['max_length'] == 16384
        assert actual_args['eval_strategy'] == 'no'
        parameters = dict(trainer.model.named_parameters())
        assert all(p.requires_grad and p.dtype == torch.float32 for p in parameters.values())
        encoder = trainer.model.thinker.audio_tower
        assert encoder.forward.__func__.__module__ == 'open_audio_llm.integrations.ms_swift.audio_batching'
        assert encoder.is_gradient_checkpointing and trainer.model.thinker.model.is_gradient_checkpointing
        teacher = trainer.retention_teacher
        assert not any(p.requires_grad for p in teacher.parameters())
        assert teacher.thinker.audio_tower.forward.__func__.__module__ != encoder.forward.__func__.__module__
        ids = {id(p): module_group(n) for n, p in parameters.items()}
        self.groups = []
        expected = {'encoder': 1e-5, 'aligner': 2e-5, 'llm': 1e-5}
        for group in trainer.optimizer.param_groups:
            names = {ids[id(p)] for p in group['params']}
            assert len(names) == 1
            name = names.pop()
            assert group['initial_lr'] == expected[name]
            self.groups.append(name)
        assert set(self.groups) == set(expected)
        keys = [f'thinker.audio_tower.layers.{n}.self_attn.q_proj.weight' for n in (0, 12, 23)]
        keys += ['thinker.audio_tower.proj1.weight', 'thinker.audio_tower.proj2.weight']
        keys += [f'thinker.model.layers.{n}.self_attn.q_proj.weight' for n in (0, 14, 27)]
        self.initial = {key: parameters[key].detach().cpu().clone() for key in keys}
        initial = Path(self.phase['source_checkpoint'])
        weight_map = json.loads((initial / 'model.safetensors.index.json').read_text())['weight_map']
        for key, tensor in self.initial.items():
            with safe_open(initial / weight_map[key], framework='pt', device='cpu') as reader:
                assert torch.equal(tensor, reader.get_tensor(key)), key
        assert trainer.catalog_sampler.epoch == 0 and trainer.catalog_sampler.consumed == 0
        self.long_audio_seen = False
        def check_audio(module, positional, kwargs):
            mask = kwargs.get('feature_attention_mask')
            if mask is not None and int(mask.sum(-1).max()) > 3000:
                self.long_audio_seen = True
        trainer.model.register_forward_pre_hook(check_audio, with_kwargs=True)
        self.write(args, state, 'startup')

    def write(self, args, state, label, updates=None):
        trainer = self.trainer
        result = {'rank': args.process_index, 'global_step': state.global_step,
                  'world_size': args.world_size, 'gradient_accumulation_steps': args.gradient_accumulation_steps,
                  'save_only_model': args.save_only_model, 'effective_save_steps': state.save_steps, 'student_batched': True, 'teacher_native': True,
                  'trainable_parameters': sum(p.numel() for p in trainer.model.parameters()),
                  'optimizer_groups': [dict(module=name, lr=g['lr'], peak_lr=g['initial_lr'],
                      parameter_count=sum(p.numel() for p in g['params']))
                      for name, g in zip(self.groups, trainer.optimizer.param_groups)],
                  'parameter_updates': updates, 'long_audio_seen': self.long_audio_seen,
                  'scheduler_last_epoch': trainer.lr_scheduler.last_epoch,
                  'optimizer_steps': sorted({int(s['step']) for s in trainer.optimizer.state.values()}),
                  'sampler': trainer.catalog_sampler.state_dict()}
        (Path(args.output_dir) / f'audit-{label}-rank{args.process_index}.json').write_text(json.dumps(result, indent=2))

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in (5, 20):
            assert self.trainer.lr_scheduler.last_epoch == state.global_step
            assert {int(s['step']) for s in self.trainer.optimizer.state.values()} == {state.global_step}
            parameters = dict(self.trainer.model.named_parameters())
            updates = {}
            for key, initial in self.initial.items():
                delta = parameters[key].detach().cpu() - initial
                assert torch.isfinite(delta).all() and torch.count_nonzero(delta) > 0, key
                updates[key] = {'changed_elements': int(torch.count_nonzero(delta)),
                                'max_abs_delta': float(delta.abs().max())}
            assert self.long_audio_seen, 'No complete audio over 30 seconds reached the model'
            self.write(args, state, f'step-{state.global_step}', updates)
        return control


callbacks_map['sot_training_audit'] = SOTTrainingAudit
