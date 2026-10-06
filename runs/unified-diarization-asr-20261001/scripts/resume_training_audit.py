"""Check that a LoRA run resumed the saved weights, optimizer, schedule and data cursor.

local_training_audit.py asserts a fresh start (global_step 0); a resumed run
must instead prove that the checkpoint state is what training continues from.
"""
import json
import math
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file
from swift.callbacks import TrainerCallback, callbacks_map

# LoRA 002 contract (docs/experiments/2026-10-06): 4,023,765 records, 34,865,152 LoRA parameters.
TRAINING_RECORDS = 4023765
TRAINABLE_PARAMETERS = 34865152


class ResumeTrainingAudit(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        trainer = self.trainer
        checkpoint = Path(args.resume_from_checkpoint)
        saved = json.loads((checkpoint / 'trainer_state.json').read_text())
        assert args.continuous_training and args.max_steps == sys.maxsize
        assert args.lr_scheduler_type == 'inverse_sqrt' and args.warmup_steps > 0
        assert state.global_step == saved['global_step'] > 0
        assert trainer.retention_teacher is None
        assert trainer.catalog_sampler.full_coverage
        assert len(trainer.train_dataset) == TRAINING_RECORDS
        sampler = trainer.catalog_sampler.state_dict()
        updates = args.world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps
        assert sampler['consumed'] * args.world_size == saved['global_step'] * updates
        parameters = dict(trainer.model.named_parameters())
        trainable = {name: p for name, p in parameters.items() if p.requires_grad}
        assert trainable and all('lora_' in name and 'thinker.model.' in name for name in trainable)
        assert sum(p.numel() for p in trainable.values()) == TRAINABLE_PARAMETERS
        adapter = load_file(str(checkpoint / 'adapter_model.safetensors'))
        selected = next(name for name in trainable if 'lora_B' in name)
        # PEFT saves "....lora_B.weight" for the parameter "....lora_B.default.weight".
        stored = adapter[selected.replace('.default.weight', '.weight')]
        assert torch.equal(trainable[selected].detach().cpu(), stored.to(trainable[selected].dtype))
        steps = {int(s['step']) for s in trainer.optimizer.state.values() if 'step' in s}
        assert steps == {saved['global_step']}, steps
        # get_last_lr() is restored state; check the schedule function that drives later steps.
        lr = trainer.lr_scheduler.get_last_lr()[0]
        for step in (state.global_step, state.global_step + 1):
            assert math.isclose(trainer.lr_scheduler.lr_lambdas[0](step), self.decay(args, step),
                                rel_tol=1e-9), step
        assert math.isclose(lr, args.learning_rate * self.decay(args, state.global_step), rel_tol=1e-6), lr
        self.selected, self.initial = selected, trainable[selected].detach().cpu().clone()
        self.frozen = next(name for name in parameters if 'audio_tower' in name and parameters[name].ndim == 2)
        self.frozen_initial = parameters[self.frozen].detach().cpu().clone()
        self.start = state.global_step
        self.summary = {'rank': args.process_index, 'resumed_from': str(checkpoint),
                        'global_step': state.global_step, 'world_size': args.world_size,
                        'samples_per_update': updates, 'training_records': len(trainer.train_dataset),
                        'trainable_parameters': TRAINABLE_PARAMETERS, 'teacher_enabled': False,
                        'adapter_restored': True, 'optimizer_step': saved['global_step'],
                        'learning_rate': lr, 'sampler': sampler, 'selected_parameter': selected}
        self.write(args, 'start')

    @staticmethod
    def decay(args, step):
        timescale = args.lr_scheduler_kwargs['timescale']
        return 1 / math.sqrt((step + timescale - args.warmup_steps) / timescale)

    def write(self, args, phase):
        (Path(args.output_dir) / f'audit-{phase}-rank{args.process_index}.json').write_text(
            json.dumps(self.summary, indent=2) + '\n')

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.start + 5:
            parameters = dict(self.trainer.model.named_parameters())
            delta = parameters[self.selected].detach().cpu() - self.initial
            assert torch.isfinite(delta).all() and torch.count_nonzero(delta) > 0
            assert torch.equal(parameters[self.frozen].detach().cpu(), self.frozen_initial)
            expected = args.learning_rate * self.decay(args, state.global_step)
            assert all(math.isclose(g['lr'], expected, rel_tol=1e-6)
                       for g in self.trainer.optimizer.param_groups), expected
            self.summary.update(global_step=state.global_step,
                                changed_elements=int(torch.count_nonzero(delta)),
                                max_abs_delta=float(delta.abs().max()),
                                frozen_audio_unchanged=True,
                                learning_rates=[group['lr'] for group in self.trainer.optimizer.param_groups])
            self.write(args, f'step-{state.global_step}')
        return control


callbacks_map['resume_training_audit'] = ResumeTrainingAudit
