"""Enforce the requested full update scope and honour the local selector's stop request.

Checkpoint selection and final evaluation run on the development machine
(scripts/local_selection.py), so training never pauses for evaluation.
"""
import json
from pathlib import Path

import torch
from swift.callbacks import TrainerCallback, callbacks_map
import yaml

# Rank 0 reads the shared stop file and broadcasts it; checking every step would
# add a collective per step for a decision that only changes every 500 steps.
STOP_CHECK_STEPS = 5


class ManySpeakerTrainingAudit(TrainerCallback):
    def on_train_begin(self,args,state,control,**kwargs):
        training = yaml.safe_load((Path(args.output_dir).parents[1]/'effective.yaml').read_text())
        parameters = list(self.trainer.model.named_parameters())
        assert all(p.requires_grad for _,p in parameters), 'Every model parameter must be trainable'
        assert not any('lora_' in n for n,_ in parameters), 'LoRA is not allowed in this run'
        assert args.world_size == training['cluster']['nodes'] * training['runtime']['distributed']['processes'] == 16
        assert self.trainer.catalog_sampler.full_coverage
        if state.is_world_process_zero:
            summary = {'world_size':args.world_size,'global_step':state.global_step,
                       'trainable_parameters':sum(p.numel() for _,p in parameters),
                       'frozen_parameters':0,'lora_parameters':0,'train_records':len(self.trainer.train_dataset),
                       'sampler':self.trainer.catalog_sampler.state_dict()}
            (Path(args.output_dir)/'actual-training-contract.json').write_text(json.dumps(summary,indent=2)+'\n')

    def on_step_end(self,args,state,control,**kwargs):
        if state.global_step % STOP_CHECK_STEPS:
            return control
        if state.is_world_process_zero:
            optimizer = self.trainer.optimizer
            groups = [{'step':state.global_step,'lr':g['lr'],'parameters':sum(p.numel() for p in g['params'])}
                      for g in optimizer.param_groups]
            assert all(g['lr'] > 0 for g in groups if g['parameters'])
            path = Path(args.output_dir)/'actual-learning-rates.json'
            if not path.exists():
                path.write_text(json.dumps(groups,indent=2)+'\n')
        request = [None]
        if state.is_world_process_zero:
            path = Path(args.output_dir)/'stop-request.json'
            request[0] = json.loads(path.read_text()) if path.exists() else None
        if torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(request,src=0)
        if request[0]:
            control.should_training_stop = True
        return control


callbacks_map['many_speaker_training_audit'] = ManySpeakerTrainingAudit
