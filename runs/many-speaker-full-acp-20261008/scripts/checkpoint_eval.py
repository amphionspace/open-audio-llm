"""Evaluate both selection panels and write the shared training stop decision."""
import json
from pathlib import Path
import subprocess
import sys

import yaml
from open_audio_llm.run_config import load_config


def evaluate(template, model, output, name, step, cached=None, result_name='summary.json'):
    config = load_config(Path(template), output/'preview')
    config.pop('config_file',None)
    config['runtime']['gpus'] = [0]
    config['parameters'].update(output={'path':'{attempt}/artifacts/evaluation'},models={'latest':str(model)},
                                primary_candidate='latest',training_step=step)
    if cached:
        config['parameters'].update(cached_predictions={'unified':cached['predictions']},
                                     cached_asr_predictions={'unified':cached['asr_predictions']})
    config['recording'] = {'attempts_dir':{'path':str(output/'attempts')},
                          'result':{'path':f'{{attempt}}/artifacts/evaluation/{result_name}'}}
    config['tracking'].update(name=name,metrics=[{'file':{'path':f'{{attempt}}/artifacts/evaluation/{result_name}'},'prefix':'evaluation'}])
    # Resolved paths stay absolute when the recipe is moved into this checkpoint.
    recipe = output/'evaluation.yaml'; output.mkdir(parents=True,exist_ok=True)
    recipe.write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True))
    subprocess.run([sys.executable,'-m','open_audio_llm.cli','eval','--config',str(recipe)],check=True)
    attempt = max((output/'attempts').glob('[0-9]*'))
    return json.loads((attempt/f'artifacts/evaluation/{result_name}').read_text())


def decision(history, baseline, chinese_baseline, tolerance, patience):
    recent = history[-patience:]
    stop = len(recent) == patience and all(r['chinese_cpCER'] > chinese_baseline+tolerance for r in recent)
    eligible = [r for r in history if r['chinese_cpCER'] <= chinese_baseline+tolerance
                and r['loops'] <= baseline['all']['repetition_loops']]
    best = min(eligible,key=lambda r:r['many_speaker_cpER']) if eligible else None
    return stop,best


def main(checkpoint, output, state=None):
    checkpoint,output = Path(checkpoint).resolve(),Path(output).resolve()
    # Selection state lives with the selector when checkpoints span resumed attempts.
    state = Path(state).resolve() if state else checkpoint.parent
    training = yaml.safe_load((checkpoint.parents[2]/'effective.yaml').read_text())
    settings = training['parameters']['checkpoint_evaluation']
    step = int(checkpoint.name.removeprefix('checkpoint-'))
    name = training['tracking']['name'] + '-' + checkpoint.parents[2].name
    baseline_root = state/'selection-baseline'
    baseline_file = baseline_root/'summary.json'
    if not baseline_file.exists():
        baseline = evaluate(settings['many_speaker_template'],training['task']['arguments']['--model'],
                            baseline_root,name+'-many-baseline',0)
        baseline_file.write_text(json.dumps(baseline,indent=2)+'\n')
    baseline = json.loads(baseline_file.read_text())
    chinese = evaluate(settings['chinese_template'],checkpoint,output/'chinese',name+f'-zh-{step}',step,
                       cached=settings['chinese_baseline'])
    many = evaluate(settings['many_speaker_template'],checkpoint,output/'many-speaker',name+f'-many-{step}',step)
    scores = chinese['diarization']['by_candidate']['latest']
    cpcer = sum(r['cpCER'] for r in scores.values()) / len(scores)
    metric = {'step':step,'checkpoint':str(checkpoint),'chinese_cpCER':cpcer,
              'many_speaker_cpER':many['macro_cpER'],'loops':many['all']['repetition_loops']}
    history_file = state/'selection-history.json'
    history = json.loads(history_file.read_text()) if history_file.exists() else []
    history.append(metric)
    history_file.write_text(json.dumps(history,indent=2)+'\n')
    stop,best = decision(history,baseline,settings['chinese_baseline_cpCER'],
                       settings['chinese_tolerance'],settings['regression_patience'])
    gate = {'step':step,'stop':stop,'reason':'consecutive_chinese_regression' if stop else None,
            'best_checkpoint':best['checkpoint'] if best else None,
            'best_many_speaker_cpER':best['many_speaker_cpER'] if best else None,
            'improved':best is not None and best['many_speaker_cpER'] < baseline['macro_cpER'],
            'baseline_many_speaker_cpER':baseline['macro_cpER'],**metric}
    metrics = {'eval/chinese/cpCER':cpcer,'eval/many_speaker/cpER':many['macro_cpER'],
               'eval/many_speaker/loops':many['all']['repetition_loops'],
               'eval/many_speaker/speaker_count_accuracy':many['all']['speaker_count_accuracy']}
    for count,scores in many['by_speakers'].items():
        metrics.update({f'eval/many_speaker/{count}spk/{k}':scores[k]
                        for k in ('cp_error_rate','speaker_count_accuracy','repetition_loops')})
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    # The collective in RetentionEvaluationCallback makes this decision visible to all ranks.
    (state/'selection-decision.json').write_text(json.dumps(gate,indent=2)+'\n')


if __name__ == '__main__':
    main(*sys.argv[1:])
