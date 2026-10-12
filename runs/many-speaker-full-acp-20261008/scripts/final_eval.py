"""Run the two fixed reporting panels after selection, before weight upload cleanup."""
import json
import os
from pathlib import Path
import sys

import yaml
from checkpoint_eval import evaluate


def main(output, effective, chosen=None):
    output = Path(output)
    training = yaml.safe_load(Path(effective).read_text())
    settings = training['parameters']['final_evaluation']
    # Model workers must not inherit the training rendezvous.
    for key in list(os.environ):
        if key in {'RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK','ROLE_RANK',
                   'ROLE_WORLD_SIZE','MASTER_ADDR','MASTER_PORT'} or key.startswith('TORCHELASTIC_'):
            os.environ.pop(key,None)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    decision = json.loads((output/'selection-decision.json').read_text())
    chosen = chosen or decision['best_checkpoint'] or decision['checkpoint']
    step = int(Path(chosen).name.removeprefix('checkpoint-'))
    name = training['tracking']['name']+'-'+Path(effective).parent.name
    fixed = evaluate(settings['fixed_template'],chosen,output/'final-evaluation/fixed',name+'-final-fixed',step,
                     cached=settings['fixed_baseline'])
    meeting = evaluate(settings['meeting_template'],chosen,output/'final-evaluation/meeting180',name+'-final-meeting180',step,
                       result_name='benchmark-summary.json')
    rates = fixed['diarization']['by_candidate']
    fixed_delta = sum(v['cpCER'] for v in rates['latest'].values())/len(rates['latest']) - sum(v['cpCER'] for v in rates['unified'].values())/len(rates['unified'])
    current,baseline = (meeting['candidates'][n] for n in ('latest','step8000'))
    checks = {'development_improved':decision['improved'],
              'fixed_chinese_retained':fixed_delta <= settings['chinese_tolerance'],
              'meeting_many_speaker_improved':current['headline_cp_error_rate']['synthetic_6-10spk'] < baseline['headline_cp_error_rate']['synthetic_6-10spk'],
              'loops_not_increased':current['all']['repetition_loops'] <= baseline['all']['repetition_loops']}
    result = {'checkpoint':chosen,'selection_passed':decision['improved'],
              'passed':all(checks.values()),'checks':checks,'fixed_cpCER_delta':fixed_delta,
              'fixed_cpCER':{k:v['cpCER'] for k,v in fixed['diarization']['by_candidate']['latest'].items()},
              'meeting180':meeting['candidates']['latest']}
    (output/'final-results.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main(*sys.argv[1:])
