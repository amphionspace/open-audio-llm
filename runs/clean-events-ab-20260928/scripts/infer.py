from settings import ROOT, RUNS, OUTPUT, SETTINGS, artifact, child
"""Batched vLLM inference with complete raw generation evidence."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('candidate', choices=['control','treatment','initial'])
    parser.add_argument('--shard', type=int, required=True)
    args = parser.parse_args()
    candidate = args.candidate
    shard = args.shard
    root = artifact('evaluation')
    plan = json.loads((root / 'plan.json').read_text())
    folder = root / candidate / f'shard-{shard}'
    folder.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    import torch
    from qwen_asr import Qwen3ASRModel
    from qwen_asr.inference.utils import parse_asr_output
    from vllm import SamplingParams
    torch.set_num_threads(4)
    engine = Qwen3ASRModel.LLM(plan['models'][candidate],
        dtype=plan['dtype'], max_model_len=plan['max_model_len'],
        gpu_memory_utilization=SETTINGS['parameters']['inference']['gpu_memory_utilization'], max_num_seqs=plan['batch_size'],
        max_num_batched_tokens=SETTINGS['parameters']['inference']['max_num_batched_tokens'],
        enforce_eager=SETTINGS['parameters']['inference']['enforce_eager'],
        tensor_parallel_size=SETTINGS['parameters']['inference']['tensor_parallel_size'], seed=plan['seed'],
        limit_mm_per_prompt={'audio': 1},
        max_inference_batch_size=plan['batch_size'], max_new_tokens=plan['max_tokens'])
    assert engine.backend == 'vllm' and type(engine.model).__module__.startswith('vllm.')
    params = SamplingParams(temperature=plan['temperature'], seed=plan['seed'],
        repetition_penalty=plan['repetition_penalty'], max_tokens=plan['max_tokens'],
        stop_token_ids=plan['stop_token_ids'], ignore_eos=False, min_tokens=0)
    runtime = {'candidate': candidate, 'backend': engine.backend,
        'engine_class': f'{type(engine.model).__module__}.{type(engine.model).__name__}',
        'dtype': str(engine.model.llm_engine.model_config.dtype),
        'packages': {k: importlib.metadata.version(k) for k in ['vllm', 'torch', 'transformers', 'qwen-asr']},
        'encoder_attention': plan['encoder_attention'], 'sampling_params': str(params), 'prompt': plan['prompt'], 'gpu': torch.cuda.get_device_name(0)}
    (folder / 'runtime.json').write_text(json.dumps(runtime, ensure_ascii=False, indent=2) + '\n')
    clips = [json.loads(line) for line in (root / 'clips.jsonl').read_text().splitlines()]
    assert all(c['duration'] * 100 <= plan['encoder_attention']['n_window_infer'] for c in clips)
    by_id = {c['id']: c for c in clips}
    positions = {c['id']: i + 1 for i, c in enumerate(clips)}
    priority = plan['priority_ids']
    ordered = [by_id[k] for k in priority] + [c for c in clips if c['id'] not in priority]
    batches = [ordered[:len(priority)]]
    pending = ordered[len(priority):]
    batches += [pending[i:i+plan['batch_size']] for i in range(0,len(pending),plan['batch_size'])]
    batches = [batch for index,batch in enumerate(batches) if index % plan['worker_count'] == shard]
    expected = {clip['id'] for batch in batches for clip in batch}
    path = folder / 'predictions.jsonl'
    resuming = SETTINGS['recording'].get('resuming', False)
    assert resuming or not path.exists(), 'Do not overwrite completed inference'
    previous = []
    if path.exists():
        raw = path.read_bytes()
        complete_bytes = raw[:raw.rfind(b'\n') + 1]
        previous = [json.loads(line) for line in complete_bytes.splitlines()]
        assert len({r['id'] for r in previous}) == len(previous)
        assert {r['id'] for r in previous} <= expected
        if complete_bytes != raw:
            path.write_bytes(complete_bytes)
    completed = {r['id'] for r in previous}
    total_seconds = 0.0
    with path.open('a', buffering=1) as output:
        for batch in batches:
            batch = [clip for clip in batch if clip['id'] not in completed]
            if not batch:
                continue
            inputs = []
            for clip in batch:
                audio, sr = sf.read(clip['audio'], dtype='float32')
                assert sr == 16000 and audio.ndim == 1
                inputs.append({'prompt': engine._build_text_prompt(context=plan['prompt'], force_language=None),
                               'multi_modal_data': {'audio': [audio]}})
            started = time.perf_counter()
            generated = engine.model.generate(inputs, sampling_params=params, use_tqdm=False)
            elapsed = time.perf_counter() - started
            assert len(generated) == len(batch)
            for clip, request in zip(batch, generated):
                result = request.outputs[0]
                language, text = parse_asr_output(result.text)
                row = {k: clip[k] for k in ['id', 'dataset', 'meeting_id', 'duration']}
                row.update(sample_index=positions[clip['id']], raw_text=result.text, text=text,
                    language=language, token_ids=list(result.token_ids), generated_tokens=len(result.token_ids),
                    finish_reason=result.finish_reason, stop_reason=result.stop_reason,
                    hit_token_limit=result.finish_reason == 'length',
                    inference_seconds=elapsed / len(batch), batch_seconds=elapsed, batch_size=len(batch))
                output.write(json.dumps(row, ensure_ascii=False) + '\n')
                completed.add(clip['id'])
            total_seconds += elapsed
            print(json.dumps({'candidate': candidate, 'completed': len(completed), 'total': len(clips),
                'batch_seconds': elapsed, 'batch_cap_hits': sum(x.outputs[0].finish_reason == 'length' for x in generated)}), flush=True)
    assert completed == expected
    (folder / 'inference-complete.json').write_text(json.dumps({'samples': len(completed), 'batch_inference_seconds': total_seconds}) + '\n')

if __name__ == '__main__':
    main()
