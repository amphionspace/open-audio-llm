"""Use the established vLLM full-clip protocol for meeting and ASR generation."""
import argparse
import importlib.metadata
from io import BytesIO
import json
from pathlib import Path
import time

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--candidate', required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    settings = config['parameters']
    root = Path(settings['output'])
    folder = root / args.candidate
    plan = json.loads((root / 'plan.json').read_text())
    import soundfile as sf
    import torch
    from qwen_asr import Qwen3ASRModel
    from qwen_asr.inference.utils import parse_asr_output
    from vllm import SamplingParams
    from open_audio_llm.data.catalog_dataset import CatalogSwiftDataset, read_data_config

    torch.set_num_threads(4)
    engine = Qwen3ASRModel.LLM(
        plan['models'][args.candidate], dtype=plan['dtype'],
        max_model_len=plan['max_model_len'],
        gpu_memory_utilization=settings['gpu_memory_utilization'],
        max_num_seqs=plan['batch_size'], max_num_batched_tokens=16384,
        enforce_eager=True, tensor_parallel_size=1, seed=plan['seed'],
        limit_mm_per_prompt={'audio': 1}, max_inference_batch_size=plan['batch_size'],
        max_new_tokens=plan['max_tokens'],
        **{key: settings[key] for key in ('attention_config', 'mm_encoder_attn_backend')
           if key in settings},
    )
    assert engine.backend == 'vllm' and type(engine.model).__module__.startswith('vllm.')
    params = SamplingParams(
        temperature=plan['temperature'], seed=plan['seed'],
        repetition_penalty=plan['repetition_penalty'], max_tokens=plan['max_tokens'],
        stop_token_ids=plan['stop_token_ids'], ignore_eos=False, min_tokens=0,
    )
    runtime = {
        'backend': engine.backend,
        'engine_class': f'{type(engine.model).__module__}.{type(engine.model).__name__}',
        'packages': {k: importlib.metadata.version(k) for k in ('vllm', 'torch', 'transformers', 'qwen-asr')},
        'encoder_attention': plan['encoder_attention'], 'sampling_params': str(params),
        'prompt': plan['prompt'], 'dtype': str(engine.model.llm_engine.model_config.dtype),
        'decoder_attention': str(engine.model.llm_engine.vllm_config.attention_config.backend),
        'mm_encoder_attention': str(engine.model.llm_engine.model_config.multimodal_config.mm_encoder_attn_backend),
        'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
    }
    (folder / 'runtime.json').write_text(json.dumps(runtime, indent=2) + '\n')

    def generate(audio, contexts):
        inputs = [{'prompt': engine._build_text_prompt(context=context, force_language=None),
                   'multi_modal_data': {'audio': [wave]}} for wave, context in zip(audio, contexts)]
        started = time.perf_counter()
        generated = engine.model.generate(inputs, sampling_params=params, use_tqdm=False)
        assert len(generated) == len(audio)
        elapsed = time.perf_counter() - started
        rows = []
        for request in generated:
            result = request.outputs[0]
            language, text = parse_asr_output(result.text)
            rows.append(dict(raw_text=result.text, text=text, language=language,
                             token_ids=list(result.token_ids), generated_tokens=len(result.token_ids),
                             finish_reason=result.finish_reason, stop_reason=result.stop_reason,
                             hit_token_limit=result.finish_reason == 'length',
                             inference_seconds=elapsed / len(audio)))
        return rows

    if args.candidate in settings['models']:
        clips = [json.loads(line) for line in (root / 'clips.jsonl').read_text().splitlines()]
        by_id = {clip['id']: clip for clip in clips}
        priority = plan['priority_ids']
        ordered = [by_id[k] for k in priority] + [c for c in clips if c['id'] not in priority]
        batches = [ordered[:len(priority)]]
        remaining = ordered[len(priority):]
        batches += [remaining[i:i + plan['batch_size']] for i in range(0, len(remaining), plan['batch_size'])]
        completed = 0
        with (folder / 'predictions.jsonl').open('w', buffering=1) as stream:
            for batch in batches:
                audio = []
                for clip in batch:
                    assert clip['duration'] * 100 <= plan['encoder_attention']['n_window_infer']
                    # The frozen clip manifest keeps original-host paths; the per-clip
                    # sha256 below proves a relocated copy is the same audio.
                    path = (Path(settings['clip_audio_dir']) / Path(clip['audio']).name
                            if settings.get('clip_audio_dir') else clip['audio'])
                    wave, sr = sf.read(path, dtype='float32')
                    assert sr == 16000 and wave.ndim == 1
                    assert hashlib_file(path) == clip['audio_sha256']
                    audio.append(wave)
                for clip, prediction in zip(batch, generate(audio, [plan['prompt']] * len(batch))):
                    row = {k: clip[k] for k in ('id', 'dataset', 'meeting_id', 'duration')}
                    stream.write(json.dumps({**row, **prediction}, ensure_ascii=False) + '\n')
                completed += len(batch)
                print(json.dumps({'task': 'diarization', 'completed': completed, 'total': len(clips)}), flush=True)

    data = read_data_config(settings['data_config'])
    data['validation'] = data['evaluation']
    dataset = CatalogSwiftDataset(data, training=False, message_format='qwen3_asr')
    rows = []
    conditions = {'plain': '', 'diarization': settings['single_speaker_prompt']}
    with (folder / 'asr-predictions.jsonl').open('w', buffering=1) as stream:
        for start in range(0, len(dataset), plan['batch_size']):
            indexes = list(range(start, min(start + plan['batch_size'], len(dataset))))
            audio = [sf.read(BytesIO(dataset[i]['audios'][0]), dtype='float32')[0] for i in indexes]
            for condition, prompt in conditions.items():
                for index, prediction in zip(indexes, generate(audio, [prompt] * len(audio))):
                    record = dataset.records[index].record
                    row = {**prediction, 'id': record.id, 'reference': record.target,
                           'dataset_id': dataset.records[index].dataset_id,
                           'reference_language': record.language, 'condition': condition}
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                    rows.append(row)
            print(json.dumps({'task': 'asr', 'completed': min(start + plan['batch_size'], len(dataset)),
                              'total': len(dataset)}), flush=True)
    (folder / 'inference-complete.json').write_text(json.dumps({'asr_rows': len(rows), 'backend': 'vllm'}) + '\n')


def hashlib_file(path):
    import hashlib
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


if __name__ == '__main__':
    main()
