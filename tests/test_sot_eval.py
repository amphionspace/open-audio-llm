import json

import pytest

from open_audio_llm.eval.sot import rescore_predictions, score_speakers
from open_audio_llm.eval.ts_asr import summarize


@pytest.mark.parametrize('chinese,reference,prediction', [
    (True, '[S1] 你好\n[S2] 世界\n[S3] 再见', '[S1] 再见\n[S2] 你好\n[S3] 世界'),
    (False, '[S1] hello there\n[S2] good morning', '[S1] Good morning!\n[S2] Hello there.'),
])
def test_permutation_and_punctuation_do_not_change_transcription_score(chinese, reference, prediction):
    result = score_speakers(reference, prediction, chinese)
    assert result['errors'] == 0 and result['format_valid']
    assert result['speaker_attribution']['accuracy'] == 1


def test_missing_extra_and_untagged_speech_are_counted():
    reference = '[S1] 一二\n[S2] 三四\n[S3] 五六'
    assert score_speakers(reference, '[S1] 一二', True)['errors'] == 4
    assert score_speakers(reference, reference + '\n[S4] 七八', True)['errors'] == 2
    assert score_speakers(reference, '你好' + reference, True)['errors'] == 2
    assert score_speakers(reference, '一二', True)['errors'] == 4
    empty = score_speakers(reference, '[S1] \n[S2] ', True)
    assert empty['errors'] == 6 and empty['empty_output'] and not empty['format_valid']


def test_punctuation_only_speaker_is_scored_with_the_other_speakers():
    result = score_speakers('[S1] hello there\n[S2] .', '[S1] hello there', False)
    assert result['errors'] == 0 and result['reference_units'] == 2
    assert result['reference_speakers'] == 1


def test_punctuation_only_turn_does_not_abort_boundary_counts():
    from open_audio_llm.eval.sot_timestamps import summarize_timed_sot
    from open_audio_llm.data.sot import TIMESTAMP_FORMAT

    reference = "[S1][0.00-2.00] hello there\n[S2][1.00-1.40] ."
    row = dict(source="sot", language="en", task="speaker_attributed_asr", duration=2.0,
               sot_output_format=TIMESTAMP_FORMAT, reference=reference, prediction=reference)
    result = summarize_timed_sot([row], False)
    assert result["cp_error_rate"] == 0
    assert result["der"] == 0


def test_same_speaker_fragments_stay_together_but_duplicate_lines_violate_format():
    result = score_speakers('[S1] 一二\n[S2] 三四', '[S1] 一\n[S2] 三四\n[S1] 二', True)
    assert result['errors'] == 0 and not result['format_valid']
    with pytest.raises(ValueError, match='reference'):
        score_speakers('not labeled', '[S1] text', False)


def test_summary_keeps_speaker_error_metrics_separate_from_plain_asr():
    result = summarize([{'source': 'sot', 'language': 'zh', 'task': 'speaker_attributed_asr',
                         'reference': '[S1] 你好\n[S2] 世界', 'prediction': '[S1] 你好'},
                        {'source': 'asr', 'language': 'zh', 'task': 'asr',
                         'reference': '你好', 'prediction': '你好'}])
    assert result['sot']['metric'] == 'cpCER' and result['sot']['error_rate'] == 0.5
    assert result['sot']['speaker_count_accuracy'] == 0
    assert result['asr']['metric'] == 'CER' and result['asr']['error_rate'] == 0


def test_bilingual_score_counts_chinese_characters_and_english_words():
    reference = '[S1] 你好OpenAI\n[S2] Good morning'
    prediction = '[S1] good evening\n[S2] 你好 OpenAI'
    result = score_speakers(reference, prediction, chinese=False, mixed=True)
    assert result['reference_units'] == 5 and result['errors'] == 1
    assert result['speaker_attribution']['scored_units'] == 4
    assert result['speaker_attribution']['accuracy'] == 1
    result = summarize([{'source': 'mixed', 'language': 'zh-en', 'task': 'speaker_attributed_asr',
                          'reference': reference, 'prediction': prediction}])['mixed']
    assert result['metric'] == 'cpMER' and result['error_rate'] == .2


def test_attribution_detects_wrong_stream_with_correct_speaker_count():
    reference = '[S1] red green blue\n[S2] cat dog bird'
    prediction = '[S1] red green bird\n[S2] cat dog blue'
    result = score_speakers(reference, prediction, False)
    assert result['reference_speakers'] == result['predicted_speakers'] == 2
    assert result['speaker_attribution']['correct_units'] == 4
    assert result['speaker_attribution']['scored_units'] == 6
    assert result['speaker_attribution']['accuracy'] == pytest.approx(4 / 6)
    assert result['speaker_attribution']['coverage'] == 1
    swapped = '[S1] cat dog blue\n[S2] red green bird'
    assert score_speakers(reference, swapped, False)['speaker_attribution'] == result['speaker_attribution']


def test_attribution_does_not_call_lexical_errors_speaker_errors():
    result = score_speakers('[S1] red green blue\n[S2] cat dog bird', '[S1] red pink\n[S2] cat', False)
    attribution = result['speaker_attribution']
    assert result['errors'] == 4
    assert attribution['scored_units'] == attribution['correct_units'] == 2
    assert attribution['accuracy'] == 1
    assert attribution['coverage'] == pytest.approx(2 / 6)


def test_tied_transcription_permutations_choose_maximum_attribution_credit():
    reference = '[S1] a b\n[S2] c d'
    predictions = ['[S1] a\n[S2] d b c', '[S1] d b c\n[S2] a']
    for prediction in predictions:
        result = score_speakers(reference, prediction, False)
        assert result['errors'] == 4
        assert result['speaker_attribution']['accuracy'] == .75


def test_shared_words_and_excess_repetitions_are_excluded():
    reference = '[S1] the red red\n[S2] the cat'
    result = score_speakers(reference, '[S1] the red red\n[S2] the red cat', False)['speaker_attribution']
    assert result['unique_reference_units'] == 3
    assert result['excess_reference_units'] == 2
    assert result['scored_units'] == result['correct_units'] == 1
    assert result['coverage'] == .2
    # Repeated reference occurrences remain countable if predictions do not exceed them.
    result = score_speakers(reference, reference, False)['speaker_attribution']
    assert result['scored_units'] == 3 and result['excess_reference_units'] == 0


@pytest.mark.parametrize('reference,prediction', [
    ('[S1] hello\n[S2] hello', '[S1] hello\n[S2] hello'),
    ('[S1] red\n[S2] cat', ''),
    ('[S1] red\n[S2] cat', '[S1] blue'),
])
def test_attribution_without_evidence_is_unknown(reference, prediction):
    result = score_speakers(reference, prediction, False)['speaker_attribution']
    assert result['accuracy'] is None and result['coverage'] == 0


@pytest.mark.parametrize('prediction', ['red green', 'red green\n[S1] cat dog'])
def test_untagged_words_never_receive_speaker_credit(prediction):
    result = score_speakers('[S1] red green\n[S2] cat dog', prediction, False)['speaker_attribution']
    assert result['scored_units'] - result['correct_units'] == 2


def test_rescore_preserves_predictions_and_micro_averages(tmp_path):
    rows = [{'source': 'sot', 'language': 'en', 'task': 'speaker_attributed_asr',
             'reference': '[S1] red green blue\n[S2] cat dog bird',
             'prediction': '[S1] red green bird\n[S2] cat dog blue'},
            {'source': 'sot', 'language': 'en', 'task': 'speaker_attributed_asr',
             'reference': '[S1] hello', 'prediction': '[S1] hello'},
            {'source': 'sot', 'language': 'en', 'task': 'speaker_attributed_asr',
             'reference': '[S1] missing', 'prediction': ''}]
    predictions, output = tmp_path / 'predictions.jsonl', tmp_path / 'scored.json'
    payload = '\n'.join(json.dumps(row) for row in rows)
    predictions.write_text(payload)
    report = rescore_predictions(predictions, output)
    assert predictions.read_text() == payload
    assert json.loads(output.read_text()) == report
    assert report['metrics'] == summarize(rows)
    attribution = report['metrics']['sot']['speaker_attribution']
    assert attribution['accuracy'] == pytest.approx(5 / 7)
    assert attribution['coverage'] == 7 / 8


def test_rescore_uses_the_same_language_aliases_as_live_evaluation(tmp_path):
    rows = [{'source': 'sot', 'language': 'mandarin', 'task': 'speaker_attributed_asr',
             'reference': '[S1] 你好', 'prediction': '[S1] 你'}]
    predictions = tmp_path / 'predictions.jsonl'
    predictions.write_text(json.dumps(rows[0]))
    assert rescore_predictions(predictions, tmp_path / 'scored.json')['metrics'] == summarize(rows)
