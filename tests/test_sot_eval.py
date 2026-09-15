import pytest

from open_audio_llm.eval.sot import score_speakers
from open_audio_llm.eval.ts_asr import summarize


@pytest.mark.parametrize('chinese,reference,prediction', [
    (True, '[S1] 你好\n[S2] 世界\n[S3] 再见', '[S1] 再见\n[S2] 你好\n[S3] 世界'),
    (False, '[S1] hello there\n[S2] good morning', '[S1] Good morning!\n[S2] Hello there.'),
])
def test_permutation_and_punctuation_do_not_change_transcription_score(chinese, reference, prediction):
    result = score_speakers(reference, prediction, chinese)
    assert result['errors'] == 0 and result['format_valid']


def test_missing_extra_and_untagged_speech_are_counted():
    reference = '[S1] 一二\n[S2] 三四\n[S3] 五六'
    assert score_speakers(reference, '[S1] 一二', True)['errors'] == 4
    assert score_speakers(reference, reference + '\n[S4] 七八', True)['errors'] == 2
    assert score_speakers(reference, '你好' + reference, True)['errors'] == 2
    assert score_speakers(reference, '一二', True)['errors'] == 4
    empty = score_speakers(reference, '[S1] \n[S2] ', True)
    assert empty['errors'] == 6 and empty['empty_output'] and not empty['format_valid']


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
