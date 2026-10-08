from open_audio_llm.eval.meeting_scores import score_recording
from open_audio_llm.eval.sot_timestamps import summarize_timed_sot
from open_audio_llm.eval.target_sot import summarize_target_sot
from open_audio_llm.data.sot import TIMESTAMP_FORMAT


def test_plain_swap_is_free_and_enrolled_swap_is_not():
    reference = "[S1][0.00-2.00] hello there\n[S2][2.00-4.00] good day"
    swapped = "[S2][0.00-2.00] hello there\n[S1][2.00-4.00] good day"
    plain = score_recording(reference, swapped, 4.0, language="en", fixed_targets=False)
    assert plain["tcp"]["5"]["error_rate"] == 0
    assert plain["der"]["error_rate"] == 0
    enrolled_ref = reference.replace("S", "T")
    enrolled_swap = swapped.replace("S", "T")
    enrolled = score_recording(
        enrolled_ref, enrolled_swap, 4.0, language="en", fixed_targets=True, target_count=2,
    )
    assert enrolled["tcp"]["5"]["error_rate"] > 0
    assert enrolled["der"]["error_rate"] > 0


def test_time_shift_changes_tcp_and_der_but_not_cp():
    reference = "[S1][0.00-2.00] hello there"
    shifted = "[S1][10.00-12.00] hello there"
    row = dict(source="sot", language="en", task="speaker_attributed_asr", duration=12.0,
               sot_output_format=TIMESTAMP_FORMAT, reference=reference, prediction=shifted)
    result = summarize_timed_sot([row], False)
    assert result["cp_error_rate"] == 0
    assert result["cp_metric"] == "cpWER"
    assert result["tcp_error_rate_5s"] > 0
    assert result["der"] > 0
    close = score_recording(reference, "[S1][0.10-2.10] hello there", 2.1, language="en", fixed_targets=False)
    assert close["der"]["error_rate"] == 0
    assert close["tcp"]["5"]["error_rate"] == 0


def test_extra_target_label_is_not_matched_to_anonymous_speech():
    reference = "[T1][0.00-2.00] apple\n[S1][2.00-4.00] pear"
    prediction = "[T2][0.00-2.00] apple\n[S1][2.00-4.00] pear"
    scored = score_recording(reference, prediction, 4.0, language="en", fixed_targets=True, target_count=2)
    assert scored["tcp"]["5"]["errors"] > 0
    assert scored["der"]["false_alarm"] > 0
    assert scored["der"]["error_rate"] > 0


def test_recipe_summary_keeps_plain_and_enrolled_apart():
    from open_audio_llm.eval.recipe_test import score_rows

    plain = dict(status="ok", split="long_en_test_plain", enrolled=False, language="en",
                 duration=2.0, reference="[S1][0.00-2.00] hello there",
                 prediction="[S2][0.00-2.00] hello there", finish_reason="stop")
    enrolled = dict(status="ok", split="dialog_en_test_2to5_enroll", enrolled=True, language="en",
                    duration=2.0, count=1, mode="all", reference="[T1][0.00-2.00] hello there",
                    prediction="[T1][0.00-2.00] hello there", finish_reason="stop")
    metrics, failures = score_rows([plain, enrolled])
    assert not failures
    assert metrics["long_en_test_plain"]["cp_metric"] == "cpWER"
    assert metrics["long_en_test_plain"]["der"] == 0
    assert metrics["dialog_en_test_2to5_enroll"]["cp_metric"] == "cpWER"
    assert metrics["dialog_en_test_2to5_enroll"]["tcp_error_rate_5s"] == 0


def test_chinese_uses_cer_names_and_perfect_copy_is_zero():
    reference = "[T1][0.00-2.00] 你好\n[S1][2.00-4.00] 世界"
    rows = [dict(reference=reference, prediction=reference, duration=4.0, count=1, mode="all", language="zh")]
    result = summarize_target_sot(rows)
    assert result["cp_metric"] == "cpCER"
    assert result["tcp_metric"] == "tcpCER"
    assert result["cp_error_rate"] == 0
    assert result["tcp_error_rate_5s"] == 0
    assert result["tcp_error_rate_1s"] == 0
    assert result["der"] == 0
    assert result["error_rate"] == 0
