"""Self-contained test dataset registry for decode.py and vLLM eval scripts.

Holds (1) the declarative table of named test datasets, (2) the default
task each dataset is associated with, and (3) the default-language
mapping used by WER normalisers. All test cuts are resolved purely via
``lhotse.load_manifest_lazy + CutSet.from_manifests`` so this module can
be imported in lightweight envs (e.g. vLLM-only env without k2).

Both ``src/decode.py`` and ``src/open_audio_llm/integrations/vllm/test_vllm_inference.py``
import from here so the two evaluation paths share the exact same set of
"known" test sets — no per-script drift.
"""

from __future__ import annotations

import argparse
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

from lhotse import CutSet, load_manifest_lazy


# ---------------------------------------------------------------------------
# Multilingual dataset registry (separate from the named ASR/SER table).
# Loaded lazily on first multilingual use.
# ---------------------------------------------------------------------------
MULTILINGUAL_REGISTRY_PATH = Path(
    "/ai_sds_wuzz/MULTILINGUAL_DATA/dataset_registry.json"
)


@lru_cache()
def _load_multilingual_registry() -> dict:
    """Load and cache the multilingual dataset registry."""
    with open(MULTILINGUAL_REGISTRY_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Declarative test set definitions
#
# Each entry's "kind" decides how cuts are materialised:
#
#   - kind="single"
#       Standard recordings + supervisions pair under a configurable base.
#       Optional ``supervisions_punc`` overrides the default punc-derived path
#       for datasets whose punctuated variant has a different stem (e.g. the
#       CommonVoice EN cleaned/orig_punc split). Optional ``post_filter``
#       names a callable in POST_FILTERS that runs after CutSet construction.
#
#   - kind="multi"
#       Multiple sub-splits returned as ``{sub_name: CutSet, ...}``. Used by
#       LibriSpeech (clean/other) and WenetSpeech (test_net/test_meeting).
#
#   - kind="ser_pattern"
#       SER convention: ``<base>/<subdir>/data/manifests/<prefix>_recordings_test``
#       and ``<prefix>_supervisions_test``. Mirrors
#       ``ManifestBackedSerDataset`` in src/ser_datamodule.py.
#
#   - kind="cuts_file"
#       Single pre-built ``cuts.jsonl[.gz]`` (used by ts_hw_test).
#
# The ``base`` field selects which CLI arg holds the root directory:
#   "manifest_dir"          → args.manifest_dir          (default ASR root)
#   "ser_manifest_dir"      → args.ser_manifest_dir      (SER/SEC root)
#   "tsasr_test_manifest_dir" → args.tsasr_test_manifest_dir (TS-ASR cuts)
#
# ``use_punc`` defaults to False; when True the supervisions path picks up
# the ``_punc`` variant whenever ``args.use_punc`` is also True. Datasets
# that never have a ``_punc`` companion just leave it False.
# ---------------------------------------------------------------------------
TEST_SET_DEFS: Dict[str, Dict[str, Any]] = {
    # ---- 单 split ASR ----
    "aishell": {
        "kind": "single", "base": "manifest_dir", "use_punc": True,
        "recordings":   "data_aishell/data/manifests/aishell_recordings_test.jsonl.gz",
        "supervisions": "data_aishell/data/manifests/aishell_supervisions_test.jsonl.gz",
    },
    "aishell2": {
        "kind": "single", "base": "manifest_dir", "use_punc": True,
        "recordings":   "data_aishell2/data/manifests/aishell2_recordings_test.jsonl.gz",
        "supervisions": "data_aishell2/data/manifests/aishell2_supervisions_test.jsonl.gz",
    },
    "aishell3": {
        "kind": "single", "base": "manifest_dir", "use_punc": True,
        "recordings":   "data_aishell3/data/manifests/aishell3_recordings_test.jsonl.gz",
        "supervisions": "data_aishell3/data/manifests/aishell3_supervisions_test.jsonl.gz",
    },
    "magicdata": {
        "kind": "single", "base": "manifest_dir", "use_punc": True,
        "recordings":   "MAGICDATA/data/manifests/magicdata_recordings_test.jsonl.gz",
        "supervisions": "MAGICDATA/data/manifests/magicdata_supervisions_test.jsonl.gz",
    },
    "kespeech": {
        "kind": "single", "base": "manifest_dir", "use_punc": True,
        "recordings":   "KeSpeech/data/manifests/kespeech-asr_recordings_test.jsonl.gz",
        "supervisions": "KeSpeech/data/manifests/kespeech-asr_supervisions_test.jsonl.gz",
    },
    "thchs30": {
        "kind": "single", "base": "manifest_dir", "use_punc": True,
        "recordings":   "data_thchs30/data/manifests/thchs_30_recordings_test.jsonl.gz",
        "supervisions": "data_thchs30/data/manifests/thchs_30_supervisions_test.jsonl.gz",
    },
    "talcs": {
        # supervisions stem is *_test_cleaned*; _maybe_punc → *_test_cleaned_punc*.
        "kind": "single", "base": "manifest_dir", "use_punc": True,
        "recordings":   "TALCS/data/manifests/talcs_recordings_test.jsonl.gz",
        "supervisions": "TALCS/data/manifests/talcs_supervisions_test_cleaned.jsonl.gz",
    },
    "gigaspeech": {
        "kind": "single", "base": "manifest_dir", "use_punc": True,
        "recordings":   "GigaSpeech/data/manifests/gigaspeech_recordings_TEST.jsonl.gz",
        "supervisions": "GigaSpeech/data/manifests/gigaspeech_supervisions_TEST.jsonl.gz",
    },
    "mls": {
        "kind": "single", "base": "manifest_dir", "use_punc": True,
        "recordings":   "MLS/data/manifests/mls-english_recordings_test.jsonl.gz",
        "supervisions": "MLS/data/manifests/mls-english_supervisions_test.jsonl.gz",
    },

    "commonvoice_en": {
        # CommonVoice EN's _punc variant has a different stem suffix
        # (`_orig_punc` vs `_cleaned`), so we declare both explicitly.
        "kind": "single", "base": "manifest_dir", "use_punc": True,
        "recordings":        "common_voice_en/data/manifests/cv-en_recordings_test.jsonl.gz",
        "supervisions":      "common_voice_en/data/manifests/cv-en_supervisions_test_cleaned.jsonl.gz",
        "supervisions_punc": "common_voice_en/data/manifests/cv-en_supervisions_test_orig_punc.jsonl.gz",
    },
    "commonvoice_zh": {
        "kind": "single", "base": "manifest_dir", "use_punc": True,
        "recordings":   "common_voice_zh/data/manifests/cv-zh-CN_recordings_test.jsonl.gz",
        "supervisions": "common_voice_zh/data/manifests/cv-zh-CN_supervisions_test.jsonl.gz",
    },

    # ---- 单 split + 固定 hotwords supervisions ----
    # CommonVoice EN hotwords manifest 放在独立目录(不在 LHOTSE 标准布局下),
    # 直接给绝对路径 — Path("/abs/...") 在 _build_single_cuts 中会覆盖
    # base_dir, 所以与现有 manifest_dir 解析逻辑零冲突。
    # 该 manifest 把 hotwords/clean 正确放到 custom 内 (与 zh 版本一致),
    # 因此复用同一个 _clean_pass post_filter 过滤 QA 不通过的 cut。
    "commonvoice_en_hotwords": {
        "kind": "single", "base": "manifest_dir", "use_punc": False,
        "recordings":   "/ai_sds_wuzz/DATA_ASR/common_voice_en/lhotse/hotwords/cv-en_recordings_test.jsonl.gz",
        "supervisions": "/ai_sds_wuzz/DATA_ASR/common_voice_en/lhotse/hotwords/cv-en_supervisions_test_orig_punc_hotwords.jsonl.gz",
        "post_filter": "_clean_pass",
    },
    "commonvoice_zh_hotwords": {
        "kind": "single", "base": "manifest_dir", "use_punc": False,
        "recordings":   "/ai_sds_wuzz/DATA_ASR/common_voice_zh/lhotse/hotwords/cv-zh-CN_recordings_test.jsonl.gz",
        "supervisions": "/ai_sds_wuzz/DATA_ASR/common_voice_zh/lhotse/hotwords/cv-zh-CN_supervisions_test_punc_hotwords.jsonl.gz",
        # Drop cuts whose QA pass is marked failed (mirrors the original
        # CommonVoiceZhHotwordsDataset filter).
        "post_filter": "_clean_pass",
    },
    # AISHELL test hotwords manifest 在各自 data_aishell*/lhotse/hotwords/
    # 目录, 与 LHOTSE 标准布局下的 plain ASR test manifest 分离。
    "aishell_hotwords": {
        "kind": "single", "base": "manifest_dir", "use_punc": False,
        "recordings":   "/ai_sds_wuzz/DATA_ASR/data_aishell/lhotse/hotwords/aishell_recordings_test.jsonl.gz",
        "supervisions": "/ai_sds_wuzz/DATA_ASR/data_aishell/lhotse/hotwords/aishell_supervisions_test_punc.jsonl.gz",
        "post_filter": "_clean_pass",
    },
    "aishell2_hotwords": {
        "kind": "single", "base": "manifest_dir", "use_punc": False,
        "recordings":   "/ai_sds_wuzz/DATA_ASR/data_aishell2/lhotse/hotwords/aishell2_recordings_test.jsonl.gz",
        "supervisions": "/ai_sds_wuzz/DATA_ASR/data_aishell2/lhotse/hotwords/aishell2_supervisions_test_hotwords.jsonl.gz",
    },
    "aishell3_hotwords": {
        "kind": "single", "base": "manifest_dir", "use_punc": False,
        "recordings":   "/ai_sds_wuzz/DATA_ASR/data_aishell3/lhotse/hotwords/aishell3_recordings_test.jsonl.gz",
        "supervisions": "/ai_sds_wuzz/DATA_ASR/data_aishell3/lhotse/hotwords/aishell3_supervisions_test_punc_hotwords.jsonl.gz",
    },

    # ---- 单 split ASR + 固定 hotwords supervisions (LibriSpeech) ----
    # hot words 来自 IS21 Deep Bias 标注 (test) / 词频法自动提取 (train)。
    # 由 local/prepare_librispeech_hotwords_supervisions.py 生成。
    "librispeech_test_clean_hotwords": {
        "kind": "single", "base": "manifest_dir", "use_punc": False,
        "recordings":   "LibriSpeech/data/manifests/librispeech_recordings_test-clean.jsonl.gz",
        "supervisions": "LibriSpeech/data/manifests/librispeech_supervisions_test-clean_hotwords.jsonl.gz",
    },
    "librispeech_test_other_hotwords": {
        "kind": "single", "base": "manifest_dir", "use_punc": False,
        "recordings":   "LibriSpeech/data/manifests/librispeech_recordings_test-other.jsonl.gz",
        "supervisions": "LibriSpeech/data/manifests/librispeech_supervisions_test-other_hotwords.jsonl.gz",
    },

    # ---- 多 split ASR ----
    "librispeech": {
        "kind": "multi", "base": "manifest_dir", "use_punc": True,
        "subsets": {
            "librispeech_test_clean": {
                "recordings":   "LibriSpeech/data/manifests/librispeech_recordings_test-clean.jsonl.gz",
                "supervisions": "LibriSpeech/data/manifests/librispeech_supervisions_test-clean.jsonl.gz",
            },
            "librispeech_test_other": {
                "recordings":   "LibriSpeech/data/manifests/librispeech_recordings_test-other.jsonl.gz",
                "supervisions": "LibriSpeech/data/manifests/librispeech_supervisions_test-other.jsonl.gz",
            },
        },
    },
    "wenetspeech": {
        "kind": "multi", "base": "manifest_dir", "use_punc": True,
        "subsets": {
            "wenetspeech_test_net": {
                "recordings":   "WenetSpeech/data/manifests/wenetspeech_recordings_TEST_NET.jsonl.gz",
                "supervisions": "WenetSpeech/data/manifests/wenetspeech_supervisions_TEST_NET.jsonl.gz",
            },
            "wenetspeech_test_meeting": {
                "recordings":   "WenetSpeech/data/manifests/wenetspeech_recordings_TEST_MEETING.jsonl.gz",
                "supervisions": "WenetSpeech/data/manifests/wenetspeech_supervisions_TEST_MEETING.jsonl.gz",
            },
        },
    },

    # ---- 多 split ASR 的子集别名 (single-kind, 与 multi 形式并存) ----
    # 评测端常希望 LibriSpeech / WenetSpeech 的 clean/other 与 net/meeting
    # 分别单独成 spec, 这样汇总表里能给出每个子集独立的 WER (而不是合并跑)。
    # 路径与上面 multi.subsets 的对应条目完全一致, 只是入口形态从 multi 转
    # single, 让 plan YAML 可直接写 ``- dataset: librispeech_test_clean``。
    "librispeech_test_clean": {
        "kind": "single", "base": "manifest_dir", "use_punc": True,
        "recordings":   "LibriSpeech/data/manifests/librispeech_recordings_test-clean.jsonl.gz",
        "supervisions": "LibriSpeech/data/manifests/librispeech_supervisions_test-clean.jsonl.gz",
    },
    "librispeech_test_other": {
        "kind": "single", "base": "manifest_dir", "use_punc": True,
        "recordings":   "LibriSpeech/data/manifests/librispeech_recordings_test-other.jsonl.gz",
        "supervisions": "LibriSpeech/data/manifests/librispeech_supervisions_test-other.jsonl.gz",
    },
    "wenetspeech_test_net": {
        "kind": "single", "base": "manifest_dir", "use_punc": True,
        "recordings":   "WenetSpeech/data/manifests/wenetspeech_recordings_TEST_NET.jsonl.gz",
        "supervisions": "WenetSpeech/data/manifests/wenetspeech_supervisions_TEST_NET.jsonl.gz",
    },
    "wenetspeech_test_meeting": {
        "kind": "single", "base": "manifest_dir", "use_punc": True,
        "recordings":   "WenetSpeech/data/manifests/wenetspeech_recordings_TEST_MEETING.jsonl.gz",
        "supervisions": "WenetSpeech/data/manifests/wenetspeech_supervisions_TEST_MEETING.jsonl.gz",
    },

    # ---- SER / SEC: standard ManifestBackedSerDataset 公式 ----
    "biic_podcast_ser": {
        "kind": "ser_pattern", "base": "ser_manifest_dir",
        "subdir": "biic_podcast", "prefix": "biic_podcast_ser",
    },
    "m3ed_ser": {
        "kind": "ser_pattern", "base": "ser_manifest_dir",
        "subdir": "m3ed", "prefix": "m3ed_ser",
    },
    "meld_ser": {
        "kind": "ser_pattern", "base": "ser_manifest_dir",
        "subdir": "meld", "prefix": "meld_ser",
    },
    "iemocap_ser": {
        "kind": "ser_pattern", "base": "ser_manifest_dir",
        "subdir": "iemocap", "prefix": "iemocap_ser",
    },
    "msp_podcast_ser": {
        # MSP-Podcast splits its test set into two halves (test1 / test2);
        # we return both as sub-cuts so per-half WER stays visible.
        "kind": "ser_pattern", "base": "ser_manifest_dir",
        "subdir": "msp_podcast", "prefix": "msp_podcast_ser",
        "splits": ["test1", "test2"],
    },
    "emotion1200_en_ser": {
        "kind": "ser_pattern", "base": "ser_manifest_dir",
        "subdir": "emotion1200_en", "prefix": "emotion1200_en_ser",
    },
    "emotion1200_zh_ser": {
        "kind": "ser_pattern", "base": "ser_manifest_dir",
        "subdir": "emotion1200_zh", "prefix": "emotion1200_zh_ser",
    },
    "emotion1200_en_sec": {
        "kind": "ser_pattern", "base": "ser_manifest_dir",
        "subdir": "emotion1200_en_sec", "prefix": "emotion1200_en_sec",
    },
    "emotion1200_zh_sec": {
        "kind": "ser_pattern", "base": "ser_manifest_dir",
        "subdir": "emotion1200_zh_sec", "prefix": "emotion1200_zh_sec",
    },
    "emotion1200_en_sepc": {
        "kind": "ser_pattern", "base": "ser_manifest_dir",
        "subdir": "emotion1200_en_sepc", "prefix": "emotion1200_en_sepc",
    },
    "emotion1200_zh_sepc": {
        "kind": "ser_pattern", "base": "ser_manifest_dir",
        "subdir": "emotion1200_zh_sepc", "prefix": "emotion1200_zh_sepc",
    },

    # ---- TS-ASR: 单个 cuts.jsonl.gz 文件 ----
    "ts_hw_test": {
        "kind": "cuts_file", "base": "tsasr_test_manifest_dir",
        "path": "ts_hw_test_cuts_all.jsonl.gz",
    },
    # ---- TS-ASR: Librimix 公开测试集 (target speaker, 全英文) ----
    # cuts 文件与 ts_hw_test_cuts_all.jsonl.gz 同目录, 共用
    # --tsasr-test-manifest-dir 即可。音频源在 cuts 内部记录绝对路径。
    "libri2mix": {
        "kind": "cuts_file", "base": "tsasr_test_manifest_dir",
        "path": "libri2mix_test_16k_min_mix_both_cuts.jsonl.gz",
    },
    "libri3mix": {
        "kind": "cuts_file", "base": "tsasr_test_manifest_dir",
        "path": "libri3mix_test_16k_min_mix_both_cuts.jsonl.gz",
    },

    # ---- TS-ASR mix_tsstyle: 自建混音对照集 ----
    # 由 tmp/mix_tsstyle/{mix_librimix_tsstyle,mix_tshw_libristyle}.py 生成,
    # 再用 tmp/mix_tsstyle/build_cuts.py 转成 lhotse cuts.jsonl.gz。
    # 路径写绝对路径, 不依赖 --tsasr-test-manifest-dir
    # (Path("/abs") / "/another/abs" == "/another/abs", base_dir 被忽略)。
    "libri2mix_tsstyle": {
        "kind": "cuts_file", "base": "tsasr_test_manifest_dir",
        "path": "/chenmingjie/mingdong/workspace/AmphionASR/tmp/mix_tsstyle/cuts/libri2mix_tsstyle_cuts.jsonl.gz",
    },
    "libri3mix_tsstyle": {
        "kind": "cuts_file", "base": "tsasr_test_manifest_dir",
        "path": "/chenmingjie/mingdong/workspace/AmphionASR/tmp/mix_tsstyle/cuts/libri3mix_tsstyle_cuts.jsonl.gz",
    },
    "ts_hw_test_libristyle": {
        "kind": "cuts_file", "base": "tsasr_test_manifest_dir",
        "path": "/chenmingjie/mingdong/workspace/AmphionASR/tmp/mix_tsstyle/cuts/ts_hw_test_libristyle_cuts.jsonl.gz",
    },

    # ---- LibriMix SNR-offset ablation (driver: mix_librimix_libristyle.py) ----
    # 用 LibriMix mix_both 算法 (mix_core.remix_record_libri) per-cut 重新合成
    # libri{2,3}mix 的 6000/9000 测试样本; 通过 --interferer-snr-offset-db N
    # 让 interferer 目标 LUFS 整体下移 N dB, 等效 ΔSNR_mean 从 0 -> +N.
    # 因为 paths.jsonl 每行已经按 _s1/_s2/_s3 标好 target/interferer 角色,
    # per-cut 渲染天然解决 target 轮换问题 (老的 csv-缩放路线有此 bug).
    # build:  python tmp/mix_tsstyle/mix_librimix_libristyle.py
    #               --paths .../libri{2,3}mix_paths.jsonl
    #               --interferer-snr-offset-db {0,5,10}
    # cuts:   python tmp/mix_tsstyle/build_cuts.py --in <paths> --out <cuts>
    # libri{2,3}mix_lib: offset=0 baseline, 新 pipeline 跑出来的等效 LibriMix.
    "libri2mix_lib": {
        "kind": "cuts_file", "base": "tsasr_test_manifest_dir",
        "path": "/chenmingjie/mingdong/workspace/AmphionASR/tmp/mix_tsstyle/output_librimix_libristyle/libri2mix/libri2mix_lib_cuts.jsonl.gz",
    },
    "libri3mix_lib": {
        "kind": "cuts_file", "base": "tsasr_test_manifest_dir",
        "path": "/chenmingjie/mingdong/workspace/AmphionASR/tmp/mix_tsstyle/output_librimix_libristyle/libri3mix/libri3mix_lib_cuts.jsonl.gz",
    },
    "libri2mix_snrp5": {
        "kind": "cuts_file", "base": "tsasr_test_manifest_dir",
        "path": "/chenmingjie/mingdong/workspace/AmphionASR/tmp/mix_tsstyle/output_librimix_libristyle/libri2mix_snrp5/libri2mix_lib_snrp5_cuts.jsonl.gz",
    },
    "libri3mix_snrp5": {
        "kind": "cuts_file", "base": "tsasr_test_manifest_dir",
        "path": "/chenmingjie/mingdong/workspace/AmphionASR/tmp/mix_tsstyle/output_librimix_libristyle/libri3mix_snrp5/libri3mix_lib_snrp5_cuts.jsonl.gz",
    },
    "libri2mix_snrp10": {
        "kind": "cuts_file", "base": "tsasr_test_manifest_dir",
        "path": "/chenmingjie/mingdong/workspace/AmphionASR/tmp/mix_tsstyle/output_librimix_libristyle/libri2mix_snrp10/libri2mix_lib_snrp10_cuts.jsonl.gz",
    },
    "libri3mix_snrp10": {
        "kind": "cuts_file", "base": "tsasr_test_manifest_dir",
        "path": "/chenmingjie/mingdong/workspace/AmphionASR/tmp/mix_tsstyle/output_librimix_libristyle/libri3mix_snrp10/libri3mix_lib_snrp10_cuts.jsonl.gz",
    },

    # ---- ESC: AudioSet ESC dev 离线混入 Emilia 前景的固定测试集 ----
    # 由 local/prepare_audioset_esc_test.py 一次性生成 (MixedCut 描述如何
    # 叠加, 不重写 wav). 名称以 _esc 结尾, decode.py 据此路由到
    # EscDataModule (跳过 AsrDataModule 上的 noise/SpecAug/CutMix 通路).
    "audioset_esc_test": {
        "kind": "cuts_file", "base": "esc_manifest_dir",
        "path": "audioset_esc_test_mixed_cuts.jsonl.gz",
    },

    # NOTE: aidatatang / primewords are intentionally absent: their dataset
    # classes in src/asr_datamodule.py expose only train_cuts() (no test
    # split materialised), so registering them as test sets would crash at
    # call time. Re-add when their test manifests exist.
    #
    # CommonVoice 退化鲁棒性测试集在 TEST_SET_DEFS 字面量声明后通过派生块
    # 批量注册 (11 种退化 x 2 语言 = 22 项), 见下方 _CV_DEGRADATIONS 循环。
}


# ---------------------------------------------------------------------------
# CommonVoice degradation robustness test sets (11 degradations x 2 langs
# = 22 specs). Manifests + audio live under
# /ai_sds_wuzz/DATA_ASR/degradation/cv_{en,zh}_test/<degradation>/, with
# fixed filenames cv_{en,zh}_test_<degradation>_{recordings,supervisions}_degraded.jsonl.gz.
#
# Paths are written absolute (mirrors the commonvoice_en_hotwords trick:
# Path("/abs/...") passed to _build_single_cuts overrides base_dir), so
# the "manifest_dir" base sentinel is harmless. use_punc=False because
# degraded sets only ship a cleaned-text supervisions file (no _punc
# variant). Default task is plain "asr" — the degradation lives in the
# audio only, the transcript is identical to the upstream CV cleaned set.
# ---------------------------------------------------------------------------
_CV_DEGRADATIONS: List[str] = [
    # 混响 (RIR)
    "rir_slr26",
    "rir_slr28_sim",
    "rir_slr28_real",
    "rir_rirmega",
    # 加性噪声
    "noise_musan_noise",
    "noise_musan_music",
    "noise_musan_speech",
    "noise_dns",
    "noise_audioset_traffic",
    "noise_audioset_esc",
    "noise_wham",
]

_CV_DEGRADED_ROOT = Path("/ai_sds_wuzz/DATA_ASR/degradation")

for _lang in ("en", "zh"):
    for _deg in _CV_DEGRADATIONS:
        _name = f"cv_{_lang}_{_deg}"
        _stem = f"cv_{_lang}_test_{_deg}"
        _mdir = _CV_DEGRADED_ROOT / f"cv_{_lang}_test" / _deg / "manifests"
        TEST_SET_DEFS[_name] = {
            "kind": "single", "base": "manifest_dir", "use_punc": False,
            "recordings": str(_mdir / f"{_stem}_recordings_degraded.jsonl.gz"),
            "supervisions": str(_mdir / f"{_stem}_supervisions_degraded.jsonl.gz"),
        }
del _lang, _deg, _name, _stem, _mdir

# CV 退化 + 热词复合测试集 (11 退化 x 2 语言 = 22 项)
# 音频取退化版 recordings，supervisions 使用专门生成的截齐版 hotwords
# manifest（local/prepare_degraded_hotwords_supervisions.py 生成）。
# 截齐的原因：退化录音比原始录音略短（RIR 卷积/加噪后精度损失），若直接
# 复用原始 hotwords supervision 文件，supervision 的结束时间会超出录音边界，
# 导致 lhotse trim_to_supervisions 丢弃 supervision 并触发断言失败。
for _lang in ("en", "zh"):
    for _deg in _CV_DEGRADATIONS:
        _name = f"cv_{_lang}_{_deg}_hotwords"
        _stem = f"cv_{_lang}_test_{_deg}"
        _mdir = _CV_DEGRADED_ROOT / f"cv_{_lang}_test" / _deg / "manifests"
        TEST_SET_DEFS[_name] = {
            "kind": "single", "base": "manifest_dir", "use_punc": False,
            "recordings": str(_mdir / f"{_stem}_recordings_degraded.jsonl.gz"),
            "supervisions": str(_mdir / f"{_stem}_supervisions_hotwords.jsonl.gz"),
            "post_filter": "_clean_pass",
        }
del _lang, _deg, _name, _stem, _mdir


# ---------------------------------------------------------------------------
# Default language for built-in datasets. ``"multi"`` means the dataset
# carries per-cut language and downstream code should aggregate per-cut.
# ---------------------------------------------------------------------------
DATASET_LANGUAGE: Dict[str, str] = {
    "librispeech":    "en",
    "librispeech_test_clean": "en",
    "librispeech_test_other": "en",
    "gigaspeech":     "en",
    "mls":            "en",
    "commonvoice_en": "en",
    "commonvoice_en_hotwords": "en",
    "librispeech_test_clean_hotwords": "en",
    "librispeech_test_other_hotwords": "en",
    "aishell":        "zh",
    "aishell2":       "zh",
    "aishell3":       "zh",
    "commonvoice_zh": "zh",
    "commonvoice_zh_hotwords": "zh",
    "aishell_hotwords":  "zh",
    "aishell2_hotwords": "zh",
    "aishell3_hotwords": "zh",
    "kespeech":       "zh",
    "talcs":          "zh",
    "thchs30":        "zh",
    "magicdata":      "zh",
    "wenetspeech":    "zh",
    "wenetspeech_test_net":     "zh",
    "wenetspeech_test_meeting": "zh",
    # SER/SEC datasets
    "biic_podcast_ser":   "zh",
    "m3ed_ser":           "zh",
    "meld_ser":           "en",
    "iemocap_ser":        "en",
    "msp_podcast_ser":    "en",
    "emotion1200_en_ser": "en",
    "emotion1200_zh_ser": "zh",
    "emotion1200_en_sec": "en",
    "emotion1200_zh_sec": "zh",
    "emotion1200_en_sepc": "en",
    "emotion1200_zh_sepc": "zh",
    # TS-ASR test sets carry mixed-language cuts (zh + en).
    "ts_hw_test":     "multi",
    # Librimix splits are LibriSpeech-derived → English only.
    "libri2mix":      "en",
    "libri3mix":      "en",
    # mix_tsstyle 对照集:
    #   libri{2,3}mix_tsstyle 沿用 LibriSpeech 文本 → 仍是 en;
    #   ts_hw_test_libristyle 跨 zh+en 多语 → multi。
    "libri2mix_tsstyle":     "en",
    "libri3mix_tsstyle":     "en",
    "ts_hw_test_libristyle": "multi",
    # LibriMix SNR-offset ablation (LibriSpeech-derived -> en)
    "libri2mix_lib":         "en",
    "libri3mix_lib":         "en",
    "libri2mix_snrp5":       "en",
    "libri3mix_snrp5":       "en",
    "libri2mix_snrp10":      "en",
    "libri3mix_snrp10":      "en",
    # ESC: cut.text defaults to caption_en (caption_zh stays in custom for
    # later analysis); decode-side metric path is generation-only so the
    # language tag here only affects logging.
    "audioset_esc_test": "en",
}

# Inject CV degraded test sets into DATASET_LANGUAGE / DATASET_TASK alongside
# their TEST_SET_DEFS entries — keep the three tables symmetric so a typo on
# any side surfaces as a load-time error rather than mid-run.
for _lang in ("en", "zh"):
    for _deg in _CV_DEGRADATIONS:
        DATASET_LANGUAGE[f"cv_{_lang}_{_deg}"] = _lang
        DATASET_LANGUAGE[f"cv_{_lang}_{_deg}_hotwords"] = _lang
del _lang, _deg


# ---------------------------------------------------------------------------
# Voices-in-the-Wild-Bench (VITW) — 16 splits, 中英混合，共 5000 条
# 音频由 local/prepare_vitw_bench.py 从 parquet 提取，manifests 存于：
#   /ai_sds_wuzz/DATA_ASR/Voices-in-the-Wild-Bench/manifests/
# splits: 8 real + 8 syn，类型：noise/far_field/echo/distortion/
#                                   mixed/obstructed/recording/dropout
# ---------------------------------------------------------------------------
_VITW_ROOT = Path("/ai_sds_wuzz/DATA_ASR/Voices-in-the-Wild-Bench")
_VITW_SPLITS: List[str] = [
    "real_distortion", "real_dropout", "real_echo",    "real_far_field",
    "real_mixed",      "real_noise",   "real_obstructed", "real_recording",
    "syn_distortion",  "syn_dropout",  "syn_echo",     "syn_far_field",
    "syn_mixed",       "syn_noise",    "syn_obstructed",  "syn_recording",
]
for _split in _VITW_SPLITS:
    TEST_SET_DEFS[f"vitw_{_split}"] = {
        "kind": "single", "base": "manifest_dir", "use_punc": False,
        "recordings":   str(_VITW_ROOT / "manifests" / f"vitw_{_split}_recordings.jsonl.gz"),
        "supervisions": str(_VITW_ROOT / "manifests" / f"vitw_{_split}_supervisions.jsonl.gz"),
    }
    DATASET_LANGUAGE[f"vitw_{_split}"] = "multi"
del _split


# ---------------------------------------------------------------------------
# Default task per dataset. Test plans can omit ``task`` for these datasets
# (the loader fills in from this table). Datasets not in this map require an
# explicit ``task`` in the plan.
# ---------------------------------------------------------------------------
DATASET_TASK: Dict[str, str] = {
    # Plain ASR
    "librispeech":    "asr",
    "librispeech_test_clean": "asr",
    "librispeech_test_other": "asr",
    "gigaspeech":     "asr",
    "mls":            "asr",
    "commonvoice_en": "asr",
    "commonvoice_zh": "asr",
    "aishell":        "asr",
    "aishell2":       "asr",
    "aishell3":       "asr",
    "kespeech":       "asr",
    "magicdata":      "asr",
    "talcs":          "asr",
    "thchs30":        "asr",
    "wenetspeech":    "asr",
    "wenetspeech_test_net":     "asr",
    "wenetspeech_test_meeting": "asr",
    # ASR + hotwords
    "commonvoice_en_hotwords": "asr_hotwords",
    "commonvoice_zh_hotwords": "asr_hotwords",
    "librispeech_test_clean_hotwords": "asr_hotwords",
    "librispeech_test_other_hotwords": "asr_hotwords",
    "aishell_hotwords":  "asr_hotwords",
    "aishell2_hotwords": "asr_hotwords",
    "aishell3_hotwords": "asr_hotwords",
    # SER / SEC
    "biic_podcast_ser":   "ser",
    "m3ed_ser":           "ser",
    "meld_ser":           "ser",
    "iemocap_ser":        "ser",
    "msp_podcast_ser":    "ser",
    "emotion1200_en_ser": "ser",
    "emotion1200_zh_ser": "ser",
    "emotion1200_en_sec": "sec",
    "emotion1200_zh_sec": "sec",
    "emotion1200_en_sepc": "sepc",
    "emotion1200_zh_sepc": "sepc",
    # TS-ASR
    "ts_hw_test":     "ts_asr",
    "libri2mix":      "ts_asr",
    "libri3mix":      "ts_asr",
    # mix_tsstyle 对照集 (同样走 ts_asr 解码路径)
    "libri2mix_tsstyle":     "ts_asr",
    "libri3mix_tsstyle":     "ts_asr",
    "ts_hw_test_libristyle": "ts_asr",
    # LibriMix SNR-offset ablation
    "libri2mix_lib":         "ts_asr",
    "libri3mix_lib":         "ts_asr",
    "libri2mix_snrp5":       "ts_asr",
    "libri3mix_snrp5":       "ts_asr",
    "libri2mix_snrp10":      "ts_asr",
    "libri3mix_snrp10":      "ts_asr",
    # ESC
    "audioset_esc_test": "esc",
}

for _lang in ("en", "zh"):
    for _deg in _CV_DEGRADATIONS:
        DATASET_TASK[f"cv_{_lang}_{_deg}"] = "asr"
        DATASET_TASK[f"cv_{_lang}_{_deg}_hotwords"] = "asr_hotwords"
del _lang, _deg

for _split in _VITW_SPLITS:
    DATASET_TASK[f"vitw_{_split}"] = "asr"
del _split


# ---------------------------------------------------------------------------
# Per-dataset post-filter callables. Keep these tiny so the table stays
# declarative; reach for a closure only when a single dataset truly needs
# a custom CutSet.filter.
# ---------------------------------------------------------------------------

def _drop_failed_clean_pass(cuts: CutSet) -> CutSet:
    """Drop cuts whose QA ``custom.clean.pass`` flag is False.

    Originally introduced for ``commonvoice_zh_hotwords`` (mirroring the
    legacy ``CommonVoiceZhHotwordsDataset.test_cuts`` filter) and now also
    used by ``commonvoice_en_hotwords`` whose new manifest carries the same
    ``custom.clean = {pass: bool, engine: [...], ...}`` QA block. Keeps the
    two hotwords test sets symmetric — no double-counting of utterances the
    QA pass already rejected.
    """
    def _ok(cut) -> bool:
        for s in cut.supervisions:
            clean = (s.custom or {}).get("clean")
            if clean is not None and not clean.get("pass", True):
                return False
        return True
    return cuts.filter(_ok)


POST_FILTERS: Dict[str, Callable[[CutSet], CutSet]] = {
    "_clean_pass": _drop_failed_clean_pass,
}


# ---------------------------------------------------------------------------
# Resolver helpers
# ---------------------------------------------------------------------------

def _maybe_punc(path: Path, use_punc: bool) -> Path:
    """Insert ``_punc`` before ``.jsonl.gz`` when *use_punc* is True.

    Mirrors ``src/asr_datamodule.py::_maybe_punc`` so the punc-suffix
    convention stays in lockstep across modules.
    """
    if not use_punc:
        return path
    return path.parent / path.name.replace(".jsonl.gz", "_punc.jsonl.gz")


def _resolve_supervisions_path(
    base_dir: Path, spec: Dict[str, Any], use_punc: bool,
) -> Path:
    """Pick the supervisions path, honouring an explicit punc-only stem.

    Some datasets (notably CommonVoice EN) use a different stem for the
    punctuated variant (``_cleaned`` vs ``_orig_punc``); when their entry
    declares ``supervisions_punc``, we use it directly. Otherwise we fall
    back to the standard ``_punc`` insertion.
    """
    if use_punc and spec.get("supervisions_punc"):
        return base_dir / spec["supervisions_punc"]
    return _maybe_punc(base_dir / spec["supervisions"], use_punc)


def _build_single_cuts(base_dir: Path, spec: Dict[str, Any], use_punc: bool) -> CutSet:
    """Materialise one (recordings, supervisions) pair into a CutSet."""
    rec_path = base_dir / spec["recordings"]
    sup_path = _resolve_supervisions_path(base_dir, spec, use_punc)
    logging.info("Loading test cuts: rec=%s sup=%s", rec_path, sup_path)
    cuts = CutSet.from_manifests(
        recordings=load_manifest_lazy(rec_path),
        supervisions=load_manifest_lazy(sup_path),
    )
    return cuts


def _resolve_multilingual_test_cuts(lang: str, dataset_name: str) -> CutSet:
    """Inline replacement for ``MultilingualDataset.test_cuts()``.

    Reads the JSON registry, formats the test recordings/supervisions
    manifest paths, and returns a CutSet. Does NOT consult use_punc — at
    test time we always use the canonical (un-punctuated) supervisions for
    multilingual datasets to keep WER comparable across languages.
    """
    registry = _load_multilingual_registry()
    if lang not in registry or dataset_name not in registry[lang]:
        available_langs = sorted(registry.keys())
        raise ValueError(
            f"Unknown multilingual dataset '{lang}:{dataset_name}'. "
            f"Available languages: {available_langs}"
        )
    info = registry[lang][dataset_name]
    splits = {s for s in info["splits"].keys() if not s.endswith("_punc")}
    if "test" not in splits:
        raise ValueError(
            f"Multilingual dataset '{lang}:{dataset_name}' has no 'test' "
            f"split. Available splits: {sorted(splits)}"
        )
    mdir = Path(info["manifests_dir"])
    prefix = info["manifest_prefix"]
    rec_path = mdir / f"{prefix}_recordings_test.jsonl.gz"
    sup_path = mdir / f"{prefix}_supervisions_test.jsonl.gz"
    logging.info(
        "Loading multilingual test cuts: lang=%s dataset=%s",
        lang, dataset_name,
    )
    return CutSet.from_manifests(
        recordings=load_manifest_lazy(rec_path),
        supervisions=load_manifest_lazy(sup_path),
    )


# ---------------------------------------------------------------------------
# Public helpers (signatures preserved for decode.py / vllm callers)
# ---------------------------------------------------------------------------

def _parse_multilingual_dataset_name(name: str) -> Union[Tuple[str, str], None]:
    """Parse multilingual dataset name into (lang, dataset_name).

    Accepts both ``"<lang>:<dataset>"`` and ``"multilingual:<lang>:<dataset>"``.
    Returns None when the name does not match either form.
    """
    if name.startswith("multilingual:"):
        parts = name.split(":", 2)
        if len(parts) == 3 and parts[1] and parts[2]:
            return parts[1], parts[2]
        return None
    if name.count(":") == 1:
        lang, dataset_name = name.split(":", 1)
        if lang and dataset_name:
            return lang, dataset_name
    return None


def _resolve_test_cuts_from_dataset_name(
    dataset_name: str,
    args: argparse.Namespace,
) -> Tuple[Union[CutSet, Dict[str, CutSet]], str]:
    """Resolve test cuts for both built-in and multilingual datasets.

    Returns ``(cuts_or_dict, resolved_dataset_name)``. For multilingual
    datasets the returned name is ``"<lang>:<dataset>"`` regardless of input
    form. Manifest roots are looked up via per-task fields on ``args`` so
    SER / TS-ASR cuts can live in separate trees from the standard ASR root.
    """
    if dataset_name in TEST_SET_DEFS:
        spec = TEST_SET_DEFS[dataset_name]
        kind = spec["kind"]
        base_attr = spec.get("base", "manifest_dir")
        if not hasattr(args, base_attr):
            raise AttributeError(
                f"args is missing required attribute '{base_attr}' "
                f"(needed by test set '{dataset_name}')."
            )
        base_dir = Path(getattr(args, base_attr))
        spec_use_punc = bool(spec.get("use_punc", False))
        use_punc = spec_use_punc and bool(getattr(args, "use_punc", False))

        if kind == "single":
            cuts = _build_single_cuts(base_dir, spec, use_punc)
            if "post_filter" in spec:
                cuts = POST_FILTERS[spec["post_filter"]](cuts)
            return cuts, dataset_name

        if kind == "multi":
            subsets: Dict[str, CutSet] = {}
            for sub_name, sub_spec in spec["subsets"].items():
                # Sub-spec inherits parent use_punc semantics; treat each
                # sub like its own "single" entry (post_filter not allowed
                # at sub level — keep multi-split entries simple).
                subsets[sub_name] = _build_single_cuts(
                    base_dir, sub_spec, use_punc,
                )
            return subsets, dataset_name

        if kind == "ser_pattern":
            mdir = base_dir / spec["subdir"] / "data" / "manifests"
            prefix = spec["prefix"]
            splits = spec.get("splits") or ["test"]

            def _build(split: str) -> CutSet:
                rec_path = mdir / f"{prefix}_recordings_{split}.jsonl.gz"
                sup_path = mdir / f"{prefix}_supervisions_{split}.jsonl.gz"
                logging.info(
                    "Loading SER cuts (%s): rec=%s sup=%s",
                    split, rec_path, sup_path,
                )
                return CutSet.from_manifests(
                    recordings=load_manifest_lazy(rec_path),
                    supervisions=load_manifest_lazy(sup_path),
                )

            if len(splits) == 1:
                return _build(splits[0]), dataset_name
            return {
                f"{spec['subdir']}_{split}": _build(split) for split in splits
            }, dataset_name

        if kind == "cuts_file":
            full_path = base_dir / spec["path"]
            logging.info("Loading test cuts file: %s", full_path)
            return load_manifest_lazy(full_path), dataset_name

        raise ValueError(
            f"Unknown TEST_SET_DEFS kind {kind!r} for dataset "
            f"'{dataset_name}'."
        )

    parsed = _parse_multilingual_dataset_name(dataset_name)
    if parsed is not None:
        lang, multilingual_dataset_name = parsed
        cuts = _resolve_multilingual_test_cuts(
            lang, multilingual_dataset_name,
        )
        return cuts, f"{lang}:{multilingual_dataset_name}"

    raise ValueError(
        f"Unknown test dataset '{dataset_name}'. "
        f"Available built-in datasets: {sorted(TEST_SET_DEFS.keys())}. "
        "For multilingual datasets, use '<lang>:<dataset_name>' or "
        "'multilingual:<lang>:<dataset_name>'."
    )


def get_default_task(dataset_name: str) -> Union[str, None]:
    """Return the default task for *dataset_name*, or ``None`` if unknown.

    Multilingual ``<lang>:<dataset>`` names always default to ``"asr"``.
    """
    if dataset_name in DATASET_TASK:
        return DATASET_TASK[dataset_name]
    parsed = _parse_multilingual_dataset_name(dataset_name)
    if parsed is not None:
        return "asr"
    return None


def _detect_language(dataset_name: str, cli_language: str | None) -> str | None:
    """Resolve the evaluation language for *dataset_name*.

    Priority: CLI ``--language`` flag > built-in mapping > multilingual
    prefix > ``"en"``. Returns ``None`` for ``"multi"`` datasets so callers
    know to aggregate per-cut.
    """
    if cli_language:
        return cli_language
    if dataset_name in DATASET_LANGUAGE:
        lang = DATASET_LANGUAGE[dataset_name]
        return None if lang == "multi" else lang
    parsed = _parse_multilingual_dataset_name(dataset_name)
    if parsed is not None:
        return parsed[0]
    return "en"


def list_named_datasets() -> List[str]:
    """Sorted list of registered dataset names (for help text / errors)."""
    return sorted(TEST_SET_DEFS.keys())
