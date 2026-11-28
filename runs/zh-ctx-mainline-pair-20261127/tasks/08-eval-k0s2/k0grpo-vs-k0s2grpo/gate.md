# 判定：k0s2grpo 对 k0grpo，think 关，配对 bootstrap 95% CI，差 = k0s2grpo − k0grpo（百分点）

两边都不带纠偏这一包：这里的“收益”“回退”只是运行间波动，第 3 节的结论规则不适用。

## 1. 收益：纠偏 v2

| 来源 · 组 · 指标（条数） | k0grpo | k0s2grpo | 差 [95% CI] |
|---|---:|---:|---|
| emilia_zh · 原转写已对 · consistency · target_ok（1） | 100.00 | 100.00 | +0.00 [+0.00, +0.00] |
| emilia_zh · 原转写已对 · explicit · target_ok（662） | 90.94 | 91.24 | +0.30 [-0.91, +1.66] |
| emilia_zh · 原转写已对 · implicit · target_ok（40） | 90.00 | 90.00 | +0.00 [-7.50, +7.50] |
| emilia_zh · 需要纠正 · consistency · target_ok（55） | 69.09 | 74.55 | +5.45 [+0.00, +10.91] |
| emilia_zh · 需要纠正 · explicit · target_ok（516） | 45.54 | 45.16 | -0.39 [-2.13, +1.36] |
| emilia_zh · 需要纠正 · implicit · target_ok（629） | 68.68 | 68.52 | -0.16 [-1.75, +1.43] |
| synthetic asr · confirm · consistency · target_alternative（100） | 5.00 | 4.00 | -1.00 [-3.00, +0.00] |
| synthetic asr · confirm · consistency · target_ok（100） | 87.00 | 88.00 | +1.00 [-2.00, +4.00] |
| synthetic asr · confirm · explicit · target_alternative（395） | 5.57 | 6.33 | +0.76 [-0.51, +2.03] |
| synthetic asr · confirm · explicit · target_ok（395） | 86.84 | 86.84 | +0.00 [-1.77, +1.77] |
| synthetic asr · confirm · implicit · target_alternative（300） | 25.00 | 23.67 | -1.33 [-2.67, -0.33] * |
| synthetic asr · confirm · implicit · target_ok（300） | 74.67 | 76.00 | +1.33 [+0.33, +2.67] * |
| synthetic asr · correct · consistency · target_ok（100） | 55.00 | 45.00 | -10.00 [-16.00, -5.00] * |
| synthetic asr · correct · explicit · target_ok（395） | 48.61 | 46.08 | -2.53 [-5.06, -0.25] * |
| synthetic asr · correct · implicit · target_ok（300） | 32.33 | 32.33 | +0.00 [-1.33, +1.33] |
| synthetic ts_asr · confirm · consistency · target_alternative（95） | 3.16 | 2.11 | -1.05 [-3.16, +0.00] |
| synthetic ts_asr · confirm · consistency · target_ok（95） | 80.00 | 84.21 | +4.21 [+1.05, +8.42] * |
| synthetic ts_asr · confirm · explicit · target_alternative（378） | 5.56 | 6.88 | +1.32 [-0.26, +2.91] |
| synthetic ts_asr · confirm · explicit · target_ok（378） | 76.46 | 77.78 | +1.32 [-1.06, +3.70] |
| synthetic ts_asr · confirm · implicit · target_alternative（293） | 22.87 | 23.89 | +1.02 [-0.34, +2.73] |
| synthetic ts_asr · confirm · implicit · target_ok（293） | 67.24 | 68.60 | +1.37 [-0.68, +3.41] |
| synthetic ts_asr · correct · consistency · target_ok（95） | 49.47 | 38.95 | -10.53 [-16.84, -4.21] * |
| synthetic ts_asr · correct · explicit · target_ok（378） | 38.89 | 38.10 | -0.79 [-3.17, +1.59] |
| synthetic ts_asr · correct · implicit · target_ok（293） | 28.67 | 29.69 | +1.02 [-0.68, +2.73] |
| synthetic_indep asr · confirm · consistency · target_alternative（100） | 1.00 | 2.00 | +1.00 [+0.00, +3.00] |
| synthetic_indep asr · confirm · consistency · target_ok（100） | 92.00 | 89.00 | -3.00 [-8.00, +1.00] |
| synthetic_indep asr · confirm · explicit · target_alternative（1000） | 3.30 | 4.10 | +0.80 [+0.10, +1.50] * |
| synthetic_indep asr · confirm · explicit · target_ok（1000） | 81.80 | 81.20 | -0.60 [-1.90, +0.80] |
| synthetic_indep asr · confirm · implicit · target_alternative（200） | 7.50 | 7.00 | -0.50 [-1.50, +0.00] |
| synthetic_indep asr · confirm · implicit · target_ok（200） | 92.50 | 93.00 | +0.50 [+0.00, +1.50] |
| synthetic_indep asr · correct · consistency · target_ok（100） | 23.00 | 24.00 | +1.00 [-4.00, +6.00] |
| synthetic_indep asr · correct · explicit · target_ok（1000） | 28.00 | 27.00 | -1.00 [-2.60, +0.50] |
| synthetic_indep asr · correct · implicit · target_ok（200） | 29.50 | 32.00 | +2.50 [+0.00, +5.00] |
| synthetic_indep ts_asr · confirm · consistency · target_alternative（92） | 0.00 | 1.09 | +1.09 [+0.00, +3.26] |
| synthetic_indep ts_asr · confirm · consistency · target_ok（92） | 85.87 | 88.04 | +2.17 [-3.26, +7.61] |
| synthetic_indep ts_asr · confirm · explicit · target_alternative（784） | 2.81 | 3.06 | +0.26 [-0.51, +1.02] |
| synthetic_indep ts_asr · confirm · explicit · target_ok（784） | 70.15 | 74.11 | +3.95 [+2.17, +5.99] * |
| synthetic_indep ts_asr · confirm · implicit · target_alternative（194） | 7.73 | 6.19 | -1.55 [-3.61, +0.00] |
| synthetic_indep ts_asr · confirm · implicit · target_ok（194） | 80.93 | 83.51 | +2.58 [-1.03, +6.19] |
| synthetic_indep ts_asr · correct · consistency · target_ok（92） | 14.13 | 10.87 | -3.26 [-7.61, +0.00] |
| synthetic_indep ts_asr · correct · explicit · target_ok（784） | 18.62 | 19.13 | +0.51 [-1.15, +2.17] |
| synthetic_indep ts_asr · correct · implicit · target_ok（194） | 22.16 | 23.71 | +1.55 [-1.03, +4.12] |
| wenetspeech_podcast · 原转写已对 · explicit · target_ok（81） | 91.36 | 90.12 | -1.23 [-3.70, +0.00] |
| wenetspeech_podcast · 需要纠正 · explicit · target_ok（45） | 53.33 | 51.11 | -2.22 [-6.67, +0.00] |
| wenetspeech_podcast · 需要纠正 · implicit · target_ok（1） | 100.00 | 100.00 | +0.00 [+0.00, +0.00] |

判定 1（Emilia 需要纠正 · explicit，516 条，CI 下限 > 0）：不成立

## 2. 回退：交付 27 组

| 项目 · 指标 | k0grpo | k0s2grpo | 差 [95% CI] |
|---|---:|---:|---|
| v12 · all/CER error_rate | 12.51 | 12.37 | -0.14 [-0.38, +0.08] |
| v12 · all/CER false_alarm_rate | 3.95 | 4.36 | +0.41 [+0.19, +0.62] * |
| v12 · all/CER miss_rate | 1.85 | 1.69 | -0.15 [-0.29, -0.02] * |
| es-asr · all/WER error_rate | 4.92 | 4.92 | -0.00 [-0.06, +0.05] |
| es-asr · all/WER miss_rate | 0.02 | 0.03 | +0.01 [+0.00, +0.02] |
| ts-es-v1 · all/WER error_rate | 26.27 | 25.97 | -0.30 [-0.66, +0.06] |
| ts-es-v1 · all/WER false_alarm_rate | 4.40 | 4.54 | +0.14 [-0.23, +0.50] |
| ts-es-v1 · all/WER miss_rate | 3.98 | 3.39 | -0.59 [-0.84, -0.34] * |
| babble4 · all/CER error_rate | 2.05 | 2.16 | +0.11 [-0.07, +0.32] |
| babble4 · all/CER false_alarm_rate | 4.91 | 3.13 | -1.78 [-2.31, -1.22] * |
| babble4 · all/CER miss_rate | 0.00 | 0.00 | +0.00 [+0.00, +0.00] |
| babble4 · all/WER error_rate | 2.15 | 2.28 | +0.13 [-0.03, +0.29] |
| babble4 · all/WER false_alarm_rate | 1.62 | 1.62 | +0.00 [-0.50, +0.50] |
| babble4 · all/WER miss_rate | 0.10 | 0.10 | +0.00 [+0.00, +0.00] |
| babble3 · all/CER error_rate | 2.05 | 2.16 | +0.11 [-0.08, +0.31] |
| babble3 · all/CER false_alarm_rate | 0.25 | 0.50 | +0.25 [+0.00, +0.79] |
| babble3 · all/CER miss_rate | 0.00 | 0.00 | +0.00 [+0.00, +0.00] |
| babble3 · all/WER error_rate | 2.15 | 2.28 | +0.13 [-0.03, +0.28] |
| babble3 · all/WER false_alarm_rate | 1.62 | 1.62 | +0.00 [-0.50, +0.50] |
| babble3 · all/WER miss_rate | 0.10 | 0.10 | +0.00 [+0.00, +0.00] |
| customer-ts · all/CER error_rate | 5.21 | 18.75 | +13.54 [+0.00, +53.42] |
| customer-ts · all/CER false_alarm_rate | 0.00 | 0.00 | +0.00 [+0.00, +0.00] |
| customer-ts · all/CER miss_rate | 0.00 | 20.00 | +20.00 [+0.00, +60.00] |
| customer-asr · all/CER error_rate | 17.81 | 26.42 | +8.61 [-1.67, +28.40] |
| customer-asr · all/CER miss_rate | 0.65 | 0.00 | -0.65 [-1.94, +0.00] |
| ts-hotwords · CER error_rate | 14.06 | 14.09 | +0.03 [-0.28, +0.35] |
| ts-hotwords · CER recall | 93.63 | 93.45 | -0.18 [-0.50, +0.14] |
| ts-en · all/WER error_rate | 13.39 | 13.19 | -0.20 [-0.44, +0.04] |
| ts-en · all/WER false_alarm_rate | 6.30 | 5.38 | -0.92 [-1.21, -0.64] * |
| ts-en · all/WER miss_rate | 2.33 | 2.59 | +0.26 [+0.07, +0.46] * |
| first-other · all/CER error_rate | 19.97 | 19.80 | -0.16 [-0.37, +0.06] |
| first-other · all/CER false_alarm_rate | 4.77 | 4.25 | -0.52 [-0.95, -0.15] * |
| first-other · all/CER miss_rate | 0.60 | 0.96 | +0.36 [+0.18, +0.56] * |
| asr-benchmark · all/CER error_rate | 3.59 | 3.57 | -0.02 [-0.04, -0.00] * |
| asr-benchmark · all/CER miss_rate | 0.13 | 0.17 | +0.05 [+0.04, +0.06] * |
| asr-benchmark · all/CER term_hit_rate | 98.67 | 98.85 | +0.17 [+0.08, +0.27] * |
| asr-benchmark · all/WER error_rate | 7.45 | 7.43 | -0.02 [-0.06, +0.03] |
| asr-benchmark · all/WER false_alarm_rate | 24.29 | 21.37 | -2.92 [-3.41, -2.47] * |
| asr-benchmark · all/WER miss_rate | 0.37 | 0.53 | +0.16 [+0.12, +0.21] * |
| hotword-bench · CER error_rate | 3.01 | 3.13 | +0.11 [+0.10, +0.13] * |
| hotword-bench · CER recall | 88.32 | 88.14 | -0.18 [-0.23, -0.12] * |
| police · all/CER error_rate | 0.26 | 0.27 | +0.01 [-0.01, +0.02] |
| police · all/CER miss_rate | 0.00 | 0.00 | +0.00 [+0.00, +0.00] |
| police · all/CER term_hit_rate | 99.68 | 99.72 | +0.05 [-0.04, +0.13] |
| police-single · all/CER error_rate | 0.37 | 0.39 | +0.02 [-0.04, +0.09] |
| police-single · all/CER miss_rate | 0.00 | 0.00 | +0.00 [+0.00, +0.00] |
| police-single · all/CER term_hit_rate | 99.14 | 99.11 | -0.03 [-0.21, +0.15] |
| crosslingual · all/CER error_rate | 32.77 | 43.04 | +10.27 [-3.96, +36.13] |
| crosslingual · all/CER false_alarm_rate | 1.70 | 2.50 | +0.80 [+0.00, +1.65] |
| crosslingual · all/CER miss_rate | 14.62 | 14.55 | -0.08 [-1.07, +0.83] |
| scene-asr · all/CER error_rate | 2.24 | 2.08 | -0.15 [-0.27, -0.03] * |
| scene-asr · all/CER miss_rate | 0.74 | 0.54 | -0.20 [-0.37, -0.04] * |
| crosslingual-en · all/WER error_rate | 18.46 | 17.51 | -0.95 [-1.76, -0.12] * |
| crosslingual-en · all/WER false_alarm_rate | 3.30 | 2.84 | -0.45 [-1.44, +0.45] |
| crosslingual-en · all/WER miss_rate | 8.94 | 8.41 | -0.53 [-1.26, +0.23] |
| scene-ts · all/CER error_rate（回退） | 3.92 | 5.31 | +1.39 [+0.68, +2.20] * |
| scene-ts · all/CER false_alarm_rate | 31.43 | 28.57 | -2.86 [-8.96, +2.70] |
| scene-ts · all/CER miss_rate | 0.27 | 0.27 | +0.00 [+0.00, +0.00] |
| ts-es-v2 · all/WER error_rate | 26.35 | 26.03 | -0.33 [-0.66, -0.00] * |
| ts-es-v2 · all/WER false_alarm_rate | 4.33 | 4.27 | -0.06 [-0.36, +0.23] |
| ts-es-v2 · all/WER miss_rate | 3.51 | 2.95 | -0.56 [-0.78, -0.33] * |
| ctx-v1 · all/CER error_rate | 6.50 | 5.06 | -1.44 [-1.91, -0.99] * |
| ctx-v1 · all/CER miss_rate | 5.02 | 3.63 | -1.39 [-1.79, -0.95] * |
| babble-es · all/WER error_rate | 6.67 | 6.66 | -0.01 [-0.40, +0.38] |
| babble-es · all/WER false_alarm_rate | 0.50 | 0.50 | +0.00 [+0.00, +0.00] |
| babble-es · all/WER miss_rate | 0.00 | 0.00 | +0.00 [+0.00, +0.00] |
| hotword-bench-es · CER error_rate | 4.66 | 4.71 | +0.05 [-0.03, +0.14] |
| hotword-bench-es · CER recall | 87.86 | 87.61 | -0.25 [-0.52, +0.01] |
| chime4 · all/WER error_rate（回退） | 8.97 | 10.28 | +1.32 [+0.45, +2.29] * |
| chime4 · all/WER miss_rate（回退） | 5.53 | 7.42 | +1.89 [+0.98, +2.88] * |
| customer-hotwords · CER error_rate | 15.78 | 15.10 | -0.68 [-3.49, +1.88] |
| customer-hotwords · CER recall | 75.69 | 77.78 | +2.08 [-2.88, +7.14] |
| badcase · CER error_rate | 8.33 | 3.89 | -4.44 [-11.67, +1.11] |
| badcase · CER recall | 90.00 | 93.33 | +3.33 [-10.00, +16.67] |
| ts-hotwords-es · CER error_rate | 25.66 | 25.11 | -0.56 [-0.97, -0.12] * |
| ts-hotwords-es · CER recall | 85.17 | 85.95 | +0.78 [+0.21, +1.35] * |
| crosslingual-es · all/WER error_rate | 24.57 | 24.32 | -0.25 [-1.72, +1.16] |
| crosslingual-es · all/WER false_alarm_rate | 1.88 | 2.08 | +0.21 [-0.64, +1.20] |
| crosslingual-es · all/WER miss_rate | 9.72 | 8.89 | -0.83 [-2.12, +0.42] |

显著更差且差 > 0.5 个点（回退）：scene-ts all/CER error_rate +1.39；chime4 all/WER error_rate +1.32；chime4 all/WER miss_rate +1.89
门槛移动（TS 误触发和漏识别一升一降、和不超过 +0.5）：无
显著更差但差 ≤ 0.5 个点（只记录）：v12 all/CER false_alarm_rate +0.41；ts-en all/WER miss_rate +0.26；first-other all/CER miss_rate +0.36；asr-benchmark all/CER miss_rate +0.05；asr-benchmark all/WER miss_rate +0.16；hotword-bench CER error_rate +0.11；hotword-bench CER recall -0.18
显著更好：v12 all/CER miss_rate -0.15；ts-es-v1 all/WER miss_rate -0.59；babble4 all/CER false_alarm_rate -1.78；ts-en all/WER false_alarm_rate -0.92；first-other all/CER false_alarm_rate -0.52；asr-benchmark all/CER error_rate -0.02；asr-benchmark all/CER term_hit_rate +0.17；asr-benchmark all/WER false_alarm_rate -2.92；scene-asr all/CER error_rate -0.15；scene-asr all/CER miss_rate -0.20；crosslingual-en all/WER error_rate -0.95；ts-es-v2 all/WER error_rate -0.33；ts-es-v2 all/WER miss_rate -0.56；ctx-v1 all/CER error_rate -1.44；ctx-v1 all/CER miss_rate -1.39；ts-hotwords-es CER error_rate -0.56；ts-hotwords-es CER recall +0.78

只记录：输出里带竖线（参考里没有）的条数，k0grpo / k0s2grpo：hotword-bench 126/39（合计 126 / 39）

## 3. 结论规则

1 不成立：在一轮 SFT → GRPO 里纠偏这一包拿不到明确说明类的收益，不进主线。
