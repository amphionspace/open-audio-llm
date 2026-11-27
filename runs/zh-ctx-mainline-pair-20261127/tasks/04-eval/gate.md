# 判定：kgrpo（带纠偏）对 k0grpo（不带），think 关，配对 bootstrap 95% CI，差 = kgrpo − k0grpo（百分点）

## 1. 收益：纠偏 v2

| 来源 · 组 · 指标（条数） | k0grpo | kgrpo | 差 [95% CI] |
|---|---:|---:|---|
| emilia_zh · 原转写已对 · consistency · target_ok（1） | 100.00 | 100.00 | +0.00 [+0.00, +0.00] |
| emilia_zh · 原转写已对 · explicit · target_ok（662） | 90.94 | 91.84 | +0.91 [-0.45, +2.27] |
| emilia_zh · 原转写已对 · implicit · target_ok（40） | 90.00 | 90.00 | +0.00 [+0.00, +0.00] |
| emilia_zh · 需要纠正 · consistency · target_ok（55） | 69.09 | 70.91 | +1.82 [+0.00, +5.45] |
| emilia_zh · 需要纠正 · explicit · target_ok（516） | 45.54 | 47.67 | +2.13 [+0.19, +4.07] * |
| emilia_zh · 需要纠正 · implicit · target_ok（629） | 68.68 | 69.16 | +0.48 [-0.96, +2.07] |
| synthetic asr · confirm · consistency · target_alternative（100） | 5.00 | 2.00 | -3.00 [-7.00, +0.00] |
| synthetic asr · confirm · consistency · target_ok（100） | 87.00 | 91.00 | +4.00 [-2.00, +11.00] |
| synthetic asr · confirm · explicit · target_alternative（395） | 5.57 | 3.54 | -2.03 [-4.30, +0.25] |
| synthetic asr · confirm · explicit · target_ok（395） | 86.84 | 90.38 | +3.54 [+0.25, +7.09] * |
| synthetic asr · confirm · implicit · target_alternative（300） | 25.00 | 25.33 | +0.33 [-0.67, +1.67] |
| synthetic asr · confirm · implicit · target_ok（300） | 74.67 | 74.00 | -0.67 [-2.00, +0.67] |
| synthetic asr · correct · consistency · target_ok（100） | 55.00 | 80.00 | +25.00 [+17.00, +34.00] * |
| synthetic asr · correct · explicit · target_ok（395） | 48.61 | 68.86 | +20.25 [+15.95, +24.56] * |
| synthetic asr · correct · implicit · target_ok（300） | 32.33 | 32.00 | -0.33 [-1.67, +1.33] |
| synthetic ts_asr · confirm · consistency · target_alternative（95） | 3.16 | 1.05 | -2.11 [-5.26, +0.00] |
| synthetic ts_asr · confirm · consistency · target_ok（95） | 80.00 | 94.74 | +14.74 [+6.32, +23.16] * |
| synthetic ts_asr · confirm · explicit · target_alternative（378） | 5.56 | 3.97 | -1.59 [-3.97, +0.53] |
| synthetic ts_asr · confirm · explicit · target_ok（378） | 76.46 | 89.68 | +13.23 [+8.99, +17.46] * |
| synthetic ts_asr · confirm · implicit · target_alternative（293） | 22.87 | 24.57 | +1.71 [-0.01, +3.75] |
| synthetic ts_asr · confirm · implicit · target_ok（293） | 67.24 | 74.40 | +7.17 [+4.10, +10.59] * |
| synthetic ts_asr · correct · consistency · target_ok（95） | 49.47 | 81.05 | +31.58 [+22.11, +42.11] * |
| synthetic ts_asr · correct · explicit · target_ok（378） | 38.89 | 70.90 | +32.01 [+26.72, +37.30] * |
| synthetic ts_asr · correct · implicit · target_ok（293） | 28.67 | 32.08 | +3.41 [+1.02, +6.14] * |
| synthetic_indep asr · confirm · consistency · target_alternative（100） | 1.00 | 0.00 | -1.00 [-3.00, +0.00] |
| synthetic_indep asr · confirm · consistency · target_ok（100） | 92.00 | 92.00 | +0.00 [-5.00, +6.00] |
| synthetic_indep asr · confirm · explicit · target_alternative（1000） | 3.30 | 4.10 | +0.80 [-0.20, +1.90] |
| synthetic_indep asr · confirm · explicit · target_ok（1000） | 81.80 | 84.00 | +2.20 [+0.40, +4.10] * |
| synthetic_indep asr · confirm · implicit · target_alternative（200） | 7.50 | 7.00 | -0.50 [-2.00, +1.00] |
| synthetic_indep asr · confirm · implicit · target_ok（200） | 92.50 | 93.00 | +0.50 [-1.00, +2.00] |
| synthetic_indep asr · correct · consistency · target_ok（100） | 23.00 | 45.00 | +22.00 [+13.97, +31.00] * |
| synthetic_indep asr · correct · explicit · target_ok（1000） | 28.00 | 41.00 | +13.00 [+10.50, +15.50] * |
| synthetic_indep asr · correct · implicit · target_ok（200） | 29.50 | 30.50 | +1.00 [-1.50, +4.00] |
| synthetic_indep ts_asr · confirm · consistency · target_alternative（92） | 0.00 | 0.00 | +0.00 [+0.00, +0.00] |
| synthetic_indep ts_asr · confirm · consistency · target_ok（92） | 85.87 | 90.22 | +4.35 [-3.26, +11.96] |
| synthetic_indep ts_asr · confirm · explicit · target_alternative（784） | 2.81 | 3.70 | +0.89 [-0.38, +2.17] |
| synthetic_indep ts_asr · confirm · explicit · target_ok（784） | 70.15 | 81.12 | +10.97 [+8.04, +13.90] * |
| synthetic_indep ts_asr · confirm · implicit · target_alternative（194） | 7.73 | 6.70 | -1.03 [-3.61, +1.55] |
| synthetic_indep ts_asr · confirm · implicit · target_ok（194） | 80.93 | 90.21 | +9.28 [+4.64, +13.92] * |
| synthetic_indep ts_asr · correct · consistency · target_ok（92） | 14.13 | 40.22 | +26.09 [+16.30, +35.87] * |
| synthetic_indep ts_asr · correct · explicit · target_ok（784） | 18.62 | 37.24 | +18.62 [+15.56, +21.56] * |
| synthetic_indep ts_asr · correct · implicit · target_ok（194） | 22.16 | 26.29 | +4.12 [+0.52, +7.22] * |
| wenetspeech_podcast · 原转写已对 · explicit · target_ok（81） | 91.36 | 91.36 | +0.00 [+0.00, +0.00] |
| wenetspeech_podcast · 需要纠正 · explicit · target_ok（45） | 53.33 | 60.00 | +6.67 [+0.00, +15.56] |
| wenetspeech_podcast · 需要纠正 · implicit · target_ok（1） | 100.00 | 100.00 | +0.00 [+0.00, +0.00] |

判定 1（Emilia 需要纠正 · explicit，516 条，CI 下限 > 0）：成立

## 2. 回退：交付 27 组

| 项目 · 指标 | k0grpo | kgrpo | 差 [95% CI] |
|---|---:|---:|---|
| v12 · all/CER error_rate | 12.51 | 12.86 | +0.35 [+0.10, +0.60] * |
| v12 · all/CER false_alarm_rate | 3.95 | 3.46 | -0.49 [-0.70, -0.29] * |
| v12 · all/CER miss_rate | 1.85 | 2.13 | +0.29 [+0.14, +0.45] * |
| es-asr · all/WER error_rate | 4.92 | 4.91 | -0.01 [-0.07, +0.04] |
| es-asr · all/WER miss_rate | 0.02 | 0.03 | +0.01 [+0.00, +0.02] |
| ts-es-v1 · all/WER error_rate | 26.27 | 26.41 | +0.14 [-0.26, +0.54] |
| ts-es-v1 · all/WER false_alarm_rate（回退） | 4.40 | 9.26 | +4.86 [+4.31, +5.38] * |
| ts-es-v1 · all/WER miss_rate | 3.98 | 2.31 | -1.67 [-1.97, -1.37] * |
| babble4 · all/CER error_rate | 2.05 | 2.28 | +0.24 [+0.02, +0.50] * |
| babble4 · all/CER false_alarm_rate | 4.91 | 3.81 | -1.10 [-1.71, -0.56] * |
| babble4 · all/CER miss_rate | 0.00 | 0.00 | +0.00 [+0.00, +0.00] |
| babble4 · all/WER error_rate | 2.15 | 2.12 | -0.03 [-0.20, +0.12] |
| babble4 · all/WER false_alarm_rate | 1.62 | 2.00 | +0.38 [+0.00, +0.86] |
| babble4 · all/WER miss_rate | 0.10 | 0.00 | -0.10 [-0.31, +0.00] |
| babble3 · all/CER error_rate | 2.05 | 2.28 | +0.24 [+0.01, +0.52] * |
| babble3 · all/CER false_alarm_rate | 0.25 | 1.00 | +0.75 [-0.26, +1.92] |
| babble3 · all/CER miss_rate | 0.00 | 0.00 | +0.00 [+0.00, +0.00] |
| babble3 · all/WER error_rate | 2.15 | 2.12 | -0.03 [-0.19, +0.11] |
| babble3 · all/WER false_alarm_rate | 1.62 | 2.00 | +0.38 [+0.00, +0.87] |
| babble3 · all/WER miss_rate | 0.10 | 0.00 | -0.10 [-0.31, +0.00] |
| customer-ts · all/CER error_rate | 5.21 | 17.71 | +12.50 [-2.36, +53.42] |
| customer-ts · all/CER false_alarm_rate | 0.00 | 0.00 | +0.00 [+0.00, +0.00] |
| customer-ts · all/CER miss_rate | 0.00 | 20.00 | +20.00 [+0.00, +60.00] |
| customer-asr · all/CER error_rate | 17.81 | 20.29 | +2.48 [+0.00, +5.12] |
| customer-asr · all/CER miss_rate | 0.65 | 0.65 | +0.00 [+0.00, +0.00] |
| ts-hotwords · CER error_rate（回退） | 14.06 | 14.57 | +0.51 [+0.23, +0.81] * |
| ts-hotwords · CER recall | 93.63 | 93.17 | -0.46 [-0.79, -0.14] * |
| ts-en · all/WER error_rate | 13.39 | 13.64 | +0.25 [+0.01, +0.48] * |
| ts-en · all/WER false_alarm_rate（回退） | 6.30 | 7.61 | +1.31 [+0.97, +1.63] * |
| ts-en · all/WER miss_rate | 2.33 | 2.21 | -0.13 [-0.31, +0.05] |
| first-other · all/CER error_rate（回退） | 19.97 | 20.62 | +0.65 [+0.38, +0.94] * |
| first-other · all/CER false_alarm_rate | 4.77 | 2.31 | -2.46 [-2.94, -1.96] * |
| first-other · all/CER miss_rate（门槛移动） | 0.60 | 2.28 | +1.68 [+1.35, +2.05] * |
| asr-benchmark · all/CER error_rate | 3.59 | 3.57 | -0.03 [-0.12, +0.03] |
| asr-benchmark · all/CER miss_rate | 0.13 | 0.14 | +0.01 [+0.01, +0.02] * |
| asr-benchmark · all/CER term_hit_rate | 98.67 | 98.64 | -0.03 [-0.13, +0.07] |
| asr-benchmark · all/WER error_rate | 7.45 | 7.44 | -0.01 [-0.05, +0.04] |
| asr-benchmark · all/WER false_alarm_rate | 24.29 | 22.51 | -1.78 [-2.21, -1.37] * |
| asr-benchmark · all/WER miss_rate | 0.37 | 0.40 | +0.03 [-0.01, +0.07] |
| hotword-bench · CER error_rate | 3.01 | 2.99 | -0.03 [-0.04, -0.01] * |
| hotword-bench · CER recall | 88.32 | 88.24 | -0.08 [-0.14, -0.03] * |
| police · all/CER error_rate | 0.26 | 0.27 | +0.00 [-0.01, +0.02] |
| police · all/CER miss_rate | 0.00 | 0.00 | +0.00 [+0.00, +0.00] |
| police · all/CER term_hit_rate | 99.68 | 99.68 | +0.00 [-0.08, +0.09] |
| police-single · all/CER error_rate | 0.37 | 0.36 | -0.01 [-0.12, +0.10] |
| police-single · all/CER miss_rate | 0.00 | 0.00 | +0.00 [+0.00, +0.00] |
| police-single · all/CER term_hit_rate | 99.14 | 99.17 | +0.03 [-0.15, +0.25] |
| crosslingual · all/CER error_rate | 32.77 | 30.60 | -2.16 [-4.70, +0.36] |
| crosslingual · all/CER false_alarm_rate | 1.70 | 1.36 | -0.34 [-0.94, +0.23] |
| crosslingual · all/CER miss_rate | 14.62 | 13.56 | -1.06 [-1.99, -0.15] * |
| scene-asr · all/CER error_rate | 2.24 | 2.03 | -0.21 [-0.40, +0.01] |
| scene-asr · all/CER miss_rate | 0.74 | 0.35 | -0.39 [-0.59, -0.20] * |
| crosslingual-en · all/WER error_rate | 18.46 | 16.23 | -2.22 [-3.24, -1.24] * |
| crosslingual-en · all/WER false_alarm_rate | 3.30 | 3.86 | +0.57 [-0.12, +1.33] |
| crosslingual-en · all/WER miss_rate | 8.94 | 7.58 | -1.36 [-2.30, -0.47] * |
| scene-ts · all/CER error_rate（回退） | 3.92 | 4.99 | +1.06 [+0.29, +1.90] * |
| scene-ts · all/CER false_alarm_rate | 31.43 | 28.57 | -2.86 [-10.94, +5.41] |
| scene-ts · all/CER miss_rate | 0.27 | 0.22 | -0.05 [-0.16, +0.00] |
| ts-es-v2 · all/WER error_rate | 26.35 | 26.42 | +0.07 [-0.28, +0.43] |
| ts-es-v2 · all/WER false_alarm_rate（回退） | 4.33 | 9.03 | +4.70 [+4.25, +5.16] * |
| ts-es-v2 · all/WER miss_rate | 3.51 | 2.00 | -1.51 [-1.78, -1.25] * |
| ctx-v1 · all/CER error_rate | 6.50 | 1.31 | -5.19 [-5.94, -4.44] * |
| ctx-v1 · all/CER miss_rate | 5.02 | 0.06 | -4.96 [-5.66, -4.22] * |
| babble-es · all/WER error_rate | 6.67 | 6.53 | -0.13 [-0.57, +0.29] |
| babble-es · all/WER false_alarm_rate | 0.50 | 0.50 | +0.00 [+0.00, +0.00] |
| babble-es · all/WER miss_rate | 0.00 | 0.00 | +0.00 [+0.00, +0.00] |
| hotword-bench-es · CER error_rate | 4.66 | 4.70 | +0.04 [-0.04, +0.12] |
| hotword-bench-es · CER recall | 87.86 | 87.58 | -0.28 [-0.54, -0.01] * |
| chime4 · all/WER error_rate（回退） | 8.97 | 11.39 | +2.42 [+1.51, +3.36] * |
| chime4 · all/WER miss_rate（回退） | 5.53 | 8.03 | +2.50 [+1.52, +3.48] * |
| customer-hotwords · CER error_rate | 15.78 | 17.32 | +1.54 [-0.25, +3.69] |
| customer-hotwords · CER recall | 75.69 | 75.00 | -0.69 [-3.36, +1.44] |
| badcase · CER error_rate | 8.33 | 10.56 | +2.22 [+0.00, +6.11] |
| badcase · CER recall | 90.00 | 83.33 | -6.67 [-16.67, +0.00] |
| ts-hotwords-es · CER error_rate（回退） | 25.66 | 27.11 | +1.45 [+0.88, +2.00] * |
| ts-hotwords-es · CER recall | 85.17 | 86.47 | +1.30 [+0.75, +1.87] * |
| crosslingual-es · all/WER error_rate | 24.57 | 20.09 | -4.48 [-6.15, -2.90] * |
| crosslingual-es · all/WER false_alarm_rate（门槛移动） | 1.88 | 4.38 | +2.50 [+1.03, +4.14] * |
| crosslingual-es · all/WER miss_rate | 9.72 | 4.86 | -4.86 [-6.52, -3.24] * |

显著更差且差 > 0.5 个点（回退）：ts-es-v1 all/WER false_alarm_rate +4.86；ts-hotwords CER error_rate +0.51；ts-en all/WER false_alarm_rate +1.31；first-other all/CER error_rate +0.65；scene-ts all/CER error_rate +1.06；ts-es-v2 all/WER false_alarm_rate +4.70；chime4 all/WER error_rate +2.42；chime4 all/WER miss_rate +2.50；ts-hotwords-es CER error_rate +1.45
门槛移动（TS 误触发和漏识别一升一降、和不超过 +0.5）：first-other miss_rate +1.68 / false_alarm_rate -2.46；crosslingual-es false_alarm_rate +2.50 / miss_rate -4.86
显著更差但差 ≤ 0.5 个点（只记录）：v12 all/CER error_rate +0.35；v12 all/CER miss_rate +0.29；babble4 all/CER error_rate +0.24；babble3 all/CER error_rate +0.24；ts-hotwords CER recall -0.46；ts-en all/WER error_rate +0.25；asr-benchmark all/CER miss_rate +0.01；hotword-bench CER recall -0.08；hotword-bench-es CER recall -0.28
显著更好：v12 all/CER false_alarm_rate -0.49；ts-es-v1 all/WER miss_rate -1.67；babble4 all/CER false_alarm_rate -1.10；first-other all/CER false_alarm_rate -2.46；asr-benchmark all/WER false_alarm_rate -1.78；hotword-bench CER error_rate -0.03；crosslingual all/CER miss_rate -1.06；scene-asr all/CER miss_rate -0.39；crosslingual-en all/WER error_rate -2.22；crosslingual-en all/WER miss_rate -1.36；ts-es-v2 all/WER miss_rate -1.51；ctx-v1 all/CER error_rate -5.19；ctx-v1 all/CER miss_rate -4.96；ts-hotwords-es CER recall +1.30；crosslingual-es all/WER error_rate -4.48；crosslingual-es all/WER miss_rate -4.86

只记录：输出里带竖线（参考里没有）的条数，k0grpo / kgrpo：hotword-bench 126/54；ctx-v1 0/3（合计 126 / 57）

## 3. 结论规则

1 成立但 2 有回退：收益和回退的全表交给用户取舍。
