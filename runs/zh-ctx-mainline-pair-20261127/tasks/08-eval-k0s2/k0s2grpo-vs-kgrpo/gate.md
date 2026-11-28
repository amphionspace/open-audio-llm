# 判定：kgrpo 对 k0s2grpo，think 关，配对 bootstrap 95% CI，差 = kgrpo − k0s2grpo（百分点）

## 1. 收益：纠偏 v2

| 来源 · 组 · 指标（条数） | k0s2grpo | kgrpo | 差 [95% CI] |
|---|---:|---:|---|
| emilia_zh · 原转写已对 · consistency · target_ok（1） | 100.00 | 100.00 | +0.00 [+0.00, +0.00] |
| emilia_zh · 原转写已对 · explicit · target_ok（662） | 91.24 | 91.84 | +0.60 [-0.60, +1.81] |
| emilia_zh · 原转写已对 · implicit · target_ok（40） | 90.00 | 90.00 | +0.00 [-7.50, +7.50] |
| emilia_zh · 需要纠正 · consistency · target_ok（55） | 74.55 | 70.91 | -3.64 [-9.09, +0.00] |
| emilia_zh · 需要纠正 · explicit · target_ok（516） | 45.16 | 47.67 | +2.52 [+0.39, +4.65] * |
| emilia_zh · 需要纠正 · implicit · target_ok（629） | 68.52 | 69.16 | +0.64 [-0.95, +2.23] |
| synthetic asr · confirm · consistency · target_alternative（100） | 4.00 | 2.00 | -2.00 [-5.00, +0.00] |
| synthetic asr · confirm · consistency · target_ok（100） | 88.00 | 91.00 | +3.00 [-3.00, +9.00] |
| synthetic asr · confirm · explicit · target_alternative（395） | 6.33 | 3.54 | -2.78 [-5.32, -0.51] * |
| synthetic asr · confirm · explicit · target_ok（395） | 86.84 | 90.38 | +3.54 [+0.25, +6.84] * |
| synthetic asr · confirm · implicit · target_alternative（300） | 23.67 | 25.33 | +1.67 [+0.00, +3.67] |
| synthetic asr · confirm · implicit · target_ok（300） | 76.00 | 74.00 | -2.00 [-4.00, -0.33] * |
| synthetic asr · correct · consistency · target_ok（100） | 45.00 | 80.00 | +35.00 [+26.00, +44.00] * |
| synthetic asr · correct · explicit · target_ok（395） | 46.08 | 68.86 | +22.78 [+18.23, +27.34] * |
| synthetic asr · correct · implicit · target_ok（300） | 32.33 | 32.00 | -0.33 [-1.67, +1.00] |
| synthetic ts_asr · confirm · consistency · target_alternative（95） | 2.11 | 1.05 | -1.05 [-3.16, +0.00] |
| synthetic ts_asr · confirm · consistency · target_ok（95） | 84.21 | 94.74 | +10.53 [+3.16, +17.89] * |
| synthetic ts_asr · confirm · explicit · target_alternative（378） | 6.88 | 3.97 | -2.91 [-5.56, -0.53] * |
| synthetic ts_asr · confirm · explicit · target_ok（378） | 77.78 | 89.68 | +11.90 [+7.67, +16.14] * |
| synthetic ts_asr · confirm · implicit · target_alternative（293） | 23.89 | 24.57 | +0.68 [-1.37, +2.73] |
| synthetic ts_asr · confirm · implicit · target_ok（293） | 68.60 | 74.40 | +5.80 [+3.07, +9.22] * |
| synthetic ts_asr · correct · consistency · target_ok（95） | 38.95 | 81.05 | +42.11 [+32.63, +52.63] * |
| synthetic ts_asr · correct · explicit · target_ok（378） | 38.10 | 70.90 | +32.80 [+27.51, +37.83] * |
| synthetic ts_asr · correct · implicit · target_ok（293） | 29.69 | 32.08 | +2.39 [+0.00, +4.78] |
| synthetic_indep asr · confirm · consistency · target_alternative（100） | 2.00 | 0.00 | -2.00 [-5.00, +0.00] |
| synthetic_indep asr · confirm · consistency · target_ok（100） | 89.00 | 92.00 | +3.00 [-2.00, +8.00] |
| synthetic_indep asr · confirm · explicit · target_alternative（1000） | 4.10 | 4.10 | +0.00 [-1.10, +1.10] |
| synthetic_indep asr · confirm · explicit · target_ok（1000） | 81.20 | 84.00 | +2.80 [+0.90, +4.80] * |
| synthetic_indep asr · confirm · implicit · target_alternative（200） | 7.00 | 7.00 | +0.00 [-1.50, +1.50] |
| synthetic_indep asr · confirm · implicit · target_ok（200） | 93.00 | 93.00 | +0.00 [-1.50, +1.50] |
| synthetic_indep asr · correct · consistency · target_ok（100） | 24.00 | 45.00 | +21.00 [+13.00, +30.00] * |
| synthetic_indep asr · correct · explicit · target_ok（1000） | 27.00 | 41.00 | +14.00 [+11.50, +16.50] * |
| synthetic_indep asr · correct · implicit · target_ok（200） | 32.00 | 30.50 | -1.50 [-3.50, +0.00] |
| synthetic_indep ts_asr · confirm · consistency · target_alternative（92） | 1.09 | 0.00 | -1.09 [-3.26, +0.00] |
| synthetic_indep ts_asr · confirm · consistency · target_ok（92） | 88.04 | 90.22 | +2.17 [-4.35, +8.70] |
| synthetic_indep ts_asr · confirm · explicit · target_alternative（784） | 3.06 | 3.70 | +0.64 [-0.77, +2.04] |
| synthetic_indep ts_asr · confirm · explicit · target_ok（784） | 74.11 | 81.12 | +7.02 [+4.34, +9.70] * |
| synthetic_indep ts_asr · confirm · implicit · target_alternative（194） | 6.19 | 6.70 | +0.52 [-2.06, +3.09] |
| synthetic_indep ts_asr · confirm · implicit · target_ok（194） | 83.51 | 90.21 | +6.70 [+3.09, +10.82] * |
| synthetic_indep ts_asr · correct · consistency · target_ok（92） | 10.87 | 40.22 | +29.35 [+20.65, +39.13] * |
| synthetic_indep ts_asr · correct · explicit · target_ok（784） | 19.13 | 37.24 | +18.11 [+15.18, +20.92] * |
| synthetic_indep ts_asr · correct · implicit · target_ok（194） | 23.71 | 26.29 | +2.58 [-0.52, +6.19] |
| wenetspeech_podcast · 原转写已对 · explicit · target_ok（81） | 90.12 | 91.36 | +1.23 [+0.00, +3.70] |
| wenetspeech_podcast · 需要纠正 · explicit · target_ok（45） | 51.11 | 60.00 | +8.89 [+2.22, +17.78] * |
| wenetspeech_podcast · 需要纠正 · implicit · target_ok（1） | 100.00 | 100.00 | +0.00 [+0.00, +0.00] |

判定 1（Emilia 需要纠正 · explicit，516 条，CI 下限 > 0）：成立

## 2. 回退：交付 27 组

| 项目 · 指标 | k0s2grpo | kgrpo | 差 [95% CI] |
|---|---:|---:|---|
| v12 · all/CER error_rate | 12.37 | 12.86 | +0.49 [+0.28, +0.71] * |
| v12 · all/CER false_alarm_rate | 4.36 | 3.46 | -0.89 [-1.12, -0.67] * |
| v12 · all/CER miss_rate | 1.69 | 2.13 | +0.44 [+0.28, +0.61] * |
| es-asr · all/WER error_rate | 4.92 | 4.91 | -0.01 [-0.07, +0.04] |
| es-asr · all/WER miss_rate | 0.03 | 0.03 | +0.00 [+0.00, +0.00] |
| ts-es-v1 · all/WER error_rate | 25.97 | 26.41 | +0.44 [+0.07, +0.78] * |
| ts-es-v1 · all/WER false_alarm_rate（回退） | 4.54 | 9.26 | +4.72 [+4.18, +5.24] * |
| ts-es-v1 · all/WER miss_rate | 3.39 | 2.31 | -1.08 [-1.33, -0.83] * |
| babble4 · all/CER error_rate | 2.16 | 2.28 | +0.12 [-0.09, +0.37] |
| babble4 · all/CER false_alarm_rate（回退） | 3.13 | 3.81 | +0.68 [+0.25, +1.13] * |
| babble4 · all/CER miss_rate | 0.00 | 0.00 | +0.00 [+0.00, +0.00] |
| babble4 · all/WER error_rate | 2.28 | 2.12 | -0.16 [-0.30, -0.03] * |
| babble4 · all/WER false_alarm_rate | 1.62 | 2.00 | +0.38 [-0.13, +0.97] |
| babble4 · all/WER miss_rate | 0.10 | 0.00 | -0.10 [-0.31, +0.00] |
| babble3 · all/CER error_rate | 2.16 | 2.28 | +0.12 [-0.09, +0.36] |
| babble3 · all/CER false_alarm_rate | 0.50 | 1.00 | +0.50 [-0.71, +1.73] |
| babble3 · all/CER miss_rate | 0.00 | 0.00 | +0.00 [+0.00, +0.00] |
| babble3 · all/WER error_rate | 2.28 | 2.12 | -0.16 [-0.30, -0.03] * |
| babble3 · all/WER false_alarm_rate | 1.62 | 2.00 | +0.38 [-0.13, +0.91] |
| babble3 · all/WER miss_rate | 0.10 | 0.00 | -0.10 [-0.31, +0.00] |
| customer-ts · all/CER error_rate | 18.75 | 17.71 | -1.04 [-2.50, +0.00] |
| customer-ts · all/CER false_alarm_rate | 0.00 | 0.00 | +0.00 [+0.00, +0.00] |
| customer-ts · all/CER miss_rate | 20.00 | 20.00 | +0.00 [+0.00, +0.00] |
| customer-asr · all/CER error_rate | 26.42 | 20.29 | -6.13 [-25.37, +4.11] |
| customer-asr · all/CER miss_rate | 0.00 | 0.65 | +0.65 [+0.00, +1.94] |
| ts-hotwords · CER error_rate | 14.09 | 14.57 | +0.48 [+0.16, +0.81] * |
| ts-hotwords · CER recall | 93.45 | 93.17 | -0.28 [-0.62, +0.07] |
| ts-en · all/WER error_rate | 13.19 | 13.64 | +0.44 [+0.24, +0.66] * |
| ts-en · all/WER false_alarm_rate（回退） | 5.38 | 7.61 | +2.24 [+1.87, +2.58] * |
| ts-en · all/WER miss_rate | 2.59 | 2.21 | -0.39 [-0.58, -0.20] * |
| first-other · all/CER error_rate（回退） | 19.80 | 20.62 | +0.82 [+0.55, +1.10] * |
| first-other · all/CER false_alarm_rate | 4.25 | 2.31 | -1.94 [-2.37, -1.51] * |
| first-other · all/CER miss_rate（门槛移动） | 0.96 | 2.28 | +1.32 [+1.01, +1.65] * |
| asr-benchmark · all/CER error_rate | 3.57 | 3.57 | -0.00 [-0.10, +0.06] |
| asr-benchmark · all/CER miss_rate | 0.17 | 0.14 | -0.03 [-0.04, -0.02] * |
| asr-benchmark · all/CER term_hit_rate | 98.85 | 98.64 | -0.21 [-0.31, -0.11] * |
| asr-benchmark · all/WER error_rate | 7.43 | 7.44 | +0.01 [-0.04, +0.05] |
| asr-benchmark · all/WER false_alarm_rate（回退） | 21.37 | 22.51 | +1.14 [+0.77, +1.52] * |
| asr-benchmark · all/WER miss_rate | 0.53 | 0.40 | -0.13 [-0.18, -0.09] * |
| hotword-bench · CER error_rate | 3.13 | 2.99 | -0.14 [-0.16, -0.13] * |
| hotword-bench · CER recall | 88.14 | 88.24 | +0.09 [+0.04, +0.15] * |
| police · all/CER error_rate | 0.27 | 0.27 | -0.00 [-0.01, +0.01] |
| police · all/CER miss_rate | 0.00 | 0.00 | +0.00 [+0.00, +0.00] |
| police · all/CER term_hit_rate | 99.72 | 99.68 | -0.05 [-0.14, +0.05] |
| police-single · all/CER error_rate | 0.39 | 0.36 | -0.03 [-0.14, +0.07] |
| police-single · all/CER miss_rate | 0.00 | 0.00 | +0.00 [+0.00, +0.00] |
| police-single · all/CER term_hit_rate | 99.11 | 99.17 | +0.06 [-0.15, +0.28] |
| crosslingual · all/CER error_rate | 43.04 | 30.60 | -12.43 [-37.99, +1.53] |
| crosslingual · all/CER false_alarm_rate | 2.50 | 1.36 | -1.14 [-2.06, -0.33] * |
| crosslingual · all/CER miss_rate | 14.55 | 13.56 | -0.98 [-1.82, -0.15] * |
| scene-asr · all/CER error_rate | 2.08 | 2.03 | -0.06 [-0.23, +0.14] |
| scene-asr · all/CER miss_rate | 0.54 | 0.35 | -0.20 [-0.35, -0.07] * |
| crosslingual-en · all/WER error_rate | 17.51 | 16.23 | -1.27 [-2.11, -0.48] * |
| crosslingual-en · all/WER false_alarm_rate（门槛移动） | 2.84 | 3.86 | +1.02 [+0.11, +1.98] * |
| crosslingual-en · all/WER miss_rate | 8.41 | 7.58 | -0.83 [-1.62, -0.08] * |
| scene-ts · all/CER error_rate | 5.31 | 4.99 | -0.32 [-1.27, +0.52] |
| scene-ts · all/CER false_alarm_rate | 28.57 | 28.57 | +0.00 [-9.59, +9.09] |
| scene-ts · all/CER miss_rate | 0.27 | 0.22 | -0.05 [-0.16, +0.00] |
| ts-es-v2 · all/WER error_rate | 26.03 | 26.42 | +0.39 [+0.08, +0.71] * |
| ts-es-v2 · all/WER false_alarm_rate（回退） | 4.27 | 9.03 | +4.76 [+4.30, +5.23] * |
| ts-es-v2 · all/WER miss_rate | 2.95 | 2.00 | -0.95 [-1.17, -0.73] * |
| ctx-v1 · all/CER error_rate | 5.06 | 1.31 | -3.75 [-4.40, -3.13] * |
| ctx-v1 · all/CER miss_rate | 3.63 | 0.06 | -3.57 [-4.22, -2.95] * |
| babble-es · all/WER error_rate | 6.66 | 6.53 | -0.12 [-0.52, +0.26] |
| babble-es · all/WER false_alarm_rate | 0.50 | 0.50 | +0.00 [+0.00, +0.00] |
| babble-es · all/WER miss_rate | 0.00 | 0.00 | +0.00 [+0.00, +0.00] |
| hotword-bench-es · CER error_rate | 4.71 | 4.70 | -0.01 [-0.10, +0.08] |
| hotword-bench-es · CER recall | 87.61 | 87.58 | -0.03 [-0.29, +0.24] |
| chime4 · all/WER error_rate | 10.28 | 11.39 | +1.10 [-0.02, +2.16] |
| chime4 · all/WER miss_rate | 7.42 | 8.03 | +0.61 [-0.53, +1.59] |
| customer-hotwords · CER error_rate（回退） | 15.10 | 17.32 | +2.22 [+0.15, +4.92] * |
| customer-hotwords · CER recall | 77.78 | 75.00 | -2.78 [-7.38, +1.44] |
| badcase · CER error_rate（回退） | 3.89 | 10.56 | +6.67 [+1.11, +13.33] * |
| badcase · CER recall | 93.33 | 83.33 | -10.00 [-20.00, +0.00] |
| ts-hotwords-es · CER error_rate（回退） | 25.11 | 27.11 | +2.00 [+1.49, +2.53] * |
| ts-hotwords-es · CER recall | 85.95 | 86.47 | +0.52 [-0.03, +1.08] |
| crosslingual-es · all/WER error_rate | 24.32 | 20.09 | -4.23 [-5.80, -2.82] * |
| crosslingual-es · all/WER false_alarm_rate（门槛移动） | 2.08 | 4.38 | +2.29 [+0.85, +3.87] * |
| crosslingual-es · all/WER miss_rate | 8.89 | 4.86 | -4.03 [-5.55, -2.62] * |

显著更差且差 > 0.5 个点（回退）：ts-es-v1 all/WER false_alarm_rate +4.72；babble4 all/CER false_alarm_rate +0.68；ts-en all/WER false_alarm_rate +2.24；first-other all/CER error_rate +0.82；asr-benchmark all/WER false_alarm_rate +1.14；ts-es-v2 all/WER false_alarm_rate +4.76；customer-hotwords CER error_rate +2.22；badcase CER error_rate +6.67；ts-hotwords-es CER error_rate +2.00
门槛移动（TS 误触发和漏识别一升一降、和不超过 +0.5）：first-other miss_rate +1.32 / false_alarm_rate -1.94；crosslingual-en false_alarm_rate +1.02 / miss_rate -0.83；crosslingual-es false_alarm_rate +2.29 / miss_rate -4.03
显著更差但差 ≤ 0.5 个点（只记录）：v12 all/CER error_rate +0.49；v12 all/CER miss_rate +0.44；ts-es-v1 all/WER error_rate +0.44；ts-hotwords CER error_rate +0.48；ts-en all/WER error_rate +0.44；asr-benchmark all/CER term_hit_rate -0.21；ts-es-v2 all/WER error_rate +0.39
显著更好：v12 all/CER false_alarm_rate -0.89；ts-es-v1 all/WER miss_rate -1.08；babble4 all/WER error_rate -0.16；babble3 all/WER error_rate -0.16；ts-en all/WER miss_rate -0.39；first-other all/CER false_alarm_rate -1.94；asr-benchmark all/CER miss_rate -0.03；asr-benchmark all/WER miss_rate -0.13；hotword-bench CER error_rate -0.14；hotword-bench CER recall +0.09；crosslingual all/CER false_alarm_rate -1.14；crosslingual all/CER miss_rate -0.98；scene-asr all/CER miss_rate -0.20；crosslingual-en all/WER error_rate -1.27；crosslingual-en all/WER miss_rate -0.83；ts-es-v2 all/WER miss_rate -0.95；ctx-v1 all/CER error_rate -3.75；ctx-v1 all/CER miss_rate -3.57；crosslingual-es all/WER error_rate -4.23；crosslingual-es all/WER miss_rate -4.03

只记录：输出里带竖线（参考里没有）的条数，k0s2grpo / kgrpo：hotword-bench 39/54；ctx-v1 0/3（合计 39 / 57）

## 3. 结论规则

1 成立但 2 有回退：收益和回退的全表交给用户取舍。
