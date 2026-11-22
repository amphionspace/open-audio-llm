# 交付评估：kgrpo 对 r1117（同一推理栈，差 = kgrpo − r1117）

| 项目 | 结果 |
|---|---|
| v12 | all/CER error_rate 12.78 → 12.86，+0.08 [-0.14, +0.32]<br>all/CER false_alarm_rate 2.46 → 3.46，+1.00 [+0.78, +1.23]* **更差**<br>all/CER miss_rate 2.76 → 2.13，-0.63 [-0.83, -0.43]* |
| es-asr | all/WER error_rate 5.25 → 4.91，-0.34 [-0.40, -0.28]*<br>all/WER miss_rate 0.02 → 0.03，+0.01 [+0.00, +0.02] |
| ts-es-v1 | all/WER error_rate 27.28 → 26.41，-0.88 [-1.29, -0.50]*<br>all/WER false_alarm_rate 3.21 → 9.26，+6.05 [+5.44, +6.63]* **更差**<br>all/WER miss_rate 3.86 → 2.31，-1.55 [-1.86, -1.24]* |
| babble4 | all/CER error_rate 1.90 → 2.28，+0.38 [+0.11, +0.68]* **更差**<br>all/CER false_alarm_rate 4.95 → 3.81，-1.14 [-1.77, -0.56]*<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 1.92 → 2.12，+0.20 [+0.05, +0.35]* **更差**<br>all/WER false_alarm_rate 1.75 → 2.00，+0.25 [+0.00, +0.65]<br>all/WER miss_rate 0.10 → 0.00，-0.10 [-0.31, +0.00] |
| babble3 | all/CER error_rate 1.90 → 2.28，+0.38 [+0.10, +0.68]* **更差**<br>all/CER false_alarm_rate 1.25 → 1.00，-0.25 [-1.42, +0.79]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 1.92 → 2.12，+0.20 [+0.05, +0.35]* **更差**<br>all/WER false_alarm_rate 1.75 → 2.00，+0.25 [+0.00, +0.63]<br>all/WER miss_rate 0.10 → 0.00，-0.10 [-0.31, +0.00] |
| customer-ts | all/CER error_rate 18.75 → 17.71，-1.04 [-2.50, +0.00]<br>all/CER false_alarm_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER miss_rate 20.00 → 20.00，+0.00 [+0.00, +0.00] |
| customer-asr | all/CER error_rate 20.15 → 20.29，+0.15 [-2.35, +2.63]<br>all/CER miss_rate 0.65 → 0.65，+0.00 [+0.00, +0.00] |
| ts-hotwords | CER 14.43 → 14.57（+0.14）<br>召回 89.9 → 93.2（+3.3） |
| ts-en | all/WER error_rate 13.59 → 13.64，+0.05 [-0.18, +0.29]<br>all/WER false_alarm_rate 6.20 → 7.61，+1.42 [+1.07, +1.75]* **更差**<br>all/WER miss_rate 2.94 → 2.21，-0.73 [-0.94, -0.52]* |
| first-other | all/CER error_rate 20.11 → 20.62，+0.51 [+0.19, +0.83]* **更差**<br>all/CER false_alarm_rate 2.58 → 2.31，-0.27 [-0.60, +0.05]<br>all/CER miss_rate 1.66 → 2.28，+0.62 [+0.28, +1.00]* **更差** |
| asr-benchmark | all/CER error_rate 3.56 → 3.57，+0.00 [-0.08, +0.06]<br>all/CER miss_rate 0.13 → 0.14，+0.01 [+0.00, +0.02]* **更差**<br>all/CER term_hit_rate 98.77 → 98.64，-0.13 [-0.23, -0.03]* **更差**<br>all/WER error_rate 7.56 → 7.44，-0.11 [-0.17, -0.06]*<br>all/WER false_alarm_rate 24.00 → 22.51，-1.49 [-1.90, -1.04]*<br>all/WER miss_rate 0.45 → 0.40，-0.05 [-0.10, +0.00] |
| hotword-bench | CER 3.27 → 2.99（-0.29）<br>召回 87.9 → 88.2（+0.4） |
| police | all/CER error_rate 0.27 → 0.27，-0.01 [-0.02, +0.01]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.75 → 99.68，-0.07 [-0.16, +0.01] |
| police-single | all/CER error_rate 0.38 → 0.36，-0.02 [-0.12, +0.10]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.08 → 99.17，+0.09 [-0.09, +0.28] |
| crosslingual | all/CER error_rate 34.29 → 30.60，-3.69 [-6.11, -1.22]*<br>all/CER false_alarm_rate 0.91 → 1.36，+0.45 [+0.00, +1.04]<br>all/CER miss_rate 14.62 → 13.56，-1.06 [-2.23, +0.00] |
| scene-asr | all/CER error_rate 4.72 → 2.03，-2.70 [-3.24, -2.22]*<br>all/CER miss_rate 0.85 → 0.35，-0.50 [-0.76, -0.28]* |
| crosslingual-en | all/WER error_rate 17.88 → 16.23，-1.64 [-2.39, -0.89]*<br>all/WER false_alarm_rate 2.73 → 3.86，+1.14 [+0.34, +2.04]* **更差**<br>all/WER miss_rate 9.17 → 7.58，-1.59 [-2.31, -0.91]* |
| scene-ts | all/CER error_rate 3.43 → 4.99，+1.55 [+0.72, +2.41]* **更差**<br>all/CER false_alarm_rate 21.43 → 28.57，+7.14 [-1.79, +17.24]<br>all/CER miss_rate 0.33 → 0.22，-0.11 [-0.27, +0.00] |
| ts-es-v2 | all/WER error_rate 27.49 → 26.42，-1.07 [-1.42, -0.71]*<br>all/WER false_alarm_rate 3.03 → 9.03，+6.00 [+5.52, +6.51]* **更差**<br>all/WER miss_rate 3.40 → 2.00，-1.40 [-1.66, -1.15]* |
| ctx-v1 | all/CER error_rate 7.87 → 1.31，-6.56 [-7.35, -5.79]*<br>all/CER miss_rate 5.32 → 0.06，-5.26 [-6.03, -4.49]* |
| babble-es | all/WER error_rate 7.37 → 6.53，-0.83 [-1.25, -0.40]*<br>all/WER false_alarm_rate 0.50 → 0.50，+0.00 [+0.00, +0.00]<br>all/WER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00] |
| hotword-bench-es | CER 4.97 → 4.70（-0.26）<br>召回 86.8 → 87.6（+0.8） |
| chime4 | all/WER error_rate 9.42 → 11.39，+1.96 [+0.98, +2.98]* **更差**<br>all/WER miss_rate 5.23 → 8.03，+2.80 [+1.74, +3.79]* **更差** |
| customer-hotwords | CER 16.30 → 17.32（+1.02）<br>召回 78.5 → 75.0（-3.5） |
| badcase | CER 10.00 → 10.56（+0.56）<br>召回 93.3 → 83.3（-10.0） |
| ts-hotwords-es | CER 26.33 → 27.11（+0.78）<br>召回 83.5 → 86.5（+3.0） |
| crosslingual-es | all/WER error_rate 23.11 → 20.09，-3.02 [-4.86, -1.37]*<br>all/WER false_alarm_rate 1.04 → 4.38，+3.33 [+1.60, +5.14]* **更差**<br>all/WER miss_rate 8.06 → 4.86，-3.19 [-4.64, -1.84]* |

显著更差的项：v12: all/CER false_alarm_rate；ts-es-v1: all/WER false_alarm_rate；babble4: all/CER error_rate, all/WER error_rate；babble3: all/CER error_rate, all/WER error_rate；ts-en: all/WER false_alarm_rate；first-other: all/CER error_rate, all/CER miss_rate；asr-benchmark: all/CER miss_rate, all/CER term_hit_rate；crosslingual-en: all/WER false_alarm_rate；scene-ts: all/CER error_rate；ts-es-v2: all/WER false_alarm_rate；chime4: all/WER error_rate, all/WER miss_rate；crosslingual-es: all/WER false_alarm_rate
