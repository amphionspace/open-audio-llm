# 交付评估：kgrpo 对 k0s2grpo（同一推理栈，差 = kgrpo − k0s2grpo）

| 项目 | 结果 |
|---|---|
| v12 | all/CER error_rate 12.37 → 12.86，+0.49 [+0.28, +0.71]* **更差**<br>all/CER false_alarm_rate 4.36 → 3.46，-0.89 [-1.12, -0.67]*<br>all/CER miss_rate 1.69 → 2.13，+0.44 [+0.28, +0.61]* **更差** |
| es-asr | all/WER error_rate 4.92 → 4.91，-0.01 [-0.07, +0.04]<br>all/WER miss_rate 0.03 → 0.03，+0.00 [+0.00, +0.00] |
| ts-es-v1 | all/WER error_rate 25.97 → 26.41，+0.44 [+0.07, +0.78]* **更差**<br>all/WER false_alarm_rate 4.54 → 9.26，+4.72 [+4.18, +5.24]* **更差**<br>all/WER miss_rate 3.39 → 2.31，-1.08 [-1.33, -0.83]* |
| babble4 | all/CER error_rate 2.16 → 2.28，+0.12 [-0.09, +0.37]<br>all/CER false_alarm_rate 3.13 → 3.81，+0.68 [+0.25, +1.13]* **更差**<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 2.28 → 2.12，-0.16 [-0.30, -0.03]*<br>all/WER false_alarm_rate 1.62 → 2.00，+0.38 [-0.13, +0.97]<br>all/WER miss_rate 0.10 → 0.00，-0.10 [-0.31, +0.00] |
| babble3 | all/CER error_rate 2.16 → 2.28，+0.12 [-0.09, +0.36]<br>all/CER false_alarm_rate 0.50 → 1.00，+0.50 [-0.71, +1.73]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 2.28 → 2.12，-0.16 [-0.30, -0.03]*<br>all/WER false_alarm_rate 1.62 → 2.00，+0.38 [-0.13, +0.91]<br>all/WER miss_rate 0.10 → 0.00，-0.10 [-0.31, +0.00] |
| customer-ts | all/CER error_rate 18.75 → 17.71，-1.04 [-2.50, +0.00]<br>all/CER false_alarm_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER miss_rate 20.00 → 20.00，+0.00 [+0.00, +0.00] |
| customer-asr | all/CER error_rate 26.42 → 20.29，-6.13 [-25.37, +4.11]<br>all/CER miss_rate 0.00 → 0.65，+0.65 [+0.00, +1.94] |
| ts-hotwords | CER 14.09 → 14.57（+0.48）<br>召回 93.4 → 93.2（-0.3） |
| ts-en | all/WER error_rate 13.19 → 13.64，+0.44 [+0.24, +0.66]* **更差**<br>all/WER false_alarm_rate 5.38 → 7.61，+2.24 [+1.87, +2.58]* **更差**<br>all/WER miss_rate 2.59 → 2.21，-0.39 [-0.58, -0.20]* |
| first-other | all/CER error_rate 19.80 → 20.62，+0.82 [+0.55, +1.10]* **更差**<br>all/CER false_alarm_rate 4.25 → 2.31，-1.94 [-2.37, -1.51]*<br>all/CER miss_rate 0.96 → 2.28，+1.32 [+1.01, +1.65]* **更差** |
| asr-benchmark | all/CER error_rate 3.57 → 3.57，-0.00 [-0.10, +0.06]<br>all/CER miss_rate 0.17 → 0.14，-0.03 [-0.04, -0.02]*<br>all/CER term_hit_rate 98.85 → 98.64，-0.21 [-0.31, -0.11]* **更差**<br>all/WER error_rate 7.43 → 7.44，+0.01 [-0.04, +0.05]<br>all/WER false_alarm_rate 21.37 → 22.51，+1.14 [+0.77, +1.52]* **更差**<br>all/WER miss_rate 0.53 → 0.40，-0.13 [-0.18, -0.09]* |
| hotword-bench | CER 3.13 → 2.99（-0.14）<br>召回 88.1 → 88.2（+0.1） |
| police | all/CER error_rate 0.27 → 0.27，-0.00 [-0.01, +0.01]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.72 → 99.68，-0.05 [-0.14, +0.05] |
| police-single | all/CER error_rate 0.39 → 0.36，-0.03 [-0.14, +0.07]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.11 → 99.17，+0.06 [-0.15, +0.28] |
| crosslingual | all/CER error_rate 43.04 → 30.60，-12.43 [-37.99, +1.53]<br>all/CER false_alarm_rate 2.50 → 1.36，-1.14 [-2.06, -0.33]*<br>all/CER miss_rate 14.55 → 13.56，-0.98 [-1.82, -0.15]* |
| scene-asr | all/CER error_rate 2.08 → 2.03，-0.06 [-0.23, +0.14]<br>all/CER miss_rate 0.54 → 0.35，-0.20 [-0.35, -0.07]* |
| crosslingual-en | all/WER error_rate 17.51 → 16.23，-1.27 [-2.11, -0.48]*<br>all/WER false_alarm_rate 2.84 → 3.86，+1.02 [+0.11, +1.98]* **更差**<br>all/WER miss_rate 8.41 → 7.58，-0.83 [-1.62, -0.08]* |
| scene-ts | all/CER error_rate 5.31 → 4.99，-0.32 [-1.27, +0.52]<br>all/CER false_alarm_rate 28.57 → 28.57，+0.00 [-9.59, +9.09]<br>all/CER miss_rate 0.27 → 0.22，-0.05 [-0.16, +0.00] |
| ts-es-v2 | all/WER error_rate 26.03 → 26.42，+0.39 [+0.08, +0.71]* **更差**<br>all/WER false_alarm_rate 4.27 → 9.03，+4.76 [+4.30, +5.23]* **更差**<br>all/WER miss_rate 2.95 → 2.00，-0.95 [-1.17, -0.73]* |
| ctx-v1 | all/CER error_rate 5.06 → 1.31，-3.75 [-4.40, -3.13]*<br>all/CER miss_rate 3.63 → 0.06，-3.57 [-4.22, -2.95]* |
| babble-es | all/WER error_rate 6.66 → 6.53，-0.12 [-0.52, +0.26]<br>all/WER false_alarm_rate 0.50 → 0.50，+0.00 [+0.00, +0.00]<br>all/WER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00] |
| hotword-bench-es | CER 4.71 → 4.70（-0.01）<br>召回 87.6 → 87.6（-0.0） |
| chime4 | all/WER error_rate 10.28 → 11.39，+1.10 [-0.02, +2.16]<br>all/WER miss_rate 7.42 → 8.03，+0.61 [-0.53, +1.59] |
| customer-hotwords | CER 15.10 → 17.32（+2.22）<br>召回 77.8 → 75.0（-2.8） |
| badcase | CER 3.89 → 10.56（+6.67）<br>召回 93.3 → 83.3（-10.0） |
| ts-hotwords-es | CER 25.11 → 27.11（+2.00）<br>召回 86.0 → 86.5（+0.5） |
| crosslingual-es | all/WER error_rate 24.32 → 20.09，-4.23 [-5.80, -2.82]*<br>all/WER false_alarm_rate 2.08 → 4.38，+2.29 [+0.85, +3.87]* **更差**<br>all/WER miss_rate 8.89 → 4.86，-4.03 [-5.55, -2.62]* |

显著更差的项：v12: all/CER error_rate, all/CER miss_rate；ts-es-v1: all/WER error_rate, all/WER false_alarm_rate；babble4: all/CER false_alarm_rate；ts-en: all/WER error_rate, all/WER false_alarm_rate；first-other: all/CER error_rate, all/CER miss_rate；asr-benchmark: all/CER term_hit_rate, all/WER false_alarm_rate；crosslingual-en: all/WER false_alarm_rate；ts-es-v2: all/WER error_rate, all/WER false_alarm_rate；crosslingual-es: all/WER false_alarm_rate
