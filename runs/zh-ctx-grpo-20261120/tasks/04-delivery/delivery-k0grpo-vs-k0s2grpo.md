# 交付评估：k0s2grpo 对 k0grpo（同一推理栈，差 = k0s2grpo − k0grpo）

| 项目 | 结果 |
|---|---|
| v12 | all/CER error_rate 12.51 → 12.37，-0.14 [-0.38, +0.08]<br>all/CER false_alarm_rate 3.95 → 4.36，+0.41 [+0.19, +0.62]* **更差**<br>all/CER miss_rate 1.85 → 1.69，-0.15 [-0.29, -0.02]* |
| es-asr | all/WER error_rate 4.92 → 4.92，-0.00 [-0.06, +0.05]<br>all/WER miss_rate 0.02 → 0.03，+0.01 [+0.00, +0.02] |
| ts-es-v1 | all/WER error_rate 26.27 → 25.97，-0.30 [-0.66, +0.06]<br>all/WER false_alarm_rate 4.40 → 4.54，+0.14 [-0.23, +0.50]<br>all/WER miss_rate 3.98 → 3.39，-0.59 [-0.84, -0.34]* |
| babble4 | all/CER error_rate 2.05 → 2.16，+0.11 [-0.07, +0.32]<br>all/CER false_alarm_rate 4.91 → 3.13，-1.78 [-2.31, -1.22]*<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 2.15 → 2.28，+0.13 [-0.03, +0.29]<br>all/WER false_alarm_rate 1.62 → 1.62，+0.00 [-0.50, +0.50]<br>all/WER miss_rate 0.10 → 0.10，+0.00 [+0.00, +0.00] |
| babble3 | all/CER error_rate 2.05 → 2.16，+0.11 [-0.08, +0.31]<br>all/CER false_alarm_rate 0.25 → 0.50，+0.25 [+0.00, +0.79]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 2.15 → 2.28，+0.13 [-0.03, +0.28]<br>all/WER false_alarm_rate 1.62 → 1.62，+0.00 [-0.50, +0.50]<br>all/WER miss_rate 0.10 → 0.10，+0.00 [+0.00, +0.00] |
| customer-ts | all/CER error_rate 5.21 → 18.75，+13.54 [+0.00, +53.42]<br>all/CER false_alarm_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER miss_rate 0.00 → 20.00，+20.00 [+0.00, +60.00] |
| customer-asr | all/CER error_rate 17.81 → 26.42，+8.61 [-1.67, +28.40]<br>all/CER miss_rate 0.65 → 0.00，-0.65 [-1.94, +0.00] |
| ts-hotwords | CER 14.06 → 14.09（+0.03）<br>召回 93.6 → 93.4（-0.2） |
| ts-en | all/WER error_rate 13.39 → 13.19，-0.20 [-0.44, +0.04]<br>all/WER false_alarm_rate 6.30 → 5.38，-0.92 [-1.21, -0.64]*<br>all/WER miss_rate 2.33 → 2.59，+0.26 [+0.07, +0.46]* **更差** |
| first-other | all/CER error_rate 19.97 → 19.80，-0.16 [-0.37, +0.06]<br>all/CER false_alarm_rate 4.77 → 4.25，-0.52 [-0.95, -0.15]*<br>all/CER miss_rate 0.60 → 0.96，+0.36 [+0.18, +0.56]* **更差** |
| asr-benchmark | all/CER error_rate 3.59 → 3.57，-0.02 [-0.04, -0.00]*<br>all/CER miss_rate 0.13 → 0.17，+0.05 [+0.04, +0.06]* **更差**<br>all/CER term_hit_rate 98.67 → 98.85，+0.17 [+0.08, +0.27]*<br>all/WER error_rate 7.45 → 7.43，-0.02 [-0.06, +0.03]<br>all/WER false_alarm_rate 24.29 → 21.37，-2.92 [-3.41, -2.47]*<br>all/WER miss_rate 0.37 → 0.53，+0.16 [+0.12, +0.21]* **更差** |
| hotword-bench | CER 3.01 → 3.13（+0.11）<br>召回 88.3 → 88.1（-0.2） |
| police | all/CER error_rate 0.26 → 0.27，+0.01 [-0.01, +0.02]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.68 → 99.72，+0.05 [-0.04, +0.13] |
| police-single | all/CER error_rate 0.37 → 0.39，+0.02 [-0.04, +0.09]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.14 → 99.11，-0.03 [-0.21, +0.15] |
| crosslingual | all/CER error_rate 32.77 → 43.04，+10.27 [-3.96, +36.13]<br>all/CER false_alarm_rate 1.70 → 2.50，+0.80 [+0.00, +1.65]<br>all/CER miss_rate 14.62 → 14.55，-0.08 [-1.07, +0.83] |
| scene-asr | all/CER error_rate 2.24 → 2.08，-0.15 [-0.27, -0.03]*<br>all/CER miss_rate 0.74 → 0.54，-0.20 [-0.37, -0.04]* |
| crosslingual-en | all/WER error_rate 18.46 → 17.51，-0.95 [-1.76, -0.12]*<br>all/WER false_alarm_rate 3.30 → 2.84，-0.45 [-1.44, +0.45]<br>all/WER miss_rate 8.94 → 8.41，-0.53 [-1.26, +0.23] |
| scene-ts | all/CER error_rate 3.92 → 5.31，+1.39 [+0.68, +2.20]* **更差**<br>all/CER false_alarm_rate 31.43 → 28.57，-2.86 [-8.96, +2.70]<br>all/CER miss_rate 0.27 → 0.27，+0.00 [+0.00, +0.00] |
| ts-es-v2 | all/WER error_rate 26.35 → 26.03，-0.33 [-0.66, -0.00]*<br>all/WER false_alarm_rate 4.33 → 4.27，-0.06 [-0.36, +0.23]<br>all/WER miss_rate 3.51 → 2.95，-0.56 [-0.78, -0.33]* |
| ctx-v1 | all/CER error_rate 6.50 → 5.06，-1.44 [-1.91, -0.99]*<br>all/CER miss_rate 5.02 → 3.63，-1.39 [-1.79, -0.95]* |
| babble-es | all/WER error_rate 6.67 → 6.66，-0.01 [-0.40, +0.38]<br>all/WER false_alarm_rate 0.50 → 0.50，+0.00 [+0.00, +0.00]<br>all/WER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00] |
| hotword-bench-es | CER 4.66 → 4.71（+0.05）<br>召回 87.9 → 87.6（-0.3） |
| chime4 | all/WER error_rate 8.97 → 10.28，+1.32 [+0.45, +2.29]* **更差**<br>all/WER miss_rate 5.53 → 7.42，+1.89 [+0.98, +2.88]* **更差** |
| customer-hotwords | CER 15.78 → 15.10（-0.68）<br>召回 75.7 → 77.8（+2.1） |
| badcase | CER 8.33 → 3.89（-4.44）<br>召回 90.0 → 93.3（+3.3） |
| ts-hotwords-es | CER 25.66 → 25.11（-0.56）<br>召回 85.2 → 86.0（+0.8） |
| crosslingual-es | all/WER error_rate 24.57 → 24.32，-0.25 [-1.72, +1.16]<br>all/WER false_alarm_rate 1.88 → 2.08，+0.21 [-0.64, +1.20]<br>all/WER miss_rate 9.72 → 8.89，-0.83 [-2.12, +0.42] |

显著更差的项：v12: all/CER false_alarm_rate；ts-en: all/WER miss_rate；first-other: all/CER miss_rate；asr-benchmark: all/CER miss_rate, all/WER miss_rate；scene-ts: all/CER error_rate；chime4: all/WER error_rate, all/WER miss_rate
