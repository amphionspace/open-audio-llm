# 交付评估：k0s2grpo 对 r1117（同一推理栈，差 = k0s2grpo − r1117）

| 项目 | 结果 |
|---|---|
| v12 | all/CER error_rate 12.78 → 12.37，-0.41 [-0.64, -0.20]*<br>all/CER false_alarm_rate 2.46 → 4.36，+1.89 [+1.65, +2.14]* **更差**<br>all/CER miss_rate 2.76 → 1.69，-1.07 [-1.27, -0.87]* |
| es-asr | all/WER error_rate 5.25 → 4.92，-0.33 [-0.39, -0.26]*<br>all/WER miss_rate 0.02 → 0.03，+0.01 [+0.00, +0.02] |
| ts-es-v1 | all/WER error_rate 27.28 → 25.97，-1.32 [-1.67, -0.95]*<br>all/WER false_alarm_rate 3.21 → 4.54，+1.33 [+0.98, +1.68]* **更差**<br>all/WER miss_rate 3.86 → 3.39，-0.47 [-0.73, -0.21]* |
| babble4 | all/CER error_rate 1.90 → 2.16，+0.26 [+0.04, +0.49]* **更差**<br>all/CER false_alarm_rate 4.95 → 3.13，-1.82 [-2.41, -1.23]*<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 1.92 → 2.28，+0.36 [+0.22, +0.51]* **更差**<br>all/WER false_alarm_rate 1.75 → 1.62，-0.13 [-0.66, +0.38]<br>all/WER miss_rate 0.10 → 0.10，+0.00 [+0.00, +0.00] |
| babble3 | all/CER error_rate 1.90 → 2.16，+0.26 [+0.05, +0.48]* **更差**<br>all/CER false_alarm_rate 1.25 → 0.50，-0.75 [-1.63, +0.00]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 1.92 → 2.28，+0.36 [+0.21, +0.51]* **更差**<br>all/WER false_alarm_rate 1.75 → 1.62，-0.13 [-0.72, +0.38]<br>all/WER miss_rate 0.10 → 0.10，+0.00 [+0.00, +0.00] |
| customer-ts | all/CER error_rate 18.75 → 18.75，+0.00 [+0.00, +0.00]<br>all/CER false_alarm_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER miss_rate 20.00 → 20.00，+0.00 [+0.00, +0.00] |
| customer-asr | all/CER error_rate 20.15 → 26.42，+6.28 [-3.99, +25.83]<br>all/CER miss_rate 0.65 → 0.00，-0.65 [-1.94, +0.00] |
| ts-hotwords | CER 14.43 → 14.09（-0.34）<br>召回 89.9 → 93.4（+3.5） |
| ts-en | all/WER error_rate 13.59 → 13.19，-0.40 [-0.64, -0.16]*<br>all/WER false_alarm_rate 6.20 → 5.38，-0.82 [-1.12, -0.52]*<br>all/WER miss_rate 2.94 → 2.59，-0.35 [-0.55, -0.14]* |
| first-other | all/CER error_rate 20.11 → 19.80，-0.31 [-0.61, -0.01]*<br>all/CER false_alarm_rate 2.58 → 4.25，+1.66 [+1.23, +2.10]* **更差**<br>all/CER miss_rate 1.66 → 0.96，-0.70 [-0.97, -0.43]* |
| asr-benchmark | all/CER error_rate 3.56 → 3.57，+0.00 [-0.11, +0.12]<br>all/CER miss_rate 0.13 → 0.17，+0.05 [+0.04, +0.06]* **更差**<br>all/CER term_hit_rate 98.77 → 98.85，+0.08 [-0.02, +0.18]<br>all/WER error_rate 7.56 → 7.43，-0.12 [-0.19, -0.06]*<br>all/WER false_alarm_rate 24.00 → 21.37，-2.63 [-3.10, -2.15]*<br>all/WER miss_rate 0.45 → 0.53，+0.08 [+0.03, +0.14]* **更差** |
| hotword-bench | CER 3.27 → 3.13（-0.15）<br>召回 87.9 → 88.1（+0.3） |
| police | all/CER error_rate 0.27 → 0.27，-0.01 [-0.02, +0.00]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.75 → 99.72，-0.03 [-0.11, +0.05] |
| police-single | all/CER error_rate 0.38 → 0.39，+0.02 [-0.11, +0.17]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.08 → 99.11，+0.03 [-0.18, +0.21] |
| crosslingual | all/CER error_rate 34.29 → 43.04，+8.75 [-5.35, +34.42]<br>all/CER false_alarm_rate 0.91 → 2.50，+1.59 [+0.69, +2.62]* **更差**<br>all/CER miss_rate 14.62 → 14.55，-0.08 [-1.24, +0.99] |
| scene-asr | all/CER error_rate 4.72 → 2.08，-2.64 [-3.19, -2.16]*<br>all/CER miss_rate 0.85 → 0.54，-0.30 [-0.52, -0.11]* |
| crosslingual-en | all/WER error_rate 17.88 → 17.51，-0.37 [-1.25, +0.53]<br>all/WER false_alarm_rate 2.73 → 2.84，+0.11 [-0.57, +0.87]<br>all/WER miss_rate 9.17 → 8.41，-0.76 [-1.57, +0.08] |
| scene-ts | all/CER error_rate 3.43 → 5.31，+1.88 [+0.96, +2.92]* **更差**<br>all/CER false_alarm_rate 21.43 → 28.57，+7.14 [+1.52, +13.89]* **更差**<br>all/CER miss_rate 0.33 → 0.27，-0.05 [-0.16, +0.00] |
| ts-es-v2 | all/WER error_rate 27.49 → 26.03，-1.46 [-1.78, -1.14]*<br>all/WER false_alarm_rate 3.03 → 4.27，+1.24 [+0.94, +1.55]* **更差**<br>all/WER miss_rate 3.40 → 2.95，-0.45 [-0.69, -0.23]* |
| ctx-v1 | all/CER error_rate 7.87 → 5.06，-2.81 [-3.35, -2.28]*<br>all/CER miss_rate 5.32 → 3.63，-1.69 [-2.22, -1.20]* |
| babble-es | all/WER error_rate 7.37 → 6.66，-0.71 [-1.16, -0.24]*<br>all/WER false_alarm_rate 0.50 → 0.50，+0.00 [+0.00, +0.00]<br>all/WER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00] |
| hotword-bench-es | CER 4.97 → 4.71（-0.26）<br>召回 86.8 → 87.6（+0.8） |
| chime4 | all/WER error_rate 9.42 → 10.28，+0.86 [-0.19, +1.94]<br>all/WER miss_rate 5.23 → 7.42，+2.20 [+1.06, +3.33]* **更差** |
| customer-hotwords | CER 16.30 → 15.10（-1.19）<br>召回 78.5 → 77.8（-0.7） |
| badcase | CER 10.00 → 3.89（-6.11）<br>召回 93.3 → 93.3（+0.0） |
| ts-hotwords-es | CER 26.33 → 25.11（-1.22）<br>召回 83.5 → 86.0（+2.5） |
| crosslingual-es | all/WER error_rate 23.11 → 24.32，+1.21 [-0.69, +2.86]<br>all/WER false_alarm_rate 1.04 → 2.08，+1.04 [+0.00, +2.22]<br>all/WER miss_rate 8.06 → 8.89，+0.83 [-0.55, +2.19] |

显著更差的项：v12: all/CER false_alarm_rate；ts-es-v1: all/WER false_alarm_rate；babble4: all/CER error_rate, all/WER error_rate；babble3: all/CER error_rate, all/WER error_rate；first-other: all/CER false_alarm_rate；asr-benchmark: all/CER miss_rate, all/WER miss_rate；crosslingual: all/CER false_alarm_rate；scene-ts: all/CER error_rate, all/CER false_alarm_rate；ts-es-v2: all/WER false_alarm_rate；chime4: all/WER miss_rate
