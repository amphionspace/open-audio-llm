# 交付评估：pm-train-ts 对 r1117（同一推理栈，差 = pm-train-ts − r1117）

| 项目 | 结果 |
|---|---|
| v12 | all/CER error_rate 12.78 → 13.55，+0.77 [+0.57, +0.99]* **更差**<br>all/CER false_alarm_rate 2.46 → 2.20，-0.26 [-0.42, -0.10]*<br>all/CER miss_rate 2.76 → 3.57，+0.81 [+0.61, +1.02]* **更差** |
| es-asr | all/WER error_rate 5.25 → 5.36，+0.11 [+0.07, +0.15]* **更差**<br>all/WER miss_rate 0.02 → 0.02，+0.00 [+0.00, +0.00] |
| ts-es-v1 | all/WER error_rate 27.28 → 28.87，+1.58 [+0.89, +2.64]* **更差**<br>all/WER false_alarm_rate 3.21 → 2.82，-0.39 [-0.68, -0.09]*<br>all/WER miss_rate 3.86 → 4.01，+0.14 [-0.11, +0.39] |
| babble4 | all/CER error_rate 1.90 → 1.99，+0.09 [-0.11, +0.28]<br>all/CER false_alarm_rate 4.95 → 4.53，-0.42 [-0.97, +0.13]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 1.92 → 2.08，+0.16 [+0.03, +0.31]* **更差**<br>all/WER false_alarm_rate 1.75 → 1.62，-0.13 [-0.39, +0.00]<br>all/WER miss_rate 0.10 → 0.10，+0.00 [+0.00, +0.00] |
| babble3 | all/CER error_rate 1.90 → 1.99，+0.09 [-0.12, +0.29]<br>all/CER false_alarm_rate 1.25 → 1.00，-0.25 [-0.77, +0.00]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 1.92 → 2.08，+0.16 [+0.02, +0.31]* **更差**<br>all/WER false_alarm_rate 1.75 → 1.62，-0.13 [-0.39, +0.00]<br>all/WER miss_rate 0.10 → 0.10，+0.00 [+0.00, +0.00] |
| customer-ts | all/CER error_rate 18.75 → 18.75，+0.00 [+0.00, +0.00]<br>all/CER false_alarm_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER miss_rate 20.00 → 20.00，+0.00 [+0.00, +0.00] |
| customer-asr | all/CER error_rate 20.15 → 22.19，+2.04 [-0.55, +5.00]<br>all/CER miss_rate 0.65 → 0.65，+0.00 [+0.00, +0.00] |
| ts-hotwords | CER 14.43 → 15.14（+0.71）<br>召回 89.9 → 88.7（-1.2） |
| ts-en | all/WER error_rate 13.59 → 14.04，+0.45 [+0.21, +0.68]* **更差**<br>all/WER false_alarm_rate 6.20 → 6.48，+0.28 [+0.01, +0.56]* **更差**<br>all/WER miss_rate 2.94 → 3.01，+0.07 [-0.12, +0.29] |
| first-other | all/CER error_rate 20.11 → 21.85，+1.74 [+1.37, +2.12]* **更差**<br>all/CER false_alarm_rate 2.58 → 1.89，-0.70 [-1.01, -0.37]*<br>all/CER miss_rate 1.66 → 4.12，+2.46 [+2.03, +2.91]* **更差** |
| asr-benchmark | all/CER error_rate 3.56 → 3.76，+0.19 [+0.04, +0.36]* **更差**<br>all/CER miss_rate 0.13 → 0.13，+0.00 [-0.01, +0.01]<br>all/CER term_hit_rate 98.77 → 98.73，-0.03 [-0.14, +0.07]<br>all/WER error_rate 7.56 → 7.59，+0.03 [-0.02, +0.08]<br>all/WER false_alarm_rate 24.00 → 24.29，+0.29 [-0.06, +0.64]<br>all/WER miss_rate 0.45 → 0.36，-0.09 [-0.13, -0.06]* |
| hotword-bench | CER 3.27 → 3.27（+0.00）<br>召回 87.9 → 87.7（-0.2） |
| police | all/CER error_rate 0.27 → 0.29，+0.02 [+0.01, +0.03]* **更差**<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.75 → 99.74，-0.01 [-0.07, +0.06] |
| police-single | all/CER error_rate 0.38 → 0.50，+0.12 [-0.02, +0.27]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.08 → 98.90，-0.18 [-0.43, +0.03] |
| crosslingual | all/CER error_rate 34.29 → 38.67，+4.38 [+1.97, +6.81]* **更差**<br>all/CER false_alarm_rate 0.91 → 3.30，+2.39 [+1.43, +3.46]* **更差**<br>all/CER miss_rate 14.62 → 15.00，+0.38 [-0.61, +1.37] |
| scene-asr | all/CER error_rate 4.72 → 4.94，+0.22 [-0.14, +0.59]<br>all/CER miss_rate 0.85 → 0.83，-0.02 [-0.20, +0.15] |
| crosslingual-en | all/WER error_rate 17.88 → 18.27，+0.39 [-0.32, +1.16]<br>all/WER false_alarm_rate 2.73 → 4.32，+1.59 [+0.68, +2.64]* **更差**<br>all/WER miss_rate 9.17 → 9.24，+0.08 [-0.60, +0.83] |
| scene-ts | all/CER error_rate 3.43 → 6.06，+2.62 [+1.69, +3.61]* **更差**<br>all/CER false_alarm_rate 21.43 → 55.71，+34.29 [+22.66, +46.05]* **更差**<br>all/CER miss_rate 0.33 → 0.33，+0.00 [+0.00, +0.00] |
| ts-es-v2 | all/WER error_rate 27.49 → 29.00，+1.51 [+0.91, +2.45]* **更差**<br>all/WER false_alarm_rate 3.03 → 2.64，-0.39 [-0.64, -0.15]*<br>all/WER miss_rate 3.40 → 3.62，+0.21 [+0.00, +0.44] |
| ctx-v1 | all/CER error_rate 7.87 → 1.47，-6.40 [-7.20, -5.63]*<br>all/CER miss_rate 5.32 → 0.00，-5.32 [-6.09, -4.55]* |
| babble-es | all/WER error_rate 7.37 → 7.65，+0.29 [-0.03, +0.61]<br>all/WER false_alarm_rate 0.50 → 0.50，+0.00 [+0.00, +0.00]<br>all/WER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00] |
| hotword-bench-es | CER 4.97 → 5.08（+0.11）<br>召回 86.8 → 86.6（-0.1） |
| chime4 | all/WER error_rate 9.42 → 7.84，-1.58 [-2.40, -0.78]*<br>all/WER miss_rate 5.23 → 3.86，-1.36 [-2.20, -0.61]* |
| customer-hotwords | CER 16.30 → 17.24（+0.94）<br>召回 78.5 → 75.0（-3.5） |
| badcase | CER 10.00 → 25.00（+15.00）<br>召回 93.3 → 90.0（-3.3） |
| ts-hotwords-es | CER 26.33 → 27.68（+1.34）<br>召回 83.5 → 82.5（-1.0） |
| crosslingual-es | all/WER error_rate 23.11 → 24.80，+1.69 [+0.33, +3.28]* **更差**<br>all/WER false_alarm_rate 1.04 → 1.67，+0.62 [+0.00, +1.42]<br>all/WER miss_rate 8.06 → 9.44，+1.39 [+0.14, +2.71]* **更差** |

显著更差的项：v12: all/CER error_rate, all/CER miss_rate；es-asr: all/WER error_rate；ts-es-v1: all/WER error_rate；babble4: all/WER error_rate；babble3: all/WER error_rate；ts-en: all/WER error_rate, all/WER false_alarm_rate；first-other: all/CER error_rate, all/CER miss_rate；asr-benchmark: all/CER error_rate；police: all/CER error_rate；crosslingual: all/CER error_rate, all/CER false_alarm_rate；crosslingual-en: all/WER false_alarm_rate；scene-ts: all/CER error_rate, all/CER false_alarm_rate；ts-es-v2: all/WER error_rate；crosslingual-es: all/WER error_rate, all/WER miss_rate
