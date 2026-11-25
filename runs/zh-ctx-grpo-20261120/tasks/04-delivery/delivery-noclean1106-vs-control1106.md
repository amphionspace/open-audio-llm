# 交付评估：control1106 对 noclean1106（同一推理栈，差 = control1106 − noclean1106）

| 项目 | 结果 |
|---|---|
| v12 | all/CER error_rate 14.51 → 13.32，-1.19 [-1.46, -0.93]*<br>all/CER false_alarm_rate 2.25 → 3.51，+1.25 [+1.04, +1.48]* **更差**<br>all/CER miss_rate 4.50 → 3.03，-1.47 [-1.71, -1.23]* |
| es-asr | all/WER error_rate 64.30 → 71.00，+6.71 [+6.07, +7.33]* **更差**<br>all/WER miss_rate 0.21 → 0.33，+0.12 [+0.07, +0.17]* **更差** |
| ts-es-v1 | all/WER error_rate 70.09 → 81.84，+11.76 [+10.80, +12.73]* **更差**<br>all/WER false_alarm_rate 1.65 → 1.99，+0.34 [+0.06, +0.60]* **更差**<br>all/WER miss_rate 4.20 → 3.91，-0.29 [-0.56, -0.01]* |
| babble4 | all/CER error_rate 2.15 → 2.24，+0.09 [-0.20, +0.39]<br>all/CER false_alarm_rate 7.28 → 6.85，-0.42 [-1.02, +0.13]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 2.39 → 2.46，+0.07 [-0.09, +0.23]<br>all/WER false_alarm_rate 2.25 → 2.25，+0.00 [-0.50, +0.49]<br>all/WER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00] |
| babble3 | all/CER error_rate 2.15 → 2.24，+0.09 [-0.20, +0.40]<br>all/CER false_alarm_rate 0.50 → 0.50，+0.00 [+0.00, +0.00]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 2.39 → 2.46，+0.07 [-0.09, +0.22]<br>all/WER false_alarm_rate 2.25 → 2.25，+0.00 [-0.50, +0.50]<br>all/WER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00] |
| customer-ts | all/CER error_rate 4.17 → 4.17，+0.00 [+0.00, +0.00]<br>all/CER false_alarm_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00] |
| customer-asr | all/CER error_rate 14.60 → 17.81，+3.21 [+1.17, +5.77]* **更差**<br>all/CER miss_rate 0.65 → 0.65，+0.00 [+0.00, +0.00] |
| ts-hotwords | CER 15.82 → 15.10（-0.72）<br>召回 92.1 → 92.2（+0.1） |
| ts-en | all/WER error_rate 14.97 → 14.60，-0.37 [-0.63, -0.11]*<br>all/WER false_alarm_rate 6.64 → 6.22，-0.42 [-0.70, -0.16]*<br>all/WER miss_rate 4.15 → 3.45，-0.71 [-0.94, -0.50]* |
| first-other | all/CER error_rate 22.16 → 22.20，+0.04 [-0.33, +0.40]<br>all/CER false_alarm_rate 3.10 → 4.10，+0.99 [+0.58, +1.43]* **更差**<br>all/CER miss_rate 1.22 → 0.74，-0.48 [-0.72, -0.26]* |
| asr-benchmark | all/CER error_rate 3.72 → 3.67，-0.05 [-0.07, -0.02]*<br>all/CER miss_rate 0.04 → 0.05，+0.01 [+0.00, +0.01]* **更差**<br>all/CER term_hit_rate 98.65 → 98.78，+0.13 [+0.03, +0.23]*<br>all/WER error_rate 7.44 → 7.42，-0.02 [-0.06, +0.02]<br>all/WER false_alarm_rate 41.59 → 38.38，-3.21 [-3.85, -2.58]*<br>all/WER miss_rate 0.09 → 0.14，+0.06 [+0.03, +0.08]* **更差** |
| hotword-bench | CER 4.04 → 3.83（-0.21）<br>召回 84.3 → 85.0（+0.8） |
| police | all/CER error_rate 0.30 → 0.27，-0.02 [-0.04, -0.01]*<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.59 → 99.70，+0.10 [+0.02, +0.19]* |
| police-single | all/CER error_rate 0.37 → 0.35，-0.02 [-0.11, +0.10]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.08 → 99.17，+0.09 [-0.12, +0.31] |
| crosslingual | all/CER error_rate 40.27 → 39.11，-1.16 [-3.86, +1.14]<br>all/CER false_alarm_rate 1.36 → 1.59，+0.23 [-0.42, +0.90]<br>all/CER miss_rate 25.91 → 22.65，-3.26 [-4.59, -1.94]* |
| scene-asr | all/CER error_rate 4.90 → 4.97，+0.07 [-0.23, +0.39]<br>all/CER miss_rate 0.41 → 0.48，+0.07 [-0.07, +0.20] |
| crosslingual-en | all/WER error_rate 23.29 → 21.86，-1.42 [-2.48, -0.45]*<br>all/WER false_alarm_rate 3.18 → 2.39，-0.80 [-1.59, -0.11]*<br>all/WER miss_rate 16.74 → 13.94，-2.80 [-3.90, -1.77]* |
| scene-ts | all/CER error_rate 3.86 → 3.38，-0.48 [-1.00, +0.02]<br>all/CER false_alarm_rate 22.86 → 24.29，+1.43 [-3.17, +6.56]<br>all/CER miss_rate 0.44 → 0.27，-0.16 [-0.38, +0.00] |
| ts-es-v2 | all/WER error_rate 70.53 → 82.78，+12.25 [+11.32, +13.13]* **更差**<br>all/WER false_alarm_rate 1.48 → 1.88，+0.39 [+0.17, +0.61]* **更差**<br>all/WER miss_rate 3.80 → 3.50，-0.30 [-0.53, -0.05]* |
| ctx-v1 | all/CER error_rate 9.79 → 7.61，-2.18 [-2.75, -1.64]*<br>all/CER miss_rate 7.48 → 5.36，-2.12 [-2.65, -1.60]* |
| babble-es | all/WER error_rate 99.33 → 110.60，+11.27 [+7.69, +14.81]* **更差**<br>all/WER false_alarm_rate 0.50 → 0.50，+0.00 [+0.00, +0.00]<br>all/WER miss_rate 0.30 → 0.20，-0.10 [-0.48, +0.20] |
| hotword-bench-es | CER 48.97 → 57.15（+8.18）<br>召回 46.9 → 43.1（-3.8） |
| chime4 | all/WER error_rate 6.04 → 7.65，+1.61 [+0.92, +2.33]* **更差**<br>all/WER miss_rate 0.76 → 3.03，+2.27 [+1.44, +3.11]* **更差** |
| customer-hotwords | CER 14.16 → 15.87（+1.71）<br>召回 75.0 → 76.4（+1.4） |
| badcase | CER 21.11 → 27.22（+6.11）<br>召回 73.3 → 63.3（-10.0） |
| ts-hotwords-es | CER 67.06 → 78.53（+11.47）<br>召回 67.5 → 61.4（-6.1） |
| crosslingual-es | all/WER error_rate 68.48 → 75.24，+6.76 [+4.42, +9.28]* **更差**<br>all/WER false_alarm_rate 0.00 → 0.21，+0.21 [+0.00, +0.65]<br>all/WER miss_rate 22.50 → 20.28，-2.22 [-3.50, -0.97]* |

显著更差的项：v12: all/CER false_alarm_rate；es-asr: all/WER error_rate, all/WER miss_rate；ts-es-v1: all/WER error_rate, all/WER false_alarm_rate；customer-asr: all/CER error_rate；first-other: all/CER false_alarm_rate；asr-benchmark: all/CER miss_rate, all/WER miss_rate；ts-es-v2: all/WER error_rate, all/WER false_alarm_rate；babble-es: all/WER error_rate；chime4: all/WER error_rate, all/WER miss_rate；crosslingual-es: all/WER error_rate
