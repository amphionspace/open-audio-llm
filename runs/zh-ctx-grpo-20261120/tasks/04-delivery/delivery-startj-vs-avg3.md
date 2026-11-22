# 交付评估：avg3 对 startj（同一推理栈，差 = avg3 − startj）

| 项目 | 结果 |
|---|---|
| v12 | all/CER error_rate 13.60 → 12.88，-0.73 [-0.92, -0.54]*<br>all/CER false_alarm_rate 2.67 → 2.73，+0.07 [-0.11, +0.24]<br>all/CER miss_rate 3.17 → 2.86，-0.32 [-0.47, -0.17]* |
| es-asr | all/WER error_rate 74.10 → 70.60，-3.50 [-4.23, -2.84]*<br>all/WER miss_rate 0.32 → 0.38，+0.06 [+0.02, +0.10]* **更差** |
| ts-es-v1 | all/WER error_rate 84.83 → 72.40，-12.43 [-13.91, -11.22]*<br>all/WER false_alarm_rate 2.75 → 4.44，+1.70 [+1.37, +2.04]* **更差**<br>all/WER miss_rate 3.55 → 2.97，-0.57 [-0.78, -0.40]* |
| babble4 | all/CER error_rate 2.06 → 1.96，-0.10 [-0.25, +0.05]<br>all/CER false_alarm_rate 6.68 → 4.78，-1.90 [-2.60, -1.23]*<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 2.50 → 2.32，-0.18 [-0.29, -0.08]*<br>all/WER false_alarm_rate 2.62 → 2.62，+0.00 [-0.92, +0.98]<br>all/WER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00] |
| babble3 | all/CER error_rate 2.06 → 1.96，-0.10 [-0.25, +0.04]<br>all/CER false_alarm_rate 0.50 → 0.75，+0.25 [-0.87, +1.43]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 2.50 → 2.32，-0.18 [-0.29, -0.08]*<br>all/WER false_alarm_rate 2.62 → 2.62，+0.00 [-1.01, +0.97]<br>all/WER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00] |
| customer-ts | all/CER error_rate 5.21 → 5.21，+0.00 [+0.00, +0.00]<br>all/CER false_alarm_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00] |
| customer-asr | all/CER error_rate 23.36 → 28.32，+4.96 [-1.47, +17.50]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00] |
| ts-hotwords | CER 15.24 → 14.32（-0.91）<br>召回 92.5 → 92.1（-0.4） |
| ts-en | all/WER error_rate 14.27 → 13.33，-0.94 [-1.13, -0.75]*<br>all/WER false_alarm_rate 7.21 → 5.64，-1.57 [-1.91, -1.25]*<br>all/WER miss_rate 3.05 → 2.75，-0.30 [-0.46, -0.14]* |
| first-other | all/CER error_rate 21.69 → 19.58，-2.11 [-2.42, -1.81]*<br>all/CER false_alarm_rate 3.65 → 3.60，-0.05 [-0.40, +0.30]<br>all/CER miss_rate 1.10 → 1.06，-0.04 [-0.22, +0.14] |
| asr-benchmark | all/CER error_rate 3.67 → 3.54，-0.13 [-0.15, -0.11]*<br>all/CER miss_rate 0.04 → 0.11，+0.06 [+0.05, +0.07]* **更差**<br>all/CER term_hit_rate 98.64 → 98.66，+0.02 [-0.05, +0.09]<br>all/WER error_rate 7.38 → 7.29，-0.09 [-0.12, -0.06]*<br>all/WER false_alarm_rate 39.79 → 24.57，-15.22 [-16.20, -14.29]*<br>all/WER miss_rate 0.11 → 0.21，+0.11 [+0.07, +0.14]* **更差** |
| hotword-bench | CER 3.77 → 3.42（-0.36）<br>召回 84.4 → 87.3（+2.9） |
| police | all/CER error_rate 0.26 → 0.26，-0.00 [-0.01, +0.01]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.64 → 99.68，+0.04 [-0.03, +0.10] |
| police-single | all/CER error_rate 0.33 → 0.39，+0.06 [+0.00, +0.15]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.20 → 99.11，-0.09 [-0.25, +0.03] |
| crosslingual | all/CER error_rate 40.29 → 34.39，-5.91 [-8.06, -3.89]*<br>all/CER false_alarm_rate 0.45 → 1.02，+0.57 [+0.11, +1.11]* **更差**<br>all/CER miss_rate 23.41 → 17.42，-5.98 [-7.31, -4.70]* |
| scene-asr | all/CER error_rate 9.01 → 7.57，-1.44 [-1.86, -1.04]*<br>all/CER miss_rate 0.28 → 0.39，+0.11 [+0.00, +0.22] |
| crosslingual-en | all/WER error_rate 20.58 → 18.07，-2.51 [-3.30, -1.76]*<br>all/WER false_alarm_rate 2.50 → 1.48，-1.02 [-1.80, -0.34]*<br>all/WER miss_rate 12.58 → 10.30，-2.27 [-3.14, -1.46]* |
| scene-ts | all/CER error_rate 3.06 → 3.26，+0.21 [-0.16, +0.60]<br>all/CER false_alarm_rate 22.86 → 25.71，+2.86 [+0.00, +7.35]<br>all/CER miss_rate 0.33 → 0.33，+0.00 [+0.00, +0.00] |
| ts-es-v2 | all/WER error_rate 85.71 → 73.12，-12.60 [-13.87, -11.56]*<br>all/WER false_alarm_rate 2.60 → 4.20，+1.60 [+1.31, +1.89]* **更差**<br>all/WER miss_rate 3.17 → 2.61，-0.56 [-0.74, -0.39]* |
| ctx-v1 | all/CER error_rate 1.61 → 1.40，-0.21 [-0.38, -0.08]*<br>all/CER miss_rate 0.37 → 0.15，-0.22 [-0.40, -0.06]* |
| babble-es | all/WER error_rate 112.19 → 103.71，-8.48 [-11.21, -5.97]*<br>all/WER false_alarm_rate 0.50 → 0.50，+0.00 [+0.00, +0.00]<br>all/WER miss_rate 0.70 → 1.80，+1.10 [+0.50, +1.79]* **更差** |
| hotword-bench-es | CER 57.19 → 52.86（-4.33）<br>召回 43.4 → 47.0（+3.6） |
| chime4 | all/WER error_rate 7.16 → 8.26，+1.10 [+0.33, +1.85]* **更差**<br>all/WER miss_rate 2.50 → 4.55，+2.05 [+1.06, +2.95]* **更差** |
| customer-hotwords | CER 17.06 → 17.06（+0.00）<br>召回 74.3 → 72.2（-2.1） |
| badcase | CER 30.56 → 17.22（-13.33）<br>召回 60.0 → 70.0（+10.0） |
| ts-hotwords-es | CER 81.51 → 70.92（-10.59）<br>召回 63.3 → 67.8（+4.5） |
| crosslingual-es | all/WER error_rate 75.06 → 65.29，-9.77 [-12.01, -7.49]*<br>all/WER false_alarm_rate 0.62 → 0.83，+0.21 [+0.00, +0.65]<br>all/WER miss_rate 20.42 → 15.00，-5.42 [-7.12, -3.83]* |

显著更差的项：es-asr: all/WER miss_rate；ts-es-v1: all/WER false_alarm_rate；asr-benchmark: all/CER miss_rate, all/WER miss_rate；crosslingual: all/CER false_alarm_rate；ts-es-v2: all/WER false_alarm_rate；babble-es: all/WER miss_rate；chime4: all/WER error_rate, all/WER miss_rate
