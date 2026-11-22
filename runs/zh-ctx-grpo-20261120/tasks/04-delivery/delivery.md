# 交付评估：avg3 对 11.17（同一节点、同一推理栈，差 = avg3 − 11.17）

| 项目 | 结果 |
|---|---|
| v12 | all/CER error_rate 12.78 → 12.88，+0.10 [-0.16, +0.38]<br>all/CER false_alarm_rate 2.46 → 2.73，+0.27 [+0.06, +0.48]* **更差**<br>all/CER miss_rate 2.76 → 2.86，+0.10 [-0.11, +0.32] |
| es-asr | all/WER error_rate 5.25 → 70.60，+65.35 [+64.38, +66.35]* **更差**<br>all/WER miss_rate 0.02 → 0.38，+0.36 [+0.28, +0.44]* **更差** |
| ts-es-v1 | all/WER error_rate 27.28 → 72.40，+45.11 [+43.96, +46.33]* **更差**<br>all/WER false_alarm_rate 3.21 → 4.44，+1.23 [+0.74, +1.75]* **更差**<br>all/WER miss_rate 3.86 → 2.97，-0.89 [-1.21, -0.59]* |
| babble4 | all/CER error_rate 1.90 → 1.96，+0.06 [-0.20, +0.30]<br>all/CER false_alarm_rate 4.95 → 4.78，-0.17 [-0.72, +0.38]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 1.92 → 2.32，+0.39 [+0.19, +0.60]* **更差**<br>all/WER false_alarm_rate 1.75 → 2.62，+0.87 [+0.13, +1.66]* **更差**<br>all/WER miss_rate 0.10 → 0.00，-0.10 [-0.31, +0.00] |
| babble3 | all/CER error_rate 1.90 → 1.96，+0.06 [-0.18, +0.30]<br>all/CER false_alarm_rate 1.25 → 0.75，-0.50 [-1.72, +0.74]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 1.92 → 2.32，+0.39 [+0.20, +0.59]* **更差**<br>all/WER false_alarm_rate 1.75 → 2.62，+0.87 [+0.13, +1.62]* **更差**<br>all/WER miss_rate 0.10 → 0.00，-0.10 [-0.31, +0.00] |
| customer-ts | all/CER error_rate 18.75 → 5.21，-13.54 [-53.42, +0.00]<br>all/CER false_alarm_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER miss_rate 20.00 → 0.00，-20.00 [-60.00, +0.00] |
| customer-asr | all/CER error_rate 20.15 → 28.32，+8.18 [-3.25, +30.39]<br>all/CER miss_rate 0.65 → 0.00，-0.65 [-1.94, +0.00] |
| ts-hotwords | CER 14.43 → 14.32（-0.11）<br>召回 89.9 → 92.1（+2.2） |
| ts-en | all/WER error_rate 13.59 → 13.33，-0.26 [-0.50, -0.02]*<br>all/WER false_alarm_rate 6.20 → 5.64，-0.56 [-0.89, -0.23]*<br>all/WER miss_rate 2.94 → 2.75，-0.19 [-0.40, +0.03] |
| first-other | all/CER error_rate 20.11 → 19.58，-0.53 [-0.85, -0.23]*<br>all/CER false_alarm_rate 2.58 → 3.60，+1.02 [+0.61, +1.46]* **更差**<br>all/CER miss_rate 1.66 → 1.06，-0.60 [-0.88, -0.32]* |
| asr-benchmark | all/CER error_rate 3.56 → 3.54，-0.02 [-0.10, +0.03]<br>all/CER miss_rate 0.13 → 0.11，-0.02 [-0.03, -0.02]*<br>all/CER term_hit_rate 98.77 → 98.66，-0.11 [-0.23, +0.01]<br>all/WER error_rate 7.56 → 7.29，-0.27 [-0.33, -0.21]*<br>all/WER false_alarm_rate 24.00 → 24.57，+0.57 [+0.11, +1.06]* **更差**<br>all/WER miss_rate 0.45 → 0.21，-0.24 [-0.29, -0.18]* |
| hotword-bench | CER 3.27 → 3.42（+0.14）<br>召回 87.9 → 87.3（-0.5） |
| police | all/CER error_rate 0.27 → 0.26，-0.02 [-0.03, +0.00]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.75 → 99.68，-0.07 [-0.18, +0.02] |
| police-single | all/CER error_rate 0.38 → 0.39，+0.02 [-0.13, +0.17]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.08 → 99.11，+0.03 [-0.21, +0.28] |
| crosslingual | all/CER error_rate 34.29 → 34.39，+0.10 [-2.64, +2.86]<br>all/CER false_alarm_rate 0.91 → 1.02，+0.11 [-0.35, +0.59]<br>all/CER miss_rate 14.62 → 17.42，+2.80 [+1.44, +4.15]* **更差** |
| scene-asr | all/CER error_rate 4.72 → 7.57，+2.84 [+2.17, +3.52]* **更差**<br>all/CER miss_rate 0.85 → 0.39，-0.46 [-0.72, -0.24]* |
| crosslingual-en | all/WER error_rate 17.88 → 18.07，+0.19 [-0.79, +1.23]<br>all/WER false_alarm_rate 2.73 → 1.48，-1.25 [-2.15, -0.34]*<br>all/WER miss_rate 9.17 → 10.30，+1.14 [+0.30, +2.03]* **更差** |
| scene-ts | all/CER error_rate 3.43 → 3.26，-0.17 [-0.80, +0.51]<br>all/CER false_alarm_rate 21.43 → 25.71，+4.29 [-5.80, +14.87]<br>all/CER miss_rate 0.33 → 0.33，+0.00 [-0.16, +0.16] |
| ts-es-v2 | all/WER error_rate 27.49 → 73.12，+45.63 [+44.62, +46.70]* **更差**<br>all/WER false_alarm_rate 3.03 → 4.20，+1.17 [+0.77, +1.59]* **更差**<br>all/WER miss_rate 3.40 → 2.61，-0.80 [-1.08, -0.53]* |
| ctx-v1 | all/CER error_rate 7.87 → 1.40，-6.47 [-7.25, -5.70]*<br>all/CER miss_rate 5.32 → 0.15，-5.17 [-5.94, -4.43]* |
| babble-es | all/WER error_rate 7.37 → 103.71，+96.35 [+93.07, +99.47]* **更差**<br>all/WER false_alarm_rate 0.50 → 0.50，+0.00 [+0.00, +0.00]<br>all/WER miss_rate 0.00 → 1.80，+1.80 [+1.01, +2.67]* **更差** |
| hotword-bench-es | CER 4.97 → 52.86（+47.89）<br>召回 86.8 → 47.0（-39.8） |
| chime4 | all/WER error_rate 9.42 → 8.26，-1.16 [-2.07, -0.29]*<br>all/WER miss_rate 5.23 → 4.55，-0.68 [-1.67, +0.23] |
| customer-hotwords | CER 16.30 → 17.06（+0.77）<br>召回 78.5 → 72.2（-6.2） |
| badcase | CER 10.00 → 17.22（+7.22）<br>召回 93.3 → 70.0（-23.3） |
| ts-hotwords-es | CER 26.33 → 70.92（+44.58）<br>召回 83.5 → 67.8（-15.7） |
| crosslingual-es | all/WER error_rate 23.11 → 65.29，+42.19 [+38.78, +45.57]* **更差**<br>all/WER false_alarm_rate 1.04 → 0.83，-0.21 [-1.47, +1.02]<br>all/WER miss_rate 8.06 → 15.00，+6.94 [+4.80, +9.23]* **更差** |

显著更差的项：v12: all/CER false_alarm_rate；es-asr: all/WER error_rate, all/WER miss_rate；ts-es-v1: all/WER error_rate, all/WER false_alarm_rate；babble4: all/WER error_rate, all/WER false_alarm_rate；babble3: all/WER error_rate, all/WER false_alarm_rate；first-other: all/CER false_alarm_rate；asr-benchmark: all/WER false_alarm_rate；crosslingual: all/CER miss_rate；scene-asr: all/CER error_rate；crosslingual-en: all/WER miss_rate；ts-es-v2: all/WER error_rate, all/WER false_alarm_rate；babble-es: all/WER error_rate, all/WER miss_rate；crosslingual-es: all/WER error_rate, all/WER miss_rate
