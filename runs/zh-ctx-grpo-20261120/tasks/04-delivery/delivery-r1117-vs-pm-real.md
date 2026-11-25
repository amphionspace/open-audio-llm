# 交付评估：pm-real 对 r1117（同一推理栈，差 = pm-real − r1117）

| 项目 | 结果 |
|---|---|
| v12 | all/CER error_rate 12.78 → 13.26，+0.48 [+0.29, +0.66]* **更差**<br>all/CER false_alarm_rate 2.46 → 2.67，+0.21 [+0.06, +0.36]* **更差**<br>all/CER miss_rate 2.76 → 2.88，+0.12 [-0.03, +0.27] |
| es-asr | all/WER error_rate 5.25 → 5.37，+0.12 [+0.08, +0.16]* **更差**<br>all/WER miss_rate 0.02 → 0.02，+0.00 [+0.00, +0.00] |
| ts-es-v1 | all/WER error_rate 27.28 → 27.68，+0.39 [+0.10, +0.68]* **更差**<br>all/WER false_alarm_rate 3.21 → 3.52，+0.31 [+0.05, +0.58]* **更差**<br>all/WER miss_rate 3.86 → 3.74，-0.13 [-0.36, +0.10] |
| babble4 | all/CER error_rate 1.90 → 2.08，+0.17 [+0.02, +0.34]* **更差**<br>all/CER false_alarm_rate 4.95 → 4.70，-0.25 [-0.67, +0.13]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 1.92 → 2.27，+0.35 [+0.20, +0.50]* **更差**<br>all/WER false_alarm_rate 1.75 → 1.62，-0.13 [-0.39, +0.00]<br>all/WER miss_rate 0.10 → 0.00，-0.10 [-0.31, +0.00] |
| babble3 | all/CER error_rate 1.90 → 2.08，+0.17 [+0.02, +0.34]* **更差**<br>all/CER false_alarm_rate 1.25 → 1.00，-0.25 [-1.04, +0.53]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 1.92 → 2.27，+0.35 [+0.20, +0.50]* **更差**<br>all/WER false_alarm_rate 1.75 → 1.62，-0.13 [-0.39, +0.00]<br>all/WER miss_rate 0.10 → 0.00，-0.10 [-0.31, +0.00] |
| customer-ts | all/CER error_rate 18.75 → 18.75，+0.00 [+0.00, +0.00]<br>all/CER false_alarm_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER miss_rate 20.00 → 20.00，+0.00 [+0.00, +0.00] |
| customer-asr | all/CER error_rate 20.15 → 21.17，+1.02 [-1.76, +3.96]<br>all/CER miss_rate 0.65 → 0.00，-0.65 [-1.94, +0.00] |
| ts-hotwords | CER 14.43 → 15.06（+0.63）<br>召回 89.9 → 89.8（-0.1） |
| ts-en | all/WER error_rate 13.59 → 13.93，+0.34 [+0.13, +0.54]* **更差**<br>all/WER false_alarm_rate 6.20 → 6.33，+0.13 [-0.09, +0.34]<br>all/WER miss_rate 2.94 → 2.95，+0.02 [-0.14, +0.18] |
| first-other | all/CER error_rate 20.11 → 20.41，+0.30 [+0.06, +0.54]* **更差**<br>all/CER false_alarm_rate 2.58 → 2.58，+0.00 [-0.27, +0.27]<br>all/CER miss_rate 1.66 → 2.22，+0.56 [+0.28, +0.84]* **更差** |
| asr-benchmark | all/CER error_rate 3.56 → 3.65，+0.08 [+0.03, +0.19]* **更差**<br>all/CER miss_rate 0.13 → 0.13，+0.01 [+0.00, +0.01]* **更差**<br>all/CER term_hit_rate 98.77 → 98.71，-0.06 [-0.14, +0.03]<br>all/WER error_rate 7.56 → 7.57，+0.01 [-0.02, +0.05]<br>all/WER false_alarm_rate 24.00 → 23.52，-0.48 [-0.73, -0.22]*<br>all/WER miss_rate 0.45 → 0.45，-0.00 [-0.03, +0.03] |
| hotword-bench | CER 3.27 → 3.23（-0.04）<br>召回 87.9 → 87.4（-0.4） |
| police | all/CER error_rate 0.27 → 0.28，+0.01 [-0.00, +0.02]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.75 → 99.65，-0.10 [-0.18, -0.03]* **更差** |
| police-single | all/CER error_rate 0.38 → 0.49，+0.11 [-0.01, +0.28]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.08 → 99.02，-0.06 [-0.25, +0.09] |
| crosslingual | all/CER error_rate 34.29 → 37.13，+2.84 [+0.45, +5.09]* **更差**<br>all/CER false_alarm_rate 0.91 → 0.80，-0.11 [-0.55, +0.23]<br>all/CER miss_rate 14.62 → 15.15，+0.53 [-0.24, +1.35] |
| scene-asr | all/CER error_rate 4.72 → 4.75，+0.03 [-0.26, +0.32]<br>all/CER miss_rate 0.85 → 0.67，-0.17 [-0.30, -0.04]* |
| crosslingual-en | all/WER error_rate 17.88 → 17.27，-0.61 [-1.30, +0.12]<br>all/WER false_alarm_rate 2.73 → 2.73，+0.00 [-0.45, +0.46]<br>all/WER miss_rate 9.17 → 8.86，-0.30 [-0.92, +0.37] |
| scene-ts | all/CER error_rate 3.43 → 3.82，+0.39 [-0.00, +0.81]<br>all/CER false_alarm_rate 21.43 → 22.86，+1.43 [-5.06, +7.94]<br>all/CER miss_rate 0.33 → 0.38，+0.05 [+0.00, +0.16] |
| ts-es-v2 | all/WER error_rate 27.49 → 27.94，+0.45 [+0.18, +0.73]* **更差**<br>all/WER false_alarm_rate 3.03 → 3.25，+0.22 [+0.01, +0.44]* **更差**<br>all/WER miss_rate 3.40 → 3.39，-0.01 [-0.22, +0.20] |
| ctx-v1 | all/CER error_rate 7.87 → 7.75，-0.12 [-0.42, +0.17]<br>all/CER miss_rate 5.32 → 5.14，-0.18 [-0.49, +0.09] |
| babble-es | all/WER error_rate 7.37 → 7.65，+0.29 [+0.03, +0.56]* **更差**<br>all/WER false_alarm_rate 0.50 → 0.50，+0.00 [+0.00, +0.00]<br>all/WER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00] |
| hotword-bench-es | CER 4.97 → 5.09（+0.12）<br>召回 86.8 → 86.2（-0.5） |
| chime4 | all/WER error_rate 9.42 → 8.98，-0.44 [-1.12, +0.17]<br>all/WER miss_rate 5.23 → 5.15，-0.08 [-0.68, +0.53] |
| customer-hotwords | CER 16.30 → 17.58（+1.28）<br>召回 78.5 → 73.6（-4.9） |
| badcase | CER 10.00 → 21.11（+11.11）<br>召回 93.3 → 93.3（+0.0） |
| ts-hotwords-es | CER 26.33 → 26.86（+0.52）<br>召回 83.5 → 83.3（-0.1） |
| crosslingual-es | all/WER error_rate 23.11 → 22.54，-0.56 [-1.89, +0.58]<br>all/WER false_alarm_rate 1.04 → 1.04，+0.00 [-0.61, +0.62]<br>all/WER miss_rate 8.06 → 7.78，-0.28 [-1.34, +0.74] |

显著更差的项：v12: all/CER error_rate, all/CER false_alarm_rate；es-asr: all/WER error_rate；ts-es-v1: all/WER error_rate, all/WER false_alarm_rate；babble4: all/CER error_rate, all/WER error_rate；babble3: all/CER error_rate, all/WER error_rate；ts-en: all/WER error_rate；first-other: all/CER error_rate, all/CER miss_rate；asr-benchmark: all/CER error_rate, all/CER miss_rate；police: all/CER term_hit_rate；crosslingual: all/CER error_rate；ts-es-v2: all/WER error_rate, all/WER false_alarm_rate；babble-es: all/WER error_rate
