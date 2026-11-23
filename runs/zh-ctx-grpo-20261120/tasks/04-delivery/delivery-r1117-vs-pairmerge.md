# 交付评估：pairmerge 对 r1117（同一推理栈，差 = pairmerge − r1117）

| 项目 | 结果 |
|---|---|
| v12 | all/CER error_rate 12.78 → 13.72，+0.94 [+0.70, +1.18]* **更差**<br>all/CER false_alarm_rate 2.46 → 2.50，+0.04 [-0.14, +0.22]<br>all/CER miss_rate 2.76 → 3.14，+0.39 [+0.20, +0.57]* **更差** |
| es-asr | all/WER error_rate 5.25 → 5.47，+0.22 [+0.17, +0.27]* **更差**<br>all/WER miss_rate 0.02 → 0.02，+0.00 [+0.00, +0.00] |
| ts-es-v1 | all/WER error_rate 27.28 → 28.47，+1.19 [+0.86, +1.51]* **更差**<br>all/WER false_alarm_rate 3.21 → 2.87，-0.34 [-0.63, -0.05]*<br>all/WER miss_rate 3.86 → 3.85，-0.01 [-0.26, +0.21] |
| babble4 | all/CER error_rate 1.90 → 2.08，+0.18 [-0.05, +0.41]<br>all/CER false_alarm_rate 4.95 → 4.36，-0.59 [-1.16, -0.04]*<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 1.92 → 2.30，+0.38 [+0.23, +0.53]* **更差**<br>all/WER false_alarm_rate 1.75 → 1.62，-0.13 [-0.59, +0.26]<br>all/WER miss_rate 0.10 → 0.00，-0.10 [-0.31, +0.00] |
| babble3 | all/CER error_rate 1.90 → 2.08，+0.18 [-0.06, +0.43]<br>all/CER false_alarm_rate 1.25 → 1.25，+0.00 [-0.72, +0.77]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 1.92 → 2.30，+0.38 [+0.24, +0.53]* **更差**<br>all/WER false_alarm_rate 1.75 → 1.62，-0.13 [-0.53, +0.26]<br>all/WER miss_rate 0.10 → 0.00，-0.10 [-0.31, +0.00] |
| customer-ts | all/CER error_rate 18.75 → 18.75，+0.00 [+0.00, +0.00]<br>all/CER false_alarm_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER miss_rate 20.00 → 20.00，+0.00 [+0.00, +0.00] |
| customer-asr | all/CER error_rate 20.15 → 21.46，+1.31 [-1.62, +4.71]<br>all/CER miss_rate 0.65 → 0.00，-0.65 [-1.94, +0.00] |
| ts-hotwords | CER 14.43 → 15.38（+0.95）<br>召回 89.9 → 89.2（-0.7） |
| ts-en | all/WER error_rate 13.59 → 14.46，+0.87 [+0.63, +1.12]* **更差**<br>all/WER false_alarm_rate 6.20 → 6.25，+0.06 [-0.21, +0.32]<br>all/WER miss_rate 2.94 → 2.87，-0.06 [-0.26, +0.14] |
| first-other | all/CER error_rate 20.11 → 21.49，+1.38 [+1.05, +1.73]* **更差**<br>all/CER false_alarm_rate 2.58 → 2.11，-0.47 [-0.80, -0.15]*<br>all/CER miss_rate 1.66 → 3.28，+1.62 [+1.26, +2.00]* **更差** |
| asr-benchmark | all/CER error_rate 3.56 → 3.77，+0.20 [+0.09, +0.37]* **更差**<br>all/CER miss_rate 0.13 → 0.12，-0.01 [-0.02, -0.01]*<br>all/CER term_hit_rate 98.77 → 98.57，-0.20 [-0.31, -0.09]* **更差**<br>all/WER error_rate 7.56 → 7.58，+0.02 [-0.03, +0.07]<br>all/WER false_alarm_rate 24.00 → 25.12，+1.12 [+0.76, +1.49]* **更差**<br>all/WER miss_rate 0.45 → 0.29，-0.16 [-0.20, -0.12]* |
| hotword-bench | CER 3.27 → 3.22（-0.06）<br>召回 87.9 → 87.4（-0.4） |
| police | all/CER error_rate 0.27 → 0.30，+0.03 [+0.01, +0.04]* **更差**<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.75 → 99.58，-0.18 [-0.28, -0.07]* **更差** |
| police-single | all/CER error_rate 0.38 → 0.47，+0.10 [-0.06, +0.25]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.08 → 98.96，-0.12 [-0.37, +0.12] |
| crosslingual | all/CER error_rate 34.29 → 38.92，+4.63 [+2.42, +7.00]* **更差**<br>all/CER false_alarm_rate 0.91 → 1.14，+0.23 [-0.45, +0.90]<br>all/CER miss_rate 14.62 → 15.45，+0.83 [-0.22, +1.89] |
| scene-asr | all/CER error_rate 4.72 → 8.63，+3.91 [+3.20, +4.62]* **更差**<br>all/CER miss_rate 0.85 → 0.59，-0.26 [-0.46, -0.07]* |
| crosslingual-en | all/WER error_rate 17.88 → 17.68，-0.20 [-1.07, +0.72]<br>all/WER false_alarm_rate 2.73 → 3.18，+0.45 [-0.33, +1.29]<br>all/WER miss_rate 9.17 → 9.24，+0.08 [-0.60, +0.75] |
| scene-ts | all/CER error_rate 3.43 → 3.38，-0.06 [-0.60, +0.46]<br>all/CER false_alarm_rate 21.43 → 17.14，-4.29 [-12.86, +4.41]<br>all/CER miss_rate 0.33 → 0.38，+0.05 [+0.00, +0.16] |
| ts-es-v2 | all/WER error_rate 27.49 → 28.83，+1.34 [+1.03, +1.67]* **更差**<br>all/WER false_alarm_rate 3.03 → 2.77，-0.27 [-0.50, -0.02]*<br>all/WER miss_rate 3.40 → 3.57，+0.17 [-0.06, +0.40] |
| ctx-v1 | all/CER error_rate 7.87 → 2.20，-5.67 [-6.39, -4.97]*<br>all/CER miss_rate 5.32 → 0.89，-4.43 [-5.11, -3.76]* |
| babble-es | all/WER error_rate 7.37 → 7.70，+0.33 [+0.02, +0.66]* **更差**<br>all/WER false_alarm_rate 0.50 → 0.50，+0.00 [+0.00, +0.00]<br>all/WER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00] |
| hotword-bench-es | CER 4.97 → 5.16（+0.19）<br>召回 86.8 → 86.2（-0.5） |
| chime4 | all/WER error_rate 9.42 → 10.16，+0.74 [-0.02, +1.48]<br>all/WER miss_rate 5.23 → 6.29，+1.06 [+0.23, +1.89]* **更差** |
| customer-hotwords | CER 16.30 → 17.75（+1.45）<br>召回 78.5 → 75.0（-3.5） |
| badcase | CER 10.00 → 29.44（+19.44）<br>召回 93.3 → 100.0（+6.7） |
| ts-hotwords-es | CER 26.33 → 27.36（+1.02）<br>召回 83.5 → 82.6（-0.8） |
| crosslingual-es | all/WER error_rate 23.11 → 24.41，+1.31 [-0.08, +2.65]<br>all/WER false_alarm_rate 1.04 → 1.25，+0.21 [-0.67, +1.09]<br>all/WER miss_rate 8.06 → 9.58，+1.53 [+0.29, +2.76]* **更差** |

显著更差的项：v12: all/CER error_rate, all/CER miss_rate；es-asr: all/WER error_rate；ts-es-v1: all/WER error_rate；babble4: all/WER error_rate；babble3: all/WER error_rate；ts-en: all/WER error_rate；first-other: all/CER error_rate, all/CER miss_rate；asr-benchmark: all/CER error_rate, all/CER term_hit_rate, all/WER false_alarm_rate；police: all/CER error_rate, all/CER term_hit_rate；crosslingual: all/CER error_rate；scene-asr: all/CER error_rate；ts-es-v2: all/WER error_rate；babble-es: all/WER error_rate；chime4: all/WER miss_rate；crosslingual-es: all/WER miss_rate
