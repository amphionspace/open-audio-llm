# 交付评估：pairmerge-g 对 r1117（同一推理栈，差 = pairmerge-g − r1117）

| 项目 | 结果 |
|---|---|
| v12 | all/CER error_rate 12.78 → 13.44，+0.66 [+0.44, +0.88]* **更差**<br>all/CER false_alarm_rate 2.46 → 2.45，-0.01 [-0.18, +0.16]<br>all/CER miss_rate 2.76 → 3.16，+0.40 [+0.22, +0.58]* **更差** |
| es-asr | all/WER error_rate 5.25 → 5.38，+0.13 [+0.09, +0.17]* **更差**<br>all/WER miss_rate 0.02 → 0.02，+0.00 [+0.00, +0.00] |
| ts-es-v1 | all/WER error_rate 27.28 → 28.38，+1.09 [+0.76, +1.42]* **更差**<br>all/WER false_alarm_rate 3.21 → 2.82，-0.39 [-0.70, -0.08]*<br>all/WER miss_rate 3.86 → 3.92，+0.06 [-0.17, +0.29] |
| babble4 | all/CER error_rate 1.90 → 2.03，+0.12 [-0.09, +0.32]<br>all/CER false_alarm_rate 4.95 → 4.99，+0.04 [-0.50, +0.60]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 1.92 → 2.19，+0.27 [+0.14, +0.40]* **更差**<br>all/WER false_alarm_rate 1.75 → 1.50，-0.25 [-0.64, +0.00]<br>all/WER miss_rate 0.10 → 0.10，+0.00 [+0.00, +0.00] |
| babble3 | all/CER error_rate 1.90 → 2.03，+0.12 [-0.07, +0.34]<br>all/CER false_alarm_rate 1.25 → 1.25，+0.00 [-0.72, +0.77]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 1.92 → 2.19，+0.27 [+0.14, +0.41]* **更差**<br>all/WER false_alarm_rate 1.75 → 1.50，-0.25 [-0.64, +0.00]<br>all/WER miss_rate 0.10 → 0.10，+0.00 [+0.00, +0.00] |
| customer-ts | all/CER error_rate 18.75 → 18.75，+0.00 [+0.00, +0.00]<br>all/CER false_alarm_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER miss_rate 20.00 → 20.00，+0.00 [+0.00, +0.00] |
| customer-asr | all/CER error_rate 20.15 → 20.15，+0.00 [-2.45, +2.48]<br>all/CER miss_rate 0.65 → 0.00，-0.65 [-1.94, +0.00] |
| ts-hotwords | CER 14.43 → 15.06（+0.62）<br>召回 89.9 → 89.0（-0.9） |
| ts-en | all/WER error_rate 13.59 → 14.01，+0.42 [+0.19, +0.66]* **更差**<br>all/WER false_alarm_rate 6.20 → 6.26，+0.07 [-0.20, +0.33]<br>all/WER miss_rate 2.94 → 2.83，-0.11 [-0.29, +0.07] |
| first-other | all/CER error_rate 20.11 → 21.33，+1.22 [+0.88, +1.56]* **更差**<br>all/CER false_alarm_rate 2.58 → 2.04，-0.55 [-0.86, -0.25]*<br>all/CER miss_rate 1.66 → 2.88，+1.22 [+0.86, +1.59]* **更差** |
| asr-benchmark | all/CER error_rate 3.56 → 3.74，+0.17 [+0.08, +0.31]* **更差**<br>all/CER miss_rate 0.13 → 0.11，-0.02 [-0.03, -0.01]*<br>all/CER term_hit_rate 98.77 → 98.69，-0.07 [-0.18, +0.03]<br>all/WER error_rate 7.56 → 7.54，-0.02 [-0.07, +0.02]<br>all/WER false_alarm_rate 24.00 → 25.73，+1.73 [+1.33, +2.14]* **更差**<br>all/WER miss_rate 0.45 → 0.30，-0.15 [-0.19, -0.11]* |
| hotword-bench | CER 3.27 → 3.27（+0.00）<br>召回 87.9 → 87.9（+0.0） |
| police | all/CER error_rate 0.27 → 0.30，+0.02 [+0.01, +0.04]* **更差**<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.75 → 99.68，-0.07 [-0.17, +0.02] |
| police-single | all/CER error_rate 0.38 → 0.48，+0.10 [-0.03, +0.24]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.08 → 98.90，-0.18 [-0.40, +0.03] |
| crosslingual | all/CER error_rate 34.29 → 36.68，+2.39 [+0.01, +4.86]* **更差**<br>all/CER false_alarm_rate 0.91 → 0.91，+0.00 [-0.55, +0.57]<br>all/CER miss_rate 14.62 → 15.00，+0.38 [-0.61, +1.37] |
| scene-asr | all/CER error_rate 4.72 → 8.67，+3.94 [+3.21, +4.65]* **更差**<br>all/CER miss_rate 0.85 → 0.65，-0.20 [-0.41, +0.00] |
| crosslingual-en | all/WER error_rate 17.88 → 18.25，+0.37 [-0.49, +1.24]<br>all/WER false_alarm_rate 2.73 → 3.30，+0.57 [-0.23, +1.37]<br>all/WER miss_rate 9.17 → 9.47，+0.30 [-0.46, +1.10] |
| scene-ts | all/CER error_rate 3.43 → 3.17，-0.26 [-0.77, +0.22]<br>all/CER false_alarm_rate 21.43 → 17.14，-4.29 [-12.91, +4.41]<br>all/CER miss_rate 0.33 → 0.44，+0.11 [+0.00, +0.27] |
| ts-es-v2 | all/WER error_rate 27.49 → 28.54，+1.05 [+0.74, +1.40]* **更差**<br>all/WER false_alarm_rate 3.03 → 2.72，-0.31 [-0.57, -0.07]*<br>all/WER miss_rate 3.40 → 3.51，+0.11 [-0.10, +0.33] |
| ctx-v1 | all/CER error_rate 7.87 → 2.29，-5.58 [-6.30, -4.88]*<br>all/CER miss_rate 5.32 → 0.92，-4.40 [-5.08, -3.72]* |
| babble-es | all/WER error_rate 7.37 → 7.57，+0.21 [-0.04, +0.46]<br>all/WER false_alarm_rate 0.50 → 0.25，-0.25 [-0.78, +0.00]<br>all/WER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00] |
| hotword-bench-es | CER 4.97 → 5.07（+0.10）<br>召回 86.8 → 86.9（+0.1） |
| chime4 | all/WER error_rate 9.42 → 11.30，+1.88 [+1.03, +2.73]* **更差**<br>all/WER miss_rate 5.23 → 7.20，+1.97 [+1.06, +2.88]* **更差** |
| customer-hotwords | CER 16.30 → 16.47（+0.17）<br>召回 78.5 → 77.8（-0.7） |
| badcase | CER 10.00 → 17.22（+7.22）<br>召回 93.3 → 96.7（+3.3） |
| ts-hotwords-es | CER 26.33 → 27.19（+0.86）<br>召回 83.5 → 82.7（-0.7） |
| crosslingual-es | all/WER error_rate 23.11 → 24.50，+1.39 [-0.08, +2.90]<br>all/WER false_alarm_rate 1.04 → 1.25，+0.21 [-0.64, +1.09]<br>all/WER miss_rate 8.06 → 9.86，+1.81 [+0.55, +3.16]* **更差** |

显著更差的项：v12: all/CER error_rate, all/CER miss_rate；es-asr: all/WER error_rate；ts-es-v1: all/WER error_rate；babble4: all/WER error_rate；babble3: all/WER error_rate；ts-en: all/WER error_rate；first-other: all/CER error_rate, all/CER miss_rate；asr-benchmark: all/CER error_rate, all/WER false_alarm_rate；police: all/CER error_rate；crosslingual: all/CER error_rate；scene-asr: all/CER error_rate；ts-es-v2: all/WER error_rate；chime4: all/WER error_rate, all/WER miss_rate；crosslingual-es: all/WER miss_rate
