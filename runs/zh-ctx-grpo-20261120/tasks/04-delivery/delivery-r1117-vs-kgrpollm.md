# 交付评估：kgrpollm 对 r1117（同一推理栈，差 = kgrpollm − r1117）

| 项目 | 结果 |
|---|---|
| v12 | all/CER error_rate 12.78 → 13.32，+0.54 [+0.35, +0.74]* **更差**<br>all/CER false_alarm_rate 2.46 → 2.38，-0.09 [-0.22, +0.04]<br>all/CER miss_rate 2.76 → 3.03，+0.28 [+0.15, +0.42]* **更差** |
| es-asr | all/WER error_rate 5.25 → 4.97，-0.28 [-0.33, -0.22]*<br>all/WER miss_rate 0.02 → 0.02，+0.00 [+0.00, +0.00] |
| ts-es-v1 | all/WER error_rate 27.28 → 27.36，+0.07 [-0.20, +0.35]<br>all/WER false_alarm_rate 3.21 → 7.08，+3.87 [+3.40, +4.35]* **更差**<br>all/WER miss_rate 3.86 → 3.41，-0.46 [-0.66, -0.26]* |
| babble4 | all/CER error_rate 1.90 → 1.98，+0.08 [-0.11, +0.25]<br>all/CER false_alarm_rate 4.95 → 3.72，-1.23 [-1.75, -0.72]*<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 1.92 → 2.22，+0.30 [+0.14, +0.46]* **更差**<br>all/WER false_alarm_rate 1.75 → 1.62，-0.13 [-0.39, +0.00]<br>all/WER miss_rate 0.10 → 0.00，-0.10 [-0.31, +0.00] |
| babble3 | all/CER error_rate 1.90 → 1.98，+0.08 [-0.11, +0.26]<br>all/CER false_alarm_rate 1.25 → 0.25，-1.00 [-2.05, -0.24]*<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 1.92 → 2.22，+0.30 [+0.13, +0.46]* **更差**<br>all/WER false_alarm_rate 1.75 → 1.62，-0.13 [-0.39, +0.00]<br>all/WER miss_rate 0.10 → 0.00，-0.10 [-0.31, +0.00] |
| customer-ts | all/CER error_rate 18.75 → 18.75，+0.00 [+0.00, +0.00]<br>all/CER false_alarm_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER miss_rate 20.00 → 20.00，+0.00 [+0.00, +0.00] |
| customer-asr | all/CER error_rate 20.15 → 19.42，-0.73 [-3.07, +1.61]<br>all/CER miss_rate 0.65 → 0.65，+0.00 [+0.00, +0.00] |
| ts-hotwords | CER 14.43 → 15.08（+0.65）<br>召回 89.9 → 92.4（+2.5） |
| ts-en | all/WER error_rate 13.59 → 13.80，+0.21 [+0.01, +0.40]* **更差**<br>all/WER false_alarm_rate 6.20 → 7.08，+0.88 [+0.63, +1.15]* **更差**<br>all/WER miss_rate 2.94 → 2.82，-0.12 [-0.25, +0.01] |
| first-other | all/CER error_rate 20.11 → 20.66，+0.55 [+0.26, +0.83]* **更差**<br>all/CER false_alarm_rate 2.58 → 2.16，-0.42 [-0.72, -0.12]*<br>all/CER miss_rate 1.66 → 3.06，+1.40 [+1.08, +1.73]* **更差** |
| asr-benchmark | all/CER error_rate 3.56 → 3.59，+0.03 [-0.06, +0.08]<br>all/CER miss_rate 0.13 → 0.14，+0.01 [+0.01, +0.02]* **更差**<br>all/CER term_hit_rate 98.77 → 98.71，-0.06 [-0.15, +0.03]<br>all/WER error_rate 7.56 → 7.54，-0.01 [-0.06, +0.03]<br>all/WER false_alarm_rate 24.00 → 22.88，-1.12 [-1.48, -0.78]*<br>all/WER miss_rate 0.45 → 0.38，-0.07 [-0.11, -0.03]* |
| hotword-bench | CER 3.27 → 3.00（-0.27）<br>召回 87.9 → 88.2（+0.4） |
| police | all/CER error_rate 0.27 → 0.27，-0.00 [-0.02, +0.01]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.75 → 99.72，-0.03 [-0.09, +0.04] |
| police-single | all/CER error_rate 0.38 → 0.36，-0.02 [-0.10, +0.06]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.08 → 99.17，+0.09 [-0.09, +0.28] |
| crosslingual | all/CER error_rate 34.29 → 34.14，-0.15 [-2.10, +1.58]<br>all/CER false_alarm_rate 0.91 → 0.80，-0.11 [-0.47, +0.23]<br>all/CER miss_rate 14.62 → 17.20，+2.58 [+1.52, +3.64]* **更差** |
| scene-asr | all/CER error_rate 4.72 → 2.62，-2.10 [-2.61, -1.63]*<br>all/CER miss_rate 0.85 → 1.13，+0.28 [+0.15, +0.46]* **更差** |
| crosslingual-en | all/WER error_rate 17.88 → 18.04，+0.16 [-0.45, +0.79]<br>all/WER false_alarm_rate 2.73 → 2.95，+0.23 [-0.46, +0.94]<br>all/WER miss_rate 9.17 → 9.85，+0.68 [+0.30, +1.15]* **更差** |
| scene-ts | all/CER error_rate 3.43 → 3.30，-0.13 [-0.52, +0.26]<br>all/CER false_alarm_rate 21.43 → 12.86，-8.57 [-15.71, -2.44]*<br>all/CER miss_rate 0.33 → 0.44，+0.11 [+0.00, +0.27] |
| ts-es-v2 | all/WER error_rate 27.49 → 27.43，-0.06 [-0.30, +0.20]<br>all/WER false_alarm_rate 3.03 → 6.91，+3.88 [+3.48, +4.30]* **更差**<br>all/WER miss_rate 3.40 → 2.98，-0.43 [-0.59, -0.25]* |
| ctx-v1 | all/CER error_rate 7.87 → 2.55，-5.32 [-6.01, -4.65]*<br>all/CER miss_rate 5.32 → 1.32，-4.00 [-4.65, -3.35]* |
| babble-es | all/WER error_rate 7.37 → 6.81，-0.56 [-0.92, -0.21]*<br>all/WER false_alarm_rate 0.50 → 0.50，+0.00 [+0.00, +0.00]<br>all/WER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00] |
| hotword-bench-es | CER 4.97 → 4.81（-0.16）<br>召回 86.8 → 87.3（+0.5） |
| chime4 | all/WER error_rate 9.42 → 9.68，+0.26 [-0.30, +0.82]<br>all/WER miss_rate 5.23 → 5.38，+0.15 [-0.45, +0.76] |
| customer-hotwords | CER 16.30 → 16.04（-0.26）<br>召回 78.5 → 77.1（-1.4） |
| badcase | CER 10.00 → 12.22（+2.22）<br>召回 93.3 → 86.7（-6.7） |
| ts-hotwords-es | CER 26.33 → 27.42（+1.09）<br>召回 83.5 → 86.0（+2.5） |
| crosslingual-es | all/WER error_rate 23.11 → 20.02，-3.09 [-4.77, -1.64]*<br>all/WER false_alarm_rate 1.04 → 3.96，+2.92 [+1.48, +4.56]* **更差**<br>all/WER miss_rate 8.06 → 4.86，-3.19 [-4.43, -1.97]* |

显著更差的项：v12: all/CER error_rate, all/CER miss_rate；ts-es-v1: all/WER false_alarm_rate；babble4: all/WER error_rate；babble3: all/WER error_rate；ts-en: all/WER error_rate, all/WER false_alarm_rate；first-other: all/CER error_rate, all/CER miss_rate；asr-benchmark: all/CER miss_rate；crosslingual: all/CER miss_rate；scene-asr: all/CER miss_rate；crosslingual-en: all/WER miss_rate；ts-es-v2: all/WER false_alarm_rate；crosslingual-es: all/WER false_alarm_rate
