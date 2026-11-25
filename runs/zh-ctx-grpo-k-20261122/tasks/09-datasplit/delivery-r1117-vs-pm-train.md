# 交付评估：pm-train 对 r1117（同一推理栈，差 = pm-train − r1117）

| 项目 | 结果 |
|---|---|
| v12 | all/CER error_rate 12.78 → 13.56，+0.79 [+0.56, +1.01]* **更差**<br>all/CER false_alarm_rate 2.46 → 2.18，-0.28 [-0.45, -0.11]*<br>all/CER miss_rate 2.76 → 3.55，+0.80 [+0.60, +0.99]* **更差** |
| es-asr | all/WER error_rate 5.25 → 5.33，+0.08 [+0.05, +0.13]* **更差**<br>all/WER miss_rate 0.02 → 0.02，+0.00 [+0.00, +0.00] |
| ts-es-v1 | all/WER error_rate 27.28 → 28.22，+0.94 [+0.61, +1.28]* **更差**<br>all/WER false_alarm_rate 3.21 → 3.38，+0.17 [-0.14, +0.48]<br>all/WER miss_rate 3.86 → 3.95，+0.09 [-0.19, +0.34] |
| babble4 | all/CER error_rate 1.90 → 2.19，+0.29 [+0.08, +0.51]* **更差**<br>all/CER false_alarm_rate 4.95 → 5.63，+0.68 [+0.17, +1.23]* **更差**<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 1.92 → 2.06，+0.14 [+0.03, +0.26]* **更差**<br>all/WER false_alarm_rate 1.75 → 1.75，+0.00 [-0.38, +0.37]<br>all/WER miss_rate 0.10 → 0.10，+0.00 [+0.00, +0.00] |
| babble3 | all/CER error_rate 1.90 → 2.19，+0.29 [+0.08, +0.50]* **更差**<br>all/CER false_alarm_rate 1.25 → 1.00，-0.25 [-0.77, +0.00]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 1.92 → 2.06，+0.14 [+0.01, +0.27]* **更差**<br>all/WER false_alarm_rate 1.75 → 1.75，+0.00 [-0.37, +0.38]<br>all/WER miss_rate 0.10 → 0.10，+0.00 [+0.00, +0.00] |
| customer-ts | all/CER error_rate 18.75 → 18.75，+0.00 [+0.00, +0.00]<br>all/CER false_alarm_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER miss_rate 20.00 → 20.00，+0.00 [+0.00, +0.00] |
| customer-asr | all/CER error_rate 20.15 → 18.54，-1.61 [-4.41, +0.81]<br>all/CER miss_rate 0.65 → 0.00，-0.65 [-1.94, +0.00] |
| ts-hotwords | CER 14.43 → 14.93（+0.50）<br>召回 89.9 → 89.7（-0.2） |
| ts-en | all/WER error_rate 13.59 → 13.96，+0.37 [+0.14, +0.60]* **更差**<br>all/WER false_alarm_rate 6.20 → 6.21，+0.01 [-0.25, +0.26]<br>all/WER miss_rate 2.94 → 3.28，+0.34 [+0.14, +0.54]* **更差** |
| first-other | all/CER error_rate 20.11 → 22.04，+1.92 [+1.58, +2.30]* **更差**<br>all/CER false_alarm_rate 2.58 → 2.11，-0.47 [-0.78, -0.15]*<br>all/CER miss_rate 1.66 → 3.52，+1.86 [+1.46, +2.27]* **更差** |
| asr-benchmark | all/CER error_rate 3.56 → 3.73，+0.17 [+0.05, +0.30]* **更差**<br>all/CER miss_rate 0.13 → 0.13，-0.00 [-0.01, +0.00]<br>all/CER term_hit_rate 98.77 → 98.69，-0.08 [-0.19, +0.02]<br>all/WER error_rate 7.56 → 7.52，-0.04 [-0.09, +0.01]<br>all/WER false_alarm_rate 24.00 → 25.58，+1.58 [+1.20, +1.98]* **更差**<br>all/WER miss_rate 0.45 → 0.27，-0.18 [-0.22, -0.14]* |
| hotword-bench | CER 3.27 → 3.31（+0.04）<br>召回 87.9 → 88.2（+0.4） |
| police | all/CER error_rate 0.27 → 0.27，-0.00 [-0.01, +0.01]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.75 → 99.72，-0.03 [-0.10, +0.05] |
| police-single | all/CER error_rate 0.38 → 0.43，+0.06 [-0.06, +0.18]<br>all/CER miss_rate 0.00 → 0.03，+0.03 [+0.00, +0.09]<br>all/CER term_hit_rate 99.08 → 99.02，-0.06 [-0.21, +0.12] |
| crosslingual | all/CER error_rate 34.29 → 38.90，+4.61 [+2.37, +7.24]* **更差**<br>all/CER false_alarm_rate 0.91 → 0.91，+0.00 [-0.56, +0.56]<br>all/CER miss_rate 14.62 → 16.44，+1.82 [+0.76, +2.86]* **更差** |
| scene-asr | all/CER error_rate 4.72 → 6.25，+1.52 [+0.77, +2.58]* **更差**<br>all/CER miss_rate 0.85 → 1.00，+0.15 [-0.02, +0.35] |
| crosslingual-en | all/WER error_rate 17.88 → 17.83，-0.04 [-0.74, +0.72]<br>all/WER false_alarm_rate 2.73 → 2.61，-0.11 [-0.91, +0.69]<br>all/WER miss_rate 9.17 → 8.86，-0.30 [-1.00, +0.45] |
| scene-ts | all/CER error_rate 3.43 → 3.56，+0.13 [-0.34, +0.61]<br>all/CER false_alarm_rate 21.43 → 10.00，-11.43 [-19.23, -4.28]*<br>all/CER miss_rate 0.33 → 0.49，+0.16 [+0.00, +0.38] |
| ts-es-v2 | all/WER error_rate 27.49 → 28.45，+0.97 [+0.67, +1.28]* **更差**<br>all/WER false_alarm_rate 3.03 → 3.24，+0.21 [-0.05, +0.48]<br>all/WER miss_rate 3.40 → 3.58，+0.18 [-0.05, +0.43] |
| ctx-v1 | all/CER error_rate 7.87 → 12.84，+4.97 [+4.10, +5.88]* **更差**<br>all/CER miss_rate 5.32 → 10.90，+5.57 [+4.77, +6.37]* **更差** |
| babble-es | all/WER error_rate 7.37 → 7.44，+0.07 [-0.19, +0.32]<br>all/WER false_alarm_rate 0.50 → 0.75，+0.25 [+0.00, +0.79]<br>all/WER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00] |
| hotword-bench-es | CER 4.97 → 5.05（+0.08）<br>召回 86.8 → 86.8（+0.0） |
| chime4 | all/WER error_rate 9.42 → 9.28，-0.14 [-0.88, +0.64]<br>all/WER miss_rate 5.23 → 4.77，-0.45 [-1.29, +0.38] |
| customer-hotwords | CER 16.30 → 16.72（+0.43）<br>召回 78.5 → 75.0（-3.5） |
| badcase | CER 10.00 → 16.67（+6.67）<br>召回 93.3 → 96.7（+3.3） |
| ts-hotwords-es | CER 26.33 → 27.28（+0.94）<br>召回 83.5 → 83.1（-0.4） |
| crosslingual-es | all/WER error_rate 23.11 → 23.38，+0.28 [-0.99, +1.60]<br>all/WER false_alarm_rate 1.04 → 1.88，+0.83 [+0.20, +1.75]* **更差**<br>all/WER miss_rate 8.06 → 8.33，+0.28 [-0.83, +1.36] |

显著更差的项：v12: all/CER error_rate, all/CER miss_rate；es-asr: all/WER error_rate；ts-es-v1: all/WER error_rate；babble4: all/CER error_rate, all/CER false_alarm_rate, all/WER error_rate；babble3: all/CER error_rate, all/WER error_rate；ts-en: all/WER error_rate, all/WER miss_rate；first-other: all/CER error_rate, all/CER miss_rate；asr-benchmark: all/CER error_rate, all/WER false_alarm_rate；crosslingual: all/CER error_rate, all/CER miss_rate；scene-asr: all/CER error_rate；ts-es-v2: all/WER error_rate；ctx-v1: all/CER error_rate, all/CER miss_rate；crosslingual-es: all/WER false_alarm_rate
