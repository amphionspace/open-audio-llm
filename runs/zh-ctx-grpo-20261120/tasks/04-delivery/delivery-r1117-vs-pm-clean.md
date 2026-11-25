# 交付评估：pm-clean 对 r1117（同一推理栈，差 = pm-clean − r1117）

| 项目 | 结果 |
|---|---|
| v12 | all/CER error_rate 12.78 → 13.03，+0.26 [+0.04, +0.46]* **更差**<br>all/CER false_alarm_rate 2.46 → 4.52，+2.05 [+1.81, +2.31]* **更差**<br>all/CER miss_rate 2.76 → 1.98，-0.77 [-0.97, -0.58]* |
| es-asr | all/WER error_rate 5.25 → 5.57，+0.33 [+0.27, +0.39]* **更差**<br>all/WER miss_rate 0.02 → 0.03，+0.01 [+0.00, +0.03] |
| ts-es-v1 | all/WER error_rate 27.28 → 29.28，+1.99 [+1.28, +3.08]* **更差**<br>all/WER false_alarm_rate 3.21 → 4.54，+1.33 [+1.00, +1.67]* **更差**<br>all/WER miss_rate 3.86 → 3.56，-0.30 [-0.54, -0.08]* |
| babble4 | all/CER error_rate 1.90 → 2.07，+0.16 [+0.01, +0.33]* **更差**<br>all/CER false_alarm_rate 4.95 → 4.10，-0.85 [-1.29, -0.46]*<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 1.92 → 2.11，+0.19 [+0.04, +0.33]* **更差**<br>all/WER false_alarm_rate 1.75 → 1.25，-0.50 [-1.01, -0.12]*<br>all/WER miss_rate 0.10 → 0.10，+0.00 [+0.00, +0.00] |
| babble3 | all/CER error_rate 1.90 → 2.07，+0.16 [+0.01, +0.33]* **更差**<br>all/CER false_alarm_rate 1.25 → 0.75，-0.50 [-1.26, +0.00]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 1.92 → 2.11，+0.19 [+0.04, +0.34]* **更差**<br>all/WER false_alarm_rate 1.75 → 1.25，-0.50 [-1.03, -0.12]*<br>all/WER miss_rate 0.10 → 0.10，+0.00 [+0.00, +0.00] |
| customer-ts | all/CER error_rate 18.75 → 17.71，-1.04 [-2.50, +0.00]<br>all/CER false_alarm_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER miss_rate 20.00 → 20.00，+0.00 [+0.00, +0.00] |
| customer-asr | all/CER error_rate 20.15 → 25.84，+5.69 [+2.31, +10.19]* **更差**<br>all/CER miss_rate 0.65 → 0.00，-0.65 [-1.94, +0.00] |
| ts-hotwords | CER 14.43 → 15.84（+1.41）<br>召回 89.9 → 88.8（-1.2） |
| ts-en | all/WER error_rate 13.59 → 14.29，+0.70 [+0.46, +0.95]* **更差**<br>all/WER false_alarm_rate 6.20 → 8.06，+1.86 [+1.54, +2.16]* **更差**<br>all/WER miss_rate 2.94 → 2.33，-0.61 [-0.79, -0.43]* |
| first-other | all/CER error_rate 20.11 → 21.41，+1.30 [+0.98, +1.61]* **更差**<br>all/CER false_alarm_rate 2.58 → 3.50，+0.92 [+0.58, +1.28]* **更差**<br>all/CER miss_rate 1.66 → 1.62，-0.04 [-0.32, +0.26] |
| asr-benchmark | all/CER error_rate 3.56 → 3.75，+0.19 [+0.09, +0.34]* **更差**<br>all/CER miss_rate 0.13 → 0.14，+0.01 [+0.01, +0.02]* **更差**<br>all/CER term_hit_rate 98.77 → 98.78，+0.01 [-0.07, +0.10]<br>all/WER error_rate 7.56 → 7.77，+0.21 [+0.16, +0.27]* **更差**<br>all/WER false_alarm_rate 24.00 → 22.40，-1.60 [-1.99, -1.23]*<br>all/WER miss_rate 0.45 → 0.68，+0.23 [+0.18, +0.28]* **更差** |
| hotword-bench | CER 3.27 → 3.23（-0.05）<br>召回 87.9 → 87.6（-0.2） |
| police | all/CER error_rate 0.27 → 0.28，+0.01 [-0.00, +0.02]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.75 → 99.72，-0.03 [-0.10, +0.04] |
| police-single | all/CER error_rate 0.38 → 0.35，-0.03 [-0.14, +0.08]<br>all/CER miss_rate 0.00 → 0.03，+0.03 [+0.00, +0.09]<br>all/CER term_hit_rate 99.08 → 99.17，+0.09 [-0.09, +0.28] |
| crosslingual | all/CER error_rate 34.29 → 34.48，+0.19 [-2.28, +2.55]<br>all/CER false_alarm_rate 0.91 → 2.27，+1.36 [+0.68, +2.16]* **更差**<br>all/CER miss_rate 14.62 → 13.26，-1.36 [-2.29, -0.47]* |
| scene-asr | all/CER error_rate 4.72 → 5.13，+0.41 [+0.12, +0.70]* **更差**<br>all/CER miss_rate 0.85 → 1.02，+0.17 [+0.02, +0.35]* **更差** |
| crosslingual-en | all/WER error_rate 17.88 → 17.43，-0.45 [-1.28, +0.37]<br>all/WER false_alarm_rate 2.73 → 4.09，+1.36 [+0.65, +2.25]* **更差**<br>all/WER miss_rate 9.17 → 7.12，-2.05 [-2.86, -1.27]* |
| scene-ts | all/CER error_rate 3.43 → 3.78，+0.35 [-0.15, +0.88]<br>all/CER false_alarm_rate 21.43 → 22.86，+1.43 [+0.00, +4.84]<br>all/CER miss_rate 0.33 → 0.33，+0.00 [-0.22, +0.22] |
| ts-es-v2 | all/WER error_rate 27.49 → 29.30，+1.81 [+1.21, +2.70]* **更差**<br>all/WER false_alarm_rate 3.03 → 4.35，+1.32 [+1.02, +1.60]* **更差**<br>all/WER miss_rate 3.40 → 3.11，-0.30 [-0.50, -0.08]* |
| ctx-v1 | all/CER error_rate 7.87 → 6.27，-1.60 [-2.07, -1.13]*<br>all/CER miss_rate 5.32 → 3.91，-1.42 [-1.88, -0.98]* |
| babble-es | all/WER error_rate 7.37 → 7.78，+0.41 [+0.10, +0.75]* **更差**<br>all/WER false_alarm_rate 0.50 → 0.50，+0.00 [+0.00, +0.00]<br>all/WER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00] |
| hotword-bench-es | CER 4.97 → 5.25（+0.28）<br>召回 86.8 → 86.1（-0.7） |
| chime4 | all/WER error_rate 9.42 → 12.80，+3.38 [+2.37, +4.42]* **更差**<br>all/WER miss_rate 5.23 → 9.47，+4.24 [+3.18, +5.38]* **更差** |
| customer-hotwords | CER 16.30 → 18.94（+2.65）<br>召回 78.5 → 77.8（-0.7） |
| badcase | CER 10.00 → 32.78（+22.78）<br>召回 93.3 → 83.3（-10.0） |
| ts-hotwords-es | CER 26.33 → 28.45（+2.12）<br>召回 83.5 → 82.7（-0.8） |
| crosslingual-es | all/WER error_rate 23.11 → 22.09，-1.02 [-2.63, +0.39]<br>all/WER false_alarm_rate 1.04 → 1.88，+0.83 [+0.00, +1.89]<br>all/WER miss_rate 8.06 → 5.14，-2.92 [-4.23, -1.80]* |

显著更差的项：v12: all/CER error_rate, all/CER false_alarm_rate；es-asr: all/WER error_rate；ts-es-v1: all/WER error_rate, all/WER false_alarm_rate；babble4: all/CER error_rate, all/WER error_rate；babble3: all/CER error_rate, all/WER error_rate；customer-asr: all/CER error_rate；ts-en: all/WER error_rate, all/WER false_alarm_rate；first-other: all/CER error_rate, all/CER false_alarm_rate；asr-benchmark: all/CER error_rate, all/CER miss_rate, all/WER error_rate, all/WER miss_rate；crosslingual: all/CER false_alarm_rate；scene-asr: all/CER error_rate, all/CER miss_rate；crosslingual-en: all/WER false_alarm_rate；ts-es-v2: all/WER error_rate, all/WER false_alarm_rate；babble-es: all/WER error_rate；chime4: all/WER error_rate, all/WER miss_rate
