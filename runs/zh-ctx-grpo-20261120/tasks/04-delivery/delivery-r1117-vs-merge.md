# 交付评估：merge 对 r1117（同一推理栈，差 = merge − r1117）

| 项目 | 结果 |
|---|---|
| v12 | all/CER error_rate 12.78 → 13.28，+0.50 [+0.29, +0.71]* **更差**<br>all/CER false_alarm_rate 2.46 → 2.52，+0.06 [-0.12, +0.24]<br>all/CER miss_rate 2.76 → 2.89，+0.13 [-0.05, +0.31] |
| es-asr | all/WER error_rate 5.25 → 5.42，+0.17 [+0.12, +0.21]* **更差**<br>all/WER miss_rate 0.02 → 0.02，+0.00 [+0.00, +0.00] |
| ts-es-v1 | all/WER error_rate 27.28 → 28.48，+1.20 [+0.67, +1.93]* **更差**<br>all/WER false_alarm_rate 3.21 → 1.99，-1.22 [-1.56, -0.92]*<br>all/WER miss_rate 3.86 → 4.25，+0.39 [+0.14, +0.64]* **更差** |
| babble4 | all/CER error_rate 1.90 → 2.12，+0.22 [-0.02, +0.45]<br>all/CER false_alarm_rate 4.95 → 4.86，-0.08 [-0.59, +0.46]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 1.92 → 2.28，+0.36 [+0.22, +0.50]* **更差**<br>all/WER false_alarm_rate 1.75 → 2.88，+1.12 [+0.48, +1.95]* **更差**<br>all/WER miss_rate 0.10 → 0.10，+0.00 [+0.00, +0.00] |
| babble3 | all/CER error_rate 1.90 → 2.12，+0.22 [-0.03, +0.48]<br>all/CER false_alarm_rate 1.25 → 1.00，-0.25 [-0.77, +0.00]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 1.92 → 2.28，+0.36 [+0.21, +0.50]* **更差**<br>all/WER false_alarm_rate 1.75 → 2.88，+1.12 [+0.40, +1.91]* **更差**<br>all/WER miss_rate 0.10 → 0.10，+0.00 [+0.00, +0.00] |
| customer-ts | all/CER error_rate 18.75 → 18.75，+0.00 [+0.00, +0.00]<br>all/CER false_alarm_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER miss_rate 20.00 → 20.00，+0.00 [+0.00, +0.00] |
| customer-asr | all/CER error_rate 20.15 → 21.90，+1.75 [-1.41, +5.90]<br>all/CER miss_rate 0.65 → 0.00，-0.65 [-1.94, +0.00] |
| ts-hotwords | CER 14.43 → 14.91（+0.48）<br>召回 89.9 → 87.0（-2.9） |
| ts-en | all/WER error_rate 13.59 → 14.01，+0.42 [+0.20, +0.64]* **更差**<br>all/WER false_alarm_rate 6.20 → 4.92，-1.28 [-1.58, -1.00]*<br>all/WER miss_rate 2.94 → 3.12，+0.19 [+0.02, +0.36]* **更差** |
| first-other | all/CER error_rate 20.11 → 19.98，-0.13 [-0.40, +0.13]<br>all/CER false_alarm_rate 2.58 → 3.33，+0.75 [+0.41, +1.11]* **更差**<br>all/CER miss_rate 1.66 → 1.34，-0.32 [-0.58, -0.08]* |
| asr-benchmark | all/CER error_rate 3.56 → 3.67，+0.11 [-0.01, +0.23]<br>all/CER miss_rate 0.13 → 0.15，+0.02 [+0.02, +0.03]* **更差**<br>all/CER term_hit_rate 98.77 → 98.52，-0.25 [-0.36, -0.14]* **更差**<br>all/WER error_rate 7.56 → 7.55，-0.01 [-0.06, +0.04]<br>all/WER false_alarm_rate 24.00 → 22.03，-1.96 [-2.40, -1.52]*<br>all/WER miss_rate 0.45 → 0.40，-0.05 [-0.09, -0.00]* |
| hotword-bench | CER 3.27 → 3.10（-0.18）<br>召回 87.9 → 87.3（-0.6） |
| police | all/CER error_rate 0.27 → 0.32，+0.04 [+0.03, +0.06]* **更差**<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.75 → 99.50，-0.25 [-0.37, -0.14]* **更差** |
| police-single | all/CER error_rate 0.38 → 0.54，+0.16 [+0.03, +0.29]* **更差**<br>all/CER miss_rate 0.00 → 0.03，+0.03 [+0.00, +0.09]<br>all/CER term_hit_rate 99.08 → 98.90，-0.18 [-0.40, +0.03] |
| crosslingual | all/CER error_rate 34.29 → 34.58，+0.30 [-2.09, +2.72]<br>all/CER false_alarm_rate 0.91 → 0.34，-0.57 [-1.11, -0.11]*<br>all/CER miss_rate 14.62 → 16.21，+1.59 [+0.69, +2.48]* **更差** |
| scene-asr | all/CER error_rate 4.72 → 7.32，+2.59 [+1.96, +3.24]* **更差**<br>all/CER miss_rate 0.85 → 0.54，-0.30 [-0.54, -0.09]* |
| crosslingual-en | all/WER error_rate 17.88 → 18.93，+1.05 [+0.23, +1.92]* **更差**<br>all/WER false_alarm_rate 2.73 → 1.25，-1.48 [-2.44, -0.65]*<br>all/WER miss_rate 9.17 → 10.53，+1.36 [+0.59, +2.19]* **更差** |
| scene-ts | all/CER error_rate 3.43 → 2.90，-0.53 [-1.13, +0.08]<br>all/CER false_alarm_rate 21.43 → 14.29，-7.14 [-14.67, +0.00]<br>all/CER miss_rate 0.33 → 0.33，+0.00 [+0.00, +0.00] |
| ts-es-v2 | all/WER error_rate 27.49 → 28.66，+1.17 [+0.70, +1.76]* **更差**<br>all/WER false_alarm_rate 3.03 → 1.93，-1.10 [-1.36, -0.84]*<br>all/WER miss_rate 3.40 → 3.88，+0.48 [+0.25, +0.72]* **更差** |
| ctx-v1 | all/CER error_rate 7.87 → 2.40，-5.47 [-6.17, -4.79]*<br>all/CER miss_rate 5.32 → 1.20，-4.12 [-4.80, -3.48]* |
| babble-es | all/WER error_rate 7.37 → 7.54，+0.17 [-0.14, +0.47]<br>all/WER false_alarm_rate 0.50 → 0.50，+0.00 [+0.00, +0.00]<br>all/WER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00] |
| hotword-bench-es | CER 4.97 → 5.14（+0.17）<br>召回 86.8 → 85.9（-0.9） |
| chime4 | all/WER error_rate 9.42 → 9.82，+0.40 [-0.38, +1.17]<br>all/WER miss_rate 5.23 → 5.61，+0.38 [-0.45, +1.14] |
| customer-hotwords | CER 16.30 → 19.62（+3.33）<br>召回 78.5 → 71.5（-6.9） |
| badcase | CER 10.00 → 5.56（-4.44）<br>召回 93.3 → 93.3（+0.0） |
| ts-hotwords-es | CER 26.33 → 27.01（+0.68）<br>召回 83.5 → 80.0（-3.5） |
| crosslingual-es | all/WER error_rate 23.11 → 24.87，+1.76 [+0.51, +2.97]* **更差**<br>all/WER false_alarm_rate 1.04 → 0.21，-0.83 [-1.75, -0.20]*<br>all/WER miss_rate 8.06 → 10.69，+2.64 [+1.39, +4.02]* **更差** |

显著更差的项：v12: all/CER error_rate；es-asr: all/WER error_rate；ts-es-v1: all/WER error_rate, all/WER miss_rate；babble4: all/WER error_rate, all/WER false_alarm_rate；babble3: all/WER error_rate, all/WER false_alarm_rate；ts-en: all/WER error_rate, all/WER miss_rate；first-other: all/CER false_alarm_rate；asr-benchmark: all/CER miss_rate, all/CER term_hit_rate；police: all/CER error_rate, all/CER term_hit_rate；police-single: all/CER error_rate；crosslingual: all/CER miss_rate；scene-asr: all/CER error_rate；crosslingual-en: all/WER error_rate, all/WER miss_rate；ts-es-v2: all/WER error_rate, all/WER miss_rate；crosslingual-es: all/WER error_rate, all/WER miss_rate
