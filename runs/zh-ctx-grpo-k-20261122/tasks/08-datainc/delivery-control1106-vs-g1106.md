# 交付评估：g1106 对 control1106（同一推理栈，差 = g1106 − control1106）

| 项目 | 结果 |
|---|---|
| v12 | all/CER error_rate 13.32 → 13.40，+0.07 [-0.14, +0.29]<br>all/CER false_alarm_rate 3.51 → 3.57，+0.06 [-0.11, +0.23]<br>all/CER miss_rate 3.03 → 2.91，-0.12 [-0.28, +0.03] |
| es-asr | all/WER error_rate 71.00 → 76.02，+5.01 [+4.11, +5.98]* **更差**<br>all/WER miss_rate 0.33 → 0.33，+0.00 [-0.03, +0.03] |
| ts-es-v1 | all/WER error_rate 81.84 → 85.67，+3.82 [+3.03, +4.63]* **更差**<br>all/WER false_alarm_rate 1.99 → 2.38，+0.39 [+0.15, +0.62]* **更差**<br>all/WER miss_rate 3.91 → 3.97，+0.06 [-0.17, +0.30] |
| babble4 | all/CER error_rate 2.24 → 2.20，-0.04 [-0.21, +0.15]<br>all/CER false_alarm_rate 6.85 → 6.94，+0.08 [-0.30, +0.51]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 2.46 → 2.46，+0.00 [-0.14, +0.14]<br>all/WER false_alarm_rate 2.25 → 2.50，+0.25 [-0.24, +0.75]<br>all/WER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00] |
| babble3 | all/CER error_rate 2.24 → 2.20，-0.04 [-0.21, +0.14]<br>all/CER false_alarm_rate 0.50 → 0.50，+0.00 [+0.00, +0.00]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 2.46 → 2.46，+0.00 [-0.13, +0.13]<br>all/WER false_alarm_rate 2.25 → 2.50，+0.25 [-0.25, +0.78]<br>all/WER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00] |
| customer-ts | all/CER error_rate 4.17 → 4.17，+0.00 [+0.00, +0.00]<br>all/CER false_alarm_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00] |
| customer-asr | all/CER error_rate 17.81 → 19.85，+2.04 [-0.31, +5.03]<br>all/CER miss_rate 0.65 → 0.00，-0.65 [-1.94, +0.00] |
| ts-hotwords | CER 15.10 → 15.12（+0.02）<br>召回 92.2 → 93.0（+0.8） |
| ts-en | all/WER error_rate 14.60 → 14.60，-0.00 [-0.22, +0.23]<br>all/WER false_alarm_rate 6.22 → 6.23，+0.01 [-0.21, +0.23]<br>all/WER miss_rate 3.45 → 3.15，-0.30 [-0.47, -0.12]* |
| first-other | all/CER error_rate 22.20 → 21.39，-0.81 [-1.05, -0.56]*<br>all/CER false_alarm_rate 4.10 → 4.02，-0.07 [-0.42, +0.28]<br>all/CER miss_rate 0.74 → 0.96，+0.22 [+0.06, +0.41]* **更差** |
| asr-benchmark | all/CER error_rate 3.67 → 3.69，+0.01 [-0.05, +0.13]<br>all/CER miss_rate 0.05 → 0.06，+0.00 [+0.00, +0.01]<br>all/CER term_hit_rate 98.78 → 98.74，-0.04 [-0.13, +0.05]<br>all/WER error_rate 7.42 → 7.40，-0.02 [-0.05, +0.01]<br>all/WER false_alarm_rate 38.38 → 36.71，-1.67 [-2.17, -1.20]*<br>all/WER miss_rate 0.14 → 0.14，+0.00 [-0.02, +0.02] |
| hotword-bench | CER 3.83 → 3.85（+0.02）<br>召回 85.0 → 85.1（+0.1） |
| police | all/CER error_rate 0.27 → 0.28，+0.00 [-0.01, +0.01]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.70 → 99.69，-0.01 [-0.08, +0.06] |
| police-single | all/CER error_rate 0.35 → 0.33，-0.02 [-0.07, +0.00]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.17 → 99.23，+0.06 [+0.00, +0.15] |
| crosslingual | all/CER error_rate 39.11 → 38.02，-1.08 [-3.00, +0.85]<br>all/CER false_alarm_rate 1.59 → 1.59，+0.00 [-0.65, +0.64]<br>all/CER miss_rate 22.65 → 23.41，+0.76 [-0.15, +1.65] |
| scene-asr | all/CER error_rate 4.97 → 4.61，-0.35 [-0.77, +0.03]<br>all/CER miss_rate 0.48 → 0.41，-0.07 [-0.22, +0.07] |
| crosslingual-en | all/WER error_rate 21.86 → 20.19，-1.67 [-2.51, -0.90]*<br>all/WER false_alarm_rate 2.39 → 2.50，+0.11 [-0.57, +0.79]<br>all/WER miss_rate 13.94 → 12.42，-1.52 [-2.45, -0.67]* |
| scene-ts | all/CER error_rate 3.38 → 3.65，+0.27 [-0.25, +0.77]<br>all/CER false_alarm_rate 24.29 → 27.14，+2.86 [+0.00, +7.35]<br>all/CER miss_rate 0.27 → 0.22，-0.05 [-0.16, +0.00] |
| ts-es-v2 | all/WER error_rate 82.78 → 86.61，+3.83 [+3.11, +4.53]* **更差**<br>all/WER false_alarm_rate 1.88 → 2.20，+0.32 [+0.14, +0.53]* **更差**<br>all/WER miss_rate 3.50 → 3.56，+0.06 [-0.15, +0.26] |
| ctx-v1 | all/CER error_rate 7.61 → 7.72，+0.11 [-0.16, +0.39]<br>all/CER miss_rate 5.36 → 5.45，+0.09 [-0.18, +0.37] |
| babble-es | all/WER error_rate 110.60 → 114.93，+4.33 [+1.43, +7.33]* **更差**<br>all/WER false_alarm_rate 0.50 → 0.50，+0.00 [+0.00, +0.00]<br>all/WER miss_rate 0.20 → 0.60，+0.40 [+0.10, +0.82]* **更差** |
| hotword-bench-es | CER 57.15 → 60.86（+3.70）<br>召回 43.1 → 42.2（-0.9） |
| chime4 | all/WER error_rate 7.65 → 7.16，-0.50 [-0.99, -0.07]*<br>all/WER miss_rate 3.03 → 2.50，-0.53 [-0.98, -0.15]* |
| customer-hotwords | CER 15.87 → 16.98（+1.11）<br>召回 76.4 → 74.3（-2.1） |
| badcase | CER 27.22 → 28.89（+1.67）<br>召回 63.3 → 63.3（+0.0） |
| ts-hotwords-es | CER 78.53 → 82.49（+3.96）<br>召回 61.4 → 61.1（-0.3） |
| crosslingual-es | all/WER error_rate 75.24 → 77.25，+2.01 [+0.11, +3.96]* **更差**<br>all/WER false_alarm_rate 0.21 → 0.21，+0.00 [+0.00, +0.00]<br>all/WER miss_rate 20.28 → 20.42，+0.14 [-1.10, +1.30] |

显著更差的项：es-asr: all/WER error_rate；ts-es-v1: all/WER error_rate, all/WER false_alarm_rate；first-other: all/CER miss_rate；ts-es-v2: all/WER error_rate, all/WER false_alarm_rate；babble-es: all/WER error_rate, all/WER miss_rate；crosslingual-es: all/WER error_rate
