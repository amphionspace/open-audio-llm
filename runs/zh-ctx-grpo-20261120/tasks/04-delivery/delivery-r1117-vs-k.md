# 交付评估：k 对 r1117（同一推理栈，差 = k − r1117）

| 项目 | 结果 |
|---|---|
| v12 | all/CER error_rate 12.78 → 14.38，+1.60 [+1.31, +1.90]* **更差**<br>all/CER false_alarm_rate 2.46 → 2.05，-0.41 [-0.61, -0.22]*<br>all/CER miss_rate 2.76 → 3.75，+1.00 [+0.76, +1.23]* **更差** |
| es-asr | all/WER error_rate 5.25 → 4.98，-0.27 [-0.34, -0.21]*<br>all/WER miss_rate 0.02 → 0.01，-0.01 [-0.03, +0.00] |
| ts-es-v1 | all/WER error_rate 27.28 → 27.65，+0.37 [-0.04, +0.76]<br>all/WER false_alarm_rate 3.21 → 8.78，+5.57 [+5.04, +6.12]* **更差**<br>all/WER miss_rate 3.86 → 2.61，-1.25 [-1.56, -0.94]* |
| babble4 | all/CER error_rate 1.90 → 2.23，+0.32 [+0.08, +0.57]* **更差**<br>all/CER false_alarm_rate 4.95 → 6.73，+1.78 [+1.08, +2.49]* **更差**<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 1.92 → 2.28，+0.36 [+0.19, +0.52]* **更差**<br>all/WER false_alarm_rate 1.75 → 2.25，+0.50 [+0.00, +1.13]<br>all/WER miss_rate 0.10 → 0.00，-0.10 [-0.31, +0.00] |
| babble3 | all/CER error_rate 1.90 → 2.23，+0.32 [+0.08, +0.56]* **更差**<br>all/CER false_alarm_rate 1.25 → 0.50，-0.75 [-1.71, +0.00]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 1.92 → 2.28，+0.36 [+0.19, +0.53]* **更差**<br>all/WER false_alarm_rate 1.75 → 2.25，+0.50 [-0.00, +1.14]<br>all/WER miss_rate 0.10 → 0.00，-0.10 [-0.31, +0.00] |
| customer-ts | all/CER error_rate 18.75 → 17.71，-1.04 [-2.50, +0.00]<br>all/CER false_alarm_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER miss_rate 20.00 → 20.00，+0.00 [+0.00, +0.00] |
| customer-asr | all/CER error_rate 20.15 → 21.61，+1.46 [-1.05, +4.27]<br>all/CER miss_rate 0.65 → 0.65，+0.00 [+0.00, +0.00] |
| ts-hotwords | CER 14.43 → 15.59（+1.16）<br>召回 89.9 → 91.8（+1.9） |
| ts-en | all/WER error_rate 13.59 → 14.62，+1.03 [+0.78, +1.30]* **更差**<br>all/WER false_alarm_rate 6.20 → 6.69，+0.49 [+0.13, +0.85]* **更差**<br>all/WER miss_rate 2.94 → 3.04，+0.10 [-0.12, +0.33] |
| first-other | all/CER error_rate 20.11 → 22.33，+2.22 [+1.88, +2.53]* **更差**<br>all/CER false_alarm_rate 2.58 → 2.58，+0.00 [-0.34, +0.34]<br>all/CER miss_rate 1.66 → 1.88，+0.22 [-0.10, +0.54] |
| asr-benchmark | all/CER error_rate 3.56 → 3.70，+0.14 [+0.06, +0.19]* **更差**<br>all/CER miss_rate 0.13 → 0.07，-0.06 [-0.07, -0.05]*<br>all/CER term_hit_rate 98.77 → 98.75，-0.01 [-0.13, +0.10]<br>all/WER error_rate 7.56 → 7.45，-0.11 [-0.17, -0.05]*<br>all/WER false_alarm_rate 24.00 → 33.00，+9.00 [+8.29, +9.77]* **更差**<br>all/WER miss_rate 0.45 → 0.12，-0.33 [-0.38, -0.27]* |
| hotword-bench | CER 3.27 → 3.19（-0.08）<br>召回 87.9 → 88.2（+0.3） |
| police | all/CER error_rate 0.27 → 0.27，-0.01 [-0.02, +0.01]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.75 → 99.69，-0.06 [-0.14, +0.01] |
| police-single | all/CER error_rate 0.38 → 0.35，-0.02 [-0.14, +0.09]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.08 → 99.17，+0.09 [-0.09, +0.28] |
| crosslingual | all/CER error_rate 34.29 → 38.72，+4.44 [+1.82, +6.94]* **更差**<br>all/CER false_alarm_rate 0.91 → 0.34，-0.57 [-1.22, +0.00]<br>all/CER miss_rate 14.62 → 22.50，+7.88 [+6.18, +9.59]* **更差** |
| scene-asr | all/CER error_rate 4.72 → 2.01，-2.72 [-3.26, -2.25]*<br>all/CER miss_rate 0.85 → 0.22，-0.63 [-0.87, -0.41]* |
| crosslingual-en | all/WER error_rate 17.88 → 19.91，+2.03 [+1.09, +2.95]* **更差**<br>all/WER false_alarm_rate 2.73 → 2.84，+0.11 [-0.78, +1.09]<br>all/WER miss_rate 9.17 → 11.36，+2.20 [+1.31, +3.18]* **更差** |
| scene-ts | all/CER error_rate 3.43 → 3.62，+0.19 [-0.33, +0.76]<br>all/CER false_alarm_rate 21.43 → 20.00，-1.43 [-8.45, +6.25]<br>all/CER miss_rate 0.33 → 0.33，+0.00 [+0.00, +0.00] |
| ts-es-v2 | all/WER error_rate 27.49 → 27.67，+0.18 [-0.19, +0.54]<br>all/WER false_alarm_rate 3.03 → 8.58，+5.54 [+5.06, +6.04]* **更差**<br>all/WER miss_rate 3.40 → 2.25，-1.15 [-1.41, -0.90]* |
| ctx-v1 | all/CER error_rate 7.87 → 2.14，-5.73 [-6.45, -5.03]*<br>all/CER miss_rate 5.32 → 0.98，-4.34 [-5.05, -3.66]* |
| babble-es | all/WER error_rate 7.37 → 6.82，-0.55 [-1.00, -0.07]*<br>all/WER false_alarm_rate 0.50 → 0.75，+0.25 [+0.00, +0.79]<br>all/WER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00] |
| hotword-bench-es | CER 4.97 → 4.80（-0.17）<br>召回 86.8 → 87.5（+0.7） |
| chime4 | all/WER error_rate 9.42 → 6.54，-2.88 [-3.89, -1.86]*<br>all/WER miss_rate 5.23 → 1.67，-3.56 [-4.55, -2.58]* |
| customer-hotwords | CER 16.30 → 17.83（+1.54）<br>召回 78.5 → 75.7（-2.8） |
| badcase | CER 10.00 → 22.22（+12.22）<br>召回 93.3 → 76.7（-16.7） |
| ts-hotwords-es | CER 26.33 → 28.19（+1.86）<br>召回 83.5 → 86.4（+2.9） |
| crosslingual-es | all/WER error_rate 23.11 → 23.39，+0.29 [-1.61, +2.11]<br>all/WER false_alarm_rate 1.04 → 4.17，+3.12 [+1.49, +4.92]* **更差**<br>all/WER miss_rate 8.06 → 7.92，-0.14 [-1.67, +1.36] |

显著更差的项：v12: all/CER error_rate, all/CER miss_rate；ts-es-v1: all/WER false_alarm_rate；babble4: all/CER error_rate, all/CER false_alarm_rate, all/WER error_rate；babble3: all/CER error_rate, all/WER error_rate；ts-en: all/WER error_rate, all/WER false_alarm_rate；first-other: all/CER error_rate；asr-benchmark: all/CER error_rate, all/WER false_alarm_rate；crosslingual: all/CER error_rate, all/CER miss_rate；crosslingual-en: all/WER error_rate, all/WER miss_rate；ts-es-v2: all/WER false_alarm_rate；crosslingual-es: all/WER false_alarm_rate
