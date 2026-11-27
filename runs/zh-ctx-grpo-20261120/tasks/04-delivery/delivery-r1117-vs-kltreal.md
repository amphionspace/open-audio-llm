# 交付评估：kltreal 对 r1117（同一推理栈，差 = kltreal − r1117）

| 项目 | 结果 |
|---|---|
| v12 | all/CER error_rate 12.78 → 13.11，+0.33 [+0.13, +0.55]* **更差**<br>all/CER false_alarm_rate 2.46 → 2.65，+0.19 [+0.04, +0.33]* **更差**<br>all/CER miss_rate 2.76 → 2.93，+0.18 [+0.02, +0.33]* **更差** |
| es-asr | all/WER error_rate 5.25 → 5.27，+0.03 [-0.01, +0.06]<br>all/WER miss_rate 0.02 → 0.02，+0.00 [+0.00, +0.00] |
| ts-es-v1 | all/WER error_rate 27.28 → 28.13，+0.85 [+0.57, +1.15]* **更差**<br>all/WER false_alarm_rate 3.21 → 3.01，-0.20 [-0.50, +0.08]<br>all/WER miss_rate 3.86 → 4.09，+0.23 [+0.04, +0.44]* **更差** |
| babble4 | all/CER error_rate 1.90 → 2.12，+0.22 [+0.03, +0.43]* **更差**<br>all/CER false_alarm_rate 4.95 → 4.61，-0.34 [-0.78, +0.09]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 1.92 → 2.05，+0.12 [+0.01, +0.23]* **更差**<br>all/WER false_alarm_rate 1.75 → 1.62，-0.13 [-0.39, +0.00]<br>all/WER miss_rate 0.10 → 0.00，-0.10 [-0.31, +0.00] |
| babble3 | all/CER error_rate 1.90 → 2.12，+0.22 [+0.03, +0.41]* **更差**<br>all/CER false_alarm_rate 1.25 → 0.75，-0.50 [-1.26, +0.00]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 1.92 → 2.05，+0.12 [+0.02, +0.23]* **更差**<br>all/WER false_alarm_rate 1.75 → 1.62，-0.13 [-0.39, +0.00]<br>all/WER miss_rate 0.10 → 0.00，-0.10 [-0.31, +0.00] |
| customer-ts | all/CER error_rate 18.75 → 17.71，-1.04 [-2.50, +0.00]<br>all/CER false_alarm_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER miss_rate 20.00 → 20.00，+0.00 [+0.00, +0.00] |
| customer-asr | all/CER error_rate 20.15 → 22.34，+2.19 [-1.03, +5.90]<br>all/CER miss_rate 0.65 → 0.00，-0.65 [-1.94, +0.00] |
| ts-hotwords | CER 14.43 → 14.83（+0.40）<br>召回 89.9 → 90.0（+0.0） |
| ts-en | all/WER error_rate 13.59 → 13.94，+0.35 [+0.14, +0.56]* **更差**<br>all/WER false_alarm_rate 6.20 → 5.57，-0.63 [-0.87, -0.41]*<br>all/WER miss_rate 2.94 → 3.41，+0.47 [+0.30, +0.64]* **更差** |
| first-other | all/CER error_rate 20.11 → 20.57，+0.46 [+0.21, +0.69]* **更差**<br>all/CER false_alarm_rate 2.58 → 2.43，-0.15 [-0.40, +0.12]<br>all/CER miss_rate 1.66 → 1.84，+0.18 [-0.06, +0.42] |
| asr-benchmark | all/CER error_rate 3.56 → 3.64，+0.08 [-0.00, +0.13]<br>all/CER miss_rate 0.13 → 0.12，-0.00 [-0.01, +0.00]<br>all/CER term_hit_rate 98.77 → 98.79，+0.02 [-0.07, +0.11]<br>all/WER error_rate 7.56 → 7.72，+0.17 [+0.11, +0.23]* **更差**<br>all/WER false_alarm_rate 24.00 → 23.61，-0.39 [-0.81, +0.06]<br>all/WER miss_rate 0.45 → 0.56，+0.11 [+0.07, +0.16]* **更差** |
| hotword-bench | CER 3.27 → 3.33（+0.06）<br>召回 87.9 → 87.8（-0.1） |
| police | all/CER error_rate 0.27 → 0.29，+0.02 [+0.00, +0.03]* **更差**<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.75 → 99.70，-0.06 [-0.13, +0.02] |
| police-single | all/CER error_rate 0.38 → 0.43，+0.05 [-0.06, +0.16]<br>all/CER miss_rate 0.00 → 0.03，+0.03 [+0.00, +0.09]<br>all/CER term_hit_rate 99.08 → 98.99，-0.09 [-0.34, +0.12] |
| crosslingual | all/CER error_rate 34.29 → 35.95，+1.66 [-0.38, +3.86]<br>all/CER false_alarm_rate 0.91 → 1.02，+0.11 [-0.35, +0.59]<br>all/CER miss_rate 14.62 → 15.76，+1.14 [+0.38, +1.93]* **更差** |
| scene-asr | all/CER error_rate 4.72 → 4.66，-0.06 [-0.40, +0.27]<br>all/CER miss_rate 0.85 → 1.00，+0.15 [+0.02, +0.30]* **更差** |
| crosslingual-en | all/WER error_rate 17.88 → 18.30，+0.42 [-0.29, +1.17]<br>all/WER false_alarm_rate 2.73 → 2.16，-0.57 [-1.32, +0.12]<br>all/WER miss_rate 9.17 → 10.53，+1.36 [+0.69, +2.17]* **更差** |
| scene-ts | all/CER error_rate 3.43 → 4.08，+0.65 [+0.19, +1.17]* **更差**<br>all/CER false_alarm_rate 21.43 → 24.29，+2.86 [-2.56, +8.82]<br>all/CER miss_rate 0.33 → 0.33，+0.00 [-0.16, +0.16] |
| ts-es-v2 | all/WER error_rate 27.49 → 28.35，+0.86 [+0.58, +1.14]* **更差**<br>all/WER false_alarm_rate 3.03 → 2.86，-0.17 [-0.41, +0.07]<br>all/WER miss_rate 3.40 → 3.79，+0.38 [+0.18, +0.59]* **更差** |
| ctx-v1 | all/CER error_rate 7.87 → 6.31，-1.56 [-1.99, -1.14]*<br>all/CER miss_rate 5.32 → 3.97，-1.35 [-1.79, -0.92]* |
| babble-es | all/WER error_rate 7.37 → 7.70，+0.33 [+0.03, +0.63]* **更差**<br>all/WER false_alarm_rate 0.50 → 0.50，+0.00 [+0.00, +0.00]<br>all/WER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00] |
| hotword-bench-es | CER 4.97 → 4.98（+0.01）<br>召回 86.8 → 86.7（-0.1） |
| chime4 | all/WER error_rate 9.42 → 9.26，-0.16 [-0.78, +0.42]<br>all/WER miss_rate 5.23 → 5.30，+0.08 [-0.61, +0.76] |
| customer-hotwords | CER 16.30 → 17.66（+1.37）<br>召回 78.5 → 76.4（-2.1） |
| badcase | CER 10.00 → 11.67（+1.67）<br>召回 93.3 → 96.7（+3.3） |
| ts-hotwords-es | CER 26.33 → 27.07（+0.73）<br>召回 83.5 → 83.6（+0.1） |
| crosslingual-es | all/WER error_rate 23.11 → 24.57，+1.46 [+0.05, +2.85]* **更差**<br>all/WER false_alarm_rate 1.04 → 1.46，+0.42 [-0.40, +1.28]<br>all/WER miss_rate 8.06 → 8.75，+0.69 [-0.29, +1.76] |

显著更差的项：v12: all/CER error_rate, all/CER false_alarm_rate, all/CER miss_rate；ts-es-v1: all/WER error_rate, all/WER miss_rate；babble4: all/CER error_rate, all/WER error_rate；babble3: all/CER error_rate, all/WER error_rate；ts-en: all/WER error_rate, all/WER miss_rate；first-other: all/CER error_rate；asr-benchmark: all/WER error_rate, all/WER miss_rate；police: all/CER error_rate；crosslingual: all/CER miss_rate；scene-asr: all/CER miss_rate；crosslingual-en: all/WER miss_rate；scene-ts: all/CER error_rate；ts-es-v2: all/WER error_rate, all/WER miss_rate；babble-es: all/WER error_rate；crosslingual-es: all/WER error_rate
