# 交付评估：noclean1106 对 base1106（同一推理栈，差 = noclean1106 − base1106）

| 项目 | 结果 |
|---|---|
| v12 | all/CER error_rate 12.92 → 14.51，+1.59 [+1.37, +1.81]* **更差**<br>all/CER false_alarm_rate 3.17 → 2.25，-0.91 [-1.16, -0.67]*<br>all/CER miss_rate 2.82 → 4.50，+1.68 [+1.43, +1.92]* **更差** |
| es-asr | all/WER error_rate 58.09 → 64.30，+6.21 [+5.50, +6.88]* **更差**<br>all/WER miss_rate 0.42 → 0.21，-0.21 [-0.28, -0.15]* |
| ts-es-v1 | all/WER error_rate 70.30 → 70.09，-0.21 [-1.12, +0.64]<br>all/WER false_alarm_rate 4.20 → 1.65，-2.55 [-2.97, -2.14]*<br>all/WER miss_rate 3.18 → 4.20，+1.02 [+0.76, +1.29]* **更差** |
| babble4 | all/CER error_rate 2.08 → 2.15，+0.07 [-0.20, +0.31]<br>all/CER false_alarm_rate 4.82 → 7.28，+2.45 [+1.81, +3.12]* **更差**<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 2.35 → 2.39，+0.05 [-0.11, +0.20]<br>all/WER false_alarm_rate 1.75 → 2.25，+0.50 [-0.12, +1.10]<br>all/WER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00] |
| babble3 | all/CER error_rate 2.08 → 2.15，+0.07 [-0.22, +0.31]<br>all/CER false_alarm_rate 0.50 → 0.50，+0.00 [+0.00, +0.00]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 2.35 → 2.39，+0.05 [-0.10, +0.19]<br>all/WER false_alarm_rate 1.75 → 2.25，+0.50 [+0.00, +1.13]<br>all/WER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00] |
| customer-ts | all/CER error_rate 4.17 → 4.17，+0.00 [+0.00, +0.00]<br>all/CER false_alarm_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00] |
| customer-asr | all/CER error_rate 17.96 → 14.60，-3.36 [-6.11, -1.24]*<br>all/CER miss_rate 0.00 → 0.65，+0.65 [+0.00, +1.94] |
| ts-hotwords | CER 15.20 → 15.82（+0.63）<br>召回 92.9 → 92.1（-0.7） |
| ts-en | all/WER error_rate 13.73 → 14.97，+1.24 [+0.96, +1.52]* **更差**<br>all/WER false_alarm_rate 8.35 → 6.64，-1.71 [-2.04, -1.38]*<br>all/WER miss_rate 2.97 → 4.15，+1.18 [+0.93, +1.44]* **更差** |
| first-other | all/CER error_rate 20.69 → 22.16，+1.47 [+1.15, +1.81]* **更差**<br>all/CER false_alarm_rate 3.03 → 3.10，+0.07 [-0.30, +0.43]<br>all/CER miss_rate 1.14 → 1.22，+0.08 [-0.14, +0.30] |
| asr-benchmark | all/CER error_rate 3.55 → 3.72，+0.17 [+0.14, +0.20]* **更差**<br>all/CER miss_rate 0.09 → 0.04，-0.05 [-0.05, -0.04]*<br>all/CER term_hit_rate 98.83 → 98.65，-0.18 [-0.28, -0.09]* **更差**<br>all/WER error_rate 7.34 → 7.44，+0.10 [+0.05, +0.14]* **更差**<br>all/WER false_alarm_rate 27.95 → 41.59，+13.64 [+12.77, +14.56]* **更差**<br>all/WER miss_rate 0.23 → 0.09，-0.14 [-0.18, -0.11]* |
| hotword-bench | CER 3.68 → 4.04（+0.35）<br>召回 86.8 → 84.3（-2.5） |
| police | all/CER error_rate 0.26 → 0.30，+0.03 [+0.02, +0.05]* **更差**<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.74 → 99.59，-0.15 [-0.24, -0.06]* **更差** |
| police-single | all/CER error_rate 0.36 → 0.37，+0.01 [-0.07, +0.09]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.11 → 99.08，-0.03 [-0.21, +0.15] |
| crosslingual | all/CER error_rate 37.99 → 40.27，+2.28 [-0.17, +4.69]<br>all/CER false_alarm_rate 1.70 → 1.36，-0.34 [-0.94, +0.23]<br>all/CER miss_rate 17.42 → 25.91，+8.48 [+6.95, +10.04]* **更差** |
| scene-asr | all/CER error_rate 4.90 → 4.90，-0.00 [-0.38, +0.36]<br>all/CER miss_rate 0.83 → 0.41，-0.41 [-0.61, -0.24]* |
| crosslingual-en | all/WER error_rate 19.03 → 23.29，+4.26 [+3.12, +5.47]* **更差**<br>all/WER false_alarm_rate 5.00 → 3.18，-1.82 [-2.90, -0.81]*<br>all/WER miss_rate 10.61 → 16.74，+6.14 [+4.88, +7.53]* **更差** |
| scene-ts | all/CER error_rate 3.83 → 3.86，+0.03 [-0.60, +0.70]<br>all/CER false_alarm_rate 34.29 → 22.86，-11.43 [-20.59, -3.39]*<br>all/CER miss_rate 0.33 → 0.44，+0.11 [+0.00, +0.27] |
| ts-es-v2 | all/WER error_rate 70.65 → 70.53，-0.12 [-0.90, +0.67]<br>all/WER false_alarm_rate 4.04 → 1.48，-2.56 [-2.91, -2.23]*<br>all/WER miss_rate 2.87 → 3.80，+0.93 [+0.68, +1.18]* **更差** |
| ctx-v1 | all/CER error_rate 6.29 → 9.79，+3.50 [+2.86, +4.19]* **更差**<br>all/CER miss_rate 3.76 → 7.48，+3.72 [+3.08, +4.40]* **更差** |
| babble-es | all/WER error_rate 94.35 → 99.33，+4.98 [+1.36, +8.75]* **更差**<br>all/WER false_alarm_rate 0.50 → 0.50，+0.00 [+0.00, +0.00]<br>all/WER miss_rate 0.80 → 0.30，-0.50 [-0.99, -0.10]* |
| hotword-bench-es | CER 48.10 → 48.97（+0.88）<br>召回 49.3 → 46.9（-2.4） |
| chime4 | all/WER error_rate 7.72 → 6.04，-1.69 [-2.51, -0.93]*<br>all/WER miss_rate 4.02 → 0.76，-3.26 [-4.24, -2.35]* |
| customer-hotwords | CER 14.16 → 14.16（+0.00）<br>召回 80.6 → 75.0（-5.6） |
| badcase | CER 26.11 → 21.11（-5.00）<br>召回 76.7 → 73.3（-3.3） |
| ts-hotwords-es | CER 71.33 → 67.06（-4.26）<br>召回 66.7 → 67.5（+0.8） |
| crosslingual-es | all/WER error_rate 65.60 → 68.48，+2.88 [+0.45, +5.37]* **更差**<br>all/WER false_alarm_rate 1.46 → 0.00，-1.46 [-2.67, -0.44]*<br>all/WER miss_rate 18.06 → 22.50，+4.44 [+2.63, +6.20]* **更差** |

显著更差的项：v12: all/CER error_rate, all/CER miss_rate；es-asr: all/WER error_rate；ts-es-v1: all/WER miss_rate；babble4: all/CER false_alarm_rate；ts-en: all/WER error_rate, all/WER miss_rate；first-other: all/CER error_rate；asr-benchmark: all/CER error_rate, all/CER term_hit_rate, all/WER error_rate, all/WER false_alarm_rate；police: all/CER error_rate, all/CER term_hit_rate；crosslingual: all/CER miss_rate；crosslingual-en: all/WER error_rate, all/WER miss_rate；ts-es-v2: all/WER miss_rate；ctx-v1: all/CER error_rate, all/CER miss_rate；babble-es: all/WER error_rate；crosslingual-es: all/WER error_rate, all/WER miss_rate
