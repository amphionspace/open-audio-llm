# 交付评估：startj 对 control1106（同一推理栈，差 = startj − control1106）

| 项目 | 结果 |
|---|---|
| v12 | all/CER error_rate 13.32 → 13.60，+0.28 [+0.03, +0.53]* **更差**<br>all/CER false_alarm_rate 3.51 → 2.67，-0.84 [-1.06, -0.62]*<br>all/CER miss_rate 3.03 → 3.17，+0.14 [-0.06, +0.33] |
| es-asr | all/WER error_rate 71.00 → 74.10，+3.09 [+2.38, +3.92]* **更差**<br>all/WER miss_rate 0.33 → 0.32，-0.01 [-0.05, +0.03] |
| ts-es-v1 | all/WER error_rate 81.84 → 84.83，+2.98 [+1.83, +4.43]* **更差**<br>all/WER false_alarm_rate 1.99 → 2.75，+0.76 [+0.46, +1.07]* **更差**<br>all/WER miss_rate 3.91 → 3.55，-0.36 [-0.62, -0.09]* |
| babble4 | all/CER error_rate 2.24 → 2.06，-0.18 [-0.42, +0.05]<br>all/CER false_alarm_rate 6.85 → 6.68，-0.17 [-0.68, +0.42]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 2.46 → 2.50，+0.03 [-0.12, +0.20]<br>all/WER false_alarm_rate 2.25 → 2.62，+0.38 [-0.25, +1.09]<br>all/WER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00] |
| babble3 | all/CER error_rate 2.24 → 2.06，-0.18 [-0.43, +0.04]<br>all/CER false_alarm_rate 0.50 → 0.50，+0.00 [+0.00, +0.00]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 2.46 → 2.50，+0.03 [-0.13, +0.20]<br>all/WER false_alarm_rate 2.25 → 2.62，+0.38 [-0.25, +1.10]<br>all/WER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00] |
| customer-ts | all/CER error_rate 4.17 → 5.21，+1.04 [+0.00, +2.50]<br>all/CER false_alarm_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00] |
| customer-asr | all/CER error_rate 17.81 → 23.36，+5.55 [-0.12, +14.91]<br>all/CER miss_rate 0.65 → 0.00，-0.65 [-1.94, +0.00] |
| ts-hotwords | CER 15.10 → 15.24（+0.14）<br>召回 92.2 → 92.5（+0.2） |
| ts-en | all/WER error_rate 14.60 → 14.27，-0.33 [-0.58, -0.06]*<br>all/WER false_alarm_rate 6.22 → 7.21，+0.99 [+0.69, +1.30]* **更差**<br>all/WER miss_rate 3.45 → 3.05，-0.39 [-0.60, -0.18]* |
| first-other | all/CER error_rate 22.20 → 21.69，-0.51 [-0.83, -0.21]*<br>all/CER false_alarm_rate 4.10 → 3.65，-0.45 [-0.83, -0.07]*<br>all/CER miss_rate 0.74 → 1.10，+0.36 [+0.18, +0.56]* **更差** |
| asr-benchmark | all/CER error_rate 3.67 → 3.67，+0.00 [-0.03, +0.03]<br>all/CER miss_rate 0.05 → 0.04，-0.01 [-0.01, -0.00]*<br>all/CER term_hit_rate 98.78 → 98.64，-0.14 [-0.26, -0.03]* **更差**<br>all/WER error_rate 7.42 → 7.38，-0.04 [-0.08, -0.01]*<br>all/WER false_alarm_rate 38.38 → 39.79，+1.41 [+0.81, +2.04]* **更差**<br>all/WER miss_rate 0.14 → 0.11，-0.04 [-0.06, -0.01]* |
| hotword-bench | CER 3.83 → 3.77（-0.05）<br>召回 85.0 → 84.4（-0.6） |
| police | all/CER error_rate 0.27 → 0.26，-0.01 [-0.03, +0.00]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.70 → 99.64，-0.06 [-0.16, +0.04] |
| police-single | all/CER error_rate 0.35 → 0.33，-0.02 [-0.14, +0.08]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.17 → 99.20，+0.03 [-0.15, +0.21] |
| crosslingual | all/CER error_rate 39.11 → 40.29，+1.19 [-1.20, +3.79]<br>all/CER false_alarm_rate 1.59 → 0.45，-1.14 [-1.94, -0.42]*<br>all/CER miss_rate 22.65 → 23.41，+0.76 [-0.60, +2.11] |
| scene-asr | all/CER error_rate 4.97 → 9.01，+4.04 [+3.27, +4.80]* **更差**<br>all/CER miss_rate 0.48 → 0.28，-0.20 [-0.37, -0.04]* |
| crosslingual-en | all/WER error_rate 21.86 → 20.58，-1.29 [-2.28, -0.28]*<br>all/WER false_alarm_rate 2.39 → 2.50，+0.11 [-0.58, +0.83]<br>all/WER miss_rate 13.94 → 12.58，-1.36 [-2.45, -0.30]* |
| scene-ts | all/CER error_rate 3.38 → 3.06，-0.33 [-0.98, +0.32]<br>all/CER false_alarm_rate 24.29 → 22.86，-1.43 [-10.53, +8.33]<br>all/CER miss_rate 0.27 → 0.33，+0.05 [+0.00, +0.16] |
| ts-es-v2 | all/WER error_rate 82.78 → 85.71，+2.94 [+1.88, +4.20]* **更差**<br>all/WER false_alarm_rate 1.88 → 2.60，+0.73 [+0.47, +1.00]* **更差**<br>all/WER miss_rate 3.50 → 3.17，-0.33 [-0.58, -0.11]* |
| ctx-v1 | all/CER error_rate 7.61 → 1.61，-6.00 [-6.74, -5.28]*<br>all/CER miss_rate 5.36 → 0.37，-4.99 [-5.69, -4.28]* |
| babble-es | all/WER error_rate 110.60 → 112.19，+1.59 [-1.40, +4.74]<br>all/WER false_alarm_rate 0.50 → 0.50，+0.00 [+0.00, +0.00]<br>all/WER miss_rate 0.20 → 0.70，+0.50 [+0.10, +1.00]* **更差** |
| hotword-bench-es | CER 57.15 → 57.19（+0.04）<br>召回 43.1 → 43.4（+0.2） |
| chime4 | all/WER error_rate 7.65 → 7.16，-0.49 [-1.08, +0.08]<br>all/WER miss_rate 3.03 → 2.50，-0.53 [-1.21, +0.15] |
| customer-hotwords | CER 15.87 → 17.06（+1.19）<br>召回 76.4 → 74.3（-2.1） |
| badcase | CER 27.22 → 30.56（+3.33）<br>召回 63.3 → 60.0（-3.3） |
| ts-hotwords-es | CER 78.53 → 81.51（+2.98）<br>召回 61.4 → 63.3（+1.8） |
| crosslingual-es | all/WER error_rate 75.24 → 75.06，-0.18 [-2.45, +1.98]<br>all/WER false_alarm_rate 0.21 → 0.62，+0.42 [+0.00, +1.06]<br>all/WER miss_rate 20.28 → 20.42，+0.14 [-1.11, +1.41] |

显著更差的项：v12: all/CER error_rate；es-asr: all/WER error_rate；ts-es-v1: all/WER error_rate, all/WER false_alarm_rate；ts-en: all/WER false_alarm_rate；first-other: all/CER miss_rate；asr-benchmark: all/CER term_hit_rate, all/WER false_alarm_rate；scene-asr: all/CER error_rate；ts-es-v2: all/WER error_rate, all/WER false_alarm_rate；babble-es: all/WER miss_rate
