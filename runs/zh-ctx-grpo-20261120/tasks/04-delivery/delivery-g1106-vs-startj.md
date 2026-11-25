# 交付评估：startj 对 g1106（同一推理栈，差 = startj − g1106）

| 项目 | 结果 |
|---|---|
| v12 | all/CER error_rate 13.40 → 13.60，+0.21 [-0.04, +0.46]<br>all/CER false_alarm_rate 3.57 → 2.67，-0.90 [-1.11, -0.67]*<br>all/CER miss_rate 2.91 → 3.17，+0.26 [+0.08, +0.45]* **更差** |
| es-asr | all/WER error_rate 76.02 → 74.10，-1.92 [-3.01, -0.85]*<br>all/WER miss_rate 0.33 → 0.32，-0.01 [-0.04, +0.04] |
| ts-es-v1 | all/WER error_rate 85.67 → 84.83，-0.84 [-1.95, +0.53]<br>all/WER false_alarm_rate 2.38 → 2.75，+0.37 [+0.09, +0.68]* **更差**<br>all/WER miss_rate 3.97 → 3.55，-0.42 [-0.68, -0.17]* |
| babble4 | all/CER error_rate 2.20 → 2.06，-0.14 [-0.37, +0.07]<br>all/CER false_alarm_rate 6.94 → 6.68，-0.25 [-0.80, +0.30]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 2.46 → 2.50，+0.03 [-0.13, +0.19]<br>all/WER false_alarm_rate 2.50 → 2.62，+0.12 [-0.39, +0.68]<br>all/WER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00] |
| babble3 | all/CER error_rate 2.20 → 2.06，-0.14 [-0.38, +0.07]<br>all/CER false_alarm_rate 0.50 → 0.50，+0.00 [+0.00, +0.00]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 2.46 → 2.50，+0.03 [-0.13, +0.19]<br>all/WER false_alarm_rate 2.50 → 2.62，+0.12 [-0.39, +0.70]<br>all/WER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00] |
| customer-ts | all/CER error_rate 4.17 → 5.21，+1.04 [+0.00, +2.50]<br>all/CER false_alarm_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00] |
| customer-asr | all/CER error_rate 19.85 → 23.36，+3.50 [-2.85, +13.76]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00] |
| ts-hotwords | CER 15.12 → 15.24（+0.12）<br>召回 93.0 → 92.5（-0.6） |
| ts-en | all/WER error_rate 14.60 → 14.27，-0.33 [-0.59, -0.05]*<br>all/WER false_alarm_rate 6.23 → 7.21，+0.98 [+0.69, +1.30]* **更差**<br>all/WER miss_rate 3.15 → 3.05，-0.10 [-0.31, +0.10] |
| first-other | all/CER error_rate 21.39 → 21.69，+0.30 [-0.00, +0.61]<br>all/CER false_alarm_rate 4.02 → 3.65，-0.37 [-0.77, +0.02]<br>all/CER miss_rate 0.96 → 1.10，+0.14 [-0.10, +0.38] |
| asr-benchmark | all/CER error_rate 3.69 → 3.67，-0.01 [-0.12, +0.05]<br>all/CER miss_rate 0.06 → 0.04，-0.01 [-0.02, -0.01]*<br>all/CER term_hit_rate 98.74 → 98.64，-0.10 [-0.20, +0.00]<br>all/WER error_rate 7.40 → 7.38，-0.02 [-0.06, +0.01]<br>all/WER false_alarm_rate 36.71 → 39.79，+3.08 [+2.46, +3.75]* **更差**<br>all/WER miss_rate 0.14 → 0.11，-0.04 [-0.06, -0.02]* |
| hotword-bench | CER 3.85 → 3.77（-0.07）<br>召回 85.1 → 84.4（-0.7） |
| police | all/CER error_rate 0.28 → 0.26，-0.02 [-0.03, -0.00]*<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.69 → 99.64，-0.05 [-0.14, +0.05] |
| police-single | all/CER error_rate 0.33 → 0.33，+0.00 [-0.10, +0.10]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.23 → 99.20，-0.03 [-0.18, +0.12] |
| crosslingual | all/CER error_rate 38.02 → 40.29，+2.27 [+0.20, +4.38]* **更差**<br>all/CER false_alarm_rate 1.59 → 0.45，-1.14 [-1.89, -0.54]*<br>all/CER miss_rate 23.41 → 23.41，+0.00 [-1.29, +1.28] |
| scene-asr | all/CER error_rate 4.61 → 9.01，+4.39 [+3.63, +5.17]* **更差**<br>all/CER miss_rate 0.41 → 0.28，-0.13 [-0.28, +0.02] |
| crosslingual-en | all/WER error_rate 20.19 → 20.58，+0.38 [-0.45, +1.26]<br>all/WER false_alarm_rate 2.50 → 2.50，+0.00 [-0.82, +0.81]<br>all/WER miss_rate 12.42 → 12.58，+0.15 [-0.75, +1.10] |
| scene-ts | all/CER error_rate 3.65 → 3.06，-0.60 [-1.18, -0.01]*<br>all/CER false_alarm_rate 27.14 → 22.86，-4.29 [-13.64, +5.64]<br>all/CER miss_rate 0.22 → 0.33，+0.11 [+0.00, +0.27] |
| ts-es-v2 | all/WER error_rate 86.61 → 85.71，-0.90 [-1.86, +0.27]<br>all/WER false_alarm_rate 2.20 → 2.60，+0.41 [+0.15, +0.67]* **更差**<br>all/WER miss_rate 3.56 → 3.17，-0.39 [-0.62, -0.17]* |
| ctx-v1 | all/CER error_rate 7.72 → 1.61，-6.11 [-6.89, -5.36]*<br>all/CER miss_rate 5.45 → 0.37，-5.08 [-5.82, -4.31]* |
| babble-es | all/WER error_rate 114.93 → 112.19，-2.74 [-5.90, +0.44]<br>all/WER false_alarm_rate 0.50 → 0.50，+0.00 [+0.00, +0.00]<br>all/WER miss_rate 0.60 → 0.70，+0.10 [-0.20, +0.50] |
| hotword-bench-es | CER 60.86 → 57.19（-3.67）<br>召回 42.2 → 43.4（+1.2） |
| chime4 | all/WER error_rate 7.16 → 7.16，+0.01 [-0.49, +0.55]<br>all/WER miss_rate 2.50 → 2.50，+0.00 [-0.68, +0.61] |
| customer-hotwords | CER 16.98 → 17.06（+0.09）<br>召回 74.3 → 74.3（+0.0） |
| badcase | CER 28.89 → 30.56（+1.67）<br>召回 63.3 → 60.0（-3.3） |
| ts-hotwords-es | CER 82.49 → 81.51（-0.98）<br>召回 61.1 → 63.3（+2.1） |
| crosslingual-es | all/WER error_rate 77.25 → 75.06，-2.19 [-4.19, -0.20]*<br>all/WER false_alarm_rate 0.21 → 0.62，+0.42 [+0.00, +1.06]<br>all/WER miss_rate 20.42 → 20.42，+0.00 [-1.25, +1.26] |

显著更差的项：v12: all/CER miss_rate；ts-es-v1: all/WER false_alarm_rate；ts-en: all/WER false_alarm_rate；asr-benchmark: all/WER false_alarm_rate；crosslingual: all/CER error_rate；scene-asr: all/CER error_rate；ts-es-v2: all/WER false_alarm_rate
