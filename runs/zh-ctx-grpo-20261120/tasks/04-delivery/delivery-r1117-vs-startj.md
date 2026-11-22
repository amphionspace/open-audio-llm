# 交付评估：startj 对 r1117（同一推理栈，差 = startj − r1117）

| 项目 | 结果 |
|---|---|
| v12 | all/CER error_rate 12.78 → 13.60，+0.83 [+0.56, +1.12]* **更差**<br>all/CER false_alarm_rate 2.46 → 2.67，+0.20 [-0.02, +0.43]<br>all/CER miss_rate 2.76 → 3.17，+0.42 [+0.19, +0.63]* **更差** |
| es-asr | all/WER error_rate 5.25 → 74.10，+68.85 [+67.67, +69.99]* **更差**<br>all/WER miss_rate 0.02 → 0.32，+0.30 [+0.22, +0.38]* **更差** |
| ts-es-v1 | all/WER error_rate 27.28 → 84.83，+57.54 [+56.17, +59.15]* **更差**<br>all/WER false_alarm_rate 3.21 → 2.75，-0.46 [-0.93, +0.02]<br>all/WER miss_rate 3.86 → 3.55，-0.32 [-0.62, +0.00] |
| babble4 | all/CER error_rate 1.90 → 2.06，+0.15 [-0.13, +0.43]<br>all/CER false_alarm_rate 4.95 → 6.68，+1.73 [+0.98, +2.51]* **更差**<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 1.92 → 2.50，+0.57 [+0.37, +0.78]* **更差**<br>all/WER false_alarm_rate 1.75 → 2.62，+0.87 [+0.00, +1.74]<br>all/WER miss_rate 0.10 → 0.00，-0.10 [-0.31, +0.00] |
| babble3 | all/CER error_rate 1.90 → 2.06，+0.15 [-0.11, +0.42]<br>all/CER false_alarm_rate 1.25 → 0.50，-0.75 [-1.71, +0.00]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 1.92 → 2.50，+0.57 [+0.37, +0.78]* **更差**<br>all/WER false_alarm_rate 1.75 → 2.62，+0.87 [+0.00, +1.78]<br>all/WER miss_rate 0.10 → 0.00，-0.10 [-0.31, +0.00] |
| customer-ts | all/CER error_rate 18.75 → 5.21，-13.54 [-53.42, +0.00]<br>all/CER false_alarm_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER miss_rate 20.00 → 0.00，-20.00 [-60.00, +0.00] |
| customer-asr | all/CER error_rate 20.15 → 23.36，+3.21 [-2.87, +13.07]<br>all/CER miss_rate 0.65 → 0.00，-0.65 [-1.94, +0.00] |
| ts-hotwords | CER 14.43 → 15.24（+0.81）<br>召回 89.9 → 92.5（+2.5） |
| ts-en | all/WER error_rate 13.59 → 14.27，+0.68 [+0.43, +0.95]* **更差**<br>all/WER false_alarm_rate 6.20 → 7.21，+1.01 [+0.59, +1.44]* **更差**<br>all/WER miss_rate 2.94 → 3.05，+0.11 [-0.11, +0.34] |
| first-other | all/CER error_rate 20.11 → 21.69，+1.58 [+1.21, +1.95]* **更差**<br>all/CER false_alarm_rate 2.58 → 3.65，+1.07 [+0.66, +1.49]* **更差**<br>all/CER miss_rate 1.66 → 1.10，-0.56 [-0.85, -0.26]* |
| asr-benchmark | all/CER error_rate 3.56 → 3.67，+0.11 [+0.03, +0.17]* **更差**<br>all/CER miss_rate 0.13 → 0.04，-0.09 [-0.10, -0.07]*<br>all/CER term_hit_rate 98.77 → 98.64，-0.13 [-0.24, -0.02]* **更差**<br>all/WER error_rate 7.56 → 7.38，-0.18 [-0.24, -0.11]*<br>all/WER false_alarm_rate 24.00 → 39.79，+15.79 [+14.84, +16.75]* **更差**<br>all/WER miss_rate 0.45 → 0.11，-0.34 [-0.40, -0.29]* |
| hotword-bench | CER 3.27 → 3.77（+0.50）<br>召回 87.9 → 84.4（-3.5） |
| police | all/CER error_rate 0.27 → 0.26，-0.01 [-0.03, +0.00]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.75 → 99.64，-0.11 [-0.21, -0.02]* **更差** |
| police-single | all/CER error_rate 0.38 → 0.33，-0.05 [-0.17, +0.08]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.08 → 99.20，+0.12 [-0.09, +0.34] |
| crosslingual | all/CER error_rate 34.29 → 40.29，+6.01 [+2.98, +8.92]* **更差**<br>all/CER false_alarm_rate 0.91 → 0.45，-0.45 [-1.11, +0.12]<br>all/CER miss_rate 14.62 → 23.41，+8.79 [+7.11, +10.57]* **更差** |
| scene-asr | all/CER error_rate 4.72 → 9.01，+4.28 [+3.50, +5.06]* **更差**<br>all/CER miss_rate 0.85 → 0.28，-0.57 [-0.80, -0.35]* |
| crosslingual-en | all/WER error_rate 17.88 → 20.58，+2.70 [+1.58, +3.90]* **更差**<br>all/WER false_alarm_rate 2.73 → 2.50，-0.23 [-1.27, +0.90]<br>all/WER miss_rate 9.17 → 12.58，+3.41 [+2.33, +4.56]* **更差** |
| scene-ts | all/CER error_rate 3.43 → 3.06，-0.38 [-0.89, +0.20]<br>all/CER false_alarm_rate 21.43 → 22.86，+1.43 [-7.81, +11.32]<br>all/CER miss_rate 0.33 → 0.33，+0.00 [-0.16, +0.16] |
| ts-es-v2 | all/WER error_rate 27.49 → 85.71，+58.23 [+56.94, +59.62]* **更差**<br>all/WER false_alarm_rate 3.03 → 2.60，-0.43 [-0.80, -0.03]*<br>all/WER miss_rate 3.40 → 3.17，-0.24 [-0.53, +0.05] |
| ctx-v1 | all/CER error_rate 7.87 → 1.61，-6.26 [-7.02, -5.50]*<br>all/CER miss_rate 5.32 → 0.37，-4.96 [-5.69, -4.22]* |
| babble-es | all/WER error_rate 7.37 → 112.19，+104.83 [+101.70, +107.96]* **更差**<br>all/WER false_alarm_rate 0.50 → 0.50，+0.00 [+0.00, +0.00]<br>all/WER miss_rate 0.00 → 0.70，+0.70 [+0.20, +1.28]* **更差** |
| hotword-bench-es | CER 4.97 → 57.19（+52.22）<br>召回 86.8 → 43.4（-43.4） |
| chime4 | all/WER error_rate 9.42 → 7.16，-2.26 [-3.36, -1.20]*<br>all/WER miss_rate 5.23 → 2.50，-2.73 [-3.79, -1.67]* |
| customer-hotwords | CER 16.30 → 17.06（+0.77）<br>召回 78.5 → 74.3（-4.2） |
| badcase | CER 10.00 → 30.56（+20.56）<br>召回 93.3 → 60.0（-33.3） |
| ts-hotwords-es | CER 26.33 → 81.51（+55.17）<br>召回 83.5 → 63.3（-20.2） |
| crosslingual-es | all/WER error_rate 23.11 → 75.06，+51.96 [+48.43, +55.58]* **更差**<br>all/WER false_alarm_rate 1.04 → 0.62，-0.42 [-1.65, +0.81]<br>all/WER miss_rate 8.06 → 20.42，+12.36 [+9.97, +14.89]* **更差** |

显著更差的项：v12: all/CER error_rate, all/CER miss_rate；es-asr: all/WER error_rate, all/WER miss_rate；ts-es-v1: all/WER error_rate；babble4: all/CER false_alarm_rate, all/WER error_rate；babble3: all/WER error_rate；ts-en: all/WER error_rate, all/WER false_alarm_rate；first-other: all/CER error_rate, all/CER false_alarm_rate；asr-benchmark: all/CER error_rate, all/CER term_hit_rate, all/WER false_alarm_rate；police: all/CER term_hit_rate；crosslingual: all/CER error_rate, all/CER miss_rate；scene-asr: all/CER error_rate；crosslingual-en: all/WER error_rate, all/WER miss_rate；ts-es-v2: all/WER error_rate；babble-es: all/WER error_rate, all/WER miss_rate；crosslingual-es: all/WER error_rate, all/WER miss_rate
