# 交付评估：kgrpo 对 k0grpo（同一推理栈，差 = kgrpo − k0grpo）

| 项目 | 结果 |
|---|---|
| v12 | all/CER error_rate 12.51 → 12.86，+0.35 [+0.10, +0.60]* **更差**<br>all/CER false_alarm_rate 3.95 → 3.46，-0.49 [-0.70, -0.29]*<br>all/CER miss_rate 1.85 → 2.13，+0.29 [+0.14, +0.45]* **更差** |
| es-asr | all/WER error_rate 4.92 → 4.91，-0.01 [-0.07, +0.04]<br>all/WER miss_rate 0.02 → 0.03，+0.01 [+0.00, +0.02] |
| ts-es-v1 | all/WER error_rate 26.27 → 26.41，+0.14 [-0.26, +0.54]<br>all/WER false_alarm_rate 4.40 → 9.26，+4.86 [+4.31, +5.38]* **更差**<br>all/WER miss_rate 3.98 → 2.31，-1.67 [-1.97, -1.37]* |
| babble4 | all/CER error_rate 2.05 → 2.28，+0.24 [+0.02, +0.50]* **更差**<br>all/CER false_alarm_rate 4.91 → 3.81，-1.10 [-1.71, -0.56]*<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 2.15 → 2.12，-0.03 [-0.20, +0.12]<br>all/WER false_alarm_rate 1.62 → 2.00，+0.38 [+0.00, +0.86]<br>all/WER miss_rate 0.10 → 0.00，-0.10 [-0.31, +0.00] |
| babble3 | all/CER error_rate 2.05 → 2.28，+0.24 [+0.01, +0.52]* **更差**<br>all/CER false_alarm_rate 0.25 → 1.00，+0.75 [-0.26, +1.92]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 2.15 → 2.12，-0.03 [-0.19, +0.11]<br>all/WER false_alarm_rate 1.62 → 2.00，+0.38 [+0.00, +0.87]<br>all/WER miss_rate 0.10 → 0.00，-0.10 [-0.31, +0.00] |
| customer-ts | all/CER error_rate 5.21 → 17.71，+12.50 [-2.36, +53.42]<br>all/CER false_alarm_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER miss_rate 0.00 → 20.00，+20.00 [+0.00, +60.00] |
| customer-asr | all/CER error_rate 17.81 → 20.29，+2.48 [+0.00, +5.12]<br>all/CER miss_rate 0.65 → 0.65，+0.00 [+0.00, +0.00] |
| ts-hotwords | CER 14.06 → 14.57（+0.51）<br>召回 93.6 → 93.2（-0.5） |
| ts-en | all/WER error_rate 13.39 → 13.64，+0.25 [+0.01, +0.48]* **更差**<br>all/WER false_alarm_rate 6.30 → 7.61，+1.31 [+0.97, +1.63]* **更差**<br>all/WER miss_rate 2.33 → 2.21，-0.13 [-0.31, +0.05] |
| first-other | all/CER error_rate 19.97 → 20.62，+0.65 [+0.38, +0.94]* **更差**<br>all/CER false_alarm_rate 4.77 → 2.31，-2.46 [-2.94, -1.96]*<br>all/CER miss_rate 0.60 → 2.28，+1.68 [+1.35, +2.05]* **更差** |
| asr-benchmark | all/CER error_rate 3.59 → 3.57，-0.03 [-0.12, +0.03]<br>all/CER miss_rate 0.13 → 0.14，+0.01 [+0.01, +0.02]* **更差**<br>all/CER term_hit_rate 98.67 → 98.64，-0.03 [-0.13, +0.07]<br>all/WER error_rate 7.45 → 7.44，-0.01 [-0.05, +0.04]<br>all/WER false_alarm_rate 24.29 → 22.51，-1.78 [-2.21, -1.37]*<br>all/WER miss_rate 0.37 → 0.40，+0.03 [-0.01, +0.07] |
| hotword-bench | CER 3.01 → 2.99（-0.03）<br>召回 88.3 → 88.2（-0.1） |
| police | all/CER error_rate 0.26 → 0.27，+0.00 [-0.01, +0.02]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.68 → 99.68，+0.00 [-0.08, +0.09] |
| police-single | all/CER error_rate 0.37 → 0.36，-0.01 [-0.12, +0.10]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.14 → 99.17，+0.03 [-0.15, +0.25] |
| crosslingual | all/CER error_rate 32.77 → 30.60，-2.16 [-4.70, +0.36]<br>all/CER false_alarm_rate 1.70 → 1.36，-0.34 [-0.94, +0.23]<br>all/CER miss_rate 14.62 → 13.56，-1.06 [-1.99, -0.15]* |
| scene-asr | all/CER error_rate 2.24 → 2.03，-0.21 [-0.40, +0.01]<br>all/CER miss_rate 0.74 → 0.35，-0.39 [-0.59, -0.20]* |
| crosslingual-en | all/WER error_rate 18.46 → 16.23，-2.22 [-3.24, -1.24]*<br>all/WER false_alarm_rate 3.30 → 3.86，+0.57 [-0.12, +1.33]<br>all/WER miss_rate 8.94 → 7.58，-1.36 [-2.30, -0.47]* |
| scene-ts | all/CER error_rate 3.92 → 4.99，+1.06 [+0.29, +1.90]* **更差**<br>all/CER false_alarm_rate 31.43 → 28.57，-2.86 [-10.94, +5.41]<br>all/CER miss_rate 0.27 → 0.22，-0.05 [-0.16, +0.00] |
| ts-es-v2 | all/WER error_rate 26.35 → 26.42，+0.07 [-0.28, +0.43]<br>all/WER false_alarm_rate 4.33 → 9.03，+4.70 [+4.25, +5.16]* **更差**<br>all/WER miss_rate 3.51 → 2.00，-1.51 [-1.78, -1.25]* |
| ctx-v1 | all/CER error_rate 6.50 → 1.31，-5.19 [-5.94, -4.44]*<br>all/CER miss_rate 5.02 → 0.06，-4.96 [-5.66, -4.22]* |
| babble-es | all/WER error_rate 6.67 → 6.53，-0.13 [-0.57, +0.29]<br>all/WER false_alarm_rate 0.50 → 0.50，+0.00 [+0.00, +0.00]<br>all/WER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00] |
| hotword-bench-es | CER 4.66 → 4.70（+0.04）<br>召回 87.9 → 87.6（-0.3） |
| chime4 | all/WER error_rate 8.97 → 11.39，+2.42 [+1.51, +3.36]* **更差**<br>all/WER miss_rate 5.53 → 8.03，+2.50 [+1.52, +3.48]* **更差** |
| customer-hotwords | CER 15.78 → 17.32（+1.54）<br>召回 75.7 → 75.0（-0.7） |
| badcase | CER 8.33 → 10.56（+2.22）<br>召回 90.0 → 83.3（-6.7） |
| ts-hotwords-es | CER 25.66 → 27.11（+1.45）<br>召回 85.2 → 86.5（+1.3） |
| crosslingual-es | all/WER error_rate 24.57 → 20.09，-4.48 [-6.15, -2.90]*<br>all/WER false_alarm_rate 1.88 → 4.38，+2.50 [+1.03, +4.14]* **更差**<br>all/WER miss_rate 9.72 → 4.86，-4.86 [-6.52, -3.24]* |

显著更差的项：v12: all/CER error_rate, all/CER miss_rate；ts-es-v1: all/WER false_alarm_rate；babble4: all/CER error_rate；babble3: all/CER error_rate；ts-en: all/WER error_rate, all/WER false_alarm_rate；first-other: all/CER error_rate, all/CER miss_rate；asr-benchmark: all/CER miss_rate；scene-ts: all/CER error_rate；ts-es-v2: all/WER false_alarm_rate；chime4: all/WER error_rate, all/WER miss_rate；crosslingual-es: all/WER false_alarm_rate
