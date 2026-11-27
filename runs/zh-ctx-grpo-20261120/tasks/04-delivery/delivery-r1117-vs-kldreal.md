# 交付评估：kldreal 对 r1117（同一推理栈，差 = kldreal − r1117）

| 项目 | 结果 |
|---|---|
| v12 | all/CER error_rate 12.78 → 13.03，+0.25 [+0.03, +0.49]* **更差**<br>all/CER false_alarm_rate 2.46 → 3.78，+1.32 [+1.09, +1.56]* **更差**<br>all/CER miss_rate 2.76 → 2.25，-0.51 [-0.71, -0.32]* |
| es-asr | all/WER error_rate 5.25 → 5.31，+0.06 [+0.02, +0.10]* **更差**<br>all/WER miss_rate 0.02 → 0.02，+0.00 [+0.00, +0.00] |
| ts-es-v1 | all/WER error_rate 27.28 → 28.62，+1.34 [+1.01, +1.67]* **更差**<br>all/WER false_alarm_rate 3.21 → 2.84，-0.37 [-0.71, -0.05]*<br>all/WER miss_rate 3.86 → 4.28，+0.42 [+0.19, +0.66]* **更差** |
| babble4 | all/CER error_rate 1.90 → 1.99，+0.09 [-0.11, +0.28]<br>all/CER false_alarm_rate 4.95 → 8.21，+3.26 [+2.49, +4.03]* **更差**<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 1.92 → 2.17，+0.25 [+0.13, +0.38]* **更差**<br>all/WER false_alarm_rate 1.75 → 3.25，+1.50 [+0.64, +2.43]* **更差**<br>all/WER miss_rate 0.10 → 0.00，-0.10 [-0.31, +0.00] |
| babble3 | all/CER error_rate 1.90 → 1.99，+0.09 [-0.11, +0.27]<br>all/CER false_alarm_rate 1.25 → 0.50，-0.75 [-1.63, +0.00]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 1.92 → 2.17，+0.25 [+0.13, +0.38]* **更差**<br>all/WER false_alarm_rate 1.75 → 3.25，+1.50 [+0.64, +2.44]* **更差**<br>all/WER miss_rate 0.10 → 0.00，-0.10 [-0.31, +0.00] |
| customer-ts | all/CER error_rate 18.75 → 17.71，-1.04 [-2.50, +0.00]<br>all/CER false_alarm_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER miss_rate 20.00 → 20.00，+0.00 [+0.00, +0.00] |
| customer-asr | all/CER error_rate 20.15 → 24.09，+3.94 [+0.32, +8.41]* **更差**<br>all/CER miss_rate 0.65 → 0.00，-0.65 [-1.94, +0.00] |
| ts-hotwords | CER 14.43 → 15.46（+1.03）<br>召回 89.9 → 90.8（+0.9） |
| ts-en | all/WER error_rate 13.59 → 13.88，+0.29 [+0.04, +0.54]* **更差**<br>all/WER false_alarm_rate 6.20 → 7.38，+1.18 [+0.86, +1.49]* **更差**<br>all/WER miss_rate 2.94 → 2.46，-0.48 [-0.67, -0.29]* |
| first-other | all/CER error_rate 20.11 → 20.92，+0.80 [+0.49, +1.12]* **更差**<br>all/CER false_alarm_rate 2.58 → 2.36，-0.22 [-0.63, +0.15]<br>all/CER miss_rate 1.66 → 1.80，+0.14 [-0.14, +0.44] |
| asr-benchmark | all/CER error_rate 3.56 → 3.69，+0.13 [+0.02, +0.27]* **更差**<br>all/CER miss_rate 0.13 → 0.08，-0.05 [-0.06, -0.04]*<br>all/CER term_hit_rate 98.77 → 98.71，-0.05 [-0.15, +0.03]<br>all/WER error_rate 7.56 → 7.39，-0.17 [-0.23, -0.11]*<br>all/WER false_alarm_rate 24.00 → 28.33，+4.33 [+3.76, +4.94]* **更差**<br>all/WER miss_rate 0.45 → 0.16，-0.30 [-0.35, -0.24]* |
| hotword-bench | CER 3.27 → 3.12（-0.15）<br>召回 87.9 → 88.0（+0.2） |
| police | all/CER error_rate 0.27 → 0.28，+0.01 [-0.00, +0.02]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.75 → 99.71，-0.04 [-0.10, +0.02] |
| police-single | all/CER error_rate 0.38 → 0.41，+0.03 [-0.08, +0.14]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.08 → 99.02，-0.06 [-0.31, +0.18] |
| crosslingual | all/CER error_rate 34.29 → 33.32，-0.97 [-3.23, +1.50]<br>all/CER false_alarm_rate 0.91 → 2.05，+1.14 [+0.36, +1.97]* **更差**<br>all/CER miss_rate 14.62 → 11.89，-2.73 [-3.78, -1.75]* |
| scene-asr | all/CER error_rate 4.72 → 4.12，-0.61 [-0.96, -0.27]*<br>all/CER miss_rate 0.85 → 0.43，-0.41 [-0.61, -0.24]* |
| crosslingual-en | all/WER error_rate 17.88 → 16.97，-0.90 [-1.69, -0.15]*<br>all/WER false_alarm_rate 2.73 → 2.27，-0.45 [-1.26, +0.33]<br>all/WER miss_rate 9.17 → 8.18，-0.98 [-1.75, -0.22]* |
| scene-ts | all/CER error_rate 3.43 → 4.56，+1.12 [+0.61, +1.68]* **更差**<br>all/CER false_alarm_rate 21.43 → 20.00，-1.43 [-9.46, +6.06]<br>all/CER miss_rate 0.33 → 0.27，-0.05 [-0.16, +0.00] |
| ts-es-v2 | all/WER error_rate 27.49 → 28.81，+1.32 [+1.02, +1.61]* **更差**<br>all/WER false_alarm_rate 3.03 → 2.89，-0.14 [-0.43, +0.15]<br>all/WER miss_rate 3.40 → 3.83，+0.43 [+0.23, +0.64]* **更差** |
| ctx-v1 | all/CER error_rate 7.87 → 4.94，-2.93 [-3.51, -2.38]*<br>all/CER miss_rate 5.32 → 2.55，-2.77 [-3.35, -2.22]* |
| babble-es | all/WER error_rate 7.37 → 7.77，+0.40 [+0.14, +0.70]* **更差**<br>all/WER false_alarm_rate 0.50 → 2.25，+1.75 [+0.52, +3.09]* **更差**<br>all/WER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00] |
| hotword-bench-es | CER 4.97 → 4.98（+0.01）<br>召回 86.8 → 86.7（-0.1） |
| chime4 | all/WER error_rate 9.42 → 5.76，-3.66 [-4.72, -2.65]*<br>all/WER miss_rate 5.23 → 0.76，-4.47 [-5.61, -3.41]* |
| customer-hotwords | CER 16.30 → 18.94（+2.65）<br>召回 78.5 → 77.8（-0.7） |
| badcase | CER 10.00 → 15.56（+5.56）<br>召回 93.3 → 93.3（+0.0） |
| ts-hotwords-es | CER 26.33 → 27.49（+1.16）<br>召回 83.5 → 83.0（-0.4） |
| crosslingual-es | all/WER error_rate 23.11 → 20.96，-2.15 [-3.83, -0.71]*<br>all/WER false_alarm_rate 1.04 → 1.88，+0.83 [-0.22, +2.05]<br>all/WER miss_rate 8.06 → 5.28，-2.78 [-4.10, -1.55]* |

显著更差的项：v12: all/CER error_rate, all/CER false_alarm_rate；es-asr: all/WER error_rate；ts-es-v1: all/WER error_rate, all/WER miss_rate；babble4: all/CER false_alarm_rate, all/WER error_rate, all/WER false_alarm_rate；babble3: all/WER error_rate, all/WER false_alarm_rate；customer-asr: all/CER error_rate；ts-en: all/WER error_rate, all/WER false_alarm_rate；first-other: all/CER error_rate；asr-benchmark: all/CER error_rate, all/WER false_alarm_rate；crosslingual: all/CER false_alarm_rate；scene-ts: all/CER error_rate；ts-es-v2: all/WER error_rate, all/WER miss_rate；babble-es: all/WER error_rate, all/WER false_alarm_rate
