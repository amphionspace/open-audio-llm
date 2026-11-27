# 交付评估：k0grpo 对 r1117（同一推理栈，差 = k0grpo − r1117）

| 项目 | 结果 |
|---|---|
| v12 | all/CER error_rate 12.78 → 12.51，-0.27 [-0.53, -0.01]*<br>all/CER false_alarm_rate 2.46 → 3.95，+1.49 [+1.26, +1.72]* **更差**<br>all/CER miss_rate 2.76 → 1.85，-0.91 [-1.10, -0.73]* |
| es-asr | all/WER error_rate 5.25 → 4.92，-0.32 [-0.39, -0.26]*<br>all/WER miss_rate 0.02 → 0.02，+0.00 [+0.00, +0.00] |
| ts-es-v1 | all/WER error_rate 27.28 → 26.27，-1.02 [-1.36, -0.67]*<br>all/WER false_alarm_rate 3.21 → 4.40，+1.19 [+0.82, +1.59]* **更差**<br>all/WER miss_rate 3.86 → 3.98，+0.11 [-0.12, +0.33] |
| babble4 | all/CER error_rate 1.90 → 2.05，+0.14 [-0.08, +0.38]<br>all/CER false_alarm_rate 4.95 → 4.91，-0.04 [-0.62, +0.55]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 1.92 → 2.15，+0.23 [+0.06, +0.39]* **更差**<br>all/WER false_alarm_rate 1.75 → 1.62，-0.13 [-0.52, +0.25]<br>all/WER miss_rate 0.10 → 0.10，+0.00 [+0.00, +0.00] |
| babble3 | all/CER error_rate 1.90 → 2.05，+0.14 [-0.10, +0.37]<br>all/CER false_alarm_rate 1.25 → 0.25，-1.00 [-2.05, -0.24]*<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 1.92 → 2.15，+0.23 [+0.07, +0.40]* **更差**<br>all/WER false_alarm_rate 1.75 → 1.62，-0.13 [-0.60, +0.26]<br>all/WER miss_rate 0.10 → 0.10，+0.00 [+0.00, +0.00] |
| customer-ts | all/CER error_rate 18.75 → 5.21，-13.54 [-53.42, +0.00]<br>all/CER false_alarm_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER miss_rate 20.00 → 0.00，-20.00 [-60.00, +0.00] |
| customer-asr | all/CER error_rate 20.15 → 17.81，-2.34 [-5.07, +0.16]<br>all/CER miss_rate 0.65 → 0.65，+0.00 [+0.00, +0.00] |
| ts-hotwords | CER 14.43 → 14.06（-0.37）<br>召回 89.9 → 93.6（+3.7） |
| ts-en | all/WER error_rate 13.59 → 13.39，-0.20 [-0.42, +0.04]<br>all/WER false_alarm_rate 6.20 → 6.30，+0.10 [-0.19, +0.40]<br>all/WER miss_rate 2.94 → 2.33，-0.60 [-0.80, -0.40]* |
| first-other | all/CER error_rate 20.11 → 19.97，-0.15 [-0.47, +0.15]<br>all/CER false_alarm_rate 2.58 → 4.77，+2.19 [+1.69, +2.68]* **更差**<br>all/CER miss_rate 1.66 → 0.60，-1.06 [-1.36, -0.78]* |
| asr-benchmark | all/CER error_rate 3.56 → 3.59，+0.03 [-0.08, +0.15]<br>all/CER miss_rate 0.13 → 0.13，-0.00 [-0.01, +0.01]<br>all/CER term_hit_rate 98.77 → 98.67，-0.09 [-0.20, +0.01]<br>all/WER error_rate 7.56 → 7.45，-0.11 [-0.17, -0.05]*<br>all/WER false_alarm_rate 24.00 → 24.29，+0.29 [-0.09, +0.71]<br>all/WER miss_rate 0.45 → 0.37，-0.08 [-0.13, -0.03]* |
| hotword-bench | CER 3.27 → 3.01（-0.26）<br>召回 87.9 → 88.3（+0.5） |
| police | all/CER error_rate 0.27 → 0.26，-0.01 [-0.03, -0.00]*<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.75 → 99.68，-0.07 [-0.16, +0.01] |
| police-single | all/CER error_rate 0.38 → 0.37，-0.01 [-0.14, +0.14]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.08 → 99.14，+0.06 [-0.18, +0.28] |
| crosslingual | all/CER error_rate 34.29 → 32.77，-1.52 [-4.32, +1.07]<br>all/CER false_alarm_rate 0.91 → 1.70，+0.80 [+0.32, +1.45]* **更差**<br>all/CER miss_rate 14.62 → 14.62，+0.00 [-1.07, +0.99] |
| scene-asr | all/CER error_rate 4.72 → 2.24，-2.49 [-3.00, -2.02]*<br>all/CER miss_rate 0.85 → 0.74，-0.11 [-0.28, +0.04] |
| crosslingual-en | all/WER error_rate 17.88 → 18.46，+0.58 [-0.39, +1.52]<br>all/WER false_alarm_rate 2.73 → 3.30，+0.57 [-0.34, +1.50]<br>all/WER miss_rate 9.17 → 8.94，-0.23 [-1.15, +0.68] |
| scene-ts | all/CER error_rate 3.43 → 3.92，+0.49 [-0.19, +1.22]<br>all/CER false_alarm_rate 21.43 → 31.43，+10.00 [+3.33, +17.86]* **更差**<br>all/CER miss_rate 0.33 → 0.27，-0.05 [-0.16, +0.00] |
| ts-es-v2 | all/WER error_rate 27.49 → 26.35，-1.13 [-1.46, -0.81]*<br>all/WER false_alarm_rate 3.03 → 4.33，+1.30 [+0.98, +1.63]* **更差**<br>all/WER miss_rate 3.40 → 3.51，+0.11 [-0.11, +0.32] |
| ctx-v1 | all/CER error_rate 7.87 → 6.50，-1.37 [-1.83, -0.89]*<br>all/CER miss_rate 5.32 → 5.02，-0.31 [-0.71, +0.12] |
| babble-es | all/WER error_rate 7.37 → 6.67，-0.70 [-1.14, -0.25]*<br>all/WER false_alarm_rate 0.50 → 0.50，+0.00 [+0.00, +0.00]<br>all/WER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00] |
| hotword-bench-es | CER 4.97 → 4.66（-0.31）<br>召回 86.8 → 87.9（+1.1） |
| chime4 | all/WER error_rate 9.42 → 8.97，-0.45 [-1.23, +0.26]<br>all/WER miss_rate 5.23 → 5.53，+0.30 [-0.53, +1.14] |
| customer-hotwords | CER 16.30 → 15.78（-0.51）<br>召回 78.5 → 75.7（-2.8） |
| badcase | CER 10.00 → 8.33（-1.67）<br>召回 93.3 → 90.0（-3.3） |
| ts-hotwords-es | CER 26.33 → 25.66（-0.67）<br>召回 83.5 → 85.2（+1.7） |
| crosslingual-es | all/WER error_rate 23.11 → 24.57，+1.46 [-0.25, +3.08]<br>all/WER false_alarm_rate 1.04 → 1.88，+0.83 [+0.00, +1.87]<br>all/WER miss_rate 8.06 → 9.72，+1.67 [+0.14, +3.21]* **更差** |

显著更差的项：v12: all/CER false_alarm_rate；ts-es-v1: all/WER false_alarm_rate；babble4: all/WER error_rate；babble3: all/WER error_rate；first-other: all/CER false_alarm_rate；crosslingual: all/CER false_alarm_rate；scene-ts: all/CER false_alarm_rate；ts-es-v2: all/WER false_alarm_rate；crosslingual-es: all/WER miss_rate
