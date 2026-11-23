# 交付评估：kgrpoaudio 对 r1117（同一推理栈，差 = kgrpoaudio − r1117）

| 项目 | 结果 |
|---|---|
| v12 | all/CER error_rate 12.78 → 12.29，-0.48 [-0.72, -0.25]*<br>all/CER false_alarm_rate 2.46 → 3.68，+1.22 [+1.00, +1.44]* **更差**<br>all/CER miss_rate 2.76 → 1.92，-0.83 [-1.03, -0.65]* |
| es-asr | all/WER error_rate 5.25 → 5.11，-0.14 [-0.19, -0.09]*<br>all/WER miss_rate 0.02 → 0.03，+0.01 [+0.00, +0.02] |
| ts-es-v1 | all/WER error_rate 27.28 → 26.13，-1.15 [-1.49, -0.82]*<br>all/WER false_alarm_rate 3.21 → 4.63，+1.42 [+1.08, +1.78]* **更差**<br>all/WER miss_rate 3.86 → 2.67，-1.19 [-1.48, -0.92]* |
| babble4 | all/CER error_rate 1.90 → 1.95，+0.05 [-0.16, +0.25]<br>all/CER false_alarm_rate 4.95 → 4.86，-0.08 [-0.59, +0.42]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 1.92 → 2.05，+0.12 [-0.01, +0.26]<br>all/WER false_alarm_rate 1.75 → 2.38，+0.62 [+0.13, +1.21]* **更差**<br>all/WER miss_rate 0.10 → 0.10，+0.00 [+0.00, +0.00] |
| babble3 | all/CER error_rate 1.90 → 1.95，+0.05 [-0.15, +0.24]<br>all/CER false_alarm_rate 1.25 → 1.50，+0.25 [-0.78, +1.32]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/WER error_rate 1.92 → 2.05，+0.12 [-0.00, +0.25]<br>all/WER false_alarm_rate 1.75 → 2.38，+0.62 [+0.13, +1.22]* **更差**<br>all/WER miss_rate 0.10 → 0.10，+0.00 [+0.00, +0.00] |
| customer-ts | all/CER error_rate 18.75 → 17.71，-1.04 [-2.50, +0.00]<br>all/CER false_alarm_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER miss_rate 20.00 → 20.00，+0.00 [+0.00, +0.00] |
| customer-asr | all/CER error_rate 20.15 → 18.98，-1.17 [-3.42, +1.05]<br>all/CER miss_rate 0.65 → 0.00，-0.65 [-1.94, +0.00] |
| ts-hotwords | CER 14.43 → 14.07（-0.36）<br>召回 89.9 → 90.6（+0.7） |
| ts-en | all/WER error_rate 13.59 → 13.37，-0.22 [-0.43, +0.01]<br>all/WER false_alarm_rate 6.20 → 6.74，+0.54 [+0.26, +0.82]* **更差**<br>all/WER miss_rate 2.94 → 2.40，-0.54 [-0.74, -0.34]* |
| first-other | all/CER error_rate 20.11 → 20.34，+0.22 [-0.21, +0.78]<br>all/CER false_alarm_rate 2.58 → 2.56，-0.02 [-0.33, +0.29]<br>all/CER miss_rate 1.66 → 1.46，-0.20 [-0.46, +0.06] |
| asr-benchmark | all/CER error_rate 3.56 → 3.53，-0.03 [-0.14, +0.08]<br>all/CER miss_rate 0.13 → 0.13，-0.00 [-0.01, +0.00]<br>all/CER term_hit_rate 98.77 → 98.76，-0.01 [-0.09, +0.07]<br>all/WER error_rate 7.56 → 7.40，-0.16 [-0.21, -0.11]*<br>all/WER false_alarm_rate 24.00 → 23.74，-0.26 [-0.55, +0.07]<br>all/WER miss_rate 0.45 → 0.41，-0.04 [-0.08, -0.00]* |
| hotword-bench | CER 3.27 → 3.21（-0.06）<br>召回 87.9 → 88.1（+0.2） |
| police | all/CER error_rate 0.27 → 0.27，-0.01 [-0.01, +0.00]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.75 → 99.70，-0.05 [-0.12, +0.02] |
| police-single | all/CER error_rate 0.38 → 0.34，-0.04 [-0.11, +0.00]<br>all/CER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00]<br>all/CER term_hit_rate 99.08 → 99.14，+0.06 [+0.00, +0.15] |
| crosslingual | all/CER error_rate 34.29 → 32.67，-1.62 [-3.70, +0.40]<br>all/CER false_alarm_rate 0.91 → 1.82，+0.91 [+0.34, +1.59]* **更差**<br>all/CER miss_rate 14.62 → 12.12，-2.50 [-3.50, -1.61]* |
| scene-asr | all/CER error_rate 4.72 → 3.87，-0.85 [-1.21, -0.52]*<br>all/CER miss_rate 0.85 → 0.20，-0.65 [-0.91, -0.43]* |
| crosslingual-en | all/WER error_rate 17.88 → 15.84，-2.04 [-2.83, -1.25]*<br>all/WER false_alarm_rate 2.73 → 3.52，+0.80 [+0.00, +1.63]<br>all/WER miss_rate 9.17 → 6.59，-2.58 [-3.42, -1.73]* |
| scene-ts | all/CER error_rate 3.43 → 4.57，+1.14 [+0.39, +1.93]* **更差**<br>all/CER false_alarm_rate 21.43 → 38.57，+17.14 [+8.69, +27.03]* **更差**<br>all/CER miss_rate 0.33 → 0.22，-0.11 [-0.27, +0.00] |
| ts-es-v2 | all/WER error_rate 27.49 → 26.22，-1.26 [-1.58, -0.95]*<br>all/WER false_alarm_rate 3.03 → 4.51，+1.48 [+1.18, +1.79]* **更差**<br>all/WER miss_rate 3.40 → 2.30，-1.11 [-1.35, -0.86]* |
| ctx-v1 | all/CER error_rate 7.87 → 4.13，-3.73 [-4.41, -3.11]*<br>all/CER miss_rate 5.32 → 1.79，-3.54 [-4.16, -2.95]* |
| babble-es | all/WER error_rate 7.37 → 6.95，-0.41 [-0.73, -0.09]*<br>all/WER false_alarm_rate 0.50 → 0.50，+0.00 [+0.00, +0.00]<br>all/WER miss_rate 0.00 → 0.00，+0.00 [+0.00, +0.00] |
| hotword-bench-es | CER 4.97 → 4.88（-0.09）<br>召回 86.8 → 87.1（+0.4） |
| chime4 | all/WER error_rate 9.42 → 9.69，+0.27 [-0.65, +1.17]<br>all/WER miss_rate 5.23 → 6.44，+1.21 [+0.23, +2.20]* **更差** |
| customer-hotwords | CER 16.30 → 16.55（+0.26）<br>召回 78.5 → 75.7（-2.8） |
| badcase | CER 10.00 → 5.00（-5.00）<br>召回 93.3 → 96.7（+3.3） |
| ts-hotwords-es | CER 26.33 → 25.62（-0.71）<br>召回 83.5 → 84.6（+1.2） |
| crosslingual-es | all/WER error_rate 23.11 → 22.22，-0.88 [-2.82, +0.84]<br>all/WER false_alarm_rate 1.04 → 1.67，+0.62 [-0.21, +1.53]<br>all/WER miss_rate 8.06 → 7.36，-0.69 [-2.19, +0.82] |

显著更差的项：v12: all/CER false_alarm_rate；ts-es-v1: all/WER false_alarm_rate；babble4: all/WER false_alarm_rate；babble3: all/WER false_alarm_rate；ts-en: all/WER false_alarm_rate；crosslingual: all/CER false_alarm_rate；scene-ts: all/CER error_rate, all/CER false_alarm_rate；ts-es-v2: all/WER false_alarm_rate；chime4: all/WER miss_rate
