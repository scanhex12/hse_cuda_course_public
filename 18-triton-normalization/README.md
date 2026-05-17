# Triton Normalization — LayerNorm + ReLU (fp16)

## Задача

Дан батч векторов `X` формы (B, N) в `float16` на GPU, а также параметры `gamma` и `beta` длины N (тоже `float16`). Нужно для каждой строки `b` посчитать LayerNorm по последней оси и применить ReLU:

\[
y_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}} \cdot \gamma_i + \beta_i,\quad
z_i = \max(y_i,\, 0)
\]

где \(\mu\) и \(\sigma^2\) — среднее и дисперсия по координатам строки \(x\) длины \(N\).
