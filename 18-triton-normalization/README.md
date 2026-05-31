# Triton Normalization — LayerNorm + ReLU (fp16)

## Задача

Дан батч векторов `X` формы (B, N) в `float16` на GPU, а также параметры `gamma` и `beta` 
длины N (тоже `float16`). Нужно для каждой строки `b` посчитать LayerNorm по последней оси 
и применить ReLU:

Аналог того же в torch:

```python
ref = torch.nn.functional.layer_norm(X, (N,), gamma, beta, eps)
ref = torch.relu(ref)
```

### Статистики по строке

$$
\mu = \frac{1}{N} \sum_{j=1}^{N} x_j
$$

$$
\sigma^2 = \frac{1}{N} \sum_{j=1}^{N} (x_j - \mu)^2
$$

### LayerNorm + ReLU

$$
y_i = \gamma_i \cdot \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \beta_i
$$

$$
z_i = \max(y_i,\ 0)
$$

На выходе — матрица `Z` формы `(B, N)`, `float16`.
