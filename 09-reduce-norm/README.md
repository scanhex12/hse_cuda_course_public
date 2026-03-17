# Reduce normalization

Нужно реализовать нормализацию массива:

```
output[i] = gamma * input[i] / mean_sqrt(input) + beta
```

где

```
mean_sqrt(input) = sqrt(sum(input[i] ^ 2) / N + eps)
```
