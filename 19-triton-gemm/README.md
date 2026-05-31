# Задача 19: Dynamic W8A8 GeMM (Triton)

Реализуйте квантованное матричное умножение для inference: активации и веса в `int8`, GEMM в Triton, затем восстановление результата в `fp16`.

Сдавать только `kernel.py`. Каркас ядер и `autotune_config.py` уже есть.

## Идея

Линейный слой `Y = X @ W`, где `X` — `(M, K)`, `W` — `(K, N)`.

### Квантование (quantize)

Сначала `fp16` → `int8` с одной шкалой на строку/столбец:

```
scale = max(|x|) / 127
x_int8 = round(x / scale)
```

Обратное преобразование (приближённое восстановление `fp16`):

```
x_hat ≈ x_int8 * scale
```

Это будем называть dequant.

Для слоя:

- `X` (M×K, fp16) → per-row квант → `X_int` (M×K, int8) + `scale_X` (M,)
- `W` (K×N, fp16) → квант по столбцам (`axis=0`) → `W_int` (K×N, int8) + `scale_W` (N,)

### Matmul + dequant

Сначала считаем целочисленное произведение, потом восстанавливаем шкалы:

```
acc = X_int @ W_int          # int32
Y[i, j] ≈ acc[i, j] * scale_X[i] * scale_W[j]
```

В коде:

```python
Y ≈ (X_int @ W_int).float() * scale_X[:, None] * scale_W[None, :]
```

## Что нужно написать

### `quantize_int8_perrow_kernel` (Triton)

Квантует fp16-матрицу активаций `fpa` (M×K) в `int8` с одной шкалой на строку. Пишет `a` (M×K, int8) и `a_scale` (M). Указатели и strides передаются явно; размеры блоков — `BLOCK_SIZE_M`, `BLOCK_SIZE_K`.

### `quantize_int8_perrow(fpa)`

Host-запуск ядра выше. Возвращает `(a, a_scale)`.

### `quantize_int8(weight, axis=0)`

Квантование весов в PyTorch по оси `axis`. Для весов слоя `[K, N]` используйте `axis=0` (шкала на столбец). Возвращает `(weight_int8, scale)`; для `axis=0` нужен layout, удобный для int8 GEMM (см. homework).

### `perrow_w8a8_matmul_kernel` (Triton)

Считает `C` — результат `int8` matmul с dequant в `fp16`:

1. `acc = A @ B` в `int32`
2. `C = acc * a_scale[:, None] * b_scale[None, :]` → `fp16`

- `A` (M×K, int8), `a_scale` (M,)
- `B` (K×N, int8), `b_scale` (N,)
- `C` (M×N, fp16)

### `matmul_int8(a, a_scale, b, b_scale)`

Обёртка: запуск `perrow_w8a8_matmul_kernel`, выход fp16 (M×N).

### `matmul_quantize_int8(fpa, b, b_scale)`

Полный путь для активаций: квант `fpa` (per-row) + `matmul_int8` с уже квантованными весами `b`, `b_scale`.
