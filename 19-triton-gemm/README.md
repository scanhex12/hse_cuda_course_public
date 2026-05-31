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

В шаблоне предлагается так разделить код, однако можно оставить только те функции, которые есть в тесте и писать как вам будет удобнее.

### `quantize_int8_perrow_kernel` (Triton)

Квантует fp16-матрицу активаций `fpa` (M×K) в `int8` с одной шкалой на строку. Пишет `a` (M×K, int8) и `a_scale` (M). Strides — см. таблицу ниже; размеры блоков — `BLOCK_SIZE_M`, `BLOCK_SIZE_K`.

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

## Страйды

В Triton-ядрах указатели и strides передаются явно.

Для 2D-тензора `T` формы `(R, C)` элемент `(i, j)` лежит по адресу:

```
ptr + i * stride_row + j * stride_col
```

Для 1D-вектора длины `R`: `ptr + i * stride`, обычно `stride = 1`.

### `quantize_int8_perrow_kernel`

| Тензор | Форма | Strides | Откуда взять |
|--------|-------|---------|--------------|
| `fpa` | (M, K) fp16 | `stride_fpam`, `stride_fpak` | `fpa.stride(0)`, `fpa.stride(1)` |
| `a` | (M, K) int8 | `stride_am`, `stride_ak` | `a.stride(0)`, `a.stride(1)` |
| `a_scale` | (M,) fp16 | `stride_asm` | `a_scale.stride(0)` |

### `perrow_w8a8_matmul_kernel`

| Тензор | Форма | Strides | Откуда взять |
|--------|-------|---------|--------------|
| `a` | (M, K) int8 | `stride_am`, `stride_ak` | `a.stride(0)`, `a.stride(1)` |
| `a_scale` | (M,) fp16 | `stride_asm` | `a_scale.stride(0)` |
| `b` | (K, N) int8 | `stride_bk`, `stride_bn` | `b.stride(0)`, `b.stride(1)` |
| `b_scale` | (N,) fp16 | `stride_bsn` | `b_scale.stride(0)` |
| `c` | (M, N) fp16 | `stride_cm`, `stride_cn` | `c.stride(0)`, `c.stride(1)` |

Пример запуска matmul из host-кода:

```python
a, a_scale, b, b_scale, c = ...  # contiguous
perrow_w8a8_matmul_kernel[grid](
    a, a_scale, b, b_scale, c,
    M, N, K,
    a.stride(0), a.stride(1),
    a_scale.stride(0),
    b.stride(0), b.stride(1),
    b_scale.stride(0),
    c.stride(0), c.stride(1),
    ...
)
```

## Указания

Чтобы сделать каст в triton используйте `.to(tl.int8)`.

Перед вызовом ядра в host-обёртках вызывайте `.contiguous()` на входных тензорах: метод при необходимости копирует данные так, чтобы строки матрицы лежали в памяти подряд. После `.t()` PyTorch не копирует матрицу, а только «перечитывает» её по-другому — соседние элементы одной строки в памяти могут оказаться далеко друг от друга; `.contiguous()` убирает это и возвращает обычный порядок.

Пример:

```python
x = torch.randn(4, 8)          # shape (4, 8)
y = x.t()                      # shape (8, 4), данные те же, порядок чтения другой

x.is_contiguous()              # True  — строки лежат подряд
y.is_contiguous()              # False — «строка» y идёт по столбцам исходной x

x.stride()                     # (8, 1): следующий элемент строки через 1, следующая строка через 8
y.stride()                     # (1, 8): следующий элемент «строки» y через 8 в памяти

z = y.contiguous()             # физически переложили в (8, 4) row-major
z.is_contiguous()              # True
z.stride()                     # (4, 1)
```

В Triton это важно, потому что ядро читает память через указатель + strides. Загрузка блока строки `(M, K)` в matmul обычно выглядит так:

```python
# host: a — (M, K), contiguous → stride_am=K, stride_ak=1
rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # индексы строк A: 0..M-1
rk = tl.arange(0, BLOCK_K)                     # индексы столбцов A: 0..K-1
a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
a = tl.load(a_ptrs)   # блок A[rm, rk]; при stride_ak=1 соседние rk — соседние адреса
```

Если передать транспонированный `a` без `.contiguous()` (`stride_am=1`, `stride_ak=K`), тот же код будет просить у GPU каждый элемент строки через K ячеек — доступы разъезжаются, ядро работает медленнее или даёт неверный результат, если strides переданы не так, как в данных.

Поэтому в обёртке проще так:

```python
a = a.contiguous()
b = b.contiguous()
# stride_am=K, stride_ak=1 — можно писать простой цикл по K без особых случаев
perrow_w8a8_matmul_kernel[grid](
    a, a_scale, b, b_scale, c,
    M, N, K,
    a.stride(0), a.stride(1),
    ...
)
```
