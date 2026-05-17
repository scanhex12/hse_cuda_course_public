# Задача 19: Dynamic W8A8 GeMM (Triton)

Реализуйте квантованное матричное умножение для inference: активации и веса в `int8`, GEMM в Triton, восстановление fp16 через шкалы (dequant).

Сдавать только `kernel.py`. Каркас ядер и `autotune_config.py` уже есть.

## Идея

Линейный слой `Y = X @ W`:

- `X` (M×K, fp16) → per-row квант в `int8` + вектор шкал длины M
- `W` (K×N) → квант по столбцам (`axis=0`) + вектор шкал длины N
- `Y ≈ (X_int @ W_int) * scale_X[:, None] * scale_W[None, :]`

Шкала: `max |x| / 127` по соответствующей оси. Округление до int8 даёт расхождение с `torch.matmul` — в тестах пороги ослаблены.

## API

### `quantize_int8_perrow_kernel` (Triton)

Квантует fp16-матрицу активаций `fpa` (M×K) в `int8` с одной шкалой на строку. Пишет `a` (M×K, int8) и `a_scale` (M). Указатели и strides передаются явно; размеры блоков — `BLOCK_SIZE_M`, `BLOCK_SIZE_K`.

### `quantize_int8_perrow(fpa)`

Host-запуск ядра выше. Возвращает `(a, a_scale)`.

### `quantize_int8(weight, axis=0)`

Квантование весов в PyTorch по оси `axis`. Для весов слоя `[K, N]` используйте `axis=0` (шкала на столбец). Возвращает `(weight_int8, scale)`; для `axis=0` нужен layout, удобный для int8 GEMM (см. homework).

### `perrow_w8a8_matmul_kernel` (Triton)

Считает `C = dequant(A_int @ B_int)`:

- `A` (M×K, int8), `a_scale` (M,)
- `B` (K×N, int8), `b_scale` (N,)
- `C` (M×N, fp16)

Каркас tiling / `SPLIT_K` задан; нужно реализовать накопление int32, цикл по K и dequant. Autotune — `get_autotune_config()`.

### `matmul_int8(a, a_scale, b, b_scale, out=None)`

Обёртка: запуск `perrow_w8a8_matmul_kernel`, выход fp16 (M×N). `out` — опциональный буфер результата.

### `matmul_quantize_int8(fpa, b, b_scale, out=None)`

Полный путь для активаций: квант `fpa` (per-row) + `matmul_int8` с уже квантованными весами `b`, `b_scale`.
