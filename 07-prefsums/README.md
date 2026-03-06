# Prefix Sum

Префиксная сумма для массива `in` длины `n` — массив `out`, где:

```
out[0] = 0
out[i] = in[0] + in[1] + ... + in[i-1]   при i > 0
```

То есть `out[i]` — сумма всех элементов `in` до (но не включая) позиции `i`.

Реализуйте функцию `exclusive_scan_cuda` в `prefsum.cuh`:

```cpp
void exclusive_scan_cuda(float *d_out, const float *d_in, int n);
```
