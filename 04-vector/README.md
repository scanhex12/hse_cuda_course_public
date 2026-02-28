# CudaVector — вектор на GPU

Реализуйте в `vector.cuh` класс **`CudaVector<T>`** — аналог `std::vector`, хранящий данные на GPU.

Требуемые методы:

1. **`reserve(size_t new_cap)`** — выделяет на устройстве память под `new_cap` элементов. При увеличении копирует старые данные.

2. **`push_back(const T& val)`** — добавляет элемент в конец. При нехватке места увеличивает `capacity` (например, в 2 раза).

3. **`set(size_t idx, const T& val)`** — записывает значение в ячейку `idx` (host→device).

4. **`get(size_t idx) const`** — возвращает значение из ячейки `idx` (device→host).

5. **`operator[](size_t idx) const`** — чтение по индексу (через `get`).

6. **`operator+`, `operator-`, `operator*`, `operator/`** — поэлементные операции. Запускает CUDA-ядро. При разном размере векторов — `std::runtime_error`.

7. **`copy_from_host(const T* host_ptr, size_t n)`** — копирует `n` элементов с хоста.

8. **`copy_to_host(T* host_ptr) const`** — копирует данные на хост.

9. **Конструкторы, деструктор, move-семантика** — реализуйте сами.

Бенчмарк: `a + b` для 10M элементов должно укладываться в 10 ms.

## Note

GPU kernel-ы можно писать с шаблонами

```
template<typename T>
__global__ void kernel_elem_op(const T* a, const T* b, T* out, size_t n, int op)
{
    ...
}
```
