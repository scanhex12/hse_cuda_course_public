# Fibonacci

Реализуйте в `fibonacci.cuh`:

1. **`fib(int n)`** — `__host__ __device__` функция: n-е число Фибоначчи (fib(0)=0, fib(1)=1, fib(n)=fib(n-1)+fib(n-2)).
2. **`print_fibonacci_kernel(int count)`** — `__global__` ядро: печатает первые `count` чисел Фибоначчи (по одному числу на строку). Должно корректно работать при любых `grid_dim` и `block_dim`: если потоков меньше, чем `count`, каждый поток печатает несколько значений (индексы `tid`, `tid + total_threads`, …).
3. **`run_print_fibonacci(int count, int grid_dim, int block_dim)`** — хост-функция: запускает ядро с заданными размерностями сетки и блока и синхронизируется с устройством.
