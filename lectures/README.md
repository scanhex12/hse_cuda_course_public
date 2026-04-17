# Лекции

## Week 1

Презентация: https://docs.google.com/presentation/d/10M-cUl65kq5iEjsu66ti7vc4R0kC9SUki3Nn69A1rRc/edit?usp=sharing

1. Что такое CPU, что такое GPU, в чем отличие
2. Почему NVIDIA популярна
3. Hello world CUDA
4. 1D, 2D и 3D блоки

## Week 2

Презентация: https://docs.google.com/presentation/d/1ZRfa_DUTil_WC0uPCBLpmRyC3Iwc0w-VXUfcBiKfvCI/edit?usp=sharing

0. Виды памяти (введение)
1. API для работы с global memory GPU
2. Аллокаторы в ML фреймворках
3. Виртуальная память
4. Unified memory
5. Обработка Page faults на GPU

## Week 3

Презентация: https://docs.google.com/presentation/d/1mR2ZoJHTcCNT6HVjv-X9hHRmXl4xBkOr4tpoFCc74Zs/edit?usp=sharing

1. Поколения видеокарт - Pascal, Volta, Ampere, Hooper
2. Тензорные ядра и их API
3. Компиляция CUDA программ - real/virtual architectures, аналогия с LLVM
4. ptx/cubin файлы
5. JIT компиляция
6. tooling - nvidia-smi

## Week 4

Презентация (draft): https://docs.google.com/presentation/d/13p3bZlVGAcTNbA57gvPNfXixpxWTU1S3heih--ER22M/edit?usp=sharing

Лучше запись лекции: https://t.me/c/3365773314/21

1. Constant memory
2. Shared memory
3. GEMM-ы и как использовать shared memory для их подсчета
4. Транспонирование матриц с использованием smem
5. Warp & Warp scheduler
6. Distributed shared memory

## Week 5

Презентация (draft): https://docs.google.com/presentation/d/1bQo2r4KEvI2CcQgq_DLmkfKr6ugGD7wPavu7Amyap04/edit?usp=sharing

1. Reduce - постановка задачи
2. Warp-level инструкции (__shfl_*_sync)

## Week 6

Презентация: https://docs.google.com/presentation/d/14xu7KDmZKvDUvUd0eHrRSynt70PA1SNuTg2FLp8lBy4/edit?usp=sharing

1. PyTorch/CUDA streams
2. PyTorch/CUDA graphs
3. Транзакции
4. Типы данных с меньшей размерностью

## Week 7

Презентация: https://docs.google.com/presentation/d/1_7upuE6Iozapnwgiox9tnkf-ytsYwnRaav2kkUF0ZBU/edit?usp=sharing

1. Compute sanitizer
2. Pytorch profiler
3. NVIDIA nsight systems
4. NVIDIA nsight compute

## Week 8

Презентация: https://docs.google.com/presentation/d/1JxtA2LyO-9WWX1rRe-HfItNYLVBkgXyAQGfayDcWmSI/edit?usp=sharing

1. ONNX формат
2. TensorRT - что это и какие оптимизации есть
3. TensorRT LLM - фичи
3.1 LoRA
3.2 KV cache
3.3 Paged attention
3.4 Speculative decoding/EAGLE

# Семинары

## Алгоритмы 1

Презентация: https://docs.google.com/presentation/d/18snIEGWOsgAKW7MYsRHMKM5HdT9KYaTS5YSHTBRyZPg/edit?usp=sharing

1. Merge sort
2. Prefix sort
3. Bitonic sort
4. Radix sort

## Алгоритмы 2

Презентация: https://docs.google.com/presentation/d/1wQQPp0ViExYgFupx22qqnAzhfXf_7OuqqxkG1k80Luo/edit?usp=sharing

1. PageRank на GPU
2. BFS на GPU, sparse GEMM
3. дейкстра - Delta stepping

## Flash attention

Paper: https://arxiv.org/pdf/2205.14135

1. Flash attention v1/v2/v3/v4