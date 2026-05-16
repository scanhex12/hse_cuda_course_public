# Графы: кратчайшие пути (SSSP), delta-stepping на CUDA

## Задача

Дан взвешенный ориентированный граф с неотрицательными весами рёбер в формате CSR на CPU. Нужно для одной стартовой вершины `source` найти длины кратчайших путей до всех вершин и записать их в массив `distances_out`.

## Delta stepping

Алгоритм был разобран на семинаре, но идею можно посмотреть тут

https://ics-websites.science.uu.nl/docs/vakken/b3cc/resources/07-Delta-stepping.pdf

## Формат CSR

- Вершины нумеруются `0 … num_vertices - 1`.
- `csr_row_ptr` длины `num_vertices + 1`, монотонно неубывающий.
- Исходящие рёбра вершины `u` занимают индексы  
  `[csr_row_ptr[u], csr_row_ptr[u + 1])` в массивах `csr_col_idx` и `edge_weight`.
- Число рёбер  
  `nnz = csr_row_ptr[num_vertices] - csr_row_ptr[0]`  
  (в тестах обычно `csr_row_ptr[0] == 0`).

