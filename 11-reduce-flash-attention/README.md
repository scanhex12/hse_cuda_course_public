## flash attention

Реализуйте Flash Attention. Для прохождения бенчмарков достаточно первой версии. 

Формула attention: `output = softmax(Q @ K^T / sqrt(d)) @ V`

- Q, V: N x d (row-major)
- K: d x N (transposed layout)
- O: N x d (output)
