#pragma once

#include <cuda_helpers.h>

struct Matrix {
    int *data;
    size_t N;
    size_t M;
};

Matrix allocMatrixHost(size_t n, size_t m);
Matrix allocMatrixDevice(size_t n, size_t m);
void copyMatrixDeviceToHost(const Matrix &src_matrix, Matrix &dst_matrix);
void copyMatrixHostToDevice(const Matrix &src_matrix, Matrix &dst_matrix);
Matrix findNearestOnes(const Matrix &src_matrix);
void freeMatrixHost(Matrix &src_matrix);
void freeMatrixDevice(Matrix &src_matrix);

Matrix solve(const Matrix &src_matrix);
