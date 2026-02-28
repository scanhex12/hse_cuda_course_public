#pragma once

#include <algorithm>
#include <cuda_helpers.h>
#include <cuda_runtime.h>
#include <stdexcept>

template <typename T> 
class CudaVector {
  public:
    CudaVector();
    explicit CudaVector(size_t n);
    ~CudaVector();

    CudaVector(const CudaVector &) = delete;
    CudaVector &operator=(const CudaVector &) = delete;
    CudaVector(CudaVector &&o) noexcept;
    CudaVector &operator=(CudaVector &&o) noexcept;

    size_t size() const { return size_; }
    size_t capacity() const { return cap_; }

    void reserve(size_t new_cap);
    void push_back(const T &val);
    void set(size_t idx, const T &val);
    T get(size_t idx) const;

    T operator[](size_t idx) const { return get(idx); }

    T *data() { return d_data; }
    const T *data() const { return d_data; }

    CudaVector operator+(const CudaVector &other) const;
    CudaVector operator-(const CudaVector &other) const;
    CudaVector operator*(const CudaVector &other) const;
    CudaVector operator/(const CudaVector &other) const;

    void copy_from_host(const T *host_ptr, size_t n);
    void copy_to_host(T *host_ptr) const;

  private:
    T *d_data;
    size_t size_;
    size_t cap_;
};
