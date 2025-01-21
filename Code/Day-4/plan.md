#### **PyTorch Custom CUDA Kernel**

PyTorch와 CUDA 커널의 통합을 통해 맞춤형 연산을 GPU에서 실행할 수 있습니다. `torch.utils.cpp_extension` 모듈을 활용하여 C++/CUDA 코드를 PyTorch와 통합할 수 있습니다.

#### **기본 개념**

- **CUDA Threading**: GPU에서 스레드 구조를 활용해 병렬 처리를 수행.
- **CUDA Memory Management**: CUDA의 메모리 계층(Global, Shared, Constant)을 효과적으로 활용.
- **torch::Tensor**: PyTorch에서 제공하는 텐서를 C++/CUDA로 조작 가능.

### 예제 1) Element-wise Square Operation

#### **문제 1: 텐서의 요소를 제곱하는 CUDA 커널 작성**

```C
#include <torch/extension.h>

// CUDA 요소 제곱 커널
__global__ void square_kernel(const float* x, float* out, int64_t N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = x[idx] * x[idx];
    }
}

// C++ 함수 정의
torch::Tensor square(torch::Tensor x) {
    auto out = torch::zeros_like(x);
    int64_t N = x.size(0);

    int threads = 1024;
    int blocks = (N + threads - 1) / threads;
    square_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), N);

    cudaDeviceSynchronize();
    return out;
}

// PyTorch 모듈 정의
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("square", &square, "Element-wise square");
}

```
```python
import torch
import torch.utils.cpp_extension

# CUDA 확장 모듈 컴파일 및 로드
module_name = "custom_square"
module_file = "square_kernel.cu"

torch.utils.cpp_extension.load(
    name=module_name,
    sources=[module_file],
    verbose=True
)

# 모듈 사용
import custom_square

x = torch.rand(1000, device='cuda')
out = custom_square.square(x)

print(out)
print(torch.allclose(out, x**2))

```

### 예제 2) Matrix Transpose

#### **문제 2: 행렬을 전치(transpose)하는 CUDA 커널 작성**
```C
#include <torch/extension.h>

// CUDA 전치 커널
__global__ void transpose_kernel(const float* A, float* B, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        B[col * rows + row] = A[row * cols + col];
    }
}

// C++ 함수 정의
torch::Tensor transpose(torch::Tensor A) {
    int rows = A.size(0);
    int cols = A.size(1);
    auto B = torch::zeros({cols, rows}, A.options());

    dim3 threads(16, 16);
    dim3 blocks((rows + threads.x - 1) / threads.x, (cols + threads.y - 1) / threads.y);
    transpose_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), rows, cols);

    cudaDeviceSynchronize();
    return B;
}

// PyTorch 모듈 정의
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("transpose", &transpose, "Matrix transpose");
}

```
```python
import torch
import torch.utils.cpp_extension

module_name = "custom_transpose"
module_file = "transpose_kernel.cu"

torch.utils.cpp_extension.load(
    name=module_name,
    sources=[module_file],
    verbose=True
)

import custom_transpose

A = torch.rand(32, 64, device='cuda')
B = custom_transpose.transpose(A)

print(B)
print(torch.allclose(B, A.T))

```

### **문제 3: 행렬-벡터 곱 (Matrix-Vector Multiplication)**

#### **설명**

- 행렬 `A`와 벡터 `v`를 입력으로 받아 `A * v`를 계산하는 CUDA 커널을 작성하세요.

#### **요구사항**

1. 행렬 `A`는 크기 `(M, N)`, 벡터 `v`는 크기 `(N,)`입니다.
2. 결과 벡터 `y`의 크기는 `(M,)`입니다.
3. 블록과 스레드를 적절히 설정하여 병렬로 연산을 수행하세요.
4. CUDA 메모리 동기화를 사용하여 정확한 결과를 반환하세요.

```C
#include <torch/extension.h>

// CUDA 행렬-벡터 곱 커널
__global__ void matrix_vector_kernel(const float* A, const float* v, float* y, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M) {
        float value = 0.0;
        for (int col = 0; col < N; ++col) {
            value += A[row * N + col] * v[col];
        }
        y[row] = value;
    }
}

// C++ 함수 정의
torch::Tensor matrix_vector(torch::Tensor A, torch::Tensor v) {
    int M = A.size(0);
    int N = A.size(1);

    auto y = torch::zeros({M}, A.options());

    int threads = 256;
    int blocks = (M + threads - 1) / threads;
    matrix_vector_kernel<<<blocks, threads>>>(A.data_ptr<float>(), v.data_ptr<float>(), y.data_ptr<float>(), M, N);

    cudaDeviceSynchronize();
    return y;
}

// PyTorch 모듈 정의
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matrix_vector", &matrix_vector, "Matrix-Vector Multiplication");
}

```
```python
import torch
import torch.utils.cpp_extension

# CUDA 확장 모듈 컴파일 및 로드
module_name = "matrix_vector_op"
module_file = "matrix_vector_kernel.cu"

torch.utils.cpp_extension.load(
    name=module_name,
    sources=[module_file],
    verbose=True
)

# 모듈 사용
import matrix_vector_op

A = torch.randn(64, 128, device='cuda')
v = torch.randn(128, device='cuda')
y = matrix_vector_op.matrix_vector(A, v)

print(y)
print(torch.allclose(y, A @ v))

```


### **문제 4: Batch 행렬 곱 (Batch Matrix Multiplication)**

#### **설명**

- 두 3D 텐서 `A`와 `B`의 Batch 행렬 곱을 계산하세요.
- `A`는 크기 `(B, M, N)`, `B`는 크기 `(B, N, K)`이며 결과는 `(B, M, K)`입니다.

#### **요구사항**

1. Batch 크기 `B`를 고려해 CUDA 블록을 구성하세요.
2. Shared Memory를 활용하여 성능을 최적화하세요.
3. 커널 동기화를 활용하여 데이터 충돌을 방지하세요.

```C
#include <torch/extension.h>

// CUDA Batch 행렬 곱 커널
__global__ void batch_matmul_kernel(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) {
    int batch = blockIdx.z;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch < BATCH && row < M && col < K) {
        float value = 0.0;
        for (int i = 0; i < N; ++i) {
            value += A[batch * M * N + row * N + i] * B[batch * N * K + i * K + col];
        }
        C[batch * M * K + row * K + col] = value;
    }
}

// C++ 함수 정의
torch::Tensor batch_matmul(torch::Tensor A, torch::Tensor B) {
    int BATCH = A.size(0);
    int M = A.size(1);
    int N = A.size(2);
    int K = B.size(2);

    auto C = torch::zeros({BATCH, M, K}, A.options());

    dim3 threads(16, 16);
    dim3 blocks((M + threads.x - 1) / threads.x, (K + threads.y - 1) / threads.y, BATCH);
    batch_matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), BATCH, M, N, K);

    cudaDeviceSynchronize();
    return C;
}

// PyTorch 모듈 정의
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batch_matmul", &batch_matmul, "Batch Matrix Multiplication");
}

```
```python
import torch
import torch.utils.cpp_extension

# CUDA 확장 모듈 컴파일 및 로드
module_name = "batch_matmul_op"
module_file = "batch_matmul_kernel.cu"

torch.utils.cpp_extension.load(
    name=module_name,
    sources=[module_file],
    verbose=True
)

# 모듈 사용
import batch_matmul_op

A = torch.randn(8, 32, 64, device='cuda')
B = torch.randn(8, 64, 16, device='cuda')
C = batch_matmul_op.batch_matmul(A, B)

print(C)
print(torch.allclose(C, torch.matmul(A, B)))

```
