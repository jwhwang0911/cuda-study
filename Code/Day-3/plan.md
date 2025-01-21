### 관련 지식

#### **PyTorch C++ 확장**

PyTorch는 C++ 확장 모듈을 통해 커스텀 연산을 정의하고, 이를 Python에서 사용할 수 있게 합니다. `torch/extension.h`를 사용하면 CUDA 커널을 포함한 C++ 코드를 PyTorch와 통합할 수 있습니다.

#### **기본 개념**

- **C++ 확장 모듈**: PyTorch는 `pybind11` 라이브러리를 사용하여 C++ 코드를 Python에 바인딩합니다.
- **CUDA 커널**: GPU에서 실행되는 CUDA 코드를 포함합니다.
- **torch::Tensor**: PyTorch의 텐서를 C++에서 사용합니다.
-  `{c} cudaDeviceSyncronize()` : 모든 연산을 실행 한 이후, gpu 데이터를 cpu(host)에 넘기는 함수

### **Day 3: PyTorch와 C++/CUDA 통합**
### 예제 1) Custom Add Operation

#### **문제 1: 간단한 덧셈 연산을 수행하는 C++/CUDA 모듈 작성**

##### **코드 예시**

**add_kernel.cu**:
```C
#include <torch/extension.h>

// CUDA 벡터 덧셈 커널
__global__ void add_kernel(float* x, float* y, float* out, int64_t N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = x[idx] + y[idx];
    }
}

// C++ 함수 정의
torch::Tensor add(torch::Tensor x, torch::Tensor y) {
    auto out = torch::zeros_like(x);
    int64_t N = x.size(0);

    int threads = 1024;
    int blocks = (N + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), out.data_ptr<float>(), N);

	cudaDeviceSyncronize():
    return out;
}

// PyTorch 모듈 정의
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add, "Add two tensors");
}

```

**test_add.py**:

```python
import torch
import torch.utils.cpp_extension
import os

# CUDA 확장 모듈 컴파일 및 로드
module_name = "custom_op"
module_file = "add_kernel.cu"
extra_cuda_cflags = ["--expt-relaxed-constexpr"]

torch.utils.cpp_extension.load(
    name=module_name,
    sources=[module_file],
    extra_cuda_cflags=extra_cuda_cflags,
    verbose=True # 컴파일 상세정보 출력
)
# `extra_cuda_cflags`는 CUDA 컴파일러에 추가 플래그를 전달하여 컴파일 과정을 제어

# 모듈 사용
import custom_op

x = torch.randn(1000, device='cuda')
y = torch.randn(1000, device='cuda')
z = custom_op.add(x, y)

print(z)
print(torch.allclose(z, x + y))

```

##### **활동**

1. `add_kernel.cu` 파일에서 `add_kernel` CUDA 커널을 작성하여 각 요소를 더하는 함수를 완성합니다.
2. `test_add.py` 파일에서 `torch.utils.cpp_extension.load`를 사용하여 `add_kernel.cu`를 컴파일하고 로드합니다.
3. `test_add.py`를 실행하여 결과를 확인합니다.

---

### 예제 2) Custom Scalar Multiplication

#### **문제 2: 텐서의 모든 요소에 스칼라 값을 곱하는 C++/CUDA 모듈 작성**

##### **코드 예시**

**scalar_mult_kernel.cu**:
```C
#include <torch/extension.h>

// CUDA 스칼라 곱 커널
__global__ void scalar_mult_kernel(float* x, float scalar, float* out, int64_t N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = x[idx] * scalar;
    }
}

// C++ 함수 정의
torch::Tensor scalar_mult(torch::Tensor x, float scalar) {
    auto out = torch::zeros_like(x);
    int64_t N = x.size(0);

    int threads = 1024;
    int blocks = (N + threads - 1) / threads;
    scalar_mult_kernel<<<blocks, threads>>>(x.data_ptr<float>(), scalar, out.data_ptr<float>(), N);

	cudaDeviceSyncronize():
    return out;
}

// PyTorch 모듈 정의
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scalar_mult", &scalar_mult, "Multiply tensor by scalar");
}

```

**test_scalar_mult.py**:
```python
import torch
import torch.utils.cpp_extension
import os

# CUDA 확장 모듈 컴파일 및 로드
module_name = "custom_op"
module_file = "scalar_mult_kernel.cu"
extra_cuda_cflags = ["--expt-relaxed-constexpr"]

torch.utils.cpp_extension.load(
    name=module_name,
    sources=[module_file],
    extra_cuda_cflags=extra_cuda_cflags,
    verbose=True
)

# 모듈 사용
import custom_op

x = torch.randn(1000, device='cuda')
scalar = 2.0
z = custom_op.scalar_mult(x, scalar)

print(z)
print(torch.allclose(z, x * scalar))

```

##### **활동**

1. `scalar_mult_kernel.cu` 파일에서 `scalar_mult_kernel` CUDA 커널을 작성하여 각 요소에 스칼라 값을 곱하는 함수를 완성합니다.
2. `test_scalar_mult.py` 파일에서 `torch.utils.cpp_extension.load`를 사용하여 `scalar_mult_kernel.cu`를 컴파일하고 로드합니다.
3. `test_scalar_mult.py`를 실행하여 결과를 확인합니다.
---



### 예제 3) Custom Matrix Multiplication

#### **문제 3: 행렬 곱셈을 수행하는 C++/CUDA 모듈 작성**

##### **코드 예시**

**matrix_mult_kernel.cu**:

```c
#include <torch/extension.h>

// CUDA 행렬 곱 커널
__global__ void matrix_mult_kernel(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < K) {
        float value = 0;
        for (int i = 0; i < N; ++i) {
            value += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = value;
    }
}

// C++ 함수 정의
torch::Tensor matrix_mult(torch::Tensor A, torch::Tensor B) {
    auto C = torch::zeros({A.size(0), B.size(1)}, A.options());

    int M = A.size(0);
    int N = A.size(1);
    int K = B.size(1);

    dim3 threads(16, 16);
    dim3 blocks((M + threads.x - 1) / threads.x, (K + threads.y - 1) / threads.y);
    matrix_mult_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

	cudaDeviceSyncronize():
    return C;
}

// PyTorch 모듈 정의
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matrix_mult", &matrix_mult, "Multiply two matrices");
}

```

**test_matrix_mult.py**:
```python
import torch
import torch.utils.cpp_extension
import os

# CUDA 확장 모듈 컴파일 및 로드
module_name = "custom_op"
module_file = "matrix_mult_kernel.cu"
extra_cuda_cflags = ["--expt-relaxed-constexpr"]

torch.utils.cpp_extension.load(
    name=module_name,
    sources=[module_file],
    extra_cuda_cflags=extra_cuda_cflags,
    verbose=True
)

# 모듈 사용
import custom_op

A = torch.randn(32, 64, device='cuda')
B = torch.randn(64, 128, device='cuda')
C = custom_op.matrix_mult(A, B)

print(C)
print(torch.allclose(C, torch.matmul(A, B)))

```


---

### 예제 4) Custom Element-wise Operation

#### **문제 4: 텐서의 요소별 연산을 수행하는 C++/CUDA 모듈 작성**

##### **코드 예시**

**elementwise_op_kernel.cu**:

```C
#include <torch/extension.h>

// CUDA 커널
__global__ void elementwise_op_kernel(float* x, float* out, int64_t N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = x[idx] * x[idx] + 2 * x[idx] + 1;  // 예: (x^2 + 2x + 1)
    }
}

// C++ 함수 정의
torch::Tensor elementwise_op(torch::Tensor x) {
    auto out = torch::zeros_like(x);
    int64_t N = x.size(0);

    int threads = 1024;
    int blocks = (N + threads - 1) / threads;
    elementwise_op_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), N);

	cudaDeviceSyncronize():
    return

```

Next Day 4.
[[Cuda/PyTorch Custom CUDA Kernel.md]]