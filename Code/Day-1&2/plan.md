### 예시 문제

#### **문제 1: CUDA 환경 설정 및 벡터 덧셈 커널 작성**

##### **문제 설명**

1. CUDA Toolkit을 설치하고, nvcc 컴파일러가 올바르게 설치되었는지 확인합니다.
2. 간단한 벡터 덧셈 연산을 수행하는 CUDA 커널을 작성하세요.
3. 호스트 메모리에서 데이터를 초기화하고, 이를 디바이스 메모리로 복사한 뒤, CUDA 커널을 호출하여 벡터 덧셈을 수행합니다.
4. 결과를 다시 호스트 메모리로 복사하고 올바르게 수행되었는지 검증합니다.

**세부 단계**:

- CUDA Toolkit 설치
- 벡터 덧셈 커널 코드 작성 (`vector_add.cu`)
- nvcc를 사용하여 컴파일하고 실행

#### **문제 2: PyTorch CUDA Tensor 테스트**

##### **문제 설명**

1. PyTorch가 설치되어 있는지 확인하고, CUDA를 사용할 수 있는지 테스트합니다.
2. CUDA를 사용하여 텐서를 생성하고, 간단한 연산을 수행합니다.
3. GPU에서 연산이 제대로 수행되는지 확인합니다.

**세부 단계**:

- PyTorch 설치 확인
- CUDA 사용 가능 여부 확인
- CUDA 텐서 생성 및 연산 수행

---

### 예시 코드

#### **문제 1: CUDA 환경 설정 및 벡터 덧셈 커널 작성**
```C++
#include <cuda_runtime.h>
#include <iostream>

// CUDA 벡터 덧셈 커널
__global__ void vector_add(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 1000;
    size_t size = N * sizeof(float);

    // 호스트 메모리 할당 및 초기화
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    // 디바이스 메모리 할당
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // 호스트에서 디바이스로 데이터 복사
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // CUDA 커널 호출
    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    vector_add<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C, N);

    // 결과를 디바이스에서 호스트로 복사
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 결과 검증
    for (int i = 0; i < N; i++) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            std::cerr << "Mismatch at index " << i << "!" << std::endl;
            break;
        }
    }

    // 메모리 해제
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    std::cout << "Done!" << std::endl;
    return 0;
}

```


---

#### **문제 2: PyTorch CUDA Tensor 테스트**

##### **코드 예시**

python

코드 복사

`import torch  # CUDA 사용 가능한지 확인 print(torch.cuda.is_available())  # CUDA 디바이스 정보 출력 print(torch.cuda.get_device_name(0))  # 텐서 생성 및 GPU로 이동 x = torch.randn(3, 3) x = x.to('cuda') print(x)  # 간단한 연산 수행 y = x * 2 print(y)`

---

#### **목표: CUDA 프로그래밍의 기초 이해 및 간단한 CUDA 커널 작성**

- **CUDA 프로그래밍**
    - CUDA 프로그래밍 모델 이해 (스레드, 블록, 그리드)
    - 메모리 관리 (Global, Shared, Constant, Texture Memory)
    - 간단한 CUDA 커널 작성 및 실행

**활동:**

- 간단한 벡터 덧셈 CUDA 커널 작성
- PyTorch CUDA Tensors 사용해보기

### **관련 지식**

#### **CUDA 프로그래밍 모델 이해**

- **스레드 (Thread)**: CUDA 프로그램의 기본 실행 단위입니다.
- **블록 (Block)**: 여러 스레드로 구성되며, 블록 내의 모든 스레드는 공유 메모리를 통해 데이터를 공유할 수 있습니다.
- **그리드 (Grid)**: 여러 블록으로 구성되며, 동일한 커널을 실행하는 모든 스레드가 포함됩니다.
![[Pasted image 20241220143419.png]]
<center>(2개의 block으로 4개의 thread를 각각 사용하는 이미지)
</center>
	Thread별로 value를 따로따로 가지고 있기 때문에 해당 Thread에 access하기 위해 
	`blockIdx.x * blockDim.x + threadIdx.x` 로 index로 access 해야함

#### **메모리 모델**

- **Global Memory**: 모든 스레드가 접근 가능한 메모리입니다. 가장 느리지만 용량이 큽니다.
- **Shared Memory**: 같은 블록 내의 스레드들이 공유하는 메모리입니다. Global Memory보다 빠릅니다.
- **Register**: 각 스레드가 사용하는 가장 빠른 메모리입니다.
- **Constant and Texture Memory**: 읽기 전용 메모리로, 빠른 접근을 제공합니다.

## 커널 함수 및 메모리 할당

1. 명령을 수행할 grid와 block 정의 (2D data를 다루는 경우에 활용)
	`{c}dim3 dimBlock(t_x, t_y)` 
	  각 block별 thread 개수 설정. x로 t_x, y로 t_y개
	`{c}dim3 dimGrid((N+dimBlock.x - 1)/ dimBlock.x , (M + dimBlock.y - 1) / dimBlock.y)` 
	  예를 들어 (N, M)의 크기 이미지를 다루는 경우 thread마다 하나의 값을 할당하는 것이 효율적 (병렬처리). 
	  따라서 이미지를 각 dimBlock의 크기만큼 split하여 grid로 나누면 데이터들이 해당 block의 thread로 할당됨.
	  `{c}dimBlock.x - 1` 은 나머지 부분을 버리지 않고 block에 채워야하므로 더해줌.
2. Kernel 정의
3.  함수 수행
	`{c}MatrixTranspose<<<dimBlock, dimGrid>>>(**args)` : args는 입력해줄 데이터(e.g. Tensor)와 결과를 받을 데이터, size가 포함되어야함.

#### Host(CPU) to Device(GPU) 메모리 할당 및 연산

-  `{c}cudaMalloc(float **ptr, size_t size)` : 포인터의 주소와 size인자를 받아 포인터에 **Device** 메모리를 할당해주는 함수
- `{c} cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind)` : **src**의 값을 모두 **dst**로 **count** 만큼 보내주는 함수
	 ```C
enum __device_builtin__ cudaMemcpyKind
{

	cudaMemcpyHostToHost = 0, /**< Host -> Host */
	cudaMemcpyHostToDevice = 1, /**< Host -> Device */
	cudaMemcpyDeviceToHost = 2, /**< Device -> Host */
	cudaMemcpyDeviceToDevice = 3, /**< Device -> Device */
	cudaMemcpyDefault = 4 /**< Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing */
};
```
- `{c} dim3` : Matrix 연산에 효율적으로 활용할 수 있는 자료형
```C
dim3 dimBlock(16, 16);
dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
// 연산에서 다음과 같이 활용됨. operation은 kernel function of cuda
operation<<<dimBlock, dimGrid>>>(..args);
```

### 예제 및 문제

#### 예제 1) vector_add는 두 개의 벡터를 더하는 문제입니다. 이를 실행하기 위한 코드는 다음과 같습니다.
```C
#include <cuda_runtime.h>
#include <iostream>

// CUDA 벡터 덧셈 커널
__global__ void vector_add(float* A, float* B, float* C, int N) {
    // TODO: 두 벡터를 더하는 함수를 작성하시오.
}

int main() {
    int N = 1000;
    size_t size = N * sizeof(float);

    // 호스트 메모리 할당 및 초기화
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    // 디바이스 메모리 할당
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // 호스트에서 디바이스로 데이터 복사
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // CUDA 커널 호출
    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    vector_add<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C, N);

    // 결과를 디바이스에서 호스트로 복사
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 결과 검증
    for (int i = 0; i < N; i++) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            std::cerr << "Mismatch at index " << i << "!" << std::endl;
            break;
        }
    }

    // 메모리 해제
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    std::cout << "Done!" << std::endl;
    return 0;
}

```

#### **문제 1: 벡터 덧셈 CUDA 커널 작성**

##### **문제 설명**

1. 위 예제에서 `vector_add` 커널 함수를 완성합니다.
2. 각 스레드는 벡터 `A`와 `B`의 요소를 더하여 결과를 벡터 `C`에 저장합니다.

**힌트**: `blockIdx.x`, `blockDim.x`, `threadIdx.x`를 사용하여 각 스레드의 인덱스를 계산하세요.

정답:
```c
#include <cuda_runtime.h>
#include <iostream>

// CUDA 벡터 덧셈 커널
__global__ void vector_add(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}
```

#### 예제 2) matrix_add는 두 개의 행렬을 더하는 문제입니다. 이를 실행하기 위한 코드는 다음과 같습니다.
```c
#include <cuda_runtime.h>
#include <iostream>

// CUDA 행렬 덧셈 커널
__global__ void matrix_add(float* A, float* B, float* C, int N, int M) {
    // TODO: 두 행렬을 더하는 함수를 작성하시오.
}

int main() {
    int N = 1000;
    int M = 1000;
    size_t size = N * M * sizeof(float);

    // 호스트 메모리 할당 및 초기화
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            h_A[i * M + j] = static_cast<float>(i + j);
            h_B[i * M + j] = static_cast<float>(i - j);
        }
    }

    // 디바이스 메모리 할당
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // 호스트에서 디바이스로 데이터 복사
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // CUDA 커널 호출
    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid((N + threads_per_block.x - 1) / threads_per_block.x,
                         (M + threads_per_block.y - 1) / threads_per_block.y);
    matrix_add<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C, N, M);

    // 결과를 디바이스에서 호스트로 복사
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 결과 검증
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            if (h_C[i * M + j] != h_A[i * M + j] + h_B[i * M + j]) {
                std::cerr << "Mismatch at index (" << i << ", " << j << ")!" << std::endl;
                break;
            }
        }
    }

    // 메모리 해제
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    std::cout << "Done!" << std::endl;
    return 0;
}

```

정답:

#### 예제 3) scalar_mult는 벡터의 모든 요소에 스칼라 값을 곱하는 문제입니다. 이를 실행하기 위한 코드는 다음과 같습니다.
```c
#include <cuda_runtime.h>
#include <iostream>

// CUDA 스칼라 곱 커널
__global__ void scalar_mult(float* A, float scalar, float* B, int N) {
    // TODO: 벡터의 모든 요소에 스칼라 값을 곱하는 함수를 작성하시오.
}

int main() {
    int N = 1000;
    float scalar = 2.0f;
    size_t size = N * sizeof(float);

    // 호스트 메모리 할당 및 초기화
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
    }

    // 디바이스 메모리 할당
    float *d_A, *d_B;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);

    // 호스트에서 디바이스로 데이터 복사
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    // CUDA 커널 호출
    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    scalar_mult<<<blocks_per_grid, threads_per_block>>>(d_A, scalar, d_B, N);

    // 결과를 디바이스에서 호스트로 복사
    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);

    // 결과 검증
    for (int i = 0; i < N; i++) {
        if (h_B[i] != h_A[i] * scalar) {
            std::cerr << "Mismatch at index " << i << "!" << std::endl;
            break;
        }
    }

    // 메모리 해제
    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);

    std::cout << "Done!" << std::endl;
    return 0;
}

```

정답:

#### 예제 4) matrix_transpose는 행렬의 전치를 계산하는 문제입니다. 이를 실행하기 위한 코드는 다음과 같습니다.
```c
#include <cuda_runtime.h>
#include <iostream>

// CUDA 행렬 전치 커널
__global__ void matrix_transpose(float* A, float* B, int N, int M) {
    // TODO: 행렬의 전치를 계산하는 함수를 작성하시오.
}

int main() {
    int N = 1000;
    int M = 1000;
    size_t size = N * M * sizeof(float);

    // 호스트 메모리 할당 및 초기화
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            h_A[i * M + j] = static_cast<float>(i * M + j);
        }
    }

    // 디바이스 메모리 할당
    float *d_A, *d_B;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);

    // 호스트에서 디바이스로 데이터 복사
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    // CUDA 커널 호출
    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid((N + threads_per_block.x - 1) / threads_per_block.x,
                         (M + threads_per_block.y - 1) / threads_per_block.y);
    matrix_transpose<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, N, M);

    // 결과를 디바이스에서 호스트로 복사
    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);

    // 결과 검증
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            if (h_B[j * N + i] != h_A[i * M + j]) {
                std::cerr << "Mismatch at index (" << i << ", " << j << ")!" << std::endl;
                break;
            }
        }
    }

    // 메모리 해제
    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);

    std::cout << "Done!" << std::endl;
    return 0;
}

```
