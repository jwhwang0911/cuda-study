#include <torch/extension.h>
#include <iostream>

// |예제 1) Element-wise Square Operation
// | Expect that input is 1D

__global__ void elem_square_kernel(const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> X, torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> Out, const int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
    {
        Out[idx] = X[idx] * X[idx];
    }
}

at::Tensor elem_square(at::Tensor X)
{
    at::Tensor Out = torch::zeros_like(X);
    int N = Out.size(0);
    int blocks = 1024;
    int grids = (N + blocks - 1) / blocks;

    elem_square_kernel<<<blocks, grids>>>(
        X.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        Out.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        N);

    cudaDeviceSynchronize();

    return Out;
}

/*
| 예제 2) Matrix Transpose
| Expect that input is 2D
*/
__global__ void transpose_kernel(
    const int N,
    const int M,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> target,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> output)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < N && col < M)
    {
        output[row][col] = target[col][row];
    }
}

at::Tensor transpose(at::Tensor X)
{
    at::Tensor out = torch::zeros({X.size(1), X.size(0)}, X.options());
    dim3 dimBlocks(16, 16);
    dim3 dimGrid((X.size(1) + dimBlocks.x - 1) / dimBlocks.x, (X.size(0) + dimBlocks.y - 1) / dimBlocks.y);

    transpose_kernel<<<dimBlocks, dimGrid>>>(X.size(1), X.size(0), X.packed_accessor32<float, 2, torch::RestrictPtrTraits>(), out.packed_accessor32<float, 2, torch::RestrictPtrTraits>());

    cudaDeviceSynchronize();

    return out;
}

/*
| **문제 3: 행렬-벡터 곱 (Matrix-Vector Multiplication)**
|
| **설명**
|- 행렬 `A`와 벡터 `v`를 입력으로 받아 `A * v`를 계산하는 CUDA 커널을 작성하세요.
| **요구사항**
|
| 1. 행렬 `A`는 크기 `(M, N)`, 벡터 `v`는 크기 `(N,)`입니다.
| 2. 결과 벡터 `y`의 크기는 `(M,)`입니다.
| 3. 블록과 스레드를 적절히 설정하여 병렬로 연산을 수행하세요.
| 4. CUDA 메모리 동기화를 사용하여 정확한 결과를 반환하세요.
*/
__global__ void matmul_kernel(
    int M,
    int N,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> v,
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> result)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < M)
    {
        for (auto i = 0; i < N; i++)
        {
            result[idx] += A[idx][i] * v[i];
        }
    }
}

at::Tensor matmul(at::Tensor A, at::Tensor v)
{
    int M = A.size(0);
    int N = A.size(1);

    int N_v = v.size(0);

    at::Tensor result = torch::zeros({M}, v.options());

    if (N != N_v)
    {
        throw std::runtime_error("A and v size mismatch");
    }

    int dimblock = 16;
    int dimgrid = (M + dimblock - 1) / dimblock;

    matmul_kernel<<<dimblock, dimgrid>>>(M, N,
                                         A.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                                         v.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                                         result.packed_accessor32<float, 1, torch::RestrictPtrTraits>());

    cudaDeviceSynchronize();

    return result;
}

/*
| ### **문제 4: Batch 행렬 곱 (Batch Matrix Multiplication)**
|
| #### **설명**
|
| - 두 3D 텐서 `A`와 `B`의 Batch 행렬 곱을 계산하세요.
| - `A`는 크기 `(B, M, N)`, `B`는 크기 `(B, N, K)`이며 결과는 `(B, M, K)`입니다.
|
| #### **요구사항**
|
| 1. Batch 크기 `B`를 고려해 CUDA 블록을 구성하세요.
| 2. Shared Memory를 활용하여 성능을 최적화하세요.
| 3. 커널 동기화를 활용하여 데이터 충돌을 방지하세요.
*/

__global__ void batchmul_kernel(
    const int Batch,
    const int M,
    const int N,
    const int K,
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> output)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < M && col < K)
    {
        for (int i = 0; i < Batch; i++)
        {
            for (int j = 0; j < N; j++)
            {
                output[i][row][col] += A[i][row][j] * B[i][j][col];
            }
        }
    }
}

at::Tensor batchmul(at::Tensor A, at::Tensor B)
{
    int size_A[3] = {A.size(0), A.size(1), A.size(2)};
    int size_B[3] = {B.size(0), B.size(1), B.size(2)};

    at::Tensor output = torch::zeros({size_A[0], size_A[1], size_B[2]}, A.options());

    if (size_A[0] != size_B[0])
        std::cerr
            << "Batch size is different" << "A : " << size_A[0] << "\tB : " << size_B[0] << std::endl;

    dim3 dimBlocks(16, 16);
    dim3 dimGrid(
        (size_A[1] + dimBlocks.x - 1) / dimBlocks.x,
        (size_B[2] + dimBlocks.y - 1) / dimBlocks.y);

    batchmul_kernel<<<dimBlocks, dimGrid>>>(size_A[0], size_A[1], size_A[2], size_B[2],
                                            A.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                                            B.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                                            output.packed_accessor32<float, 3, torch::RestrictPtrTraits>());

    cudaDeviceSynchronize();

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("elem_square", &elem_square, "Matrix Element-wise square operation");
    m.def("T", &transpose, "Transpose 2D matrix");
    m.def("matmul", &matmul, "Matrix(2D), vector multiplication");
    m.def("batchmul", &batchmul, "Batchwise multiplication");
}