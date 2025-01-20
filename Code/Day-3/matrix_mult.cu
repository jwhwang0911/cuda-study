#include <torch/extension.h>

__global__ void matrix_mult_kernel(const int M, const int N, const int K,
                                   const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> X,
                                   const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> Y,
                                   torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> Out)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < M && col < K)
    {
        for (int i = 0; i < N; i++)
        {
            Out[row][col] += X[row][i] * Y[i][col];
        }
    }
}

// [INFO] rix x and y should be 2D tensor
at::Tensor matrix_mult(at::Tensor x, at::Tensor y)
{
    const int M = x.size(0);
    const int N = x.size(1);
    const int K = y.size(1);

    if (N != y.size(0))
        assert("Invalid Multiplication");

    at::Tensor C = torch::zeros({M, K}, x.options());

    dim3 dimBlock(16, 16);
    dim3 dimGrid((M + dimBlock.x - 1) / dimBlock.x, (K + dimBlock.y - 1) / dimBlock.y);

    matrix_mult_kernel<<<dimGrid, dimBlock>>>(M, N, K,
                                              x.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                                              y.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                                              C.packed_accessor32<float, 2, torch::RestrictPtrTraits>());

    cudaDeviceSynchronize();

    return C;
    // matrix_mult_kernel<<<dimGrid, dimBlock>>>(M, N, K, )
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("matrix_mult", &matrix_mult, "Multiply two matrices");
}
