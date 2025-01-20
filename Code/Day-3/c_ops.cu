#include <iostream>
#include <torch/extension.h>

__global__ void add_kernel(float *x, float *y, float *out, int64_t size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        out[i] = x[i] + y[i];
    }
}
// 1-D case
torch::Tensor add(torch::Tensor x, torch::Tensor y)
{
    if (x.size(0) != y.size(0))
    {
        if (y.size(0) == 1)
        {
            y = y.expand_as(x); // y를 x와 동일한 크기로 브로드캐스트
        }
        else if (x.size(0) == 1)
        {
            x = x.expand_as(y); // x를 y와 동일한 크기로 브로드캐스트
        }
        else
        {
            throw std::invalid_argument("Input tensors must be broadcastable or have the same size.");
        }
    }

    auto output = torch::zeros_like(x);
    int64_t size = x.size(0);

    int threads = 1024;
    int blocks = (size + threads - 1) / threads;

    add_kernel<<<threads, blocks>>>(x.data_ptr<float>(), y.data_ptr<float>(), output.data_ptr<float>(), size);
    cudaDeviceSynchronize();

    return output;
}

/*
 * 1. `scalar_mult_kernel.cu` 파일에서 `scalar_mult_kernel` CUDA 커널을 작성하여 각 요소에 스칼라 값을 곱하는 함수를 완성합니다.
 * 2. `test_scalar_mult.py` 파일에서 `torch.utils.cpp_extension.load`를 사용하여 `scalar_mult_kernel.cu`를 컴파일하고 로드합니다.
 * 3. `test_scalar_mult.py`를 실행하여 결과를 확인합니다.
 */

__global__ void mult_kernel(float *x, float *out, float scalar, int64_t total_size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < total_size)
    {
        out[idx] = x[idx] * scalar;
    }
}

// [INFO] Expect that tensor x has 1D vector
torch::Tensor mult(torch::Tensor x, float c)
{
    torch::Tensor output = torch::zeros_like(x);

    int64_t size = x.size(0);
    int threads = 1024;
    int blocks = (size + threads - 1) / threads;
    mult_kernel<<<threads, blocks>>>(x.data_ptr<float>(), output.data_ptr<float>(), c, size);
    cudaDeviceSynchronize();

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("add", &add, "Add two 1-D tensors");
    m.def("mult", &mult, "Multiple constant to tensor");
}