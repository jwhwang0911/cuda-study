#include <torch/extension.h>
#include <cmath>

/*
| **문제 1: Element-wise Sigmoid 연산**

| **Instruction**

| 1. 입력 텐서의 각 요소에 대해 Sigmoid 연산을 수행하는 CUDA 커널을 작성하세요.
| 2. 입력 텐서 크기: `(B, C, H, W)`
|    - `B`: 배치 크기, `C`: 채널 수, `H`: 높이, `W`: 너비.
| 3. 출력 텐서는 입력 텐서와 동일한 크기를 가집니다.
| 4. 각 스레드가 입력 텐서의 하나의 요소를 담당하며 Sigmoid 값을 계산.
*/

__global__ void sigmoid_kernel(int numel, const float *const data, float *output)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < numel)
    {
        output[idx] = 1 / (1 + exp(-data[idx]));
    }
}

at::Tensor sigmoid(at::Tensor X)
{
    int numel = X.numel();
    // | total number of elements
    at::Tensor output = torch::zeros_like(X);

    int dimBlocks = 16;
    int dimGrids = (numel + dimBlocks - 1) / dimBlocks;

    sigmoid_kernel<<<dimBlocks, dimGrids>>>(numel, X.data_ptr<float>(), output.data_ptr<float>());

    cudaDeviceSynchronize();

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("sigmoid", &sigmoid, "Sigmoid function for any shape of tensor");
}