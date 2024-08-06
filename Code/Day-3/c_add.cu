#include<torch/extension.h>
#include<iostream>

__global__ void add_kernel(float *x, float *y, float *out, int64_t size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i < size ) {
        out[i] = x[i] + y[i];
    }
}

// 1-D case
torch::Tensor add(torch::Tensor x, torch::Tensor y) {
    auto output = torch::zeros_like(x);
    int64_t size = x.size(0);

    int threads = 1024;
    int blocks = (size + threads - 1) / threads;

    add_kernel<<<threads, blocks>>>(x.data_ptr<float>(), y.data_ptr<float>(), output.data_ptr<float>(), size);
    
    cudaDeviceSynchronize();

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add, "Add two 1-D tensors");
}