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
