#include <cuda_runtime.h>
#include <iostream>

// CUDA 스칼라 곱 커널
__global__ void scalar_mult(float* A, float scalar, float* B, int N) {
    // TODO: 벡터의 모든 요소에 스칼라 값을 곱하는 함수를 작성하시오.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) B[i] = A[i] * scalar;
}

int main() {
    int N = 10;
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

    std::cout << "A" << std::endl;
    // 결과 검증
    for (int i = 0; i < N; i++) {
        if (h_B[i] != h_A[i] * scalar) {
            std::cerr << "Mismatch at index " << i << "!" << std::endl;
            break;
        }
        else {
                std::cout << h_A[i ] << "\t";
            }
    }

    std::cout << "\nB" << std::endl;
    for (int i = 0; i < N; i++) {
        std::cout<<h_B[i] << "\t";
    }

    // 메모리 해제
    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);

    std::cout << "Done!" << std::endl;
    return 0;
}