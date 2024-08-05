#include <cuda_runtime.h>
#include <iostream>

// CUDA 행렬 덧셈 커널
__global__ void matrix_add(float* A, float* B, float* C, int N, int M) {
    // TODO: 두 행렬을 더하는 함수를 작성하시오.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < N && j < M)
        C[i * M + j] = A[i * M + j] + B[i * M + j]; 
}

int main() {
    int N = 4;
    int M = 4;
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
    std::cout << "A" << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            if (h_C[i * M + j] != h_A[i * M + j] + h_B[i * M + j]) {
                std::cerr << "Mismatch at index (" << i << ", " << j << ")!" << std::endl;
                break;
            } else {
                std::cout << h_A[i * M + j];
            }
        }
        std::cout << std::endl;
    }

    std::cout << "B" << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
                std::cout << h_B[i * M + j];
        }
        std::cout << std::endl;
    }

    std::cout << "C" << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
                std::cout << h_C[i * M + j];
        }
        std::cout << std::endl;
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
