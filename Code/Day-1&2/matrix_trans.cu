/* 
    1. 데이터 생성
    2. 데이터 cpu -> gpu
    3. 함수 실행
    4. 데이터 gpu -> cpu
    5. 확인
*/
#include<cuda_runtime.h>
#include<iostream>
#define N 4
#define M 5

inline int idx(int i, int j) {
    return i * M + j;
}

__global__ void MatrixTranspose(float* dst, float* src) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i >= N || j >= M) return;

    dst[j * N + i] = src[i * M + j];
}

int main() {
    // 1. 데이터 생성
    
    // 2d array를 생성하여 gpu에 할당하는 것은 좋지 못한 선택임.
    // 2d array를 single array로 구현하여 사용하는 것이 gpu의 활용도를 높임.

    const size_t size = N * M * sizeof(float);
    float* M_A = new float[size];
    float* M_B = new float[size];

    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++) {
            M_A[idx(i, j)] = idx(i, j);
        }
    }

    // 2. 데이터 cpu(host) -> gpu(device)
    // gpu 메모리 할당
    float *d_A;
    float *d_B;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);

    cudaMemcpy(d_A, M_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, M_B, size, cudaMemcpyHostToDevice);

    // Matrix 연산에서 효율적으로 활용할 수 있는 dim3
    dim3 dimBlock(16, 16);
    dim3 dimGrid( (N+dimBlock.x - 1)/ dimBlock.x , (M + dimBlock.y - 1) / dimBlock.y);
    // 3. 함수실행
    MatrixTranspose<<<dimBlock, dimGrid>>>(d_B, d_A);

    //데이터 gpu -> cpu
    cudaMemcpy(M_A, d_A, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(M_B, d_B, size, cudaMemcpyDeviceToHost);

    // 결과 검증
    std::cout << "A" << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            if (M_B[j * N + i] != M_A[i * M + j]) {
                std::cerr << "Mismatch at index (" << i << ", " << j << ")!" << std::endl;
                break;
            } else {
                std::cout << M_A[i * M + j] << "\t";
            }
        }
        std::cout << std::endl;
    }

    std::cout << "B" << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
                std::cout << M_B[i * N + j] << "\t";
        }
        std::cout << std::endl;
    }
}