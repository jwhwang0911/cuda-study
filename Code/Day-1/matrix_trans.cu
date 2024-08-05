/* 
    1. 데이터 생성
    2. 데이터 cpu -> gpu
    3. 함수 실행
    4. 데이터 gpu -> cpu
    5. 확인
*/
#include<iostream>
int main() {
    // 1. 데이터 생성
    int N = 4;
    int M = 5;
    float **M_A = new float*[N];
    float **M_B = new float*[M];
    
    int large = (N < M) ? M : N;
    int small = (N < M) ? N : M;

    
}