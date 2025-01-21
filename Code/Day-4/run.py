from torch.utils.cpp_extension import load
import torch

matrix_op_ = load(
    "custrom_op",
    sources="matrix_op.cu",
    extra_cuda_cflags=[
        "--expt-relaxed-constexpr"
    ],
    verbose=True
)

class matrix_op:
    elem_square = matrix_op_.elem_square
    T = matrix_op_.T
    matmul = matrix_op_.matmul
    batchmul = matrix_op_.batchmul
        

print("Ex 1.")
x = torch.rand(1000, device="cuda")
out = matrix_op.elem_square(x)
print(torch.allclose(out, x**2))

print("______________________\n")

print("Ex 2.")
x = torch.rand(32, 64, device="cuda")
y = matrix_op.T(x)

print(torch.allclose(x.T, y))

print("______________________\n")

print("Ex 3.")

A = torch.randn(64, 128, device='cuda')
v = torch.randn(128, device='cuda')
y = matrix_op.matmul(A, v)

print(torch.allclose(y, A @ v))

print("______________________\n")
print("Ex 4.")


A = torch.randn(8, 3, 4, device='cuda')
B = torch.randn(8, 4, 2, device='cuda')
C = matrix_op.batchmul(A, B)

temp = torch.matmul(A, B)

print(torch.allclose(C, torch.matmul(A, B)))