import torch
from torch.utils.cpp_extension import load

custom_ops = load(
    name="custom_ops",
    sources="matrix_mult.cu",
    extra_cuda_cflags=[
        "--expt-relaxed-constexpr"
    ],
    verbose=True
)

X = torch.Tensor(
    [
        [1, 2],
        [3, 4]
    ]
).to("cuda")

Y = torch.Tensor(
    [
        [1, 2],
        [3, 4]
    ]
).to("cuda")


Z =  custom_ops.matrix_mult(X, Y)
print(Z)