from torch.utils.cpp_extension import load

custom_ops = load(
    name="custom_ops",
    sources="c_ops.cu",
    extra_cuda_cflags=[
        "--expt-relaxed-constexpr"
    ],
    verbose=True
)

import torch

x = torch.randint(low=0, high=10, size=(10,), device="cuda")
y = torch.randint(low=0, high=20, size=(10,), device="cuda")
x = x.type(torch.FloatTensor).to("cuda")
y = y.type(torch.FloatTensor).to("cuda")
z = custom_ops.add(x, y)

print(f"x : {x}\n y : {y}")
print(f"add result : {z}")

print("___________________________")

# x = torch.Tensor([1.1, 2.2, 3.3, 4.4])
y = torch.Tensor([1, 2, 4, 6]).to("cuda")
c = 3

print(y)
z = custom_ops.mult(y, c)
print(z)