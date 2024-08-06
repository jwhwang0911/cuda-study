from torch.utils.cpp_extension import load

custom_add = load(
    name="custom_add",
    sources="c_add.cu",
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
z = custom_add.add(x, y)

print(f"x : {x}\n y : {y}")
print(f"add result : {z}")