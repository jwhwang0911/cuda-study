from torch.utils.cpp_extension import load
import torch

custom_op = load(
    name="sigmoid",
    sources=["sigmoid.cu"],
    verbose=True
)

B, C, H, W = 1, 3, 32, 32
input_tensor = torch.randn(B, C, H, W).to("cuda")

output_custom = custom_op.sigmoid(input_tensor)

output_torch = torch.sigmoid(input_tensor)

print("Custom Sigmoid Output:")
print(output_custom)
print("Torch Sigmoid Output:")
print(output_torch)
print("Allclose:", torch.allclose(output_custom, output_torch))