import torch

print("PyTorch version :", torch.__version__)
print("CUDA available  :", torch.cuda.is_available())

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("GPU name        :", props.name)
    print("GPU memory (GB) :", round(props.total_memory / 1024**3, 1))
    print("bf16 supported  :", torch.cuda.is_bf16_supported())

    # Tiny real computation on the GPU to prove it works end to end
    x = torch.randn(2000, 2000, device="cuda")
    y = x @ x
    print("Matmul on GPU OK, result shape:", tuple(y.shape))