import torch
import gymnasium as gym
import numpy as np

print("--- Environment Verification ---")
print(f"PyTorch Version: {torch.__version__}")

# Detect hardware acceleration (NVIDIA GPU, Apple Silicon, or CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Target Device: NVIDIA GPU (CUDA accelerated)")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Target Device: Apple Silicon GPU (MPS accelerated)")
else:
    device = torch.device("cpu")
    print("Target Device: Standard CPU (No GPU acceleration detected)")

print("\nTesting tiny 2-layer EMS Neural Network...")
# Mock input: [Vehicle Speed, Torque Request, Battery State-of-Charge]
model = torch.nn.Sequential(
    torch.nn.Linear(3, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 1)
).to(device)

dummy_state = torch.randn(1, 3).to(device)
output_action = model(dummy_state)
print(f"Success! Model output value: {output_action.item():.4f}")