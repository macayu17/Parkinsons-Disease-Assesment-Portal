"""
Test CUDA setup and GPU functionality
"""
import torch
import numpy as np

print("=" * 60)
print("CUDA Setup Test")
print("=" * 60)

# Check PyTorch version
print(f"\nPyTorch Version: {torch.__version__}")

# Check CUDA availability
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
        print(f"  Memory Reserved: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
        print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    
    # Test CUDA computation
    print("\n" + "=" * 60)
    print("Testing CUDA Computation Speed")
    print("=" * 60)
    
    # Create large tensors
    size = 5000
    
    # CPU test
    print("\nCPU Test:")
    x_cpu = torch.randn(size, size)
    y_cpu = torch.randn(size, size)
    
    import time
    start = time.time()
    z_cpu = torch.matmul(x_cpu, y_cpu)
    cpu_time = time.time() - start
    print(f"  Matrix multiplication ({size}x{size}): {cpu_time:.4f} seconds")
    
    # GPU test
    print("\nGPU Test:")
    x_gpu = torch.randn(size, size).cuda()
    y_gpu = torch.randn(size, size).cuda()
    
    # Warm up
    z_gpu = torch.matmul(x_gpu, y_gpu)
    torch.cuda.synchronize()
    
    start = time.time()
    z_gpu = torch.matmul(x_gpu, y_gpu)
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"  Matrix multiplication ({size}x{size}): {gpu_time:.4f} seconds")
    
    speedup = cpu_time / gpu_time
    print(f"\n🚀 GPU Speedup: {speedup:.2f}x faster than CPU!")
    
    print("\n" + "=" * 60)
    print("✅ CUDA is properly configured and working!")
    print("=" * 60)
else:
    print("\n❌ CUDA is not available. Training will use CPU.")
    print("Make sure you have:")
    print("  1. NVIDIA GPU")
    print("  2. CUDA Toolkit installed")
    print("  3. PyTorch with CUDA support installed")
