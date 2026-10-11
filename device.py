import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DeviceManager:
    """
    Dynamic Hardware Detection & Strategy Dispatcher for AetherKV.
    Routes execution to Triton/CUDA on NVIDIA GPUs, or PyTorch CPU Vectorized routines on CPU systems.
    """
    def __init__(self):
        self.has_cuda = torch.cuda.is_available()
        self.has_triton = False
        
        if self.has_cuda:
            try:
                import triton
                self.has_triton = True
            except ImportError:
                self.has_triton = False

        self.device = torch.device("cuda" if self.has_cuda else "cpu")
        self._log_system_capabilities()

    def _log_system_capabilities(self):
        if self.has_cuda:
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logging.info(f"⚡ GPU Detected: {gpu_name} | VRAM: {vram_gb:.2f} GB")
            logging.info(f"🚀 Triton Kernels Available: {self.has_triton}")
        else:
            logging.info("💻 No GPU detected. Running in High-Performance CPU Vectorized Fallback Mode.")

    def get_device(self) -> torch.device:
        return self.device

    def is_gpu_active(self) -> bool:
        return self.has_cuda and self.has_triton

# Singleton instance for system-wide dispatch
device_manager = DeviceManager()