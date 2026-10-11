import torch
import logging
from typing import Dict, Tuple, Optional
from aetherkv.device import device_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class AsyncPCIeOffloader:
    """
    Module 3: AsyncPCIeOffloader
    Asynchronously transfers low-entropy passive KV-cache blocks between GPU VRAM and CPU Host RAM
    using dedicated CUDA streams and pinned host memory buffers to prevent blocking forward pass decode steps.
    """
    def __init__(self):
        self.is_gpu = device_manager.is_gpu_active()
        
        # Dedicated CUDA Stream for PCIe Transfers (GPU Mode)
        self.transfer_stream = torch.cuda.Stream() if self.is_gpu else None
        
        # In-memory storage table for CPU-offloaded blocks
        # Key: block_id (str), Value: (quantized_tensor, scale_tensor)
        self.cpu_host_registry: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def offload_block_async(
        self, 
        block_id: str, 
        q_tensor: torch.Tensor, 
        scale_tensor: torch.Tensor
    ) -> bool:
        """
        Asynchronously streams a quantized passive KV block from GPU to Pinned CPU RAM.
        """
        try:
            if self.is_gpu and self.transfer_stream is not None:
                with torch.cuda.stream(self.transfer_stream):
                    # Copy to CPU Host pinned memory (non_blocking=True for PCIe overlap)
                    pinned_q = q_tensor.to("cpu", non_blocking=True)
                    pinned_scale = scale_tensor.to("cpu", non_blocking=True)
                    self.cpu_host_registry[block_id] = (pinned_q, pinned_scale)
            else:
                # CPU Fallback Path: In-memory Host Buffer register
                self.cpu_host_registry[block_id] = (q_tensor.detach().clone(), scale_tensor.detach().clone())
            
            return True
        except Exception as e:
            logging.error(f"Failed to offload block {block_id}: {str(e)}")
            return False

    def fetch_block_async(
        self, 
        block_id: str, 
        target_device: torch.device
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Retrieves an offloaded KV block back from CPU Host RAM to GPU VRAM asynchronously.
        """
        if block_id not in self.cpu_host_registry:
            logging.warning(f"Block ID {block_id} not found in CPU host registry!")
            return None

        pinned_q, pinned_scale = self.cpu_host_registry[block_id]

        if self.is_gpu and self.transfer_stream is not None:
            with torch.cuda.stream(self.transfer_stream):
                # Asynchronous PCIe transfer back to target GPU VRAM
                gpu_q = pinned_q.to(target_device, non_blocking=True)
                gpu_scale = pinned_scale.to(target_device, non_blocking=True)
                
            # Synchronize transfer stream before compute utilization
            torch.cuda.current_stream().wait_stream(self.transfer_stream)
            return gpu_q, gpu_scale
        else:
            # CPU Fallback Path
            return pinned_q.to(target_device), pinned_scale.to(target_device)

    def evict_host_block(self, block_id: str):
        """
        Frees host memory when a request completes generation.
        """
        if block_id in self.cpu_host_registry:
            del self.cpu_host_registry[block_id]

    def clear_registry(self):
        """
        Clears all cached host buffers.
        """
        self.cpu_host_registry.clear()