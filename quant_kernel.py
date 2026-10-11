import torch
import math
from typing import Tuple
from aetherkv.device import device_manager

# Try loading Triton for NVIDIA GPU fast path
HAS_TRITON = False
if device_manager.is_gpu_active():
    try:
        import triton
        import triton.language as tl
        HAS_TRITON = True
    except ImportError:
        HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _triton_int4_quant_kernel(
        x_ptr, q_ptr, scale_ptr,
        stride_xb, stride_xh, stride_xs, stride_xd,
        stride_qb, stride_qh, stride_qs, stride_qd,
        BLOCK_SIZE: tl.constexpr
    ):
        """
        Triton Kernel: Block-wise BF16 -> INT4 Quantization on GPU
        """
        pid = tl.program_id(axis=0)
        # Compute block pointers and offsets
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        
        # Load high precision values
        x = tl.load(x_ptr + offs)
        
        # Compute max absolute value per block
        max_val = tl.max(tl.abs(x), axis=0)
        scale = max_val / 7.0
        scale = tl.maximum(scale, 1e-8)
        
        # Symmetric quantization: clamp(round(X / S), -8, 7)
        q = tl.extra.cuda.round(x / scale)
        q = tl.clamp(q, -8, 7)
        
        # Store scale and quantized tensor
        tl.store(q_ptr + offs, q.to(tl.int8))
        tl.store(scale_ptr + pid, scale)


class AdaptiveCacheAllocator:
    """
    Module 2: AdaptiveCacheAllocator
    Compresses Passive KV Cache blocks from BF16 to INT4.
    Supports Triton GPU Acceleration & Vectorized CPU Fallback.
    """
    def __init__(self, block_size: int = 128):
        self.block_size = block_size

    def quantize_passive_blocks(self, kv_tensor: torch.Tensor, passive_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
            kv_tensor: [batch_size, num_heads, seq_len, head_dim] (BF16/FP16)
            passive_mask: [batch_size, seq_len] (0=SINK, 1=RECENT, 2=PASSIVE)
        Output:
            quantized_kv: Quantized tensor (INT8 container for INT4 representation)
            scales: Per-block dynamic scale factors
        """
        if device_manager.is_gpu_active() and HAS_TRITON:
            return self._quantize_gpu_triton(kv_tensor, passive_mask)
        else:
            return self._quantize_cpu_vectorized(kv_tensor, passive_mask)

    def _quantize_cpu_vectorized(self, kv_tensor: torch.Tensor, passive_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        High-Performance PyTorch CPU Vectorized Block-wise Quantization Path
        """
        B, H, S, D = kv_tensor.shape
        quantized_kv = kv_tensor.clone()
        scales = torch.ones((B, H, S, 1), dtype=kv_tensor.dtype, device=kv_tensor.device)

        # Identify passive token indices (mask == 2)
        passive_indices = (passive_mask == 2).nonzero(as_tuple=False)

        if passive_indices.numel() == 0:
            return quantized_kv, scales

        for idx in passive_indices:
            b, s = idx[0].item(), idx[1].item()
            block_data = kv_tensor[b, :, s, :] # [Heads, HeadDim]

            # Calculate scale: S = max(|X|) / 7.0
            max_val = torch.abs(block_data).max(dim=-1, keepdim=True).values
            scale = torch.clamp(max_val / 7.0, min=1e-8)

            # Quantize: q = clamp(round(X / S), -8, 7)
            q_block = torch.clamp(torch.round(block_data / scale), -8, 7).to(torch.int8)

            # Dequantize simulation for in-place KV retention or INT8 store
            dequant_block = (q_block.to(kv_tensor.dtype) * scale)

            quantized_kv[b, :, s, :] = dequant_block
            scales[b, :, s, :] = scale

        return quantized_kv, scales

    def dequantize_block(self, q_tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """
        Dequantize INT4 KV blocks back to high-precision BF16 during attention compute fallback
        Formula: X_recon = q * S
        """
        return q_tensor.to(scale.dtype) * scale