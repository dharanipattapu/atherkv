import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from aetherkv.device import device_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class SequenceRequest:
    request_id: str
    prompt: str
    prompt_tokens: int
    max_gen_tokens: int
    arrival_time: float = field(default_factory=time.time)
    allocated_blocks: List[str] = field(default_factory=list)
    passive_block_count: int = 0
    offloaded_block_count: int = 0
    status: str = "WAITING"  # WAITING, RUNNING, COMPLETED


class EntropyAwareScheduler:
    """
    Module 4: EntropyAwareScheduler
    Continuous batching request scheduler that dynamically recalculates active VRAM capacity 
    based on AetherKV compression ratios and admits new requests to maximize concurrency.
    """
    def __init__(self, total_vram_mb: float = 24576.0, max_vram_threshold: float = 0.85):
        self.total_vram_mb = total_vram_mb
        self.max_vram_threshold = max_vram_threshold
        
        self.waiting_queue: List[SequenceRequest] = []
        self.running_queue: List[SequenceRequest] = []
        self.completed_requests: List[SequenceRequest] = []
        
        # Memory estimation weights per 1K tokens
        self.bf16_mb_per_1k = 128.0  # Approx 128MB per 1k tokens in BF16 (32K context scaling)
        self.int4_mb_per_1k = 32.0    # 4x compressed footprint
        
    def add_request(self, request_id: str, prompt: str, prompt_tokens: int, max_gen_tokens: int = 128):
        req = SequenceRequest(
            request_id=request_id,
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            max_gen_tokens=max_gen_tokens
        )
        self.waiting_queue.append(req)

    def calculate_active_vram_usage(self) -> float:
        """
        Calculates total VRAM footprint across all running requests considering compression savings.
        """
        total_usage_mb = 0.0
        for req in self.running_queue:
            total_tokens = req.prompt_tokens + req.max_gen_tokens
            tokens_k = total_tokens / 1000.0
            
            # Estimate uncompressed vs compressed allocations
            active_bf16_tokens_k = max(0.0, tokens_k - (req.passive_block_count * 0.128))
            
            req_vram = (active_bf16_tokens_k * self.bf16_mb_per_1k) + (req.passive_block_count * self.int4_mb_per_1k)
            total_usage_mb += req_vram

        return total_usage_mb

    def schedule_step(self) -> Dict[str, Any]:
        """
        Main continuous batching step: Admits waiting requests if VRAM usage remains under threshold.
        """
        current_vram_mb = self.calculate_active_vram_usage()
        capacity_limit_mb = self.total_vram_mb * self.max_vram_threshold
        
        admitted_count = 0
        
        while self.waiting_queue:
            next_req = self.waiting_queue[0]
            req_est_vram = (next_req.prompt_tokens / 1000.0) * self.bf16_mb_per_1k
            
            if current_vram_mb + req_est_vram <= capacity_limit_mb:
                req = self.waiting_queue.pop(0)
                req.status = "RUNNING"
                self.running_queue.append(req)
                current_vram_mb += req_est_vram
                admitted_count += 1
            else:
                # VRAM threshold reached; cannot admit more requests in this step
                break

        return {
            "running_count": len(self.running_queue),
            "waiting_count": len(self.waiting_queue),
            "admitted_this_step": admitted_count,
            "vram_usage_mb": round(current_vram_mb, 2),
            "vram_utilization_pct": round((current_vram_mb / self.total_vram_mb) * 100.0, 2)
        }

    def update_request_compression_stats(self, request_id: str, passive_blocks: int, offloaded_blocks: int):
        """
        Feedback loop from Modules 1-3 updating request memory profiles after forward pass profiling.
        """
        for req in self.running_queue:
            if req.request_id == request_id:
                req.passive_block_count = passive_blocks
                req.offloaded_block_count = offloaded_blocks
                break