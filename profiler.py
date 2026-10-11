import torch

try:
    import aetherkv_cpp
except ImportError:
    aetherkv_cpp = None


class EntropyProfiler:
    """
    Module 1: EntropyProfiler
    Calculates Shannon Entropy on raw attention matrix and classifies tokens into Sink, Recent, or Passive tiers.
    """
    def __init__(self, sink_size: int = 16, recent_size: int = 128, entropy_threshold: float = 2.5):
        self.sink_size = sink_size
        self.recent_size = recent_size
        self.entropy_threshold = entropy_threshold

    def profile_attention(self, attn_weights: torch.Tensor):
        """
        attn_weights shape: [batch, heads, seq_len, seq_len]
        Returns:
            entropy: Token-level Shannon entropy tensor
            classifications: Token mask tensor (0=SINK, 1=RECENT, 2=PASSIVE)
        """
        if aetherkv_cpp is not None:
            entropy = aetherkv_cpp.compute_token_entropy(attn_weights)
            classifications = aetherkv_cpp.classify_tokens(
                entropy, self.sink_size, self.recent_size, self.entropy_threshold
            )
            return entropy, classifications
        else:
            # High-Performance PyTorch CPU Fallback Path
            p = torch.clamp(attn_weights, 1e-9, 1.0)
            entropy = -torch.sum(p * torch.log2(p), dim=-1).mean(dim=1)
            
            classifications = torch.full_like(entropy, 2, dtype=torch.int32)
            seq_len = attn_weights.shape[-1]
            
            # Sinks
            classifications[:, :min(self.sink_size, seq_len)] = 0
            # Recent
            classifications[:, max(0, seq_len - self.recent_size):] = 1
            
            return entropy, classifications