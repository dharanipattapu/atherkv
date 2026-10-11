import torch
import math
from typing import List, Dict, Any

class AetherKVMetrics:
    """
    Evaluation suite for measuring accuracy preservation, classification precision/recall, 
    and system performance metrics across KV cache management runs.
    """

    @staticmethod
    def calculate_selection_precision_recall(
        predicted_passive_masks: torch.Tensor, 
        ground_truth_low_attn_masks: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculates Precision, Recall, and F1-Score for Passive Token Selection.
        - True Positive (TP): Correctly identified low-importance passive tokens.
        - False Positive (FP): Important tokens mistakenly classified as passive (Risk of accuracy drop).
        """
        pred_bool = (predicted_passive_masks == 2)
        gt_bool = (ground_truth_low_attn_masks == 1)

        tp = (pred_bool & gt_bool).sum().item()
        fp = (pred_bool & ~gt_bool).sum().item()
        fn = (~pred_bool & gt_bool).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4)
        }

    @staticmethod
    def calculate_perplexity(logits: torch.Tensor, target_ids: torch.Tensor) -> float:
        """
        Computes generation Perplexity (PPL).
        Baseline PPL vs AetherKV PPL comparison proves if model quality remains stable.
        """
        loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
        shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        shift_labels = target_ids[..., 1:].contiguous().view(-1)
        
        loss = loss_fn(shift_logits, shift_labels)
        perplexity = math.exp(loss.item())
        return round(perplexity, 4)

    @staticmethod
    def evaluate_niah_retrieval(retrieved_text: str, secret_passkey: str) -> bool:
        """
        Needle-In-A-Haystack (NIAH) Factual Retrieval Verification.
        Returns True if the exact passkey was successfully extracted from long context.
        """
        return secret_passkey.strip().lower() in retrieved_text.strip().lower()

    @staticmethod
    def calculate_vram_reduction(baseline_vram_mb: float, compressed_vram_mb: float) -> Dict[str, float]:
        """
        Calculates total VRAM footprint savings.
        """
        reduction_percentage = ((baseline_vram_mb - compressed_vram_mb) / baseline_vram_mb) * 100.0
        compression_ratio = baseline_vram_mb / (compressed_vram_mb + 1e-8)
        
        return {
            "vram_saved_mb": round(baseline_vram_mb - compressed_vram_mb, 2),
            "reduction_percentage": round(reduction_percentage, 2),
            "compression_ratio": round(compression_ratio, 2)
        }