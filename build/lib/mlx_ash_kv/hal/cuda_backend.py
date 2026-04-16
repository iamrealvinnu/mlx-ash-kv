import logging
import torch
import numpy as np
from typing import Any, List, Optional
from .base_backend import NeuralHealer

class CudaHealer(NeuralHealer):
    """
    NVIDIA CUDA/Triton Backend.
    Uses PyTorch for high-performance tensor operations on NVIDIA hardware.
    """

    def __init__(self):
        if not torch.cuda.is_available():
            logging.warning("[WARN] CUDA hardware not detected - Running in CPU Emulation Mode.")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda")
            logging.info(f"CUDA Healer Active on {torch.cuda.get_device_name(0)}")

    def _generate_gaussian_mask_cuda(self, seq_len: int, strike_indices: torch.Tensor, strike_sigmas: torch.Tensor, head_bitmasks: torch.Tensor, num_heads: int) -> torch.Tensor:
        """
        CUDA implementation of the Gaussian Penalty Mask.
        """
        # t: (S)
        t = torch.arange(seq_len, device=self.device, dtype=torch.float16)
        
        # mu, sigma: (N, 1)
        mu = strike_indices.unsqueeze(1)
        sigma = strike_sigmas.unsqueeze(1)
        
        # dist_sq: (N, S)
        dist_sq = torch.square(t.unsqueeze(0) - mu)
        penalty = -10000.0 * torch.exp(-dist_sq / (2 * torch.square(sigma) + 1e-6))
        
        # Causal mask: only penalize tokens at or after the strike index
        valid = (t.unsqueeze(0) >= mu) & (t.unsqueeze(0) > 0)
        penalty = torch.where(valid, penalty, torch.zeros_like(penalty))
        
        # Broadcast across heads: (N, 1, S) * (N, H, 1) -> (N, H, S)
        strike_masks = penalty.unsqueeze(1) * head_bitmasks.unsqueeze(2)
        
        # Aggregate across strikes: (H, S)
        mask, _ = torch.min(strike_masks, dim=0)
        
        # Final shape: (1, H, 1, S)
        return mask.view(1, num_heads, 1, seq_len)

    def generate_mask(self, seq_len: int, strikes: List[dict], num_heads: int) -> torch.Tensor:
        if not strikes:
            return torch.zeros((1, num_heads, 1, seq_len), device=self.device, dtype=torch.float16)
            
        indices = torch.tensor([s["index"] for s in strikes], device=self.device, dtype=torch.float16)
        sigmas = torch.tensor([s["sigma"] for s in strikes], device=self.device, dtype=torch.float16)
        
        h_masks = []
        for s in strikes:
            h_row = np.zeros(num_heads)
            h_row[s["heads"]] = 1.0
            h_masks.append(h_row)
        head_bitmasks = torch.tensor(np.array(h_masks), device=self.device, dtype=torch.float16)
        
        return self._generate_gaussian_mask_cuda(seq_len, indices, sigmas, head_bitmasks, num_heads)

    def eval_arrays(self, *arrays: Any) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def concat_arrays(self, arrays: List[torch.Tensor], axis: int) -> torch.Tensor:
        return torch.cat(arrays, dim=axis)

    def take_arrays(self, array: torch.Tensor, indices: Any, axis: int) -> torch.Tensor:
        # Convert indices to torch if they aren't already
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, device=self.device)
        return torch.index_select(array, dim=axis, index=indices)

    def page_to_disk(self, array: torch.Tensor, path: str) -> None:
        """Saves PyTorch tensor to disk."""
        torch.save(array.cpu(), path)

    def page_from_disk(self, path: str, shape: tuple, dtype: Any) -> torch.Tensor:
        """Loads PyTorch tensor from disk to the active device."""
        return torch.load(path).to(self.device)
