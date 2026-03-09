"""Utility functions for reproducible research and device management."""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible results.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for additional reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU).
    
    Returns:
        PyTorch device object.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
    **kwargs
) -> None:
    """Save model checkpoint.
    
    Args:
        model: PyTorch model.
        optimizer: Optimizer.
        epoch: Current epoch.
        loss: Current loss.
        filepath: Path to save checkpoint.
        **kwargs: Additional data to save.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        **kwargs
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    filepath: str,
    device: torch.device
) -> dict:
    """Load model checkpoint.
    
    Args:
        model: PyTorch model.
        optimizer: Optimizer (optional).
        filepath: Path to checkpoint file.
        device: Device to load checkpoint on.
        
    Returns:
        Checkpoint dictionary.
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint
