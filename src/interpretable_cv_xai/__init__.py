"""Interpretable Computer Vision Models with Explainable AI.

This package provides state-of-the-art explainable AI methods for computer vision tasks,
focusing on post-hoc local interpretability techniques.
"""

__version__ = "1.0.0"
__author__ = "XAI Research Team"

from .explainers import XAIExplainer
from .models import SimpleCNN, ResNet18
from .data import CIFAR10DataModule, ImageNetDataModule
from .metrics import ExplanationEvaluator

__all__ = [
    "XAIExplainer",
    "SimpleCNN", 
    "ResNet18",
    "CIFAR10DataModule",
    "ImageNetDataModule",
    "ExplanationEvaluator",
]
