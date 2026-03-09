"""High-level explanation interfaces and evaluation metrics."""

from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

from .methods import (
    GradCAMExplainer,
    IntegratedGradientsExplainer,
    LIMEExplainer,
    SHAPExplainer,
    SmoothGradExplainer,
    OcclusionExplainer,
)
from ..utils import get_device


class XAIExplainer:
    """Unified interface for multiple XAI methods.
    
    Provides a high-level interface to various explanation methods with
    consistent API and evaluation capabilities.
    """
    
    def __init__(
        self,
        model: nn.Module,
        methods: List[str] = None,
        device: Optional[torch.device] = None
    ):
        """Initialize XAI explainer.
        
        Args:
            model: PyTorch model to explain.
            methods: List of explanation methods to use.
            device: Device to run computations on.
        """
        self.model = model
        self.device = device or get_device()
        self.model.to(self.device)
        self.model.eval()
        
        # Available methods
        self.available_methods = {
            'gradcam': GradCAMExplainer,
            'integrated_gradients': IntegratedGradientsExplainer,
            'lime': LIMEExplainer,
            'shap': SHAPExplainer,
            'smoothgrad': SmoothGradExplainer,
            'occlusion': OcclusionExplainer,
        }
        
        # Initialize selected methods
        self.methods = methods or ['gradcam', 'integrated_gradients']
        self.explainers = {}
        
        for method_name in self.methods:
            if method_name not in self.available_methods:
                raise ValueError(f"Unknown method: {method_name}")
            
            method_class = self.available_methods[method_name]
            self.explainers[method_name] = method_class(model, self.device)
    
    def explain(
        self,
        inputs: torch.Tensor,
        target: Optional[Union[int, torch.Tensor]] = None,
        method: Optional[str] = None,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Generate explanations for inputs.
        
        Args:
            inputs: Input tensor.
            target: Target class index or tensor.
            method: Specific method to use. If None, uses all methods.
            **kwargs: Additional arguments for explanation methods.
            
        Returns:
            Explanations from specified method(s).
        """
        inputs = inputs.to(self.device)
        
        if method is not None:
            if method not in self.explainers:
                raise ValueError(f"Method {method} not initialized")
            return self.explainers[method].explain(inputs, target, **kwargs)
        
        # Generate explanations from all methods
        explanations = {}
        for method_name, explainer in self.explainers.items():
            try:
                explanations[method_name] = explainer.explain(inputs, target, **kwargs)
            except Exception as e:
                print(f"Warning: Failed to generate {method_name} explanation: {e}")
                continue
        
        return explanations
    
    def visualize(
        self,
        inputs: torch.Tensor,
        explanations: Union[torch.Tensor, Dict[str, torch.Tensor]],
        method: Optional[str] = None,
        save_path: Optional[str] = None,
        **kwargs
    ) -> None:
        """Visualize explanations.
        
        Args:
            inputs: Original input images.
            explanations: Explanations to visualize.
            method: Specific method to visualize.
            save_path: Path to save visualization.
            **kwargs: Additional visualization arguments.
        """
        if method is not None:
            if method not in self.explainers:
                raise ValueError(f"Method {method} not initialized")
            
            if isinstance(explanations, dict):
                explanations = explanations[method]
            
            if hasattr(self.explainers[method], 'visualize'):
                viz = self.explainers[method].visualize(inputs, explanations, **kwargs)
                
                if save_path:
                    plt.figure(figsize=(10, 5))
                    for i, img in enumerate(viz[:4]):  # Show first 4 images
                        plt.subplot(1, 4, i + 1)
                        plt.imshow(img)
                        plt.axis('off')
                        plt.title(f'Image {i + 1}')
                    plt.tight_layout()
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close()
            else:
                # Generic visualization for methods without custom visualize
                self._generic_visualize(inputs, explanations, save_path)
        else:
            # Visualize all methods
            if isinstance(explanations, dict):
                for method_name, method_explanations in explanations.items():
                    self.visualize(inputs, method_explanations, method_name, save_path, **kwargs)
    
    def _generic_visualize(
        self,
        inputs: torch.Tensor,
        explanations: torch.Tensor,
        save_path: Optional[str] = None
    ) -> None:
        """Generic visualization for explanations.
        
        Args:
            inputs: Original input images.
            explanations: Explanations to visualize.
            save_path: Path to save visualization.
        """
        batch_size = min(4, inputs.shape[0])
        
        fig, axes = plt.subplots(2, batch_size, figsize=(4 * batch_size, 8))
        if batch_size == 1:
            axes = axes.reshape(2, 1)
        
        for i in range(batch_size):
            # Original image
            img = inputs[i].cpu().detach().numpy().transpose(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min())
            
            axes[0, i].imshow(img)
            axes[0, i].set_title(f'Original {i + 1}')
            axes[0, i].axis('off')
            
            # Explanation
            attr = explanations[i].cpu().detach().numpy()
            if attr.ndim == 3:  # Multi-channel attribution
                attr = np.mean(attr, axis=0)
            
            im = axes[1, i].imshow(attr, cmap='RdBu_r')
            axes[1, i].set_title(f'Attribution {i + 1}')
            axes[1, i].axis('off')
            plt.colorbar(im, ax=axes[1, i])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


class ExplanationEvaluator:
    """Evaluator for explanation quality metrics.
    
    Provides comprehensive evaluation of explanation methods including
    faithfulness, stability, and fidelity metrics.
    """
    
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        """Initialize explanation evaluator.
        
        Args:
            model: PyTorch model to evaluate.
            device: Device to run computations on.
        """
        self.model = model
        self.device = device or get_device()
        self.model.to(self.device)
        self.model.eval()
    
    def faithfulness_deletion(
        self,
        inputs: torch.Tensor,
        explanations: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        num_steps: int = 10
    ) -> float:
        """Compute faithfulness using deletion test.
        
        Measures how well explanations identify important features by
        progressively removing the most important features and measuring
        the drop in model confidence.
        
        Args:
            inputs: Input tensor.
            explanations: Attribution tensor.
            target: Target class tensor.
            num_steps: Number of deletion steps.
            
        Returns:
            Deletion AUC score.
        """
        if target is None:
            with torch.no_grad():
                outputs = self.model(inputs)
                target = torch.argmax(outputs, dim=1)
        
        batch_size = inputs.shape[0]
        deletion_scores = []
        
        for i in range(batch_size):
            input_img = inputs[i:i+1]
            attr = explanations[i]
            target_class = target[i]
            
            # Get original confidence
            with torch.no_grad():
                original_output = self.model(input_img)
                original_conf = torch.softmax(original_output, dim=1)[0, target_class].item()
            
            # Sort pixels by attribution importance
            flat_attr = attr.flatten()
            sorted_indices = torch.argsort(flat_attr, descending=True)
            
            # Progressive deletion
            confidences = [original_conf]
            for step in range(1, num_steps + 1):
                # Create masked input
                masked_input = input_img.clone()
                mask_size = int(len(sorted_indices) * step / num_steps)
                mask_indices = sorted_indices[:mask_size]
                
                # Apply mask (set to mean value)
                flat_input = masked_input.view(-1)
                flat_input[mask_indices] = flat_input.mean()
                masked_input = flat_input.view(input_img.shape)
                
                # Get confidence on masked input
                with torch.no_grad():
                    masked_output = self.model(masked_input)
                    masked_conf = torch.softmax(masked_output, dim=1)[0, target_class].item()
                
                confidences.append(masked_conf)
            
            # Compute AUC
            deletion_scores.append(np.trapz(confidences, dx=1/num_steps))
        
        return np.mean(deletion_scores)
    
    def faithfulness_insertion(
        self,
        inputs: torch.Tensor,
        explanations: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        num_steps: int = 10
    ) -> float:
        """Compute faithfulness using insertion test.
        
        Measures how well explanations identify important features by
        progressively adding the most important features and measuring
        the increase in model confidence.
        
        Args:
            inputs: Input tensor.
            explanations: Attribution tensor.
            target: Target class tensor.
            num_steps: Number of insertion steps.
            
        Returns:
            Insertion AUC score.
        """
        if target is None:
            with torch.no_grad():
                outputs = self.model(inputs)
                target = torch.argmax(outputs, dim=1)
        
        batch_size = inputs.shape[0]
        insertion_scores = []
        
        for i in range(batch_size):
            input_img = inputs[i:i+1]
            attr = explanations[i]
            target_class = target[i]
            
            # Start with baseline (zeros or mean)
            baseline = torch.zeros_like(input_img)
            
            # Sort pixels by attribution importance
            flat_attr = attr.flatten()
            sorted_indices = torch.argsort(flat_attr, descending=True)
            
            # Progressive insertion
            confidences = []
            for step in range(num_steps + 1):
                # Create input with top features
                inserted_input = baseline.clone()
                mask_size = int(len(sorted_indices) * step / num_steps)
                mask_indices = sorted_indices[:mask_size]
                
                # Insert important features
                flat_input = input_img.view(-1)
                flat_inserted = inserted_input.view(-1)
                flat_inserted[mask_indices] = flat_input[mask_indices]
                inserted_input = flat_inserted.view(input_img.shape)
                
                # Get confidence
                with torch.no_grad():
                    output = self.model(inserted_input)
                    conf = torch.softmax(output, dim=1)[0, target_class].item()
                
                confidences.append(conf)
            
            # Compute AUC
            insertion_scores.append(np.trapz(confidences, dx=1/num_steps))
        
        return np.mean(insertion_scores)
    
    def stability_spearman(
        self,
        explanations1: torch.Tensor,
        explanations2: torch.Tensor
    ) -> float:
        """Compute stability using Spearman correlation.
        
        Args:
            explanations1: First set of explanations.
            explanations2: Second set of explanations.
            
        Returns:
            Mean Spearman correlation coefficient.
        """
        correlations = []
        
        for i in range(explanations1.shape[0]):
            attr1 = explanations1[i].flatten().cpu().numpy()
            attr2 = explanations2[i].flatten().cpu().numpy()
            
            corr, _ = spearmanr(attr1, attr2)
            if not np.isnan(corr):
                correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0.0
    
    def stability_kendall(
        self,
        explanations1: torch.Tensor,
        explanations2: torch.Tensor
    ) -> float:
        """Compute stability using Kendall's tau.
        
        Args:
            explanations1: First set of explanations.
            explanations2: Second set of explanations.
            
        Returns:
            Mean Kendall's tau coefficient.
        """
        correlations = []
        
        for i in range(explanations1.shape[0]):
            attr1 = explanations1[i].flatten().cpu().numpy()
            attr2 = explanations2[i].flatten().cpu().numpy()
            
            corr, _ = kendalltau(attr1, attr2)
            if not np.isnan(corr):
                correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0.0
    
    def evaluate_comprehensive(
        self,
        inputs: torch.Tensor,
        explanations: Dict[str, torch.Tensor],
        target: Optional[torch.Tensor] = None
    ) -> Dict[str, Dict[str, float]]:
        """Comprehensive evaluation of explanation methods.
        
        Args:
            inputs: Input tensor.
            explanations: Dictionary of explanations from different methods.
            target: Target class tensor.
            
        Returns:
            Dictionary of evaluation metrics for each method.
        """
        results = {}
        
        for method_name, method_explanations in explanations.items():
            method_results = {}
            
            # Faithfulness metrics
            method_results['deletion_auc'] = self.faithfulness_deletion(
                inputs, method_explanations, target
            )
            method_results['insertion_auc'] = self.faithfulness_insertion(
                inputs, method_explanations, target
            )
            
            # Stability metrics (compare with random baseline)
            random_explanations = torch.randn_like(method_explanations)
            method_results['spearman_vs_random'] = self.stability_spearman(
                method_explanations, random_explanations
            )
            method_results['kendall_vs_random'] = self.stability_kendall(
                method_explanations, random_explanations
            )
            
            results[method_name] = method_results
        
        return results
