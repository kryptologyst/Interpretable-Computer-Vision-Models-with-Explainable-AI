"""XAI method implementations using Captum and custom implementations."""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from captum.attr import (
    GradCAM,
    IntegratedGradients,
    Saliency,
    GuidedBackprop,
    SmoothGrad,
    NoiseTunnel,
    Occlusion,
    ShapleyValueSampling,
    GradientShap,
    DeepLift,
    DeepLiftShap,
)
from captum.attr._utils.attribution import Attribution
import cv2


class GradCAMExplainer:
    """Gradient-weighted Class Activation Mapping (Grad-CAM) explainer.
    
    Grad-CAM uses gradients flowing into the last convolutional layer to produce
    a coarse localization map highlighting important regions in the image.
    """
    
    def __init__(self, model: nn.Module, target_layer: Optional[str] = None):
        """Initialize Grad-CAM explainer.
        
        Args:
            model: PyTorch model to explain.
            target_layer: Name of the target layer. If None, uses the last conv layer.
        """
        self.model = model
        self.model.eval()
        
        # Find target layer
        if target_layer is None:
            self.target_layer = self._find_last_conv_layer()
        else:
            self.target_layer = target_layer
        
        # Initialize Captum GradCAM
        self.gradcam = GradCAM(model, self.target_layer)
    
    def _find_last_conv_layer(self) -> str:
        """Find the last convolutional layer in the model."""
        conv_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append(name)
        
        if not conv_layers:
            raise ValueError("No convolutional layers found in the model")
        
        return conv_layers[-1]
    
    def explain(
        self, 
        inputs: torch.Tensor, 
        target: Optional[Union[int, torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        """Generate Grad-CAM explanations.
        
        Args:
            inputs: Input tensor of shape (batch_size, channels, height, width).
            target: Target class index or tensor. If None, uses predicted class.
            **kwargs: Additional arguments for GradCAM.
            
        Returns:
            Grad-CAM attributions of shape (batch_size, height, width).
        """
        if target is None:
            with torch.no_grad():
                outputs = self.model(inputs)
                target = torch.argmax(outputs, dim=1)
        
        attributions = self.gradcam.attribute(inputs, target=target, **kwargs)
        return attributions
    
    def visualize(
        self, 
        inputs: torch.Tensor, 
        attributions: torch.Tensor,
        alpha: float = 0.4
    ) -> np.ndarray:
        """Visualize Grad-CAM attributions.
        
        Args:
            inputs: Original input images.
            attributions: Grad-CAM attributions.
            alpha: Blending factor for visualization.
            
        Returns:
            Visualized images as numpy array.
        """
        batch_size = inputs.shape[0]
        visualized = []
        
        for i in range(batch_size):
            # Get original image
            img = inputs[i].cpu().detach().numpy()
            img = np.transpose(img, (1, 2, 0))
            
            # Normalize image
            img = (img - img.min()) / (img.max() - img.min())
            
            # Get attribution
            attr = attributions[i].cpu().detach().numpy()
            attr = np.squeeze(attr)
            
            # Resize attribution to match image size
            attr_resized = cv2.resize(attr, (img.shape[1], img.shape[0]))
            
            # Normalize attribution
            attr_resized = (attr_resized - attr_resized.min()) / (attr_resized.max() - attr_resized.min())
            
            # Create heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * attr_resized), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Blend image and heatmap
            blended = cv2.addWeighted(
                np.uint8(255 * img), alpha, heatmap, 1 - alpha, 0
            )
            
            visualized.append(blended)
        
        return np.array(visualized)


class IntegratedGradientsExplainer:
    """Integrated Gradients explainer.
    
    Integrated Gradients is an axiomatic attribution method that satisfies
    the axioms of sensitivity and implementation invariance.
    """
    
    def __init__(self, model: nn.Module, baseline: Optional[torch.Tensor] = None):
        """Initialize Integrated Gradients explainer.
        
        Args:
            model: PyTorch model to explain.
            baseline: Baseline input. If None, uses zeros.
        """
        self.model = model
        self.model.eval()
        self.baseline = baseline
        self.ig = IntegratedGradients(model)
    
    def explain(
        self, 
        inputs: torch.Tensor, 
        target: Optional[Union[int, torch.Tensor]] = None,
        steps: int = 50,
        **kwargs
    ) -> torch.Tensor:
        """Generate Integrated Gradients explanations.
        
        Args:
            inputs: Input tensor.
            target: Target class index or tensor. If None, uses predicted class.
            steps: Number of integration steps.
            **kwargs: Additional arguments for IntegratedGradients.
            
        Returns:
            Integrated Gradients attributions.
        """
        if target is None:
            with torch.no_grad():
                outputs = self.model(inputs)
                target = torch.argmax(outputs, dim=1)
        
        attributions = self.ig.attribute(
            inputs, 
            target=target, 
            baselines=self.baseline,
            n_steps=steps,
            **kwargs
        )
        return attributions


class LIMEExplainer:
    """LIME (Local Interpretable Model-agnostic Explanations) explainer.
    
    LIME explains individual predictions by approximating the model locally
    with an interpretable model.
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        """Initialize LIME explainer.
        
        Args:
            model: PyTorch model to explain.
            device: Device to run computations on.
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def explain(
        self, 
        inputs: torch.Tensor, 
        target: Optional[Union[int, torch.Tensor]] = None,
        num_samples: int = 1000,
        **kwargs
    ) -> torch.Tensor:
        """Generate LIME explanations.
        
        Args:
            inputs: Input tensor.
            target: Target class index or tensor. If None, uses predicted class.
            num_samples: Number of samples for LIME.
            **kwargs: Additional arguments.
            
        Returns:
            LIME attributions.
        """
        try:
            from lime import lime_image
            from skimage.segmentation import slic
        except ImportError:
            raise ImportError("LIME requires 'lime' and 'scikit-image' packages")
        
        if target is None:
            with torch.no_grad():
                outputs = self.model(inputs)
                target = torch.argmax(outputs, dim=1)
        
        # Convert to numpy for LIME
        img_np = inputs[0].cpu().detach().numpy().transpose(1, 2, 0)
        
        # Define prediction function
        def predict_fn(images):
            batch = torch.stack([
                torch.from_numpy(img.transpose(2, 0, 1)).float()
                for img in images
            ]).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(batch)
                return F.softmax(outputs, dim=1).cpu().numpy()
        
        # Create LIME explainer
        explainer = lime_image.LimeImageExplainer()
        
        # Generate explanation
        explanation = explainer.explain_instance(
            img_np,
            predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=num_samples,
            segmentation_fn=slic
        )
        
        # Extract attributions
        image, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=False,
            num_features=10,
            hide_rest=False
        )
        
        return torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)


class SHAPExplainer:
    """SHAP (SHapley Additive exPlanations) explainer.
    
    SHAP provides a unified framework for explaining model predictions
    using game-theoretic Shapley values.
    """
    
    def __init__(self, model: nn.Module, background: Optional[torch.Tensor] = None):
        """Initialize SHAP explainer.
        
        Args:
            model: PyTorch model to explain.
            background: Background samples for SHAP. If None, uses zeros.
        """
        self.model = model
        self.model.eval()
        self.background = background
        self.gradient_shap = GradientShap(model)
    
    def explain(
        self, 
        inputs: torch.Tensor, 
        target: Optional[Union[int, torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        """Generate SHAP explanations.
        
        Args:
            inputs: Input tensor.
            target: Target class index or tensor. If None, uses predicted class.
            **kwargs: Additional arguments for GradientShap.
            
        Returns:
            SHAP attributions.
        """
        if target is None:
            with torch.no_grad():
                outputs = self.model(inputs)
                target = torch.argmax(outputs, dim=1)
        
        if self.background is None:
            background = torch.zeros_like(inputs)
        else:
            background = self.background
        
        attributions = self.gradient_shap.attribute(
            inputs,
            target=target,
            baselines=background,
            **kwargs
        )
        return attributions


class SmoothGradExplainer:
    """SmoothGrad explainer.
    
    SmoothGrad reduces noise in gradient-based explanations by averaging
    gradients over multiple noisy versions of the input.
    """
    
    def __init__(self, model: nn.Module):
        """Initialize SmoothGrad explainer.
        
        Args:
            model: PyTorch model to explain.
        """
        self.model = model
        self.model.eval()
        self.smoothgrad = SmoothGrad(Saliency(model))
    
    def explain(
        self, 
        inputs: torch.Tensor, 
        target: Optional[Union[int, torch.Tensor]] = None,
        stdevs: float = 0.15,
        n_samples: int = 25,
        **kwargs
    ) -> torch.Tensor:
        """Generate SmoothGrad explanations.
        
        Args:
            inputs: Input tensor.
            target: Target class index or tensor. If None, uses predicted class.
            stdevs: Standard deviation of noise.
            n_samples: Number of noisy samples.
            **kwargs: Additional arguments for SmoothGrad.
            
        Returns:
            SmoothGrad attributions.
        """
        if target is None:
            with torch.no_grad():
                outputs = self.model(inputs)
                target = torch.argmax(outputs, dim=1)
        
        attributions = self.smoothgrad.attribute(
            inputs,
            target=target,
            stdevs=stdevs,
            n_samples=n_samples,
            **kwargs
        )
        return attributions


class OcclusionExplainer:
    """Occlusion-based explainer.
    
    Occlusion explains predictions by systematically occluding parts of the input
    and measuring the change in prediction.
    """
    
    def __init__(self, model: nn.Module):
        """Initialize Occlusion explainer.
        
        Args:
            model: PyTorch model to explain.
        """
        self.model = model
        self.model.eval()
        self.occlusion = Occlusion(model)
    
    def explain(
        self, 
        inputs: torch.Tensor, 
        target: Optional[Union[int, torch.Tensor]] = None,
        sliding_window_shapes: Tuple[int, int] = (3, 3),
        strides: Tuple[int, int] = (1, 1),
        **kwargs
    ) -> torch.Tensor:
        """Generate Occlusion explanations.
        
        Args:
            inputs: Input tensor.
            target: Target class index or tensor. If None, uses predicted class.
            sliding_window_shapes: Shape of the sliding window.
            strides: Stride of the sliding window.
            **kwargs: Additional arguments for Occlusion.
            
        Returns:
            Occlusion attributions.
        """
        if target is None:
            with torch.no_grad():
                outputs = self.model(inputs)
                target = torch.argmax(outputs, dim=1)
        
        attributions = self.occlusion.attribute(
            inputs,
            target=target,
            sliding_window_shapes=sliding_window_shapes,
            strides=strides,
            **kwargs
        )
        return attributions
