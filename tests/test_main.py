"""Test suite for interpretable computer vision XAI package."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch

from interpretable_cv_xai.models import SimpleCNN, ResNet18, create_model
from interpretable_cv_xai.data import CIFAR10DataModule, create_synthetic_dataset
from interpretable_cv_xai.explainers import XAIExplainer
from interpretable_cv_xai.methods import GradCAMExplainer, IntegratedGradientsExplainer
from interpretable_cv_xai.metrics import FaithfulnessMetrics, StabilityMetrics
from interpretable_cv_xai.utils import set_seed, get_device, count_parameters


class TestModels:
    """Test model architectures."""
    
    def test_simple_cnn_creation(self):
        """Test SimpleCNN model creation."""
        model = SimpleCNN(num_classes=10)
        assert isinstance(model, nn.Module)
        assert count_parameters(model) > 0
    
    def test_resnet18_creation(self):
        """Test ResNet18 model creation."""
        model = ResNet18(num_classes=10, pretrained=False)
        assert isinstance(model, nn.Module)
        assert count_parameters(model) > 0
    
    def test_create_model_function(self):
        """Test create_model function."""
        model = create_model("simple_cnn", num_classes=10)
        assert isinstance(model, nn.Module)
        
        with pytest.raises(ValueError):
            create_model("invalid_model", num_classes=10)
    
    def test_model_forward_pass(self):
        """Test model forward pass."""
        model = SimpleCNN(num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        assert output.shape == (2, 10)
    
    def test_model_features_extraction(self):
        """Test feature extraction from models."""
        model = SimpleCNN(num_classes=10)
        x = torch.randn(1, 3, 32, 32)
        features = model.get_features(x)
        assert isinstance(features, list)
        assert len(features) > 0


class TestDataModule:
    """Test data loading and preprocessing."""
    
    def test_cifar10_datamodule_creation(self):
        """Test CIFAR10DataModule creation."""
        data_module = CIFAR10DataModule(batch_size=32)
        assert data_module.batch_size == 32
        assert data_module.num_workers == 4
    
    def test_synthetic_dataset_creation(self):
        """Test synthetic dataset creation."""
        images, labels = create_synthetic_dataset(num_samples=100)
        assert images.shape == (100, 3, 32, 32)
        assert labels.shape == (100,)
        assert labels.max() < 10
        assert labels.min() >= 0
    
    def test_dataloader_creation(self):
        """Test dataloader creation."""
        data_module = CIFAR10DataModule(batch_size=16)
        
        # Test train dataloader
        train_loader = data_module.train_dataloader()
        assert isinstance(train_loader, torch.utils.data.DataLoader)
        assert train_loader.batch_size == 16
        
        # Test val dataloader
        val_loader = data_module.val_dataloader()
        assert isinstance(val_loader, torch.utils.data.DataLoader)
        
        # Test test dataloader
        test_loader = data_module.test_dataloader()
        assert isinstance(test_loader, torch.utils.data.DataLoader)


class TestExplainers:
    """Test explanation methods."""
    
    def test_gradcam_explainer_creation(self):
        """Test GradCAM explainer creation."""
        model = SimpleCNN(num_classes=10)
        explainer = GradCAMExplainer(model)
        assert explainer.model == model
        assert explainer.target_layer is not None
    
    def test_integrated_gradients_explainer_creation(self):
        """Test Integrated Gradients explainer creation."""
        model = SimpleCNN(num_classes=10)
        explainer = IntegratedGradientsExplainer(model)
        assert explainer.model == model
    
    def test_xai_explainer_creation(self):
        """Test XAI explainer creation."""
        model = SimpleCNN(num_classes=10)
        explainer = XAIExplainer(model, methods=['gradcam'])
        assert 'gradcam' in explainer.explainers
    
    def test_explanation_generation(self):
        """Test explanation generation."""
        model = SimpleCNN(num_classes=10)
        explainer = GradCAMExplainer(model)
        
        # Create test input
        x = torch.randn(1, 3, 32, 32)
        
        # Generate explanation
        explanation = explainer.explain(x, target=0)
        assert explanation.shape[0] == 1  # Batch size
        assert len(explanation.shape) >= 2  # At least 2D


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_faithfulness_metrics(self):
        """Test faithfulness metrics."""
        model = SimpleCNN(num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        target = torch.tensor([0, 1])
        explanations = torch.randn(2, 1, 32, 32)
        
        # Test deletion AUC
        deletion_auc = FaithfulnessMetrics.deletion_auc(
            model, x, explanations, target, num_steps=5
        )
        assert isinstance(deletion_auc, float)
        assert 0 <= deletion_auc <= 1
        
        # Test insertion AUC
        insertion_auc = FaithfulnessMetrics.insertion_auc(
            model, x, explanations, target, num_steps=5
        )
        assert isinstance(insertion_auc, float)
        assert 0 <= insertion_auc <= 1
    
    def test_stability_metrics(self):
        """Test stability metrics."""
        explanations1 = torch.randn(2, 1, 32, 32)
        explanations2 = torch.randn(2, 1, 32, 32)
        
        # Test Spearman correlation
        spearman_corr = StabilityMetrics.spearman_correlation(explanations1, explanations2)
        assert isinstance(spearman_corr, float)
        assert -1 <= spearman_corr <= 1
        
        # Test Kendall's tau
        kendall_tau = StabilityMetrics.kendall_tau(explanations1, explanations2)
        assert isinstance(kendall_tau, float)
        assert -1 <= kendall_tau <= 1


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Test that seed is set
        torch.manual_seed(42)
        x1 = torch.randn(10)
        
        torch.manual_seed(42)
        x2 = torch.randn(10)
        
        assert torch.allclose(x1, x2)
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = SimpleCNN(num_classes=10)
        param_count = count_parameters(model)
        assert isinstance(param_count, int)
        assert param_count > 0


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_explanation(self):
        """Test end-to-end explanation pipeline."""
        # Create model
        model = SimpleCNN(num_classes=10)
        
        # Create explainer
        explainer = XAIExplainer(model, methods=['gradcam'])
        
        # Create test data
        x = torch.randn(2, 3, 32, 32)
        target = torch.tensor([0, 1])
        
        # Generate explanations
        explanations = explainer.explain(x, target)
        
        assert 'gradcam' in explanations
        assert explanations['gradcam'].shape[0] == 2
    
    def test_evaluation_pipeline(self):
        """Test evaluation pipeline."""
        from interpretable_cv_xai.metrics import ExplanationEvaluator
        
        # Create model and test data
        model = SimpleCNN(num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        target = torch.tensor([0, 1])
        explanations = {'gradcam': torch.randn(2, 1, 32, 32)}
        
        # Create evaluator
        evaluator = ExplanationEvaluator(model)
        
        # Run evaluation
        results = evaluator.evaluate_comprehensive(x, explanations, target)
        
        assert 'gradcam' in results
        assert 'deletion_auc' in results['gradcam']
        assert 'insertion_auc' in results['gradcam']


@pytest.fixture
def sample_model():
    """Fixture for sample model."""
    return SimpleCNN(num_classes=10)


@pytest.fixture
def sample_data():
    """Fixture for sample data."""
    return torch.randn(2, 3, 32, 32), torch.tensor([0, 1])


def test_model_with_sample_data(sample_model, sample_data):
    """Test model with sample data."""
    x, target = sample_data
    output = sample_model(x)
    assert output.shape == (2, 10)


if __name__ == "__main__":
    pytest.main([__file__])
