"""Model architectures for computer vision tasks."""

from typing import Optional, List
import torch
import torch.nn as nn
import torchvision.models as models


class SimpleCNN(nn.Module):
    """Simple CNN architecture for image classification.
    
    A lightweight CNN suitable for CIFAR-10 and similar datasets.
    """
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        """Initialize SimpleCNN.
        
        Args:
            num_classes: Number of output classes.
            dropout_rate: Dropout rate for regularization.
        """
        super().__init__()
        
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
        # Store layer names for Grad-CAM
        self.layer_names = ['features.6', 'features.7']  # Last conv layers
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width).
            
        Returns:
            Output tensor of shape (batch_size, num_classes).
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def get_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract features from intermediate layers.
        
        Args:
            x: Input tensor.
            
        Returns:
            List of feature tensors from different layers.
        """
        features = []
        
        # Extract features from each block
        for i, layer in enumerate(self.features):
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                features.append(x)
        
        return features


class ResNet18(nn.Module):
    """ResNet-18 architecture with modifications for XAI.
    
    Modified ResNet-18 that exposes intermediate layers for explanation methods.
    """
    
    def __init__(
        self, 
        num_classes: int = 10, 
        pretrained: bool = False,
        dropout_rate: float = 0.5
    ):
        """Initialize ResNet-18.
        
        Args:
            num_classes: Number of output classes.
            pretrained: Whether to use pretrained weights.
            dropout_rate: Dropout rate for regularization.
        """
        super().__init__()
        
        # Load pretrained ResNet-18
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Modify the final layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.backbone.fc.in_features, num_classes)
        )
        
        # Store layer names for Grad-CAM
        self.layer_names = ['backbone.layer4.1.conv2']  # Last conv layer
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width).
            
        Returns:
            Output tensor of shape (batch_size, num_classes).
        """
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract features from intermediate layers.
        
        Args:
            x: Input tensor.
            
        Returns:
            List of feature tensors from different layers.
        """
        features = []
        
        # Extract features from each ResNet block
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        features.append(x)
        
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        features.append(x)
        
        x = self.backbone.layer2(x)
        features.append(x)
        
        x = self.backbone.layer3(x)
        features.append(x)
        
        x = self.backbone.layer4(x)
        features.append(x)
        
        return features


class VGG16(nn.Module):
    """VGG-16 architecture with modifications for XAI.
    
    Modified VGG-16 that exposes intermediate layers for explanation methods.
    """
    
    def __init__(
        self, 
        num_classes: int = 10, 
        pretrained: bool = False,
        dropout_rate: float = 0.5
    ):
        """Initialize VGG-16.
        
        Args:
            num_classes: Number of output classes.
            pretrained: Whether to use pretrained weights.
            dropout_rate: Dropout rate for regularization.
        """
        super().__init__()
        
        # Load pretrained VGG-16
        self.backbone = models.vgg16(pretrained=pretrained)
        
        # Modify the classifier
        self.backbone.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, num_classes)
        )
        
        # Store layer names for Grad-CAM
        self.layer_names = ['backbone.features.30']  # Last conv layer
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width).
            
        Returns:
            Output tensor of shape (batch_size, num_classes).
        """
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract features from intermediate layers.
        
        Args:
            x: Input tensor.
            
        Returns:
            List of feature tensors from different layers.
        """
        features = []
        
        # Extract features from VGG layers
        for i, layer in enumerate(self.backbone.features):
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                features.append(x)
        
        return features


def create_model(
    model_name: str,
    num_classes: int = 10,
    pretrained: bool = False,
    **kwargs
) -> nn.Module:
    """Create a model by name.
    
    Args:
        model_name: Name of the model ('simple_cnn', 'resnet18', 'vgg16').
        num_classes: Number of output classes.
        pretrained: Whether to use pretrained weights.
        **kwargs: Additional model arguments.
        
    Returns:
        PyTorch model.
        
    Raises:
        ValueError: If model_name is not supported.
    """
    if model_name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes, **kwargs)
    elif model_name == "resnet18":
        return ResNet18(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == "vgg16":
        return VGG16(num_classes=num_classes, pretrained=pretrained, **kwargs)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
