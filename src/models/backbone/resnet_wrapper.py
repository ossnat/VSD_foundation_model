"""
2D ResNet backbone wrapper for image understanding tasks.
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNetWrapper(nn.Module):
    """
    2D ResNet backbone wrapper for image understanding.
    
    This wrapper provides a clean interface to various ResNet models from torchvision,
    with configurable output dimensions and feature extraction capabilities.
    """
    
    def __init__(self, model_name='resnet50', pretrained=True, num_classes=None, dropout=0.5):
        """
        Initialize ResNet backbone.
        
        Args:
            model_name (str): ResNet model name ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
            pretrained (bool): Whether to use pretrained weights
            num_classes (int, optional): Number of output classes. If None, returns features
            dropout (float): Dropout rate for the classifier
        """
        super(ResNetWrapper, self).__init__()
        
        # Load pretrained ResNet model
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            self.feature_dim = 512
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            self.feature_dim = 512
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        elif model_name == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            self.feature_dim = 2048
        elif model_name == 'resnet152':
            self.backbone = models.resnet152(pretrained=pretrained)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported ResNet model: {model_name}")
        
        # Remove the original classifier
        self.backbone.fc = nn.Identity()
        
        # Add custom classifier if num_classes is specified
        if num_classes is not None:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.feature_dim, num_classes)
            )
        else:
            self.classifier = None
    
    def forward(self, x):
        """
        Forward pass through the backbone.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Features or logits depending on classifier configuration
        """
        # Extract features
        features = self.backbone(x)
        
        # Apply classifier if available
        if self.classifier is not None:
            return self.classifier(features)
        
        return features
    
    def get_features(self, x):
        """
        Extract features without classification.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Feature tensor of shape (B, feature_dim)
        """
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
