"""Visualization utilities for XAI explanations."""

from typing import Dict, List, Optional, Tuple, Union
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class XAIVisualizer:
    """Comprehensive visualization tools for XAI explanations."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """Initialize XAI visualizer.
        
        Args:
            figsize: Default figure size for matplotlib plots.
        """
        self.figsize = figsize
        self.setup_style()
    
    def setup_style(self) -> None:
        """Setup matplotlib and seaborn styles."""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Custom colormap for attributions
        colors = ['#000080', '#0000FF', '#00FFFF', '#FFFF00', '#FF0000']
        n_bins = 256
        self.attribution_cmap = LinearSegmentedColormap.from_list('attribution', colors, N=n_bins)
    
    def visualize_gradcam(
        self,
        images: torch.Tensor,
        explanations: torch.Tensor,
        class_names: Optional[List[str]] = None,
        predictions: Optional[torch.Tensor] = None,
        save_path: Optional[str] = None,
        alpha: float = 0.4
    ) -> None:
        """Visualize Grad-CAM explanations.
        
        Args:
            images: Original input images.
            explanations: Grad-CAM attributions.
            class_names: List of class names.
            predictions: Model predictions.
            save_path: Path to save visualization.
            alpha: Blending factor for heatmap overlay.
        """
        batch_size = min(4, images.shape[0])
        fig, axes = plt.subplots(2, batch_size, figsize=(4 * batch_size, 8))
        
        if batch_size == 1:
            axes = axes.reshape(2, 1)
        
        for i in range(batch_size):
            # Original image
            img = images[i].cpu().detach().numpy().transpose(1, 2, 0)
            img = self._normalize_image(img)
            
            axes[0, i].imshow(img)
            title = f'Original {i + 1}'
            if predictions is not None and class_names is not None:
                pred_class = torch.argmax(predictions[i]).item()
                conf = torch.softmax(predictions[i], dim=0)[pred_class].item()
                title += f'\n{class_names[pred_class]} ({conf:.2f})'
            axes[0, i].set_title(title)
            axes[0, i].axis('off')
            
            # Grad-CAM heatmap
            attr = explanations[i].cpu().detach().numpy()
            attr = np.squeeze(attr)
            
            # Resize attribution to match image size
            attr_resized = cv2.resize(attr, (img.shape[1], img.shape[0]))
            
            # Create heatmap overlay
            heatmap = self._create_heatmap_overlay(img, attr_resized, alpha)
            
            axes[1, i].imshow(heatmap)
            axes[1, i].set_title(f'Grad-CAM {i + 1}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def visualize_attributions(
        self,
        images: torch.Tensor,
        explanations: Dict[str, torch.Tensor],
        method_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """Visualize attributions from multiple methods.
        
        Args:
            images: Original input images.
            explanations: Dictionary of explanations from different methods.
            method_names: Names of explanation methods.
            save_path: Path to save visualization.
        """
        if method_names is None:
            method_names = list(explanations.keys())
        
        batch_size = min(2, images.shape[0])
        num_methods = len(method_names)
        
        fig, axes = plt.subplots(
            batch_size, num_methods + 1, 
            figsize=(4 * (num_methods + 1), 4 * batch_size)
        )
        
        if batch_size == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(batch_size):
            # Original image
            img = images[i].cpu().detach().numpy().transpose(1, 2, 0)
            img = self._normalize_image(img)
            
            axes[i, 0].imshow(img)
            axes[i, 0].set_title('Original')
            axes[i, 0].axis('off')
            
            # Attribution maps
            for j, method_name in enumerate(method_names):
                attr = explanations[method_name][i].cpu().detach().numpy()
                
                if attr.ndim == 3:  # Multi-channel attribution
                    attr = np.mean(attr, axis=0)
                
                im = axes[i, j + 1].imshow(attr, cmap=self.attribution_cmap)
                axes[i, j + 1].set_title(method_name)
                axes[i, j + 1].axis('off')
                plt.colorbar(im, ax=axes[i, j + 1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def visualize_evaluation_metrics(
        self,
        evaluation_results: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None
    ) -> None:
        """Visualize evaluation metrics comparison.
        
        Args:
            evaluation_results: Results from explanation evaluation.
            save_path: Path to save visualization.
        """
        import pandas as pd
        
        # Convert to DataFrame
        df_data = []
        for method, metrics in evaluation_results.items():
            row = {'Method': method}
            row.update(metrics)
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Faithfulness metrics
        faithfulness_cols = ['deletion_auc', 'insertion_auc', 'sufficiency', 'necessity']
        df_faithfulness = df[['Method'] + faithfulness_cols]
        df_faithfulness.set_index('Method').plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Faithfulness Metrics')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Stability metrics
        stability_cols = ['spearman_vs_random', 'kendall_vs_random', 'iou_vs_random']
        df_stability = df[['Method'] + stability_cols]
        df_stability.set_index('Method').plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Stability Metrics')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Overall ranking
        df['overall_score'] = df[faithfulness_cols + stability_cols + ['completeness']].mean(axis=1)
        df_ranking = df[['Method', 'overall_score']].sort_values('overall_score', ascending=False)
        df_ranking.set_index('Method').plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Overall Ranking')
        axes[1, 0].set_ylabel('Overall Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Metric correlation heatmap
        metric_cols = faithfulness_cols + stability_cols + ['completeness']
        corr_matrix = df[metric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
        axes[1, 1].set_title('Metric Correlation')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def create_interactive_dashboard(
        self,
        images: torch.Tensor,
        explanations: Dict[str, torch.Tensor],
        class_names: Optional[List[str]] = None,
        predictions: Optional[torch.Tensor] = None
    ) -> None:
        """Create interactive Plotly dashboard.
        
        Args:
            images: Original input images.
            explanations: Dictionary of explanations.
            class_names: List of class names.
            predictions: Model predictions.
        """
        batch_size = min(4, images.shape[0])
        num_methods = len(explanations)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=batch_size,
            subplot_titles=[f'Image {i+1}' for i in range(batch_size)],
            specs=[[{"type": "image"} for _ in range(batch_size)],
                   [{"type": "image"} for _ in range(batch_size)]]
        )
        
        for i in range(batch_size):
            # Original image
            img = images[i].cpu().detach().numpy().transpose(1, 2, 0)
            img = self._normalize_image(img)
            
            fig.add_trace(
                go.Image(z=img),
                row=1, col=i+1
            )
            
            # First explanation method
            method_name = list(explanations.keys())[0]
            attr = explanations[method_name][i].cpu().detach().numpy()
            if attr.ndim == 3:
                attr = np.mean(attr, axis=0)
            
            fig.add_trace(
                go.Heatmap(z=attr, colorscale='RdBu'),
                row=2, col=i+1
            )
        
        fig.update_layout(
            title="Interactive XAI Dashboard",
            height=600,
            showlegend=False
        )
        
        fig.show()
    
    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range.
        
        Args:
            img: Input image array.
            
        Returns:
            Normalized image array.
        """
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return img
    
    def _create_heatmap_overlay(
        self, 
        img: np.ndarray, 
        attr: np.ndarray, 
        alpha: float
    ) -> np.ndarray:
        """Create heatmap overlay on image.
        
        Args:
            img: Original image.
            attr: Attribution map.
            alpha: Blending factor.
            
        Returns:
            Blended image with heatmap overlay.
        """
        # Normalize attribution
        attr_norm = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * attr_norm), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Blend image and heatmap
        blended = cv2.addWeighted(
            np.uint8(255 * img), alpha, heatmap, 1 - alpha, 0
        )
        
        return blended
    
    def save_explanation_gallery(
        self,
        images: torch.Tensor,
        explanations: Dict[str, torch.Tensor],
        class_names: Optional[List[str]] = None,
        predictions: Optional[torch.Tensor] = None,
        save_dir: str = "assets/explanations"
    ) -> None:
        """Save explanation gallery to files.
        
        Args:
            images: Original input images.
            explanations: Dictionary of explanations.
            class_names: List of class names.
            predictions: Model predictions.
            save_dir: Directory to save images.
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        batch_size = images.shape[0]
        
        for i in range(batch_size):
            # Create figure for each image
            fig, axes = plt.subplots(1, len(explanations) + 1, figsize=(4 * (len(explanations) + 1), 4))
            
            # Original image
            img = images[i].cpu().detach().numpy().transpose(1, 2, 0)
            img = self._normalize_image(img)
            
            axes[0].imshow(img)
            title = f'Original {i + 1}'
            if predictions is not None and class_names is not None:
                pred_class = torch.argmax(predictions[i]).item()
                conf = torch.softmax(predictions[i], dim=0)[pred_class].item()
                title += f'\n{class_names[pred_class]} ({conf:.2f})'
            axes[0].set_title(title)
            axes[0].axis('off')
            
            # Attribution maps
            for j, (method_name, attr) in enumerate(explanations.items()):
                attr_np = attr[i].cpu().detach().numpy()
                if attr_np.ndim == 3:
                    attr_np = np.mean(attr_np, axis=0)
                
                im = axes[j + 1].imshow(attr_np, cmap=self.attribution_cmap)
                axes[j + 1].set_title(method_name)
                axes[j + 1].axis('off')
                plt.colorbar(im, ax=axes[j + 1], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/explanation_{i:03d}.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"Explanation gallery saved to: {save_dir}")


class AttentionVisualizer:
    """Visualization tools for attention mechanisms."""
    
    @staticmethod
    def visualize_attention_weights(
        attention_weights: torch.Tensor,
        input_tokens: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """Visualize attention weights as heatmap.
        
        Args:
            attention_weights: Attention weight tensor.
            input_tokens: List of input tokens.
            save_path: Path to save visualization.
        """
        weights = attention_weights.cpu().detach().numpy()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(weights, annot=True, cmap='Blues', fmt='.3f')
        
        if input_tokens:
            plt.xticks(range(len(input_tokens)), input_tokens, rotation=45)
            plt.yticks(range(len(input_tokens)), input_tokens)
        
        plt.title('Attention Weights')
        plt.xlabel('Key Tokens')
        plt.ylabel('Query Tokens')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def visualize_attention_rollout(
        attention_weights: torch.Tensor,
        input_tokens: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """Visualize attention rollout.
        
        Args:
            attention_weights: Attention weight tensor.
            input_tokens: List of input tokens.
            save_path: Path to save visualization.
        """
        # Compute attention rollout
        rollout = torch.eye(attention_weights.shape[-1])
        for layer_weights in attention_weights:
            rollout = torch.matmul(layer_weights, rollout)
        
        rollout_np = rollout.cpu().detach().numpy()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(rollout_np, annot=True, cmap='Reds', fmt='.3f')
        
        if input_tokens:
            plt.xticks(range(len(input_tokens)), input_tokens, rotation=45)
            plt.yticks(range(len(input_tokens)), input_tokens)
        
        plt.title('Attention Rollout')
        plt.xlabel('Input Tokens')
        plt.ylabel('Input Tokens')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
