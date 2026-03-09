"""Evaluation metrics for explanation quality."""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns


class FaithfulnessMetrics:
    """Metrics for evaluating explanation faithfulness."""
    
    @staticmethod
    def deletion_auc(
        model: nn.Module,
        inputs: torch.Tensor,
        explanations: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        num_steps: int = 10,
        device: torch.device = None
    ) -> float:
        """Compute deletion AUC for faithfulness evaluation.
        
        Args:
            model: PyTorch model.
            inputs: Input tensor.
            explanations: Attribution tensor.
            target: Target class tensor.
            num_steps: Number of deletion steps.
            device: Device to run computations on.
            
        Returns:
            Deletion AUC score.
        """
        if device is None:
            device = inputs.device
        
        model.eval()
        if target is None:
            with torch.no_grad():
                outputs = model(inputs)
                target = torch.argmax(outputs, dim=1)
        
        batch_size = inputs.shape[0]
        deletion_scores = []
        
        for i in range(batch_size):
            input_img = inputs[i:i+1].to(device)
            attr = explanations[i]
            target_class = target[i]
            
            # Get original confidence
            with torch.no_grad():
                original_output = model(input_img)
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
                    masked_output = model(masked_input)
                    masked_conf = torch.softmax(masked_output, dim=1)[0, target_class].item()
                
                confidences.append(masked_conf)
            
            # Compute AUC
            deletion_scores.append(np.trapz(confidences, dx=1/num_steps))
        
        return np.mean(deletion_scores)
    
    @staticmethod
    def insertion_auc(
        model: nn.Module,
        inputs: torch.Tensor,
        explanations: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        num_steps: int = 10,
        device: torch.device = None
    ) -> float:
        """Compute insertion AUC for faithfulness evaluation.
        
        Args:
            model: PyTorch model.
            inputs: Input tensor.
            explanations: Attribution tensor.
            target: Target class tensor.
            num_steps: Number of insertion steps.
            device: Device to run computations on.
            
        Returns:
            Insertion AUC score.
        """
        if device is None:
            device = inputs.device
        
        model.eval()
        if target is None:
            with torch.no_grad():
                outputs = model(inputs)
                target = torch.argmax(outputs, dim=1)
        
        batch_size = inputs.shape[0]
        insertion_scores = []
        
        for i in range(batch_size):
            input_img = inputs[i:i+1].to(device)
            attr = explanations[i]
            target_class = target[i]
            
            # Start with baseline (zeros)
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
                    output = model(inserted_input)
                    conf = torch.softmax(output, dim=1)[0, target_class].item()
                
                confidences.append(conf)
            
            # Compute AUC
            insertion_scores.append(np.trapz(confidences, dx=1/num_steps))
        
        return np.mean(insertion_scores)
    
    @staticmethod
    def sufficiency(
        model: nn.Module,
        inputs: torch.Tensor,
        explanations: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        top_k: float = 0.1,
        device: torch.device = None
    ) -> float:
        """Compute sufficiency metric.
        
        Measures how much of the prediction can be explained by the top-k features.
        
        Args:
            model: PyTorch model.
            inputs: Input tensor.
            explanations: Attribution tensor.
            target: Target class tensor.
            top_k: Fraction of top features to keep.
            device: Device to run computations on.
            
        Returns:
            Sufficiency score.
        """
        if device is None:
            device = inputs.device
        
        model.eval()
        if target is None:
            with torch.no_grad():
                outputs = model(inputs)
                target = torch.argmax(outputs, dim=1)
        
        batch_size = inputs.shape[0]
        sufficiency_scores = []
        
        for i in range(batch_size):
            input_img = inputs[i:i+1].to(device)
            attr = explanations[i]
            target_class = target[i]
            
            # Get original confidence
            with torch.no_grad():
                original_output = model(input_img)
                original_conf = torch.softmax(original_output, dim=1)[0, target_class].item()
            
            # Get top-k features
            flat_attr = attr.flatten()
            k = int(len(flat_attr) * top_k)
            top_indices = torch.argsort(flat_attr, descending=True)[:k]
            
            # Create input with only top-k features
            masked_input = torch.zeros_like(input_img)
            flat_input = input_img.view(-1)
            flat_masked = masked_input.view(-1)
            flat_masked[top_indices] = flat_input[top_indices]
            masked_input = flat_masked.view(input_img.shape)
            
            # Get confidence on masked input
            with torch.no_grad():
                masked_output = model(masked_input)
                masked_conf = torch.softmax(masked_output, dim=1)[0, target_class].item()
            
            # Sufficiency is the ratio of masked to original confidence
            sufficiency_scores.append(masked_conf / original_conf if original_conf > 0 else 0)
        
        return np.mean(sufficiency_scores)
    
    @staticmethod
    def necessity(
        model: nn.Module,
        inputs: torch.Tensor,
        explanations: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        top_k: float = 0.1,
        device: torch.device = None
    ) -> float:
        """Compute necessity metric.
        
        Measures how much the prediction drops when removing the top-k features.
        
        Args:
            model: PyTorch model.
            inputs: Input tensor.
            explanations: Attribution tensor.
            target: Target class tensor.
            top_k: Fraction of top features to remove.
            device: Device to run computations on.
            
        Returns:
            Necessity score.
        """
        if device is None:
            device = inputs.device
        
        model.eval()
        if target is None:
            with torch.no_grad():
                outputs = model(inputs)
                target = torch.argmax(outputs, dim=1)
        
        batch_size = inputs.shape[0]
        necessity_scores = []
        
        for i in range(batch_size):
            input_img = inputs[i:i+1].to(device)
            attr = explanations[i]
            target_class = target[i]
            
            # Get original confidence
            with torch.no_grad():
                original_output = model(input_img)
                original_conf = torch.softmax(original_output, dim=1)[0, target_class].item()
            
            # Get top-k features
            flat_attr = attr.flatten()
            k = int(len(flat_attr) * top_k)
            top_indices = torch.argsort(flat_attr, descending=True)[:k]
            
            # Create input without top-k features
            masked_input = input_img.clone()
            flat_input = masked_input.view(-1)
            flat_input[top_indices] = flat_input.mean()  # Replace with mean
            masked_input = flat_input.view(input_img.shape)
            
            # Get confidence on masked input
            with torch.no_grad():
                masked_output = model(masked_input)
                masked_conf = torch.softmax(masked_output, dim=1)[0, target_class].item()
            
            # Necessity is the drop in confidence
            necessity_scores.append((original_conf - masked_conf) / original_conf if original_conf > 0 else 0)
        
        return np.mean(necessity_scores)


class StabilityMetrics:
    """Metrics for evaluating explanation stability."""
    
    @staticmethod
    def spearman_correlation(
        explanations1: torch.Tensor,
        explanations2: torch.Tensor
    ) -> float:
        """Compute Spearman correlation between two explanations.
        
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
    
    @staticmethod
    def kendall_tau(
        explanations1: torch.Tensor,
        explanations2: torch.Tensor
    ) -> float:
        """Compute Kendall's tau between two explanations.
        
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
    
    @staticmethod
    def intersection_over_union(
        explanations1: torch.Tensor,
        explanations2: torch.Tensor,
        threshold: float = 0.5
    ) -> float:
        """Compute IoU between two explanations.
        
        Args:
            explanations1: First set of explanations.
            explanations2: Second set of explanations.
            threshold: Threshold for binary mask.
            
        Returns:
            Mean IoU score.
        """
        ious = []
        
        for i in range(explanations1.shape[0]):
            attr1 = explanations1[i].flatten().cpu().numpy()
            attr2 = explanations2[i].flatten().cpu().numpy()
            
            # Normalize to [0, 1]
            attr1 = (attr1 - attr1.min()) / (attr1.max() - attr1.min() + 1e-8)
            attr2 = (attr2 - attr2.min()) / (attr2.max() - attr2.min() + 1e-8)
            
            # Create binary masks
            mask1 = attr1 > threshold
            mask2 = attr2 > threshold
            
            # Compute IoU
            intersection = np.sum(mask1 & mask2)
            union = np.sum(mask1 | mask2)
            
            iou = intersection / union if union > 0 else 0
            ious.append(iou)
        
        return np.mean(ious)


class FidelityMetrics:
    """Metrics for evaluating explanation fidelity."""
    
    @staticmethod
    def surrogate_accuracy(
        model: nn.Module,
        surrogate_model: nn.Module,
        inputs: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        device: torch.device = None
    ) -> float:
        """Compute accuracy of surrogate model.
        
        Args:
            model: Original model.
            surrogate_model: Surrogate model trained on explanations.
            inputs: Input tensor.
            target: Target class tensor.
            device: Device to run computations on.
            
        Returns:
            Surrogate model accuracy.
        """
        if device is None:
            device = inputs.device
        
        model.eval()
        surrogate_model.eval()
        
        if target is None:
            with torch.no_grad():
                outputs = model(inputs)
                target = torch.argmax(outputs, dim=1)
        
        with torch.no_grad():
            surrogate_outputs = surrogate_model(inputs)
            surrogate_preds = torch.argmax(surrogate_outputs, dim=1)
        
        accuracy = (surrogate_preds == target).float().mean().item()
        return accuracy
    
    @staticmethod
    def explanation_completeness(
        model: nn.Module,
        inputs: torch.Tensor,
        explanations: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        device: torch.device = None
    ) -> float:
        """Compute explanation completeness.
        
        Measures how much of the model's decision can be explained.
        
        Args:
            model: PyTorch model.
            inputs: Input tensor.
            explanations: Attribution tensor.
            target: Target class tensor.
            device: Device to run computations on.
            
        Returns:
            Completeness score.
        """
        if device is None:
            device = inputs.device
        
        model.eval()
        if target is None:
            with torch.no_grad():
                outputs = model(inputs)
                target = torch.argmax(outputs, dim=1)
        
        batch_size = inputs.shape[0]
        completeness_scores = []
        
        for i in range(batch_size):
            input_img = inputs[i:i+1].to(device)
            attr = explanations[i]
            target_class = target[i]
            
            # Get original prediction
            with torch.no_grad():
                original_output = model(input_img)
                original_conf = torch.softmax(original_output, dim=1)[0, target_class].item()
            
            # Compute explanation-based prediction
            # This is a simplified version - in practice, you'd train a surrogate model
            explanation_sum = torch.sum(attr).item()
            explanation_conf = torch.sigmoid(torch.tensor(explanation_sum)).item()
            
            # Completeness is the ratio of explanation confidence to original confidence
            completeness_scores.append(explanation_conf / original_conf if original_conf > 0 else 0)
        
        return np.mean(completeness_scores)


class ExplanationEvaluator:
    """Comprehensive evaluation of explanation methods."""
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        """Initialize explanation evaluator.
        
        Args:
            model: PyTorch model to evaluate.
            device: Device to run computations on.
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
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
            method_results['deletion_auc'] = FaithfulnessMetrics.deletion_auc(
                self.model, inputs, method_explanations, target, device=self.device
            )
            method_results['insertion_auc'] = FaithfulnessMetrics.insertion_auc(
                self.model, inputs, method_explanations, target, device=self.device
            )
            method_results['sufficiency'] = FaithfulnessMetrics.sufficiency(
                self.model, inputs, method_explanations, target, device=self.device
            )
            method_results['necessity'] = FaithfulnessMetrics.necessity(
                self.model, inputs, method_explanations, target, device=self.device
            )
            
            # Stability metrics (compare with random baseline)
            random_explanations = torch.randn_like(method_explanations)
            method_results['spearman_vs_random'] = StabilityMetrics.spearman_correlation(
                method_explanations, random_explanations
            )
            method_results['kendall_vs_random'] = StabilityMetrics.kendall_tau(
                method_explanations, random_explanations
            )
            method_results['iou_vs_random'] = StabilityMetrics.intersection_over_union(
                method_explanations, random_explanations
            )
            
            # Fidelity metrics
            method_results['completeness'] = FidelityMetrics.explanation_completeness(
                self.model, inputs, method_explanations, target, device=self.device
            )
            
            results[method_name] = method_results
        
        return results
    
    def create_leaderboard(
        self,
        evaluation_results: Dict[str, Dict[str, float]]
    ) -> None:
        """Create and display evaluation leaderboard.
        
        Args:
            evaluation_results: Results from comprehensive evaluation.
        """
        # Convert to DataFrame for better visualization
        import pandas as pd
        
        df_data = []
        for method, metrics in evaluation_results.items():
            row = {'Method': method}
            row.update(metrics)
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Display leaderboard
        print("\n" + "="*80)
        print("EXPLANATION METHOD LEADERBOARD")
        print("="*80)
        print(df.to_string(index=False, float_format='%.4f'))
        
        # Create visualization
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
        plt.savefig('assets/evaluation_leaderboard.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\nLeaderboard saved to: assets/evaluation_leaderboard.png")
