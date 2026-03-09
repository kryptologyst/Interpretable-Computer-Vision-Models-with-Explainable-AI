"""Evaluation framework for XAI methods."""

from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import time
from tqdm import tqdm

from .metrics import ExplanationEvaluator
from .viz import XAIVisualizer
from ..utils import get_device


class XAIEvaluationFramework:
    """Comprehensive evaluation framework for XAI methods."""
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        results_dir: str = "results"
    ):
        """Initialize evaluation framework.
        
        Args:
            model: PyTorch model to evaluate.
            device: Device to run computations on.
            results_dir: Directory to save evaluation results.
        """
        self.model = model
        self.device = device or get_device()
        self.model.to(self.device)
        self.model.eval()
        
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.evaluator = ExplanationEvaluator(model, self.device)
        self.visualizer = XAIVisualizer()
        
        # Store evaluation results
        self.results = {}
    
    def evaluate_methods(
        self,
        dataloader: torch.utils.data.DataLoader,
        explainer,
        methods: List[str] = None,
        max_samples: int = 100,
        save_results: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate multiple XAI methods on a dataset.
        
        Args:
            dataloader: DataLoader with test data.
            explainer: XAI explainer instance.
            methods: List of methods to evaluate.
            max_samples: Maximum number of samples to evaluate.
            save_results: Whether to save results to disk.
            
        Returns:
            Dictionary of evaluation results.
        """
        if methods is None:
            methods = explainer.methods
        
        print(f"Evaluating methods: {methods}")
        print(f"Max samples: {max_samples}")
        
        all_explanations = {method: [] for method in methods}
        all_inputs = []
        all_targets = []
        
        sample_count = 0
        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc="Generating explanations")):
            if sample_count >= max_samples:
                break
            
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Generate explanations for all methods
            explanations = explainer.explain(inputs, targets)
            
            for method in methods:
                if method in explanations:
                    all_explanations[method].append(explanations[method].cpu())
            
            all_inputs.append(inputs.cpu())
            all_targets.append(targets.cpu())
            
            sample_count += inputs.shape[0]
        
        # Concatenate all results
        all_inputs = torch.cat(all_inputs, dim=0)[:max_samples]
        all_targets = torch.cat(all_targets, dim=0)[:max_samples]
        
        for method in methods:
            if all_explanations[method]:
                all_explanations[method] = torch.cat(all_explanations[method], dim=0)[:max_samples]
        
        # Evaluate each method
        evaluation_results = {}
        for method in methods:
            if method in all_explanations and len(all_explanations[method]) > 0:
                print(f"\nEvaluating {method}...")
                
                method_results = self.evaluator.evaluate_comprehensive(
                    all_inputs.to(self.device),
                    {method: all_explanations[method].to(self.device)},
                    all_targets.to(self.device)
                )
                
                evaluation_results[method] = method_results[method]
                
                # Add timing information
                start_time = time.time()
                _ = explainer.explain(all_inputs[:1].to(self.device), method=method)
                timing = time.time() - start_time
                evaluation_results[method]['avg_time_per_sample'] = timing
        
        # Store results
        self.results = evaluation_results
        
        if save_results:
            self._save_results(evaluation_results)
        
        return evaluation_results
    
    def compare_methods(
        self,
        evaluation_results: Dict[str, Dict[str, float]],
        metric_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Compare methods using weighted scoring.
        
        Args:
            evaluation_results: Results from method evaluation.
            metric_weights: Weights for different metrics.
            
        Returns:
            Overall scores for each method.
        """
        if metric_weights is None:
            metric_weights = {
                'deletion_auc': 0.25,
                'insertion_auc': 0.25,
                'sufficiency': 0.15,
                'necessity': 0.15,
                'spearman_vs_random': 0.10,
                'kendall_vs_random': 0.05,
                'iou_vs_random': 0.05
            }
        
        method_scores = {}
        
        for method, metrics in evaluation_results.items():
            weighted_score = 0.0
            total_weight = 0.0
            
            for metric, weight in metric_weights.items():
                if metric in metrics:
                    weighted_score += metrics[metric] * weight
                    total_weight += weight
            
            if total_weight > 0:
                method_scores[method] = weighted_score / total_weight
            else:
                method_scores[method] = 0.0
        
        return method_scores
    
    def create_leaderboard(
        self,
        evaluation_results: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None
    ) -> None:
        """Create and display evaluation leaderboard.
        
        Args:
            evaluation_results: Results from method evaluation.
            save_path: Path to save leaderboard visualization.
        """
        self.evaluator.create_leaderboard(evaluation_results)
        
        if save_path:
            self.visualizer.visualize_evaluation_metrics(
                evaluation_results, 
                save_path
            )
    
    def run_ablation_study(
        self,
        dataloader: torch.utils.data.DataLoader,
        explainer,
        method: str,
        ablation_params: Dict[str, List[Any]],
        max_samples: int = 50
    ) -> Dict[str, Dict[str, float]]:
        """Run ablation study for a specific method.
        
        Args:
            dataloader: DataLoader with test data.
            explainer: XAI explainer instance.
            method: Method to ablate.
            ablation_params: Parameters to vary in ablation.
            max_samples: Maximum number of samples to evaluate.
            
        Returns:
            Results from ablation study.
        """
        print(f"Running ablation study for {method}")
        
        ablation_results = {}
        
        # Get baseline results
        baseline_results = self.evaluate_methods(
            dataloader, explainer, [method], max_samples, save_results=False
        )
        ablation_results['baseline'] = baseline_results[method]
        
        # Run ablation for each parameter
        for param_name, param_values in ablation_params.items():
            print(f"Ablating parameter: {param_name}")
            
            param_results = {}
            for value in param_values:
                print(f"  Testing {param_name}={value}")
                
                # Create explainer with modified parameter
                modified_explainer = self._create_modified_explainer(
                    explainer, method, {param_name: value}
                )
                
                # Evaluate with modified parameter
                results = self.evaluate_methods(
                    dataloader, modified_explainer, [method], max_samples, save_results=False
                )
                
                param_results[str(value)] = results[method]
            
            ablation_results[param_name] = param_results
        
        return ablation_results
    
    def _create_modified_explainer(self, explainer, method: str, params: Dict[str, Any]):
        """Create explainer with modified parameters.
        
        Args:
            explainer: Original explainer.
            method: Method to modify.
            params: Parameters to modify.
            
        Returns:
            Modified explainer.
        """
        # This is a simplified implementation
        # In practice, you'd need to properly recreate the explainer with new parameters
        return explainer
    
    def _save_results(self, results: Dict[str, Dict[str, float]]) -> None:
        """Save evaluation results to disk.
        
        Args:
            results: Evaluation results to save.
        """
        # Save as JSON
        results_file = self.results_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save as CSV
        import pandas as pd
        df_data = []
        for method, metrics in results.items():
            row = {'Method': method}
            row.update(metrics)
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        csv_file = self.results_dir / "evaluation_results.csv"
        df.to_csv(csv_file, index=False)
        
        print(f"Results saved to: {self.results_dir}")
    
    def load_results(self, results_file: str) -> Dict[str, Dict[str, float]]:
        """Load evaluation results from disk.
        
        Args:
            results_file: Path to results file.
            
        Returns:
            Loaded evaluation results.
        """
        results_path = Path(results_file)
        
        if results_path.suffix == '.json':
            with open(results_path, 'r') as f:
                results = json.load(f)
        elif results_path.suffix == '.csv':
            import pandas as pd
            df = pd.read_csv(results_path)
            results = df.set_index('Method').to_dict('index')
        else:
            raise ValueError(f"Unsupported file format: {results_path.suffix}")
        
        self.results = results
        return results
    
    def generate_report(
        self,
        evaluation_results: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None
    ) -> str:
        """Generate comprehensive evaluation report.
        
        Args:
            evaluation_results: Results from method evaluation.
            save_path: Path to save report.
            
        Returns:
            Report text.
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("XAI EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Summary statistics
        report_lines.append("SUMMARY STATISTICS")
        report_lines.append("-" * 40)
        
        for method, metrics in evaluation_results.items():
            report_lines.append(f"\n{method.upper()}:")
            for metric, value in metrics.items():
                report_lines.append(f"  {metric}: {value:.4f}")
        
        # Method comparison
        method_scores = self.compare_methods(evaluation_results)
        report_lines.append("\nMETHOD RANKING")
        report_lines.append("-" * 40)
        
        sorted_methods = sorted(method_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (method, score) in enumerate(sorted_methods, 1):
            report_lines.append(f"{i}. {method}: {score:.4f}")
        
        # Recommendations
        report_lines.append("\nRECOMMENDATIONS")
        report_lines.append("-" * 40)
        
        best_method = sorted_methods[0][0]
        report_lines.append(f"• Best overall method: {best_method}")
        
        # Find best method for each metric category
        faithfulness_metrics = ['deletion_auc', 'insertion_auc', 'sufficiency', 'necessity']
        stability_metrics = ['spearman_vs_random', 'kendall_vs_random', 'iou_vs_random']
        
        best_faithfulness = max(
            evaluation_results.items(),
            key=lambda x: np.mean([x[1].get(m, 0) for m in faithfulness_metrics])
        )[0]
        
        best_stability = max(
            evaluation_results.items(),
            key=lambda x: np.mean([x[1].get(m, 0) for m in stability_metrics])
        )[0]
        
        report_lines.append(f"• Best for faithfulness: {best_faithfulness}")
        report_lines.append(f"• Best for stability: {best_stability}")
        
        # Limitations and disclaimers
        report_lines.append("\nLIMITATIONS AND DISCLAIMERS")
        report_lines.append("-" * 40)
        report_lines.append("• These results are for research and educational purposes only")
        report_lines.append("• XAI methods may produce unstable or misleading explanations")
        report_lines.append("• Results may not generalize to different domains or contexts")
        report_lines.append("• Always validate explanations with domain experts")
        report_lines.append("• Do not use for regulated decisions without human review")
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to: {save_path}")
        
        return report_text
