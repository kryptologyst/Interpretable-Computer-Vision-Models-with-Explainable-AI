#!/usr/bin/env python3
"""Evaluation script for XAI methods."""

import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
from pathlib import Path
import json
from tqdm import tqdm

from interpretable_cv_xai.models import create_model
from interpretable_cv_xai.data import CIFAR10DataModule
from interpretable_cv_xai.explainers import XAIExplainer
from interpretable_cv_xai.eval import XAIEvaluationFramework
from interpretable_cv_xai.utils import set_seed, get_device, load_checkpoint


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main evaluation function."""
    
    # Set random seed for reproducibility
    set_seed(cfg.seed)
    
    # Get device
    if cfg.device == "auto":
        device = get_device()
    else:
        device = torch.device(cfg.device)
    
    print(f"Using device: {device}")
    
    # Create data module
    data_module = hydra.utils.instantiate(cfg.data)
    
    # Create model
    model = hydra.utils.instantiate(cfg.model)
    model = model.to(device)
    
    # Load trained model if available
    checkpoint_path = Path("checkpoints") / f"{cfg.experiment_name}_best.pth"
    if checkpoint_path.exists():
        print(f"Loading model from {checkpoint_path}")
        checkpoint = load_checkpoint(model, None, str(checkpoint_path), device)
        print(f"Model loaded from epoch {checkpoint['epoch']}")
        print(f"Validation accuracy: {checkpoint.get('val_accuracy', 'N/A'):.2f}%")
    else:
        print("No trained model found. Using randomly initialized model.")
    
    # Create XAI explainer
    explainer = XAIExplainer(
        model=model,
        methods=cfg.explanation.methods,
        device=device
    )
    
    print(f"Initialized explainer with methods: {cfg.explanation.methods}")
    
    # Create evaluation framework
    eval_framework = XAIEvaluationFramework(
        model=model,
        device=device,
        results_dir=f"results/{cfg.experiment_name}"
    )
    
    # Run evaluation
    print(f"\nStarting evaluation with max {cfg.evaluation.max_samples} samples...")
    
    evaluation_results = eval_framework.evaluate_methods(
        dataloader=data_module.test_dataloader(),
        explainer=explainer,
        methods=cfg.explanation.methods,
        max_samples=cfg.evaluation.max_samples,
        save_results=cfg.evaluation.save_results
    )
    
    # Create leaderboard
    if cfg.evaluation.create_visualizations:
        print("\nCreating evaluation leaderboard...")
        eval_framework.create_leaderboard(
            evaluation_results,
            save_path=f"assets/{cfg.experiment_name}_leaderboard.png"
        )
    
    # Generate comprehensive report
    if cfg.evaluation.generate_report:
        print("\nGenerating evaluation report...")
        report = eval_framework.generate_report(
            evaluation_results,
            save_path=f"results/{cfg.experiment_name}/evaluation_report.txt"
        )
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        # Print summary
        method_scores = eval_framework.compare_methods(evaluation_results)
        sorted_methods = sorted(method_scores.items(), key=lambda x: x[1], reverse=True)
        
        print("\nMethod Rankings:")
        for i, (method, score) in enumerate(sorted_methods, 1):
            print(f"{i}. {method}: {score:.4f}")
        
        print(f"\nBest method: {sorted_methods[0][0]}")
        print(f"Best score: {sorted_methods[0][1]:.4f}")
        
        # Print key metrics for best method
        best_method = sorted_methods[0][0]
        best_metrics = evaluation_results[best_method]
        
        print(f"\nKey metrics for {best_method}:")
        print(f"  Deletion AUC: {best_metrics.get('deletion_auc', 'N/A'):.4f}")
        print(f"  Insertion AUC: {best_metrics.get('insertion_auc', 'N/A'):.4f}")
        print(f"  Sufficiency: {best_metrics.get('sufficiency', 'N/A'):.4f}")
        print(f"  Necessity: {best_metrics.get('necessity', 'N/A'):.4f}")
        print(f"  Stability (Spearman): {best_metrics.get('spearman_vs_random', 'N/A'):.4f}")
        print(f"  Completeness: {best_metrics.get('completeness', 'N/A'):.4f}")
    
    # Demo explanations on sample images
    print("\nGenerating sample explanations...")
    
    # Get a few sample images
    sample_batch = data_module.get_sample_batch("test")
    sample_images, sample_targets = sample_batch
    sample_images = sample_images[:4]  # Take first 4 images
    sample_targets = sample_targets[:4]
    
    # Generate explanations
    explanations = explainer.explain(sample_images.to(device), sample_targets.to(device))
    
    # Visualize explanations
    if cfg.evaluation.create_visualizations:
        explainer.visualize(
            sample_images,
            explanations,
            save_path=f"assets/{cfg.experiment_name}_sample_explanations.png"
        )
    
    print(f"\nEvaluation completed!")
    print(f"Results saved to: results/{cfg.experiment_name}/")
    print(f"Visualizations saved to: assets/")
    
    # Print disclaimer
    print("\n" + "="*80)
    print("IMPORTANT DISCLAIMER")
    print("="*80)
    print("This evaluation is for research and educational purposes only.")
    print("XAI outputs may be unstable or misleading.")
    print("Do not use for regulated decisions without human review.")
    print("="*80)


if __name__ == "__main__":
    main()
