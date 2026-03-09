#!/usr/bin/env python3
"""Complete XAI pipeline demonstration script."""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from interpretable_cv_xai.models import SimpleCNN
from interpretable_cv_xai.data import CIFAR10DataModule, create_synthetic_dataset
from interpretable_cv_xai.explainers import XAIExplainer
from interpretable_cv_xai.eval import XAIEvaluationFramework
from interpretable_cv_xai.utils import set_seed, get_device


def main():
    """Run complete XAI pipeline demonstration."""
    
    print("=" * 80)
    print("INTERPRETABLE COMPUTER VISION XAI DEMONSTRATION")
    print("=" * 80)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create directories
    Path("assets").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    
    # 1. Create synthetic data for quick demonstration
    print("\n1. Creating synthetic dataset...")
    images, labels = create_synthetic_dataset(num_samples=50, num_classes=10)
    print(f"Created {len(images)} synthetic images")
    
    # 2. Create model
    print("\n2. Creating model...")
    model = SimpleCNN(num_classes=10)
    model = model.to(device)
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")
    
    # 3. Create explainer
    print("\n3. Creating XAI explainer...")
    explainer = XAIExplainer(
        model=model,
        methods=['gradcam', 'integrated_gradients'],
        device=device
    )
    print(f"Initialized explainer with methods: {explainer.methods}")
    
    # 4. Generate explanations
    print("\n4. Generating explanations...")
    sample_images = images[:4].to(device)
    sample_targets = labels[:4].to(device)
    
    explanations = explainer.explain(sample_images, sample_targets)
    print(f"Generated explanations for {len(explanations)} methods")
    
    # 5. Visualize explanations
    print("\n5. Visualizing explanations...")
    explainer.visualize(
        sample_images.cpu(),
        explanations,
        save_path="assets/demo_explanations.png"
    )
    print("Explanations saved to assets/demo_explanations.png")
    
    # 6. Evaluate explanations
    print("\n6. Evaluating explanation quality...")
    eval_framework = XAIEvaluationFramework(
        model=model,
        device=device,
        results_dir="results/demo_evaluation"
    )
    
    evaluation_results = eval_framework.evaluate_comprehensive(
        sample_images,
        explanations,
        sample_targets
    )
    
    # 7. Display results
    print("\n7. Evaluation Results:")
    print("-" * 50)
    for method, metrics in evaluation_results.items():
        print(f"\n{method.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # 8. Create leaderboard
    print("\n8. Creating leaderboard...")
    eval_framework.create_leaderboard(
        evaluation_results,
        save_path="assets/demo_leaderboard.png"
    )
    print("Leaderboard saved to assets/demo_leaderboard.png")
    
    # 9. Generate report
    print("\n9. Generating report...")
    report = eval_framework.generate_report(
        evaluation_results,
        save_path="results/demo_evaluation/report.txt"
    )
    print("Report saved to results/demo_evaluation/report.txt")
    
    # 10. Method comparison
    print("\n10. Method Comparison:")
    print("-" * 30)
    method_scores = eval_framework.compare_methods(evaluation_results)
    sorted_methods = sorted(method_scores.items(), key=lambda x: x[1], reverse=True)
    
    for i, (method, score) in enumerate(sorted_methods, 1):
        print(f"{i}. {method}: {score:.4f}")
    
    print(f"\nBest method: {sorted_methods[0][0]}")
    print(f"Best score: {sorted_methods[0][1]:.4f}")
    
    # 11. Final summary
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - assets/demo_explanations.png")
    print("  - assets/demo_leaderboard.png")
    print("  - results/demo_evaluation/report.txt")
    print("  - results/demo_evaluation/evaluation_results.json")
    print("  - results/demo_evaluation/evaluation_results.csv")
    
    print("\nTo run the interactive demo:")
    print("  streamlit run demo/app.py")
    
    print("\n" + "=" * 80)
    print("IMPORTANT DISCLAIMER")
    print("=" * 80)
    print("This demonstration is for research and educational purposes only.")
    print("XAI outputs may be unstable or misleading.")
    print("Do not use for regulated decisions without human review.")
    print("=" * 80)


if __name__ == "__main__":
    main()
