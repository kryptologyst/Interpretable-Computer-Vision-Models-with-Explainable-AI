# Interpretable Computer Vision Models with Explainable AI

## WARNING: Research and Educational Use Only

**This project is designed for research and educational purposes only. XAI outputs may be unstable or misleading and should not be used as a substitute for human judgment. Do not use for regulated decisions without human review.**

## Overview

This project implements state-of-the-art explainable AI (XAI) methods for computer vision tasks, focusing on post-hoc local interpretability techniques. We provide a comprehensive framework for understanding and visualizing how deep learning models make decisions on image data.

## Features

- **Multiple XAI Methods**: Grad-CAM, Integrated Gradients, LIME, SHAP, SmoothGrad
- **Comprehensive Evaluation**: Faithfulness, stability, and fidelity metrics
- **Interactive Demo**: Streamlit-based visualization interface
- **Modern Stack**: PyTorch 2.x, Captum, scikit-learn compatibility
- **Reproducible Research**: Deterministic seeding, proper configuration management

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Interpretable-Computer-Vision-Models-with-Explainable-AI.git
cd Interpretable-Computer-Vision-Models-with-Explainable-AI

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"
```

### Basic Usage

```python
from interpretable_cv_xai import XAIExplainer
from interpretable_cv_xai.models import SimpleCNN
from interpretable_cv_xai.data import CIFAR10DataModule

# Load data and model
data_module = CIFAR10DataModule()
model = SimpleCNN()

# Create explainer
explainer = XAIExplainer(model, methods=['gradcam', 'integrated_gradients'])

# Generate explanations
explanations = explainer.explain(image, target_class=5)
explainer.visualize(explanations)
```

### Interactive Demo

```bash
streamlit run demo/app.py
```

## Project Structure

```
src/interpretable_cv_xai/
├── methods/          # XAI method implementations
├── explainers/       # High-level explanation interfaces
├── metrics/          # Evaluation metrics
├── viz/             # Visualization utilities
├── data/            # Data loading and preprocessing
├── models/          # Model architectures
├── eval/            # Evaluation framework
└── utils/           # Utility functions

configs/             # Configuration files
scripts/             # Training and evaluation scripts
notebooks/           # Jupyter notebooks
tests/               # Test suite
assets/              # Generated visualizations
demo/                # Streamlit demo
```

## Supported Methods

### Post-hoc Local Explanations
- **Grad-CAM**: Gradient-weighted Class Activation Mapping
- **Integrated Gradients**: Axiomatic attribution method
- **LIME**: Local Interpretable Model-agnostic Explanations
- **SHAP**: SHapley Additive exPlanations
- **SmoothGrad**: Smoothed gradient-based explanations

### Evaluation Metrics
- **Faithfulness**: Deletion/Insertion AUC, Sufficiency/Necessity
- **Stability**: Explanation similarity across runs (Kendall τ, Spearman ρ)
- **Fidelity**: Surrogate model accuracy

## Configuration

The project uses Hydra for configuration management. See `configs/` for available configurations.

```bash
# Train with custom config
python scripts/train.py model=simple_cnn data=cifar10

# Evaluate explanations
python scripts/evaluate.py explainer=gradcam model=pretrained
```

## Limitations and Disclaimers

### Important Limitations
- **Instability**: XAI methods can produce inconsistent results
- **Approximation**: Explanations are approximations, not ground truth
- **Context-dependent**: Results may not generalize across domains
- **Computational cost**: Some methods are computationally expensive

### Ethical Considerations
- Be aware of potential biases in models and explanations
- Consider impact on different user groups
- Ensure transparency about method limitations
- Respect privacy and data protection requirements

### Usage Restrictions
- **Research/Education Only**: Not for production use without validation
- **No Regulated Decisions**: Do not use for medical, legal, or financial decisions
- **Human Oversight Required**: Always validate with domain experts
- **No Guarantee of Correctness**: Explanations may be misleading

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{interpretable_cv_xai,
  title={Interpretable Computer Vision Models with Explainable AI},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Interpretable-Computer-Vision-Models-with-Explainable-AI}
}
```

## Acknowledgments

- PyTorch and Captum teams for the excellent frameworks
- The XAI research community for foundational work
- Contributors and users for feedback and improvements
# Interpretable-Computer-Vision-Models-with-Explainable-AI
