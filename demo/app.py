"""Streamlit demo for interactive XAI visualization."""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from PIL import Image
import io
import base64

from interpretable_cv_xai.models import create_model
from interpretable_cv_xai.data import CIFAR10DataModule
from interpretable_cv_xai.explainers import XAIExplainer
from interpretable_cv_xai.viz import XAIVisualizer
from interpretable_cv_xai.utils import get_device, set_seed


# Page configuration
st.set_page_config(
    page_title="Interpretable Computer Vision XAI Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'explainer' not in st.session_state:
    st.session_state.explainer = None
if 'data_module' not in st.session_state:
    st.session_state.data_module = None


def load_model_and_data():
    """Load model and data module."""
    if st.session_state.model is None:
        with st.spinner("Loading model and data..."):
            # Set seed for reproducibility
            set_seed(42)
            
            # Get device
            device = get_device()
            
            # Create data module
            data_module = CIFAR10DataModule(batch_size=1)
            st.session_state.data_module = data_module
            
            # Create model
            model = create_model("simple_cnn", num_classes=10)
            model = model.to(device)
            
            # Try to load trained model
            try:
                checkpoint = torch.load("checkpoints/xai_evaluation_best.pth", map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                st.session_state.model_accuracy = checkpoint.get('val_accuracy', 'N/A')
            except FileNotFoundError:
                st.session_state.model_accuracy = "Random (not trained)"
            
            st.session_state.model = model
            
            # Create explainer
            explainer = XAIExplainer(
                model=model,
                methods=['gradcam', 'integrated_gradients', 'lime', 'shap'],
                device=device
            )
            st.session_state.explainer = explainer


def display_disclaimer():
    """Display important disclaimer."""
    st.markdown("""
    <div class="warning-box">
        <h4>⚠️ Important Disclaimer</h4>
        <p><strong>This demo is for research and educational purposes only.</strong></p>
        <ul>
            <li>XAI outputs may be unstable or misleading</li>
            <li>Not a substitute for human judgment</li>
            <li>Do not use for regulated decisions without human review</li>
            <li>Results may not generalize to different contexts</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main demo application."""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Interpretable Computer Vision XAI Demo</h1>', unsafe_allow_html=True)
    
    # Display disclaimer
    display_disclaimer()
    
    # Load model and data
    load_model_and_data()
    
    # Sidebar controls
    st.sidebar.header("🎛️ Controls")
    
    # Model info
    st.sidebar.markdown("### Model Information")
    st.sidebar.info(f"**Model:** Simple CNN\n**Accuracy:** {st.session_state.model_accuracy}")
    
    # Explanation methods
    st.sidebar.markdown("### Explanation Methods")
    selected_methods = st.sidebar.multiselect(
        "Select methods to use:",
        options=['gradcam', 'integrated_gradients', 'lime', 'shap'],
        default=['gradcam', 'integrated_gradients']
    )
    
    # Image selection
    st.sidebar.markdown("### Image Selection")
    image_source = st.sidebar.radio(
        "Choose image source:",
        options=["Sample from CIFAR-10", "Upload your own"]
    )
    
    if image_source == "Sample from CIFAR-10":
        # Sample from dataset
        sample_idx = st.sidebar.slider("Sample index:", 0, 999, 0)
        
        if st.sidebar.button("Load Sample"):
            sample_batch = st.session_state.data_module.get_sample_batch("test")
            sample_images, sample_targets = sample_batch
            st.session_state.current_image = sample_images[sample_idx:sample_idx+1]
            st.session_state.current_target = sample_targets[sample_idx:sample_idx+1]
    
    else:
        # Upload image
        uploaded_file = st.sidebar.file_uploader(
            "Upload an image:",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a 32x32 RGB image for best results"
        )
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                image = image.resize((32, 32))
                image_array = np.array(image)
                
                if len(image_array.shape) == 3:
                    # Convert to tensor format
                    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
                    image_tensor = image_tensor.unsqueeze(0)
                    
                    st.session_state.current_image = image_tensor
                    st.session_state.current_target = None  # Will be predicted
                    
                    st.sidebar.success("Image loaded successfully!")
                else:
                    st.sidebar.error("Please upload a color image (RGB)")
            except Exception as e:
                st.sidebar.error(f"Error loading image: {str(e)}")
    
    # Main content area
    if 'current_image' in st.session_state:
        
        # Display original image and prediction
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Original Image")
            
            # Convert tensor to displayable format
            img_display = st.session_state.current_image[0].permute(1, 2, 0).numpy()
            img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
            
            st.image(img_display, caption="Input Image", use_column_width=True)
            
            # Make prediction
            with torch.no_grad():
                device = get_device()
                image_tensor = st.session_state.current_image.to(device)
                outputs = st.session_state.model(image_tensor)
                predictions = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(predictions, dim=1).item()
                confidence = predictions[0, predicted_class].item()
            
            # Display prediction
            class_names = [
                'airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck'
            ]
            
            st.markdown("### Prediction")
            st.markdown(f"""
            <div class="metric-card">
                <strong>Class:</strong> {class_names[predicted_class]}<br>
                <strong>Confidence:</strong> {confidence:.3f}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Class Probabilities")
            
            # Create probability bar chart
            prob_data = predictions[0].cpu().numpy()
            fig = px.bar(
                x=class_names,
                y=prob_data,
                title="Prediction Probabilities",
                labels={'x': 'Class', 'y': 'Probability'},
                color=prob_data,
                color_continuous_scale='Blues'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Generate explanations
        if selected_methods and st.button("Generate Explanations"):
            
            with st.spinner("Generating explanations..."):
                try:
                    # Generate explanations
                    explanations = st.session_state.explainer.explain(
                        st.session_state.current_image.to(get_device()),
                        st.session_state.current_target,
                        method=None  # Use all selected methods
                    )
                    
                    # Filter explanations for selected methods
                    filtered_explanations = {
                        method: explanations[method] 
                        for method in selected_methods 
                        if method in explanations
                    }
                    
                    st.session_state.explanations = filtered_explanations
                    
                except Exception as e:
                    st.error(f"Error generating explanations: {str(e)}")
                    st.session_state.explanations = None
        
        # Display explanations
        if 'explanations' in st.session_state and st.session_state.explanations:
            
            st.markdown("### Explanations")
            
            # Create tabs for different methods
            method_tabs = st.tabs(selected_methods)
            
            for i, method in enumerate(selected_methods):
                if method in st.session_state.explanations:
                    with method_tabs[i]:
                        st.markdown(f"#### {method.upper()} Explanation")
                        
                        # Get explanation
                        explanation = st.session_state.explanations[method]
                        
                        # Create visualization
                        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                        
                        # Original image
                        img_display = st.session_state.current_image[0].permute(1, 2, 0).numpy()
                        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
                        
                        ax[0].imshow(img_display)
                        ax[0].set_title("Original Image")
                        ax[0].axis('off')
                        
                        # Attribution map
                        attr = explanation[0].cpu().detach().numpy()
                        if attr.ndim == 3:
                            attr = np.mean(attr, axis=0)
                        
                        im = ax[1].imshow(attr, cmap='RdBu_r')
                        ax[1].set_title(f"{method.upper()} Attribution")
                        ax[1].axis('off')
                        plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
                        
                        st.pyplot(fig)
                        
                        # Method-specific information
                        if method == 'gradcam':
                            st.info("Grad-CAM highlights regions that contribute most to the prediction using gradients from the last convolutional layer.")
                        elif method == 'integrated_gradients':
                            st.info("Integrated Gradients computes attributions by integrating gradients along the path from a baseline to the input.")
                        elif method == 'lime':
                            st.info("LIME approximates the model locally with an interpretable model to explain individual predictions.")
                        elif method == 'shap':
                            st.info("SHAP uses game-theoretic Shapley values to provide unified explanations.")
            
            # Comparison view
            if len(selected_methods) > 1:
                st.markdown("### Method Comparison")
                
                # Create comparison visualization
                num_methods = len(selected_methods)
                fig, axes = plt.subplots(1, num_methods + 1, figsize=(4 * (num_methods + 1), 4))
                
                if num_methods == 1:
                    axes = axes.reshape(1, -1)
                
                # Original image
                img_display = st.session_state.current_image[0].permute(1, 2, 0).numpy()
                img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
                
                axes[0].imshow(img_display)
                axes[0].set_title("Original")
                axes[0].axis('off')
                
                # Attribution maps
                for i, method in enumerate(selected_methods):
                    if method in st.session_state.explanations:
                        attr = st.session_state.explanations[method][0].cpu().detach().numpy()
                        if attr.ndim == 3:
                            attr = np.mean(attr, axis=0)
                        
                        im = axes[i + 1].imshow(attr, cmap='RdBu_r')
                        axes[i + 1].set_title(method.upper())
                        axes[i + 1].axis('off')
                        plt.colorbar(im, ax=axes[i + 1], fraction=0.046, pad=0.04)
                
                st.pyplot(fig)
        
        # Evaluation metrics (if available)
        if 'explanations' in st.session_state and st.session_state.explanations:
            st.markdown("### Evaluation Metrics")
            
            # Create a simple evaluation
            try:
                from interpretable_cv_xai.metrics import ExplanationEvaluator
                
                evaluator = ExplanationEvaluator(st.session_state.model, get_device())
                
                # Evaluate explanations
                eval_results = evaluator.evaluate_comprehensive(
                    st.session_state.current_image.to(get_device()),
                    st.session_state.explanations,
                    st.session_state.current_target
                )
                
                # Display metrics
                for method, metrics in eval_results.items():
                    st.markdown(f"#### {method.upper()}")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Deletion AUC", f"{metrics.get('deletion_auc', 0):.3f}")
                        st.metric("Insertion AUC", f"{metrics.get('insertion_auc', 0):.3f}")
                    
                    with col2:
                        st.metric("Sufficiency", f"{metrics.get('sufficiency', 0):.3f}")
                        st.metric("Necessity", f"{metrics.get('necessity', 0):.3f}")
                    
                    with col3:
                        st.metric("Stability", f"{metrics.get('spearman_vs_random', 0):.3f}")
                        st.metric("Completeness", f"{metrics.get('completeness', 0):.3f}")
                
            except Exception as e:
                st.warning(f"Could not compute evaluation metrics: {str(e)}")
    
    else:
        # Welcome message
        st.markdown("""
        ## Welcome to the Interpretable Computer Vision XAI Demo! 🎉
        
        This interactive demo allows you to:
        
        - **Upload your own images** or select samples from CIFAR-10
        - **Generate explanations** using multiple XAI methods
        - **Compare different methods** side by side
        - **View evaluation metrics** for explanation quality
        
        ### Getting Started
        
        1. **Select explanation methods** from the sidebar
        2. **Choose an image** (upload or sample from dataset)
        3. **Click "Generate Explanations"** to see results
        4. **Explore different methods** and compare their outputs
        
        ### Available Methods
        
        - **Grad-CAM**: Gradient-weighted Class Activation Mapping
        - **Integrated Gradients**: Axiomatic attribution method
        - **LIME**: Local Interpretable Model-agnostic Explanations
        - **SHAP**: SHapley Additive exPlanations
        
        ### Important Notes
        
        - This demo uses a simple CNN trained on CIFAR-10
        - Results are for research and educational purposes only
        - XAI methods may produce unstable or misleading explanations
        - Always validate explanations with domain experts
        """)


if __name__ == "__main__":
    main()
