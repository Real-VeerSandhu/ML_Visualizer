import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as animation

import time

# Set page configuration
st.set_page_config(
    page_title="ML Visualizer - Interactive Machine Learning",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hero Section
st.title("Machine Learning Visualizer")
st.subheader("Watch Machine Learning Algorithms Train in Real-Time")


st.markdown("---")

# Features Section
st.markdown("## Highlights")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### Live Training
    Watch algorithms learn step-by-step with real-time parameter updates and convergence visualization.
    """)

with col2:
    st.markdown("""
    ### Interactive Controls
    Adjust learning rates, sample sizes, and algorithm parameters to see immediate effects on training.
    """)

with col3:
    st.markdown("""
    ### Educational
    Complete mathematical explanations, equations, and detailed progress tracking for fundamental understanding.
    """)

st.markdown("---")


# Available Algorithms Section
st.markdown("## Available Algorithms")

# Supervised Learning
st.markdown("### Supervised Learning")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **Linear Regression**
    - *Regression Algorithm*
    - Watch gradient descent find the line of best fit
    - Visualize loss function optimization
    """)

with col2:
    st.markdown("""
    **Support Vector Machine**
    - *Classification Algorithm*
    - Visualize optimal decision boundary formation
    - See margin maximization in action
    """)

with col3:
    st.markdown("""
    **Logistic Regression**
    - *Classification Algorithm*
    - Observe probability-based classification learning
    - Watch sigmoid function optimization
    """)

# Unsupervised Learning
st.markdown("### Unsupervised Learning")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **K-Means Clustering**
    - *Clustering Algorithm*
    - See centroids move to find natural data clusters
    - Watch WCSS minimization process
    """)

with col2:
    st.markdown("""
    **Decision Trees**
    - *Tree-based Algorithm*
    - See how trees split data for classification
    - Visualize recursive partitioning
    """)


st.markdown("---")

# How it Works Section
st.markdown("## How It Works")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Getting Started
    
    **1. Choose Your Algorithm**
    Select from various machine learning algorithms including regression, classification, and clustering methods.
    
    **2. Adjust Parameters**
    Customize learning rates, sample sizes, noise levels, and algorithm-specific parameters using interactive controls.
    
    **3. Watch & Learn**
    Hit play and watch the algorithm train in real-time with live visualizations and progress tracking.
    """)

with col2:
    st.markdown("""
    ### Deep Understanding
    
    **4. Analyze Results**
    View detailed training progress, convergence plots, and final model performance metrics.
    
    **5. Understand Math**
    Follow along with complete mathematical explanations, equations, and gradient computations.
    
    **6. Experiment**
    Try different parameters and see how they affect training speed, accuracy, and convergence.
    """)

st.markdown("---")

# # Educational Value Section
# st.markdown("## 🎓 Perfect for Learning")

# col1, col2, col3 = st.columns(3)

# with col1:
#     st.markdown("""
#     ### 👨‍🎓 Students
#     - Visualize complex ML concepts
#     - See theory put into practice
#     - Interactive homework assistance
#     - Understand algorithm behavior
#     """)

# with col2:
#     st.markdown("""
#     ### 👩‍🏫 Educators
#     - Engaging teaching tool
#     - Live classroom demonstrations
#     - Clear step-by-step explanations
#     - Interactive learning materials
#     """)

# with col3:
#     st.markdown("""
#     ### 👨‍💻 Practitioners
#     - Algorithm comparison tools
#     - Parameter tuning insights
#     - Quick prototyping platform
#     - Visual debugging aid
#     """)

# st.markdown("---")

# Technical Details
st.markdown("## Technical Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **Built With**
    - Streamlit
    - NumPy & Pandas
    - Matplotlib
    - Scikit-learn
    """)

with col2:
    st.markdown("""
    **Key Features**
    - Real-time visualization
    - Interactive parameters
    - Mathematical explanations
    - Progress tracking
    """)

with col3:
    st.markdown("""
    **Learning Goals**
    - Understand ML algorithms
    - See convergence behavior
    - Experiment with parameters
    - Visualize complex concepts
    """)

st.markdown("---")

# Call to Action
st.markdown("## Quick Start")
st.markdown("Choose an algorithm from the sidebar and start exploring")

# Quick Start Guide
with st.expander("Quick Start Guide"):
    st.markdown("""
    ### Getting Started in 6 Easy Steps:
    
    1. **Select an Algorithm**: Use the sidebar to navigate to different ML algorithms
    2. **Adjust Parameters**: Use the sliders to customize algorithm settings
    3. **Generate Data**: Most algorithms include synthetic data generation
    4. **Start Training**: Click the "Start Training" button to begin visualization
    5. **Watch & Learn**: Observe real-time training with live plots and progress tables
    6. **Experiment**: Try different parameters to see how they affect learning
    
    ### Tips:
    - Start with default parameters for your first run
    - Try extreme values to see algorithm behavior
    - Pay attention to convergence patterns
    - Read the mathematical explanations for deeper understanding
    - Use different random seeds to see varied behavior
    - Compare results across different parameter settings
    """)

# Footer
st.markdown("---")
st.markdown("*Built using Python, NumPy, Matplotlib*")
st.markdown("Veer Sandhu - 2025")
st.caption("[Github](https://github.com/Real-VeerSandhu/ML_Visualizer)")