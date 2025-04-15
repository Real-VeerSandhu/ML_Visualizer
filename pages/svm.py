import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

# Set page configuration
st.set_page_config(page_title="SVM Classification Visualizer", layout="wide")

st.title("Support Vector Machine (SVM) Classification Visualizer")
st.write("""
Watch as the SVM model learns the optimal decision boundary to separate classes, and visualize how
the margin and support vectors evolve during training.
""")

# Sidebar for parameters
st.sidebar.header("Model Parameters")
C = st.sidebar.slider("Regularization Parameter (C)", min_value=0.1, max_value=10.0, value=1.0, step=0.1,
                     help="Controls the trade-off between smooth decision boundary and classifying training points correctly")
kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly"], 
                             help="Determines the type of decision boundary")
gamma = st.sidebar.slider("Gamma (for RBF and Poly kernels)", min_value=0.01, max_value=2.0, value=0.1, step=0.01,
                         help="Defines how much influence a single training point has")
degree = st.sidebar.slider("Degree (for Poly kernel)", min_value=2, max_value=5, value=3, step=1,
                          help="Degree of the polynomial kernel function")
iterations = st.sidebar.slider("Training Steps", min_value=5, max_value=50, value=20, step=5,
                              help="Number of incremental steps to visualize")
noise_level = st.sidebar.slider("Noise Level", min_value=0.0, max_value=1.0, value=0.2, step=0.05,
                              help="Amount of noise in the generated data")
sample_size = st.sidebar.slider("Sample Size", min_value=20, max_value=200, value=100, step=10,
                              help="Number of data points to generate")

# Sidebar for mathematical explanations
st.sidebar.header("Mathematical Foundations")
st.sidebar.markdown("""
### SVM Objective Function
The SVM solves the following optimization problem:

$$\\min_{w, b} \\frac{1}{2} ||w||^2 + C \\sum_{i=1}^{n} \\xi_i$$

subject to:
$y_i(w^T x_i + b) \\geq 1 - \\xi_i$ and $\\xi_i \\geq 0$ for all $i$

### Kernel Functions
**Linear**: $K(x_i, x_j) = x_i^T x_j$

**RBF**: $K(x_i, x_j) = \\exp(-\\gamma ||x_i - x_j||^2)$

**Polynomial**: $K(x_i, x_j) = (\\gamma x_i^T x_j + r)^d$

### Decision Function
$f(x) = \\text{sign}\\left(\\sum_{i=1}^{n} \\alpha_i y_i K(x_i, x) + b\\right)$

where:
- $w$ is the weight vector
- $b$ is the bias term
- $C$ is the regularization parameter
- $\\xi_i$ are slack variables
- $\\alpha_i$ are Lagrange multipliers
- $K$ is the kernel function
""")

# Function to generate binary classification data
def generate_data(sample_size, noise):
    # Generate features for two classes
    n_samples = sample_size // 2
    
    # Class 1: points in a circular pattern
    radius = 3
    theta = np.linspace(0, 2*np.pi, n_samples)
    x1 = radius * np.cos(theta) + np.random.randn(n_samples) * noise
    y1 = radius * np.sin(theta) + np.random.randn(n_samples) * noise
    
    # Class 2: points in the center
    x2 = np.random.randn(n_samples) * noise
    y2 = np.random.randn(n_samples) * noise
    
    # Create class labels (1 and -1)
    X1 = np.column_stack((x1, y1))
    X2 = np.column_stack((x2, y2))
    y1_labels = np.ones(n_samples)
    y2_labels = -np.ones(n_samples)
    
    # Combine data and shuffle consistently
    X = np.vstack([X1, X2])
    y = np.hstack([y1_labels, y2_labels])
    
    # Create a permutation for shuffling
    perm = np.random.permutation(len(X))
    X = X[perm]
    y = y[perm]
    
    return X, y, X1, y1_labels, X2, y2_labels

# Function to plot decision boundary
def plot_decision_boundary(model, X, y, ax):
    # Create meshgrid
    h = 0.02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Get predictions on the meshgrid
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and margins
    ax.contour(xx, yy, Z, colors=['k'], levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    
    # Create a colormap for the filled contour
    cmap = ListedColormap(['#FFAAAA', '#AAAAFF'])
    ax.contourf(xx, yy, Z > 0, alpha=0.3, cmap=cmap)
    
    # Plot support vectors
    if hasattr(model, 'support_vectors_'):
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
                  linewidth=1, facecolors='none', edgecolors='k')
    
    return ax

# Create column layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("SVM Classification Visualization")
    # Create placeholder for the model visualization
    model_plot_placeholder = st.empty()

with col2:
    st.subheader("Training Metrics")
    # Create placeholder for metrics
    metrics_placeholder = st.empty()

# Button to start training
if st.button("Start Training"):
    # Generate data - now we get additional information about each class separately
    X, y, X1, y1, X2, y2 = generate_data(sample_size, noise_level)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X1_scaled = scaler.transform(X1)
    X2_scaled = scaler.transform(X2)
    
    # Initialize training metrics dataframe
    metrics_df = pd.DataFrame(columns=["Step", "Support Vectors", "Margin Size", "Training Accuracy"])
    
    # Create final SVM model
    if kernel == "linear":
        final_model = svm.SVC(kernel=kernel, C=C)
    elif kernel == "rbf":
        final_model = svm.SVC(kernel=kernel, C=C, gamma=gamma)
    else:  # poly
        final_model = svm.SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
    
    # Show incremental learning steps
    for i in range(1, iterations + 1):
        # Calculate how many samples to include from each class to ensure both classes are present
        samples_per_class = max(2, int((len(X) / 2) * (i / iterations)))  # Ensure at least 2 samples per class
        
        # Take equal numbers from each class
        indices_class1 = np.where(y == 1)[0][:samples_per_class]
        indices_class2 = np.where(y == -1)[0][:samples_per_class]
        
        # Combine indices and get the subset
        subset_indices = np.concatenate([indices_class1, indices_class2])
        X_subset = X_scaled[subset_indices]
        y_subset = y[subset_indices]
        
        # Create and train a new model with the current subset
        if kernel == "linear":
            model = svm.SVC(kernel=kernel, C=C)
        elif kernel == "rbf":
            model = svm.SVC(kernel=kernel, C=C, gamma=gamma)
        else:  # poly
            model = svm.SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
            
        # Fit the model
        model.fit(X_subset, y_subset)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot all data points (both included and not yet included in training)
        ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=plt.cm.Paired, alpha=0.3)
        
        # Highlight points used for training
        ax.scatter(X_subset[:, 0], X_subset[:, 1], c=y_subset, cmap=plt.cm.Paired, s=50, edgecolors='k')
        
        # Plot decision boundary
        try:
            plot_decision_boundary(model, X_subset, y_subset, ax)
            
            # Calculate margin size (valid only for linear kernel)
            margin = None
            if kernel == "linear" and hasattr(model, 'coef_'):
                w_norm = np.linalg.norm(model.coef_[0])
                margin = 2 / w_norm if w_norm > 0 else "N/A"
            else:
                margin = "N/A (non-linear kernel)"
            
            # Calculate training accuracy
            accuracy = model.score(X_subset, y_subset)
            
            # Add to metrics dataframe
            new_row = {
                "Step": i,
                "Support Vectors": len(model.support_vectors_),
                "Margin Size": margin if isinstance(margin, str) else f"{margin:.4f}",
                "Training Accuracy": f"{accuracy:.4f}"
            }
            metrics_df = pd.concat([metrics_df, pd.DataFrame([new_row])], ignore_index=True)
        except Exception as e:
            st.warning(f"Error during visualization at step {i}: {e}")
        
        ax.set_title(f"Training Step {i}/{iterations} - {len(X_subset)} points")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        
        # Set consistent axis limits
        ax.set_xlim([X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1])
        ax.set_ylim([X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1])
        
        # Update plot and metrics
        model_plot_placeholder.pyplot(fig)
        metrics_placeholder.dataframe(metrics_df, height=300)
        
        # Slow down to make the animation visible
        time.sleep(0.5)
    
    # Train final model on all data
    final_model.fit(X_scaled, y)
    
    # Display final results
    st.subheader("Final Model Details")
    st.write(f"**Number of Support Vectors:** {len(final_model.support_vectors_)}")
    st.write(f"**Training Accuracy:** {final_model.score(X_scaled, y):.4f}")
    
    # Create final visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=plt.cm.Paired)
    plot_decision_boundary(final_model, X_scaled, y, ax)
    ax.set_title("Final Model")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    st.pyplot(fig)
    
    # Display kernel parameters
    st.write("**Kernel Parameters:**")
    if kernel == "linear":
        st.write(f"- Regularization Parameter C: {C}")
    elif kernel == "rbf":
        st.write(f"- Regularization Parameter C: {C}")
        st.write(f"- Gamma: {gamma}")
    else:  # poly
        st.write(f"- Regularization Parameter C: {C}")
        st.write(f"- Gamma: {gamma}")
        st.write(f"- Degree: {degree}")
    
    # Additional explanation
    st.markdown("""
    ### What to Observe:
    
    1. **Support Vectors**: Points that lie on or near the decision boundary (marked with black circles).
    2. **Decision Boundary**: The line/curve that separates the two classes.
    3. **Margins**: The region between the dashed lines on either side of the decision boundary.
    4. **Effect of C**: Lower C values create wider margins but may allow more misclassifications.
    5. **Effect of Kernel**: Different kernels create different shapes of decision boundaries:
       - Linear: Straight line boundary
       - RBF: Can create circular/curved boundaries
       - Polynomial: Can create complex curved boundaries
    """)