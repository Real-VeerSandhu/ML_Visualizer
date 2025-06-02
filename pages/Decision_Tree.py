import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import make_classification, make_blobs
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="Decision Tree Visualizer", layout="wide", page_icon="📊")

st.title("Decision Tree Visualizer")
st.write("""
Watch as the decision tree learns to classify data by creating decision boundaries, 
and see how the tree structure evolves with different parameters.
""")

# Sidebar for parameters
st.sidebar.header("Model Parameters")
max_depth = st.sidebar.slider("Maximum Depth", min_value=1, max_value=10, value=3, step=1)
min_samples_split = st.sidebar.slider("Min Samples Split", min_value=2, max_value=20, value=2, step=1)
min_samples_leaf = st.sidebar.slider("Min Samples Leaf", min_value=1, max_value=10, value=1, step=1)
criterion = st.sidebar.selectbox("Split Criterion", ["gini", "entropy"])

# Data generation parameters
st.sidebar.header("Data Parameters")
n_samples = st.sidebar.slider("Number of Samples", min_value=50, max_value=500, value=200, step=50)
n_features = 2
n_classes = st.sidebar.slider("Number of Classes", min_value=2, max_value=4, value=3, step=1)
cluster_std = st.sidebar.slider("Cluster Standard Deviation", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
dataset_type = st.sidebar.selectbox("Dataset Type", ["Blobs", "Classification"])

# Sidebar for mathematical explanations
st.sidebar.header("Equations")
st.sidebar.markdown("""
### Gini Impurity
$Gini = 1 - \\sum_{i=1}^{c} p_i^2$

### Entropy
$Entropy = -\\sum_{i=1}^{c} p_i \\log_2(p_i)$

### Information Gain
$IG = H(parent) - \\sum \\frac{N_j}{N} H(child_j)$

where:
- $c$ is number of classes
- $p_i$ is proportion of class $i$
- $H$ is the impurity measure
- $N_j$ is samples in child $j$
""")

# Function to generate data
def generate_data(n_samples, n_classes, cluster_std, dataset_type):
    if dataset_type == "Blobs":
        X, y = make_blobs(n_samples=n_samples, centers=n_classes, 
                         cluster_std=cluster_std, random_state=42)
    else:
        X, y = make_classification(n_samples=n_samples, n_features=2, 
                                 n_redundant=0, n_informative=2,
                                 n_clusters_per_class=1, n_classes=n_classes,
                                 random_state=42)
    return X, y

# Function to plot decision boundary
def plot_decision_boundary(X, y, model, ax, title="Decision Boundary"):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    if hasattr(model, 'predict'):
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    
    # Plot the data points
    colors = ['red', 'blue', 'green', 'purple']
    for i in range(len(np.unique(y))):
        ax.scatter(X[y == i, 0], X[y == i, 1], 
                  c=colors[i], marker='o', s=50, alpha=0.8,
                  label=f'Class {i}')
    
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

# Function to calculate impurity
def calculate_impurity(y, criterion='gini'):
    if len(y) == 0:
        return 0
    
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    
    if criterion == 'gini':
        return 1 - np.sum(probabilities ** 2)
    elif criterion == 'entropy':
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))

# Generate data
X, y = generate_data(n_samples, n_classes, cluster_std, dataset_type)

# Create column layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Decision Boundary Visualization")
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    plot_decision_boundary(X, y, None, ax1, "Data Points")
    boundary_plot = st.pyplot(fig1)

with col2:
    st.subheader("Decision Tree Structure")
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.text(0.5, 0.5, 'Train the model to see the tree structure', 
             ha='center', va='center', transform=ax2.transAxes, fontsize=12)
    ax2.set_xticks([])
    ax2.set_yticks([])
    tree_plot = st.pyplot(fig2)

# Training section
st.subheader("Model Training")

# Display data info
col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    st.metric("Total Samples", n_samples)
with col_info2:
    st.metric("Number of Classes", n_classes)
with col_info3:
    initial_impurity = calculate_impurity(y, criterion)
    st.metric(f"Initial {criterion.capitalize()}", f"{initial_impurity:.4f}")

# Button to start training
if st.button("Train Decision Tree"):
    with st.spinner("Training decision tree..."):
        # Create and train the decision tree
        dt = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=42
        )
        
        dt.fit(X, y)
        
        # Update decision boundary plot
        fig1.clear()
        ax1 = fig1.add_subplot(111)
        plot_decision_boundary(X, y, dt, ax1, "Decision Tree Classification")
        boundary_plot.pyplot(fig1)
        
        # Update tree structure plot
        fig2.clear()
        ax2 = fig2.add_subplot(111)
        plot_tree(dt, ax=ax2, feature_names=['Feature 1', 'Feature 2'],
                 class_names=[f'Class {i}' for i in range(n_classes)],
                 filled=True, rounded=True, fontsize=10)
        ax2.set_title("Decision Tree Structure")
        tree_plot.pyplot(fig2)
        
        # Calculate accuracy
        accuracy = dt.score(X, y)
        
        # Display results
        st.success(f"Training completed! Accuracy: {accuracy:.4f}")
        
        # Create training progress dataframe
        st.subheader("Tree Information")
        
        # Get tree information
        tree_info = {
            "Metric": [
                "Tree Depth",
                "Number of Leaves",
                "Number of Splits",
                "Training Accuracy",
                f"Final {criterion.capitalize()}",
                "Feature Importance (Feature 1)",
                "Feature Importance (Feature 2)"
            ],
            "Value": [
                dt.get_depth(),
                dt.get_n_leaves(),
                dt.tree_.node_count,
                f"{accuracy:.4f}",
                f"{calculate_impurity(y, criterion):.4f}",
                f"{dt.feature_importances_[0]:.4f}",
                f"{dt.feature_importances_[1]:.4f}"
            ]
        }
        
        tree_df = pd.DataFrame(tree_info)
        st.dataframe(tree_df, height=280)
        
        # Show split information for each node
        st.subheader("Node Split Information")
        
        # Extract split information
        tree = dt.tree_
        feature_names = ['Feature 1', 'Feature 2']
        
        split_info = []
        for node_id in range(tree.node_count):
            if tree.children_left[node_id] != tree.children_right[node_id]:  # Not a leaf
                feature = feature_names[tree.feature[node_id]]
                threshold = tree.threshold[node_id]
                impurity = tree.impurity[node_id]
                samples = tree.n_node_samples[node_id]
                
                split_info.append({
                    "Node": node_id,
                    "Split Feature": feature,
                    "Threshold": f"{threshold:.4f}",
                    "Impurity": f"{impurity:.4f}",
                    "Samples": samples
                })
        
        if split_info:
            split_df = pd.DataFrame(split_info)
            st.dataframe(split_df, height=200)
        
        # Feature importance visualization
        st.subheader("Feature Importance")
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        features = ['Feature 1', 'Feature 2']
        importances = dt.feature_importances_
        
        bars = ax3.bar(features, importances, color=['skyblue', 'lightcoral'])
        ax3.set_ylabel('Importance')
        ax3.set_title('Feature Importance')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, importance in zip(bars, importances):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{importance:.4f}', ha='center', va='bottom')
        
        st.pyplot(fig3)

# Educational content
st.markdown("---")
st.subheader("Understanding Decision Trees")

col_edu1, col_edu2 = st.columns(2)

with col_edu1:
    st.markdown("""
    **How Decision Trees Work:**
    
    1. **Splitting**: At each node, the algorithm finds the best feature and threshold to split the data
    2. **Impurity Measures**: Uses Gini impurity or entropy to measure how "pure" each split is
    3. **Recursive Process**: Continues splitting until stopping criteria are met
    4. **Prediction**: New data points follow the tree structure to reach a leaf node
    """)

with col_edu2:
    st.markdown("""
    **Key Parameters:**
    
    - **Max Depth**: Maximum depth of the tree (prevents overfitting)
    - **Min Samples Split**: Minimum samples required to split an internal node
    - **Min Samples Leaf**: Minimum samples required to be at a leaf node
    - **Criterion**: The function to measure the quality of a split
    """)

# Interactive explanation
st.subheader("Try Different Parameters")
st.write("""
Experiment with different parameters to see how they affect the decision tree:

- **Higher max_depth**: More complex trees, potential overfitting
- **Higher min_samples_split**: Simpler trees, prevents overfitting
- **Gini vs Entropy**: Different ways to measure impurity (usually similar results)
- **Different data types**: Blobs vs Classification datasets show different patterns
""")