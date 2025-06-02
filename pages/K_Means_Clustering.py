import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.datasets import make_blobs
from matplotlib.colors import ListedColormap

# Set page configuration
st.set_page_config(page_title="K-Means Clustering Visualizer", layout="wide", page_icon="📊")

st.title("K-Means Clustering Visualizer")
st.write("""
Watch as the K-Means algorithm iteratively finds cluster centers and assigns data points to clusters.
""")

# Sidebar for parameters
st.sidebar.header("Algorithm Parameters")
k_clusters = st.sidebar.slider("Number of Clusters (k)", min_value=2, max_value=8, value=3, step=1)
max_iterations = st.sidebar.slider("Max Iterations", min_value=5, max_value=50, value=20, step=5)
sample_size = st.sidebar.slider("Sample Size", min_value=50, max_value=500, value=200, step=50)
cluster_std = st.sidebar.slider("Cluster Standard Deviation", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
random_seed = st.sidebar.slider("Random Seed", min_value=1, max_value=100, value=42, step=1)

# Sidebar for mathematical explanations
st.sidebar.header("Algorithm Details")
st.sidebar.markdown("""
### K-Means Algorithm Steps

1. **Initialize** k centroids randomly
2. **Assignment Step**: Assign each point to nearest centroid
   $$c^{(i)} = \\arg\\min_j ||x^{(i)} - \\mu_j||^2$$
3. **Update Step**: Move centroids to cluster means
   $$\\mu_j = \\frac{1}{|S_j|} \\sum_{x^{(i)} \\in S_j} x^{(i)}$$
4. **Repeat** steps 2-3 until convergence

### Distance Metric
**Euclidean Distance:**
$$d(x, \\mu) = \\sqrt{(x_1 - \\mu_1)^2 + (x_2 - \\mu_2)^2}$$

### Objective Function (WCSS)
$$J = \\sum_{j=1}^{k} \\sum_{x^{(i)} \\in S_j} ||x^{(i)} - \\mu_j||^2$$
""")

# Function to generate clustered data
def generate_clustered_data(n_samples, n_clusters, std, random_state):
    X, y_true = make_blobs(
        n_samples=n_samples, 
        centers=n_clusters, 
        cluster_std=std, 
        random_state=random_state,
        center_box=(-8, 8)
    )
    return X, y_true

# K-Means implementation
class KMeans:
    def __init__(self, k=3, max_iters=100, random_state=42):
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state
        
        # History tracking
        self.centroids_history = []
        self.labels_history = []
        self.wcss_history = []
        self.convergence_history = []
        
    def initialize_centroids(self, X):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        centroids = np.random.uniform(
            low=X.min(axis=0), 
            high=X.max(axis=0), 
            size=(self.k, n_features)
        )
        return centroids
    
    def assign_clusters(self, X, centroids):
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        return labels
    
    def update_centroids(self, X, labels):
        centroids = np.zeros((self.k, X.shape[1]))
        for k in range(self.k):
            if np.sum(labels == k) > 0:  # Avoid empty clusters
                centroids[k] = X[labels == k].mean(axis=0)
        return centroids
    
    def calculate_wcss(self, X, labels, centroids):
        wcss = 0
        for k in range(self.k):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                wcss += np.sum((cluster_points - centroids[k]) ** 2)
        return wcss
    
    def fit_step_by_step(self, X):
        # Initialize
        self.centroids = self.initialize_centroids(X)
        self.centroids_history = [self.centroids.copy()]
        self.labels_history = []
        self.wcss_history = []
        self.convergence_history = []
        
        prev_centroids = None
        
        for iteration in range(self.max_iters):
            # Assignment step
            labels = self.assign_clusters(X, self.centroids)
            
            # Calculate WCSS
            wcss = self.calculate_wcss(X, labels, self.centroids)
            
            # Check convergence
            converged = False
            if prev_centroids is not None:
                centroid_shift = np.sqrt(np.sum((self.centroids - prev_centroids)**2))
                converged = centroid_shift < 1e-4
            
            # Store history
            self.labels_history.append(labels.copy())
            self.wcss_history.append(wcss)
            self.convergence_history.append(converged)
            
            if converged:
                break
                
            # Update step
            prev_centroids = self.centroids.copy()
            self.centroids = self.update_centroids(X, labels)
            self.centroids_history.append(self.centroids.copy())
            
        return len(self.wcss_history)

# Generate data
X, y_true = generate_clustered_data(sample_size, k_clusters, cluster_std, random_seed)

# Create model
kmeans = KMeans(k=k_clusters, max_iters=max_iterations, random_state=random_seed)

# Create column layout
col1, col2 = st.columns(2)

# Color maps for visualization
colors = plt.cm.Set3(np.linspace(0, 1, k_clusters))
cmap = ListedColormap(colors)

with col1:
    st.subheader("Clustering Visualization")
    cluster_plot = st.empty()

with col2:
    st.subheader("Within-Cluster Sum of Squares (WCSS)")
    wcss_plot = st.empty()

# Button to start clustering
if st.button("Start K-Means Clustering"):
    st.subheader("Training Progress")
    
    # Initialize progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    df_placeholder = st.empty()
    
    # Setup initial display
    initial_centroids = kmeans.initialize_centroids(X)
    
    # Initialize dataframe for training progress
    df_data = {
        "Iteration": [],
        "WCSS": [],
        "Centroid Changes": [],
        "Converged": []
    }
    training_df = pd.DataFrame(df_data)
    
    # Add initial state
    initial_wcss = kmeans.calculate_wcss(X, np.zeros(len(X), dtype=int), initial_centroids)
    new_row = {
        "Iteration": "Initial",
        "WCSS": f"{initial_wcss:.2f}",
        "Centroid Changes": "N/A",
        "Converged": "No"
    }
    training_df = pd.concat([training_df, pd.DataFrame([new_row])], ignore_index=True)
    df_placeholder.dataframe(training_df, height=200)
    
    # Run K-means step by step
    kmeans.centroids = initial_centroids.copy()
    kmeans.centroids_history = [initial_centroids.copy()]
    kmeans.labels_history = []
    kmeans.wcss_history = []
    kmeans.convergence_history = []
    
    prev_centroids = None
    iteration = 0
    
    for iteration in range(max_iterations):
        # Assignment step
        labels = kmeans.assign_clusters(X, kmeans.centroids)
        
        # Calculate WCSS
        wcss = kmeans.calculate_wcss(X, labels, kmeans.centroids)
        
        # Check convergence
        converged = False
        centroid_change = 0
        if prev_centroids is not None:
            centroid_change = np.sqrt(np.sum((kmeans.centroids - prev_centroids)**2))
            converged = centroid_change < 0.01
        
        # Store history
        kmeans.labels_history.append(labels.copy())
        kmeans.wcss_history.append(wcss)
        kmeans.convergence_history.append(converged)
        
        # Create new figures for this iteration
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        
        # Plot data points colored by cluster assignment
        for k in range(k_clusters):
            cluster_mask = labels == k
            if np.any(cluster_mask):
                ax1.scatter(X[cluster_mask, 0], X[cluster_mask, 1], 
                           c=[colors[k]], alpha=0.7, s=50, label=f'Cluster {k+1}')
        
        # Plot centroids
        ax1.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
                   c='red', marker='x', s=300, linewidths=4, label='Centroids')
        
        # Plot centroid history (trails)
        if len(kmeans.centroids_history) > 1:
            for k in range(k_clusters):
                centroid_trail = np.array([cent[k] for cent in kmeans.centroids_history])
                ax1.plot(centroid_trail[:, 0], centroid_trail[:, 1], 
                        'r--', alpha=0.5, linewidth=1)
        
        ax1.set_xlabel("Feature 1")
        ax1.set_ylabel("Feature 2")
        ax1.set_title(f"K-Means Clustering - Iteration {iteration + 1}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Create WCSS plot
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        iterations_so_far = list(range(1, len(kmeans.wcss_history) + 1))
        ax2.plot(iterations_so_far, kmeans.wcss_history, 'b-', marker='o', linewidth=2, markersize=6)
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("WCSS")
        ax2.set_title(f"WCSS: {wcss:.2f}")
        ax2.grid(True, alpha=0.3)
        
        # Update progress
        progress_bar.progress((iteration + 1) / max_iterations)
        status_text.text(f"Iteration {iteration + 1}/{max_iterations} - WCSS: {wcss:.2f}")
        
        # Update dataframe
        new_row = {
            "Iteration": f"{iteration + 1}",
            "WCSS": f"{wcss:.2f}",
            "Centroid Changes": f"{centroid_change:.4f}" if prev_centroids is not None else "N/A",
            "Converged": "Yes" if converged else "No"
        }
        training_df = pd.concat([training_df, pd.DataFrame([new_row])], ignore_index=True)
        df_placeholder.dataframe(training_df, height=200)
        
        # Display plots and then close them to free memory
        cluster_plot.pyplot(fig1)
        wcss_plot.pyplot(fig2)
        plt.close(fig1)
        plt.close(fig2)
        
        if converged:
            status_text.success(f"Converged after {iteration + 1} iterations!")
            break
            
        # Update step
        prev_centroids = kmeans.centroids.copy()
        kmeans.centroids = kmeans.update_centroids(X, labels)
        kmeans.centroids_history.append(kmeans.centroids.copy())
        
        # Animation delay
        time.sleep(0.05)
    
    # Final results
    final_wcss = kmeans.wcss_history[-1] if kmeans.wcss_history else 0
    st.success(f"Clustering completed!")
    st.info(f"Final WCSS: {final_wcss:.2f}")
    st.info(f"Total iterations: {len(kmeans.wcss_history)}")
    
    # Show cluster statistics
    st.subheader("Final Cluster Statistics")
    final_labels = kmeans.labels_history[-1] if kmeans.labels_history else []
    cluster_stats = []
    
    for k in range(k_clusters):
        cluster_size = np.sum(final_labels == k)
        if cluster_size > 0:
            cluster_center = kmeans.centroids[k]
            cluster_stats.append({
                "Cluster": k + 1,
                "Size": cluster_size,
                "Center X": f"{cluster_center[0]:.2f}",
                "Center Y": f"{cluster_center[1]:.2f}"
            })
    
    if cluster_stats:
        stats_df = pd.DataFrame(cluster_stats)
        st.dataframe(stats_df, hide_index=True)

st.markdown("---")
st.subheader("Understanding K-Means Clustering")

col_edu1, col_edu2 = st.columns(2)

with col_edu1:
    st.markdown("""
    **How K-Means Works:**
    
    1. **Initialization**: Randomly place k cluster centers in feature space
    2. **Assignment**: Assign each point to nearest cluster center
    3. **Update**: Move cluster centers to mean of assigned points
    4. **Iteration**: Repeat assignment and update until convergence
    """)

with col_edu2:
    st.markdown("""
    **Key Parameters:**
    
    - **K (n_clusters)**: Number of clusters to find in the data
    - **Initialization**: Method for initial centroid placement (k-means++, random)
    - **Max Iterations**: Maximum number of algorithm iterations
    - **Tolerance**: Convergence threshold for centroid movement
    """)

# Interactive explanation
st.subheader("Try Different Parameters")
st.write("""
Experiment with different parameters to see how they affect k-means:

- **Different K values**: Too few = underfitting, too many = overfitting
- **K-means++ init**: Usually better than random initialization
- **More iterations**: Ensures convergence, especially for complex data
- **Multiple runs**: Different initializations can lead to different results
""")