import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import time

# Set page configuration
st.set_page_config(page_title="Logistic Regression Visualizer", layout="wide")

st.title("Logistic Regression Training Visualizer")
st.write("""
Watch as the model learns the decision boundary, and see how the loss decreases over iterations.
""")

# Sidebar for parameters
st.sidebar.header("Model Parameters")
learning_rate = st.sidebar.slider("Learning Rate", min_value=0.001, max_value=0.5, value=0.05, step=0.001)
iterations = st.sidebar.slider("Number of Iterations", min_value=10, max_value=200, value=100, step=10)
class_separation = st.sidebar.slider("Class Separation", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
sample_size = st.sidebar.slider("Sample Size", min_value=20, max_value=200, value=100, step=10)

# Sidebar for mathematical explanations
st.sidebar.header("Mathematical Foundations")
st.sidebar.markdown("""
### Logistic Regression Model
$p(y=1|x) = \\sigma(wx + b)$ where $\\sigma(z) = \\frac{1}{1+e^{-z}}$

### Binary Cross-Entropy Loss
$L(w, b) = -\\frac{1}{n} \\sum_{i=1}^{n} [y_i \\log(p_i) + (1-y_i) \\log(1-p_i)]$

### Gradient Descent Updates
$w = w - \\alpha \\frac{\\partial L}{\\partial w}$

$b = b - \\alpha \\frac{\\partial L}{\\partial b}$

where:
- $\\alpha$ is the learning rate
- $\\frac{\\partial L}{\\partial w} = \\frac{1}{n} \\sum_{i=1}^{n} x_i(\\sigma(wx_i + b) - y_i)$
- $\\frac{\\partial L}{\\partial b} = \\frac{1}{n} \\sum_{i=1}^{n} (\\sigma(wx_i + b) - y_i)$
""")

# Function to generate data for logistic regression
def generate_data(sample_size, separation):
    # Generate two feature variables
    X = np.random.randn(sample_size, 2)
    
    # True parameters (decision boundary)
    true_w = np.array([1.0, -1.0])
    true_b = 0.0
    
    # Generate target based on which side of the decision boundary
    z = np.dot(X, true_w) + true_b
    prob = 1 / (1 + np.exp(-separation * z))
    y = (np.random.rand(sample_size) < prob).astype(float).reshape(-1, 1)
    
    return X, y, true_w, true_b

# Logistic regression model
class LogisticRegression:
    def __init__(self, learning_rate=0.01):
        self.w = np.random.randn(2)
        # self.w = np.random.randn(2).reshape(-1, 1)

        self.b = np.random.randn()
        self.learning_rate = learning_rate
        self.loss_history = []
        self.w_history = []
        self.b_history = []
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -20, 20)))  # Clip to avoid overflow
    
    # def predict_proba(self, X):
    #     z = np.dot(X, self.w) + self.b
    #     return self.sigmoid(z)

    # def predict_proba(self, X):
    #     z = np.dot(X, self.w) + self.b
    #     return self.sigmoid(z).flatten()  # Ensure output is flattened to (n,)
    
    def predict_proba(self, X):
        z = np.dot(X, self.w) + self.b
        return self.sigmoid(z)  # Should return values between 0 and 1
    
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(float)
    
    def compute_loss(self, X, y):
        n = len(X)
        probs = self.predict_proba(X)
        # Add small epsilon to avoid log(0)
        epsilon = 1e-15
        probs = np.clip(probs, epsilon, 1 - epsilon)
        loss = -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))
        return loss

    def compute_loss(self, X, y):
        n = len(X)
        probs = self.predict_proba(X)  # Shape (n,)
        
        # Flatten y to match probs
        y_flat = y.flatten()  # Shape (n,)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-15
        probs = np.clip(probs, epsilon, 1 - epsilon)
        
        # Calculate binary cross-entropy loss
        loss = -np.mean(y_flat * np.log(probs) + (1 - y_flat) * np.log(1 - probs))
        return loss
    
    # def compute_gradients(self, X, y):
    #     n = len(X)
    #     probs = self.predict_proba(X)
    #     dw = np.dot(X.T, (probs - y)) / n
    #     db = np.sum(probs - y) / n
    #     return dw, db

    # def compute_gradients(self, X, y):
    #     n = len(X)
    #     probs = self.predict_proba(X)
    #     dw = np.dot(X.T, (probs - y)) / n
    #     dw = dw.flatten()  # Flatten to ensure dw has shape (2,)
    #     db = np.sum(probs - y) / n
    #     return dw, db
    
    def compute_gradients(self, X, y):
        n = len(X)
        probs = self.predict_proba(X)
        
        # Make sure probs and y have compatible shapes
        # probs should be shape (n,) and y should be (n,1) or (n,)
        if len(probs.shape) == 1:
            probs = probs.reshape(-1, 1)
        
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        # Calculate gradients
        dw = np.dot(X.T, (probs - y)) / n
        
        # Ensure dw has shape (2,)
        if dw.shape != self.w.shape:
            dw = dw.reshape(self.w.shape)
        
        # Calculate db
        db = np.sum(probs - y) / n
        
        return dw, db

    def update_parameters(self, dw, db):
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db
    
    def fit(self, X, y, iterations=100):
        self.loss_history = []
        self.w_history = []
        self.b_history = []
        
        for i in range(iterations):
            # Compute loss and save history
            loss = self.compute_loss(X, y)
            self.loss_history.append(loss)
            self.w_history.append(self.w.copy())
            self.b_history.append(self.b)
            
            # Compute gradients
            dw, db = self.compute_gradients(X, y)
            
            # Update parameters
            self.update_parameters(dw, db)
        
        return self.w_history, self.b_history, self.loss_history

# Generate data
X, y, true_w, true_b = generate_data(sample_size, class_separation)

# Create and train model
model = LogisticRegression(learning_rate=learning_rate)

# Create column layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Fitting Visualization")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    # Scatter points with different colors for different classes
    scatter0 = ax1.scatter(X[y.flatten()==0, 0], X[y.flatten()==0, 1], color='blue', alpha=0.6, label="Class 0")
    scatter1 = ax1.scatter(X[y.flatten()==1, 0], X[y.flatten()==1, 1], color='red', alpha=0.6, label="Class 1")
    
    # Plot decision boundaries
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Plot true decision boundary
    if true_w[1] != 0:
        slope = -true_w[0] / true_w[1]
        intercept = -true_b / true_w[1]
        x_boundary = np.array([x_min, x_max])
        y_boundary = slope * x_boundary + intercept
        true_line, = ax1.plot(x_boundary, y_boundary, 'g-', label=f"True Boundary")
    
    # Placeholder for model decision boundary
    line, = ax1.plot([], [], 'r--', label="Model Boundary")
    
    ax1.set_xlabel("Feature 1")
    ax1.set_ylabel("Feature 2")
    ax1.legend()
    ax1.grid(True)
    ax1.set_xlim([x_min, x_max])
    ax1.set_ylim([y_min, y_max])
    model_plot = st.pyplot(fig1)

with col2:
    st.subheader("Loss Over Time")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    loss_line, = ax2.plot([], [], 'b-')
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Binary Cross-Entropy Loss")
    ax2.grid(True)
    loss_plot = st.pyplot(fig2)

# Button to start training
if st.button("Start Training"):
    st.subheader("Training Progress")
    df_placeholder = st.empty()
    
    # Initialize dataframe for training progress
    df_data = {
        "Iteration": [],
        "Weight 1 (w1)": [],
        "Weight 2 (w2)": [],
        "Bias (b)": [],
        "Loss": []
    }
    training_df = pd.DataFrame(df_data)
    
    # Add initial parameters
    new_row = {
        "Iteration": "Initial",
        "Weight 1 (w1)": f"{model.w[0]:.4f}",
        "Weight 2 (w2)": f"{model.w[1]:.4f}",
        "Bias (b)": f"{model.b:.4f}",
        "Loss": f"{model.compute_loss(X, y):.4f}"
    }
    training_df = pd.concat([training_df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Update dataframe display
    df_placeholder.dataframe(training_df, height=200)
    
    # Train the model step by step
    for i in range(1, iterations + 1):
        # Update model for one iteration
        current_loss = model.compute_loss(X, y)
        dw, db = model.compute_gradients(X, y)
        model.update_parameters(dw, db)
        model.loss_history.append(current_loss)
        model.w_history.append(model.w.copy())
        model.b_history.append(model.b)
        
        # Update decision boundary plot
        if model.w[1] != 0:
            slope = -model.w[0] / model.w[1]
            intercept = -model.b / model.w[1]
            x_boundary = np.array([x_min, x_max])
            y_boundary = slope * x_boundary + intercept
            line.set_data(x_boundary, y_boundary)
        
        # Update loss plot
        iterations_so_far = list(range(len(model.loss_history)))
        loss_line.set_data(iterations_so_far, model.loss_history)
        ax2.relim()
        ax2.autoscale_view()
        
        # Add current iteration to the dataframe
        new_row = {
            "Iteration": f"{i}",
            "Weight 1 (w1)": f"{model.w[0]:.4f}",
            "Weight 2 (w2)": f"{model.w[1]:.4f}",
            "Bias (b)": f"{model.b:.4f}",
            "Loss": f"{model.compute_loss(X, y):.4f}"
        }
        training_df = pd.concat([training_df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Update dataframe display
        df_placeholder.dataframe(training_df, height=200)
        
        # Refresh plots
        model_plot.pyplot(fig1)
        loss_plot.pyplot(fig2)
        
        # Slow down to make the animation visible
        if i < iterations:
            time.sleep(0.01)
    
    # Final results
    decision_eq = f"p(y=1|x) = σ({model.w[0]:.4f}*x₁ + {model.w[1]:.4f}*x₂ + {model.b:.4f})"
    true_eq = f"True boundary: {true_w[0]:.2f}*x₁ + {true_w[1]:.2f}*x₂ + {true_b:.2f} = 0"
    
    st.success(f"Training completed! Final model: {decision_eq}")
    st.info(true_eq)
    st.info(f"Final loss: {model.compute_loss(X, y):.4f}")