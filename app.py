# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# import time

# # Set page configuration
# st.set_page_config(page_title="Linear Regression Visualizer", layout="wide")

# st.title("Linear Regression Training Visualizer")
# st.write("""
# This app demonstrates the training process of a linear regression model using gradient descent.
# Watch as the model learns the best fit line, and see how the loss decreases over iterations.
# """)

# # Sidebar for parameters
# st.sidebar.header("Model Parameters")
# learning_rate = st.sidebar.slider("Learning Rate", min_value=0.001, max_value=0.1, value=0.01, step=0.001)
# iterations = st.sidebar.slider("Number of Iterations", min_value=10, max_value=200, value=100, step=10)
# noise_level = st.sidebar.slider("Noise Level", min_value=0.0, max_value=5.0, value=2.0, step=0.1)
# sample_size = st.sidebar.slider("Sample Size", min_value=20, max_value=200, value=50, step=10)

# # Sidebar for mathematical explanations
# st.sidebar.header("Mathematical Foundations")
# st.sidebar.markdown("""
# ### Linear Regression Model
# $y = wx + b$

# ### Mean Squared Error Loss
# $L(w, b) = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - (wx_i + b))^2$

# ### Gradient Descent Updates
# $w = w - \\alpha \\frac{\\partial L}{\\partial w}$

# $b = b - \\alpha \\frac{\\partial L}{\\partial b}$

# where:
# - $\\alpha$ is the learning rate
# - $\\frac{\\partial L}{\\partial w} = -\\frac{2}{n} \\sum_{i=1}^{n} x_i(y_i - (wx_i + b))$
# - $\\frac{\\partial L}{\\partial b} = -\\frac{2}{n} \\sum_{i=1}^{n} (y_i - (wx_i + b))$
# """)

# # Function to generate data
# def generate_data(sample_size, noise):
#     # True parameters
#     true_w = 2.5
#     true_b = 3.0
    
#     # Generate features
#     X = np.random.rand(sample_size, 1) * 10
    
#     # Generate target with noise
#     y = true_w * X + true_b + np.random.randn(sample_size, 1) * noise
    
#     return X, y, true_w, true_b

# # Linear regression model
# class LinearRegression:
#     def __init__(self, learning_rate=0.01):
#         self.w = np.random.randn()
#         self.b = np.random.randn()
#         self.learning_rate = learning_rate
#         self.loss_history = []
#         self.w_history = []
#         self.b_history = []
        
#     def predict(self, X):
#         return self.w * X + self.b
    
#     def compute_loss(self, X, y):
#         n = len(X)
#         predictions = self.predict(X)
#         loss = np.mean((predictions - y) ** 2)
#         return loss
    
#     def compute_gradients(self, X, y):
#         n = len(X)
#         predictions = self.predict(X)
#         dw = -2/n * np.sum(X * (y - predictions))
#         db = -2/n * np.sum(y - predictions)
#         return dw, db
    
#     def update_parameters(self, dw, db):
#         self.w = self.w - self.learning_rate * dw
#         self.b = self.b - self.learning_rate * db
        
#     def fit(self, X, y, iterations=100):
#         self.loss_history = []
#         self.w_history = []
#         self.b_history = []
        
#         for i in range(iterations):
#             # Compute loss and save history
#             loss = self.compute_loss(X, y)
#             self.loss_history.append(loss)
#             self.w_history.append(self.w)
#             self.b_history.append(self.b)
            
#             # Compute gradients
#             dw, db = self.compute_gradients(X, y)
            
#             # Update parameters
#             self.update_parameters(dw, db)
            
#         return self.w_history, self.b_history, self.loss_history

# # Generate data
# X, y, true_w, true_b = generate_data(sample_size, noise_level)

# # Create and train model
# model = LinearRegression(learning_rate=learning_rate)

# # Create placeholders for visualizations
# col1, col2 = st.columns(2)

# with col1:
#     st.subheader("Model Fitting Visualization")
#     fig1, ax1 = plt.subplots(figsize=(10, 6))
#     scatter = ax1.scatter(X, y, alpha=0.6, label="Data Points")
#     true_line, = ax1.plot([], [], 'g-', label=f"True Line: y = {true_w:.2f}x + {true_b:.2f}")
#     line, = ax1.plot([], [], 'r-', label="Predicted Line")
#     ax1.set_xlabel("X")
#     ax1.set_ylabel("y")
#     ax1.legend()
#     ax1.grid(True)
#     model_plot = st.pyplot(fig1)

# with col2:
#     st.subheader("Loss Over Time")
#     fig2, ax2 = plt.subplots(figsize=(10, 6))
#     loss_line, = ax2.plot([], [], 'b-')
#     ax2.set_xlabel("Iterations")
#     ax2.set_ylabel("Mean Squared Error")
#     ax2.grid(True)
#     loss_plot = st.pyplot(fig2)

# # Button to start training
# if st.button("Start Training"):
#     # Setup true line data for plotting
#     x_range = np.array([0, 10])
#     true_line.set_data(x_range, true_w * x_range + true_b)
    
#     # Train the model step by step
#     for i in range(iterations + 1):
#         # Update model for one iteration if not at the start
#         if i > 0:
#             current_loss = model.compute_loss(X, y)
#             dw, db = model.compute_gradients(X, y)
#             model.update_parameters(dw, db)
#             model.loss_history.append(current_loss)
#             model.w_history.append(model.w)
#             model.b_history.append(model.b)
        
#         # Update model plot
#         x_pred = np.array([0, 10])
#         y_pred = model.predict(x_pred)
#         line.set_data(x_pred, y_pred)
        
#         # Update loss plot
#         iterations_so_far = list(range(len(model.loss_history)))
#         loss_line.set_data(iterations_so_far, model.loss_history)
#         ax2.relim()
#         ax2.autoscale_view()
        
#         # Show the current parameters
#         st.sidebar.markdown(f"### Current Parameters (Iteration {i})")
#         st.sidebar.markdown(f"w = {model.w:.4f}")
#         st.sidebar.markdown(f"b = {model.b:.4f}")
#         st.sidebar.markdown(f"Loss = {model.compute_loss(X, y):.4f}")
        
#         # Refresh plots
#         model_plot.pyplot(fig1)
#         loss_plot.pyplot(fig2)
        
#         # Slow down to make the animation visible
#         if i < iterations:
#             time.sleep(0.1)
    
#     # Final results
#     st.success(f"Training completed! Final model: y = {model.w:.4f}x + {model.b:.4f}")
#     st.info(f"True model: y = {true_w:.2f}x + {true_b:.2f}")
#     st.info(f"Final loss: {model.compute_loss(X, y):.4f}")
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import time

# Set page configuration
st.set_page_config(page_title="Linear Regression Visualizer", layout="wide")

st.title("Linear Regression Training Visualizer")
st.write("""
Watch as the model learns the line of best fit, and see how the loss decreases over iterations.
""")

# Sidebar for parameters
st.sidebar.header("Model Parameters")
learning_rate = st.sidebar.slider("Learning Rate", min_value=0.001, max_value=0.1, value=0.01, step=0.001)
iterations = st.sidebar.slider("Number of Iterations", min_value=10, max_value=200, value=100, step=10)
noise_level = st.sidebar.slider("Noise Level", min_value=0.0, max_value=5.0, value=2.0, step=0.1)
sample_size = st.sidebar.slider("Sample Size", min_value=20, max_value=200, value=50, step=10)

# Sidebar for mathematical explanations
st.sidebar.header("Mathematical Foundations")
st.sidebar.markdown("""
### Linear Regression Model
$y = wx + b$

### Mean Squared Error Loss
$L(w, b) = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - (wx_i + b))^2$

### Gradient Descent Updates
$w = w - \\alpha \\frac{\\partial L}{\\partial w}$

$b = b - \\alpha \\frac{\\partial L}{\\partial b}$

where:
- $\\alpha$ is the learning rate
- $\\frac{\\partial L}{\\partial w} = -\\frac{2}{n} \\sum_{i=1}^{n} x_i(y_i - (wx_i + b))$
- $\\frac{\\partial L}{\\partial b} = -\\frac{2}{n} \\sum_{i=1}^{n} (y_i - (wx_i + b))$
""")

# Function to generate data
def generate_data(sample_size, noise):
    # True parameters
    true_w = 2.5
    true_b = 3.0
    
    # Generate features
    X = np.random.rand(sample_size, 1) * 10
    
    # Generate target with noise
    y = true_w * X + true_b + np.random.randn(sample_size, 1) * noise
    
    return X, y, true_w, true_b

# Linear regression model
class LinearRegression:
    def __init__(self, learning_rate=0.01):
        self.w = np.random.randn()
        self.b = np.random.randn()
        self.learning_rate = learning_rate
        self.loss_history = []
        self.w_history = []
        self.b_history = []
        
    def predict(self, X):
        return self.w * X + self.b
    
    def compute_loss(self, X, y):
        n = len(X)
        predictions = self.predict(X)
        loss = np.mean((predictions - y) ** 2)
        return loss
    
    def compute_gradients(self, X, y):
        n = len(X)
        predictions = self.predict(X)
        dw = -2/n * np.sum(X * (y - predictions))
        db = -2/n * np.sum(y - predictions)
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
            self.w_history.append(self.w)
            self.b_history.append(self.b)
            
            # Compute gradients
            dw, db = self.compute_gradients(X, y)
            
            # Update parameters
            self.update_parameters(dw, db)
            
        return self.w_history, self.b_history, self.loss_history

# Generate data
X, y, true_w, true_b = generate_data(sample_size, noise_level)

# Create and train model
model = LinearRegression(learning_rate=learning_rate)

# Create column layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Fitting Visualization")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    scatter = ax1.scatter(X, y, alpha=0.6, label="Data Points")
    true_line, = ax1.plot([], [], 'g-', label=f"True Line: y = {true_w:.2f}x + {true_b:.2f}")
    line, = ax1.plot([], [], 'r-', label="Predicted Line")
    ax1.set_xlabel("X")
    ax1.set_ylabel("y")
    ax1.legend()
    ax1.grid(True)
    model_plot = st.pyplot(fig1)

with col2:
    st.subheader("Loss Over Time")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    loss_line, = ax2.plot([], [], 'b-')
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Mean Squared Error")
    ax2.grid(True)
    loss_plot = st.pyplot(fig2)


# Create a section for training progress with a dataframe

# Button to start training
if st.button("Start Training"):
    st.subheader("Training Progress")
    df_placeholder = st.empty()
    # Setup true line data for plotting
    x_range = np.array([0, 10])
    true_line.set_data(x_range, true_w * x_range + true_b)
    
    # Initialize dataframe for training progress
    df_data = {
        "Iteration": [],
        "Weight (w)": [],
        "Bias (b)": [],
        "Loss": []
    }
    training_df = pd.DataFrame(df_data)
    
    # Add initial parameters
    new_row = {
        "Iteration": "Initial",
        "Weight (w)": f"{model.w:.4f}",
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
        model.w_history.append(model.w)
        model.b_history.append(model.b)
        
        # Update model plot
        x_pred = np.array([0, 10])
        y_pred = model.predict(x_pred)
        line.set_data(x_pred, y_pred)
        
        # Update loss plot
        iterations_so_far = list(range(len(model.loss_history)))
        loss_line.set_data(iterations_so_far, model.loss_history)
        ax2.relim()
        ax2.autoscale_view()
        
        # Add current iteration to the dataframe
        new_row = {
            "Iteration": f"{i}",
            "Weight (w)": f"{model.w:.4f}",
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
            time.sleep(0.1)
    
    # Final results
    st.success(f"Training completed! Final model: y = {model.w:.4f}x + {model.b:.4f}")
    st.info(f"True model: y = {true_w:.2f}x + {true_b:.2f}")
    st.info(f"Final loss: {model.compute_loss(X, y):.4f}")
