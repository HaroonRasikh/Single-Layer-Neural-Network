
import sys
import random
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QComboBox, QLabel
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Sigmoid activation function
def sigmoid(x):
    # Implementing the sigmoid function
    return 1 / (1 + (2.71828 ** -x))  # Approximate value of e

# Predict function for binary classification
def predict(X, weights, bias):
    # Compute the dot product manually
    total_activation = 0
    for i in range(len(X)):
        total_activation += X[i] * weights[i]
    total_activation += bias
    return sigmoid(total_activation)

# Training function for single-layer perceptron (binary classification)
def train_perceptron(X, y, learning_rate=0.1, epochs=50):
    weights = np.random.uniform(-1, 1, X.shape[1])  # Random weights for each feature
    bias = random.uniform(-1, 1)  # Random bias

    for epoch in range(epochs):
        for i in range(len(X)):
            prediction = predict(X[i], weights, bias)
            error = y[i] - prediction # d - o
            print(error)

            # Update weights and bias using gradient descent
            weights += learning_rate * error * prediction * (1 - prediction) * X[i] # graaient olan error = error * o * ( 1 - o)
            bias += learning_rate * error * prediction * (1 - prediction) # c*r = c * r * o * (1 - o)

    return weights, bias

# Plot decision boundary for binary classification
def plot_decision_boundary(ax, X, y, weights, bias):
    ax.clear()  # Clear the previous plot
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.axhline(0, color='black', linewidth=1)  # Draw horizontal axis line
    ax.axvline(0, color='black', linewidth=1)  # Draw vertical axis line
    ax.grid(True)

    # Plot data points
    for i, label in enumerate(y):
        color = 'red' if label == 0 else 'blue'  # Class 0 is red, class 1 is blue
        ax.scatter(X[i][0], X[i][1], color=color)

    # Calculate and plot decision boundary
    x_vals = np.linspace(-10, 10, 100)
    #  calculation of y = (w0*x + b) / w1
    y_vals = -(weights[0] * x_vals + bias) / weights[1]  # Equation of decision boundary: w0*x + w1*y + b = 0
    ax.plot(x_vals, y_vals, color='green', label="Decision Boundary")

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Single-Layer Perceptron - Decision Boundary")
    ax.legend()

class PerceptronApp(QWidget):
    def __init__(self):
        super().__init__()
        self.points = []  # List to store points as (x, y, class)
        self.X = []  # Input features
        self.y = []  # Labels
        self.weights = None  # To store trained weights
        self.bias = None  # To store trained bias
        self.current_class = 0  # Default class is 0

        self.setWindowTitle("Single-Layer Perceptron")
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Dropdown for selecting the current class
        layout.addWidget(QLabel("Select current class:"))
        self.class_selector = QComboBox()
        self.class_selector.addItem("Class 0")
        self.class_selector.addItem("Class 1")
        self.class_selector.setCurrentIndex(0)  # Default to class 0
        self.class_selector.currentIndexChanged.connect(self.select_class)
        layout.addWidget(self.class_selector)

        # Matplotlib plot
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Button to train the perceptron
        self.train_button = QPushButton("Train Model")
        self.train_button.clicked.connect(self.train_model)
        layout.addWidget(self.train_button)

        # Clear button to reset points
        self.clear_button = QPushButton("Clear Points")
        self.clear_button.clicked.connect(self.clear_points)
        layout.addWidget(self.clear_button)

        # Connect click event on canvas
        self.canvas.mpl_connect("button_press_event", self.onclick)

        # Initialize the plot with proper axes
        self.initialize_plot()

    def initialize_plot(self):
        # Set initial axes limits and grid
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.axhline(0, color='black', linewidth=1)
        self.ax.axvline(0, color='black', linewidth=1)
        self.ax.grid(True)
        self.canvas.draw()

    def select_class(self, index):
        # Update the current class based on dropdown selection
        self.current_class = index

    def onclick(self, event):
        # Only add points if a valid click (xdata and ydata exist)
        if event.xdata is not None and event.ydata is not None:
            self.points.append((event.xdata, event.ydata, self.current_class))
            self.X.append([event.xdata, event.ydata])
            self.y.append(self.current_class)

            # Plot the point in the selected class color
            color = 'red' if self.current_class == 0 else 'blue'
            self.ax.scatter(event.xdata, event.ydata, color=color)
            self.canvas.draw()

    def train_model(self):
        # Convert points to numpy arrays for training
        X = np.array(self.X)
        y = np.array(self.y)

        # Train the perceptron
        self.weights, self.bias = train_perceptron(X, y)

        # Plot decision boundary
        plot_decision_boundary(self.ax, X, y, self.weights, self.bias)
        self.canvas.draw()

    def clear_points(self):
        # Clear the list of points, X, and y
        self.points = []
        self.X = []
        self.y = []
        
        # Re-initialize the plot with grid and axes intact
        self.ax.clear()  # Clear all existing data on the axes
        self.initialize_plot()  # Reinitialize the axes with grid and limits
        
        # Redraw the canvas
        self.canvas.draw()

# Run the application
app = QApplication(sys.argv)
window = PerceptronApp()
window.show()
sys.exit(app.exec_())
