
import sys
import random
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QComboBox, QLabel
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import math
# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + (2.718281828459045) ** -x)

# Predict function for multiclass using the sigmoid activation function
def predict_multiclass(X, weights, biases):
    outputs = []
    for i in range(len(weights)):
        total_activation = 0
        for j in range(len(X)):
            total_activation += X[j] * weights[i][j]
        total_activation += biases[i]
        outputs.append(sigmoid(total_activation))  # Apply the sigmoid function
    
    return outputs


# Training function with error propagation for multiclass
## epochs = 50
def train_multicategory_perceptron(X, y, num_classes, learning_rate=1, epochs=50):
    # Initialize weights and biases randomly
    weights = [[random.uniform(-1, 1) for _ in range(len(X[0]))] for _ in range(num_classes)]
    biases = [random.uniform(-1, 1) for _ in range(num_classes)]

    for epoch in range(epochs):
        for i in range(len(X)):
            # Forward pass: get outputs
            outputs = predict_multiclass(X[i], weights, biases)
            
            # Calculate error for each class (binary error)
            errors = [y[i][j] - outputs[j] for j in range(num_classes)]
            print('errors',errors)
            
            # Backward pass: update weights and biases based on error gradients
            for k in range(num_classes):
                # Calculate gradient for each weight and update weights
                for j in range(len(weights[k])):
                    weights[k][j] += learning_rate * errors[k] * outputs[k] * (1 - outputs[k]) * X[i][j]
                
                # Update bias with error propagation
                biases[k] += learning_rate * errors[k] * outputs[k] * (1 - outputs[k])

    return weights, biases

# Plot decision boundary for multiclass classification
def plot_decision_boundary_multiclass(ax, X, y, weights, biases, num_classes):
    ax.clear()  # Clear the previous plot

    # Set axis limits and draw grid and axes lines
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.axhline(0, color='black', linewidth=1)  # Draw horizontal axis line
    ax.axvline(0, color='black', linewidth=1)  # Draw vertical axis line
    ax.grid(True)
 # Find x_min, x_max, y_min, y_max without using min and max
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    for x in X:
        if x[0] < x_min:
            x_min = x[0]
        if x[0] > x_max:
            x_max = x[0]
        if x[1] < y_min:
            y_min = x[1]
        if x[1] > y_max:
            y_max = x[1]
    # Adjust the boundaries
    x_min, x_max = x_min - 1, x_max + 1
    y_min, y_max = y_min - 1, y_max + 1
   # Generate grid points for decision boundary
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

    # Predict class for each grid point
    Z = np.array([
        predict_multiclass([xx[i, j], yy[i, j]], weights, biases).index(max(predict_multiclass([xx[i, j], yy[i, j]], weights, biases)))
        for i in range(xx.shape[0]) for j in range(xx.shape[1])
    ])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary by color regions
    ax.contourf(xx, yy, Z, cmap=plt.cm.rainbow, alpha=0.3)

    # Plot data points
    colors = ['red', 'blue', 'green', 'purple']  # Adjust colors if you have more classes
    for i, label in enumerate(y):
        class_idx = label.index(1)  # Assuming one-hot encoding
        ax.scatter(X[i][0], X[i][1], color=colors[class_idx], label=f'Class {class_idx}' if i == 0 or label not in y[:i] else "")

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(f"Multiclass Classification with Decision Boundaries ({num_classes} classes)")
    ax.legend()

class InteractivePlot(QWidget):
    def __init__(self):
        super().__init__()
        self.num_classes = 4  # Default number of classes to 4
        self.points = []  # Store points as (x, y, class)
        self.current_class = 0  # Default class

        # PyQt Layout
        self.setWindowTitle("Multiclass Perceptron Input")
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Dropdown for selecting the number of classes
        layout.addWidget(QLabel("Select number of classes:"))
        self.class_count_selector = QComboBox()
        for i in range(2, 11):  # Allow up to 10 classes for flexibility
            self.class_count_selector.addItem(str(i))
        self.class_count_selector.setCurrentText(str(self.num_classes))  # Default to 4 classes
        self.class_count_selector.currentIndexChanged.connect(self.update_class_count)
        layout.addWidget(self.class_count_selector)

        # Dropdown for selecting the current class
        layout.addWidget(QLabel("Select current class:"))
        self.class_selector = QComboBox()
        self.update_class_selector()  # Initialize with the default number of classes
        self.class_selector.currentIndexChanged.connect(self.select_class)
        layout.addWidget(self.class_selector)

        # Matplotlib plot
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.ax.set_xlim(-20, 20)  # Set limits for Cartesian grid (4 quadrants)
        self.ax.set_ylim(-20, 20)
        self.ax.axhline(0, color='black', linewidth=1)  # Draw horizontal axis line
        self.ax.axvline(0, color='black', linewidth=1)  # Draw vertical axis line
        self.ax.grid(True)  # Add grid for clarity
        layout.addWidget(self.canvas)

        # Button to start training after adding points
        self.train_button = QPushButton("Train Model")
        self.train_button.clicked.connect(self.train_model)
        layout.addWidget(self.train_button)

        # Clear button to reset points
        self.clear_button = QPushButton("Clear Points")
        self.clear_button.clicked.connect(self.clear_points)
        layout.addWidget(self.clear_button)

        # Connect click event on canvas
        self.canvas.mpl_connect("button_press_event", self.onclick)

    def update_class_count(self):
        # Update the number of classes based on dropdown selection
        self.num_classes = int(self.class_count_selector.currentText())
        self.update_class_selector()  # Update class dropdown options

    def update_class_selector(self):
        # Update the class selection dropdown based on the number of classes
        self.class_selector.clear()
        for i in range(self.num_classes):
            self.class_selector.addItem(f"Class {i}")

    def select_class(self, index):
        self.current_class = index

    def onclick(self, event):
        # Capture click on plot and add point with selected class
        if event.xdata is not None and event.ydata is not None:
            self.points.append((event.xdata, event.ydata, self.current_class))
            self.ax.plot(event.xdata, event.ydata, 'o', label=f"Class {self.current_class}", color=f"C{self.current_class}")
            self.canvas.draw()

###################################epochs#########################
    def train_model(self):
        # Prepare X and y from clicked points
        X = [[p[0], p[1]] for p in self.points]
        y = [[1 if p[2] == i else 0 for i in range(self.num_classes)] for p in self.points]
        
        # Call the training function
        weights, biases = train_multicategory_perceptron(X, y, self.num_classes, learning_rate=0.1, epochs=50)
        
        # Plot decision boundary on the same canvas
        plot_decision_boundary_multiclass(self.ax, X, y, weights, biases, self.num_classes)
        self.canvas.draw()

    def clear_points(self):
        # Clear points and reset the plot
        self.points = []
        self.ax.clear()
        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(-20, 20)
        self.ax.axhline(0, color='black', linewidth=1)
        self.ax.axvline(0, color='black', linewidth=1)
        self.ax.grid(True)
        self.canvas.draw()

# Run the application
app = QApplication(sys.argv)
window = InteractivePlot()
window.show()
sys.exit(app.exec_())
