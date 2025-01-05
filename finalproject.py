import matplotlib.pyplot as plt

# Sample data for Training and Validation Accuracy
epochs = list(range(1, 31))  # Assuming 30 epochs
training_accuracy = [0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69,
                     0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79,
                     0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89,]
validation_accuracy = [0.50, 0.5110, 0.5221, 0.5331, 0.5441, 0.5552, 0.5662, 0.5772, 0.5883, 0.5993,
                       0.6103, 0.6214, 0.6324, 0.6434, 0.6545, 0.6655, 0.6766, 0.6876, 0.6986, 0.7097,
                       0.7207, 0.7317, 0.7428, 0.7538, 0.7648, 0.7759, 0.7869, 0.7979, 0.8090, 0.8200]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, training_accuracy, label='Training Accuracy', marker='o')
plt.plot(epochs, validation_accuracy, label='Validation Accuracy', marker='o', linestyle='--')

# Adding labels and title
plt.title('Training and Validation Accuracy vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Display the plot
plt.show()
