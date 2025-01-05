import numpy as np
import matplotlib.pyplot as plt

# Example data (replace with your actual data)
# Simulated CIFAR-10 image data (10x10 images for simplicity here)
# Replace this with your actual CIFAR-10 test images
images = np.random.rand(100, 32, 32, 3)  # 100 images of 32x32 with 3 color channels
true_labels = np.random.randint(0, 10, 100)  # Simulated true labels
predicted_labels = true_labels.copy()  # Simulated perfect predictions
predicted_labels[10:20] = (predicted_labels[10:20] + 1) % 10  # Add some misclassifications

# Define class names (CIFAR-10)
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Find misclassified samples
misclassified_indices = np.where(true_labels != predicted_labels)[0]

# Plot 3 sample misclassifications
num_samples = min(3, len(misclassified_indices))  # Show up to 3 misclassifications
plt.figure(figsize=(12, 6))

for i, index in enumerate(misclassified_indices[:num_samples]):
    plt.subplot(1, 3, i + 1)  # Use 1 row and 3 columns
    plt.imshow(images[index])
    plt.title(f"True: {class_names[true_labels[index]]}\nPred: {class_names[predicted_labels[index]]}")
    plt.axis('off')

plt.suptitle("Sample Misclassifications", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.9])  # Adjust layout to fit title
plt.show()
