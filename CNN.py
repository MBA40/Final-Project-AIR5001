import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Example true and predicted labels (replace these with your actual results)
true_labels = np.random.randint(0, 10, 1000)  # Simulated ground truth labels
cnn_predictions = true_labels.copy()  # Simulated CNN predictions
cnn_predictions[50:150] = (cnn_predictions[50:150] + 1) % 10  # Add some CNN misclassifications

# Corrected probabilities for random predictions
lr_probabilities = [0.35] + [0.07222222] * 9  # Probabilities summing to 1 for Logistic Regression
lr_probabilities[-1] = 1 - sum(lr_probabilities[:-1])  # Adjust last value to ensure total = 1

svm_probabilities = [0.5] + [0.05555556] * 9  # Probabilities summing to 1 for SVM
svm_probabilities[-1] = 1 - sum(svm_probabilities[:-1])  # Adjust last value to ensure total = 1

# Generate predictions using corrected probabilities
lr_predictions = np.random.choice(10, 1000, p=lr_probabilities)
svm_predictions = np.random.choice(10, 1000, p=svm_probabilities)

# Define class names (CIFAR-10)
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Generate confusion matrices
cm_cnn = confusion_matrix(true_labels, cnn_predictions)
cm_lr = confusion_matrix(true_labels, lr_predictions)
cm_svm = confusion_matrix(true_labels, svm_predictions)

# Plot confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# CNN Confusion Matrix
sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[0])
axes[0].set_title('CNN Confusion Matrix', fontsize=14)
axes[0].set_xlabel('Predicted Class', fontsize=12)
axes[0].set_ylabel('True Class', fontsize=12)

# Logistic Regression Confusion Matrix
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Oranges', xticklabels=class_names, yticklabels=class_names, ax=axes[1])
axes[1].set_title('Logistic Regression Confusion Matrix', fontsize=14)
axes[1].set_xlabel('Predicted Class', fontsize=12)
axes[1].set_ylabel('True Class', fontsize=12)

# SVM Confusion Matrix
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names, ax=axes[2])
axes[2].set_title('SVM Confusion Matrix', fontsize=14)
axes[2].set_xlabel('Predicted Class', fontsize=12)
axes[2].set_ylabel('True Class', fontsize=12)

# Adjust layout for better spacing
plt.subplots_adjust(wspace=0.5)  # Increase horizontal space between subplots
plt.tight_layout()
plt.show()
