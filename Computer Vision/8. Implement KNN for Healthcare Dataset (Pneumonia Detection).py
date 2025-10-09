import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Dataset path (use raw string to handle backslashes in Windows paths)
data_path = 'C:/Users/student/Desktop/Computer Vision/dataset of pnemonia'

# List categories (folder names)
categories = os.listdir(data_path)  # e.g., ["NORMAL", "PNEUMONIA"]

data = []
labels = []

# Load images and labels
for label, category in enumerate(categories):
    folder = os.path.join(data_path, category)
    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue  # Skip if image not loaded
        img = cv2.resize(img, (64, 64))  # Resize to 64x64 pixels
        data.append(img.flatten())       # Flatten image to 1D array
        labels.append(label)

data = np.array(data)
labels = np.array(labels)

print("Dataset size:", data.shape, "Labels:", len(labels))

# Split dataset into train and test sets (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# Ensure n_neighbors is at most number of training samples
n_neighbors = min(5, len(X_train))

knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(X_train, y_train)

# Predict on test set
y_pred = knn.predict(X_test)

# Evaluate and print accuracy and classification report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=categories))
