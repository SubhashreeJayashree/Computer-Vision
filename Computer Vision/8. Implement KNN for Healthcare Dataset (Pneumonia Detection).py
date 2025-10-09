import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Dataset path (change this to your dataset folder)
data_path = "C:/Users/student/Desktop/Computer Vision/dataset of pnemonia"  # e.g., "C:/Users/student/Desktop/Computer Vision/dataset of pnemonia"

categories = os.listdir(data_path)  # e.g., ["NORMAL", "PNEUMONIA"]

data = []
labels = []

# Load images
for label, category in enumerate(categories):
    folder = os.path.join(data_path, category)
    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (64, 64))  # resize to 64x64
        data.append(img.flatten())       # flatten image
        labels.append(label)

data = np.array(data)
labels = np.array(labels)

print("Dataset size:", data.shape, "Labels:", len(labels))

# Train-test split with stratification to keep class balance
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42, stratify=labels
)

# Make sure n_neighbors <= number of training samples
n_neighbors = min(5, len(X_train))

# KNN classifier
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(X_train, y_train)

# Predictions
y_pred = knn.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=categories))
