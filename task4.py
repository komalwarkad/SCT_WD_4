import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load images and extract features
def load_hand_gesture_data(base_path):
    X = []
    y = []
    gesture_labels = os.listdir(base_path)

    for label in gesture_labels:
        gesture_path = os.path.join(base_path, label)
        if not os.path.isdir(gesture_path):
            continue
        for file in os.listdir(gesture_path):
            file_path = os.path.join(gesture_path, file)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (64, 64))  # Resize to 64x64
                img_flatten = img.flatten()
                X.append(img_flatten)
                y.append(label)
    return np.array(X), np.array(y)

# Step 2: Load the dataset
X, y = load_hand_gesture_data("hand_gestures_dataset")  # Change folder name as per your dataset

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Predict and evaluate
y_pred = model.predict(X_test)

# Step 6: Display Results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))