import pandas as pd
import numpy as np
import serial
import time
from sklearn.semi_supervised import LabelPropagation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
file_path = r"C:\Users\Anuragine\Downloads\synthetic_spectral_data.csv"
df = pd.read_csv(file_path)

# Separate features and labels
X = df.iloc[:, :-1].values  # All spectral bands as features
y = df["Label"].values  # Labels

# Randomly remove labels to simulate a semi-supervised scenario
np.random.seed(42)
unlabeled_mask = np.random.rand(len(y)) < 0.6  # 60% of labels removed
y[unlabeled_mask] = -1  # Assign -1 to unlabeled data

# Split labeled data into train-test for evaluation
labeled_mask = y != -1
X_train, X_test, y_train, y_test = train_test_split(X[labeled_mask], y[labeled_mask], test_size=0.2, random_state=42)

# Train a semi-supervised Label Propagation model
lp_model = LabelPropagation(kernel="knn")

# Pseudo-labeling iterations (fixed to 5 iterations)
for iteration in range(1, 6):
    lp_model.fit(X, y)
    pseudo_labels = lp_model.transduction_
    new_labels = (y == -1) & (pseudo_labels != -1)
    added_labels = np.sum(new_labels)
    y[new_labels] = pseudo_labels[new_labels]
    print(f"Iteration {iteration}: Added {added_labels} pseudo-labeled samples.")
    if added_labels == 0:
        break

# Predict on test data
y_pred = lp_model.predict(X_test)

# Evaluate accuracy on labeled test data
accuracy = accuracy_score(y_test, y_pred)
print("\nFinal Model Accuracy:", round(accuracy, 3))

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Initialize Arduino connection
Arduino = serial.Serial(port='COM7', baudrate=115200, timeout=0.1)
time.sleep(2)  # Allow time for connection to establish

print("Waiting for Arduino data...")

while True:
    try:
        if Arduino.in_waiting:
            line = Arduino.readline().decode('utf-8').strip()  # Read and decode data
            if line:
                print("Received from Arduino:", line)
                try:
                    spectral_values = [float(val) for val in line.split(',')]
                    if len(spectral_values) == X.shape[1]:  # Ensure correct feature count
                        spectral_values = np.array(spectral_values).reshape(1, -1)
                        prediction = lp_model.predict(spectral_values)[0]
                        print("Prediction:", "Blood Present" if prediction == 1 else "No Blood")
                    else:
                        print("Warning: Incorrect number of features received.")
                except ValueError:
                    print("Warning: Non-numeric data received. Skipping.")
    except Exception as e:
        print("Error processing Arduino data:", e)
