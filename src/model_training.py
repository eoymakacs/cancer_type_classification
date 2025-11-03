"""
🎯 Goal:
Predict cancer type (e.g., breast, lung, kidney, etc.) using gene expression profiles.
This is a multiclass classification problem with high-dimensional features (thousands of genes).

📊 Dataset Options:

1. TCGA (The Cancer Genome Atlas)
- Large, real-world dataset
- Contains gene expression profiles (RNA-Seq) and clinical labels
- Access via GDC Data Portal or Firebrowse

2. Kaggle Example Dataset
- Pan-Cancer Gene Expression Dataset
- Preprocessed CSV, ready for ML
- ~9,000 samples, 20,000 genes, multiple cancer types
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Assuming model_training.py is in /src and data is in /data at root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_CSV_FILE = os.path.join(ROOT_DIR, "data", "data.csv")
LABEL_CSV_FILE = os.path.join(ROOT_DIR, "data", "labels.csv")


# Step 1: Load Data
# Example: Pan-Cancer dataset
# Load gene expression data
#data_df = pd.read_csv("data/data.csv", index_col=0)  # Use first column (sample_0, ...) as index
data_df = pd.read_csv(DATA_CSV_FILE, index_col=0)  # Use first column (sample_0, ...) as index
print(data_df.shape)  # Should be (num_samples, 20531) including sample index column removed
print(data_df.head())

# Load labels
#labels_df = pd.read_csv("data/labels.csv", index_col=0)  # Use sample IDs as index
labels_df = pd.read_csv(LABEL_CSV_FILE, index_col=0)  # Use sample IDs as index
print(labels_df.head())

# Step 2: Align and merge
# Ensure the samples in both files are in the same order
data_df = data_df.loc[labels_df.index]

# Add cancer_type column to features
data_df['cancer_type'] = labels_df['Class']
print(data_df.head())


# Step 3: Prepare features and labels for ML
# Features
X = data_df.drop('cancer_type', axis=1).values

# Scale gene expression (important for neural networks)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Labels
y = data_df['cancer_type'].values
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)  # One-hot for NN

# Step 4: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_cat, test_size=0.2, random_state=42
)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# Step 5: Build Neural Network
"""Since gene expression datasets are high-dimensional (~20k genes), a fully connected feedforward 
neural network (dense network) is suitable"""

input_dim = X_train.shape[1]  # number of genes (~20530)
num_classes = y_train.shape[1]  # number of cancer types

# Define model
# Dropout helps prevent overfitting on high-dimensional data.
# Softmax output is for multiclass classification.
model = Sequential([
    Dense(512, activation='relu', input_shape=(input_dim,)),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')  # output layer
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Step 6: Train the Model
# validation_split=0.2 → uses 20% of training data for validation
# Adjust epochs and batch size depending on dataset size
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=1
)

# Step 7: Evaluate the Model
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.2f}")

# Predict classes
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

""" Provides precision, recall, F1-score per cancer type
Useful for imbalanced datasets """
print(classification_report(y_true, y_pred, target_names=le.classes_))

# Step 8: Visualize Training
# Helps detect overfitting if validation accuracy diverges from training accuracy
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
