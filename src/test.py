import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras import layers, models


# ---------------------------------------------
# 1. Load Data
# ---------------------------------------------

column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target"
]

df = pd.read_csv("processed.cleveland.data", header=None, names=column_names)

print("Original dataset shape:", df.shape)

# ---------------------------------------------
# 2. Cleaning Missing Data
# ---------------------------------------------

df = df.replace("?", np.nan)
df = df.dropna()        # remove rows with missing values
df = df.astype(float)   # convert everything to numeric

print("Cleaned dataset shape:", df.shape)


# ---------------------------------------------
# 3. Feature-target split
# ---------------------------------------------

X = df.drop("target", axis=1).values
y = df["target"].values

print("Unique class labels:", np.unique(y))

# ---------------------------------------------
# 4. Scaling the data
# ---------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------
# 5. Train / Validation / Test split
# ---------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape)

# ---------------------------------------------
# 6. Build Neural Network
# ---------------------------------------------

model = models.Sequential([
    layers.Dense(32, activation="relu", input_shape=(X_train.shape[1],),
                 kernel_initializer="he_normal"),
    layers.Dense(16, activation="relu", kernel_initializer="he_normal"),
    layers.Dense(8, activation="relu", kernel_initializer="he_normal"),
    layers.Dense(5, activation="softmax")  # 5 classes: 0–4
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------------------------------------
# 7. Train the model
# ---------------------------------------------

history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=16,
    validation_data=(X_val, y_val),
    verbose=1
)

# ---------------------------------------------
# 8. Plot training curves
# ---------------------------------------------

plt.figure(figsize=(14, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"])

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train", "Validation"])

plt.show()

# ---------------------------------------------
# 9. Evaluate on test set
# ---------------------------------------------

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print("\nTest Accuracy:", test_acc)

# ---------------------------------------------
# 10. Confusion Matrix
# ---------------------------------------------

y_pred = np.argmax(model.predict(X_test), axis=1)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))