# train_model.py (Improved Version)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import time

# --- Step 1: Load the feature dataset ---
print("📥 Loading dataset...")
df = pd.read_csv("voice_features.csv")

# Drop any missing rows
df.dropna(inplace=True)

# Separate features and labels
X = df.drop("emotion", axis=1)
y = df["emotion"]

# Encode labels (e.g., happy -> 0, sad -> 1, etc.)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Save encoder for Streamlit use
with open("label_encoder.pkl", "wb") as file:
    pickle.dump(encoder, file)

# --- Step 2: Split data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# --- Step 3: Normalize features ---
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

# --- Step 4: Train improved model ---
print("🎙️ Training MLP Neural Network Classifier...")
start = time.time()

model = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    learning_rate="adaptive",
    max_iter=600,
    random_state=42,
)

model.fit(X_train_scaled, y_train)
end = time.time()
print(f"✅ Model trained in {end - start:.2f} seconds")

# --- Step 5: Evaluate performance ---
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\n🎯 Improved Accuracy: {acc * 100:.2f}%\n")

print("📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# --- Step 6: Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=encoder.classes_,
            yticklabels=encoder.classes_)
plt.title("Confusion Matrix for Voice Emotion Recognition (Improved MLP)")
plt.xlabel("Predicted Emotion")
plt.ylabel("True Emotion")
plt.tight_layout()
plt.show()

# --- Step 7: Save trained model ---
with open("voice_emotion_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\n✅ Model saved as 'voice_emotion_model.pkl'")
print("✅ Label encoder saved as 'label_encoder.pkl'")
print("✅ Scaler saved as 'scaler.pkl'")
print("\n🎉 Training complete! Your accuracy should now be significantly higher 🚀")
