import librosa
import numpy as np
import os
import pandas as pd

# --- Step 1: Define feature extraction function ---
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=3, offset=0.5)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
        return np.hstack([mfcc, chroma, contrast, zcr])
    except Exception as e:
        print("Error processing", file_path, ":", e)
        return None


# --- Step 2: Map emotion codes to labels ---
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}


# --- Step 3: Loop through dataset and extract features ---
DATASET_PATH = r"C:\Users\govin\VoiceSentimentAnalysis\dataset"
data = []

for actor_folder in os.listdir(DATASET_PATH):
    actor_path = os.path.join(DATASET_PATH, actor_folder)
    if not os.path.isdir(actor_path):
        continue
    for file in os.listdir(actor_path):
        if not file.endswith(".wav"):
            continue
        file_path = os.path.join(actor_path, file)
        emotion_code = file.split("-")[2]
        emotion = emotion_map.get(emotion_code)
        features = extract_features(file_path)
        if features is not None:
            data.append([*features, emotion])

# --- Step 4: Save extracted data to CSV ---
columns = [f"feature_{i}" for i in range(len(data[0]) - 1)] + ["emotion"]
df = pd.DataFrame(data, columns=columns)
df.to_csv("voice_features.csv", index=False)
print("✅ Feature extraction complete! Saved as voice_features.csv")
