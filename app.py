# app.py — Streamlit Voice Emotion Recognition 🎙️
import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import pickle
from sklearn.preprocessing import StandardScaler
import tempfile
import os

# --- Page setup ---
st.set_page_config(page_title="Voice Emotion Recognition", page_icon="🎧", layout="centered")
st.title("🎙️ Voice Emotion Recognition App")
st.write("Upload a `.wav` audio file to analyze the emotion behind the voice.")

# --- Load trained assets ---
@st.cache_resource
def load_assets():
    with open("voice_emotion_model.pkl", "rb") as file:
        model = pickle.load(file)
    with open("label_encoder.pkl", "rb") as file:
        encoder = pickle.load(file)
    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)
    return model, encoder, scaler

model, encoder, scaler = load_assets()

# --- Feature extraction function ---
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
    features = np.hstack([mfcc, chroma, contrast, zcr])
    return features

# --- File upload section ---
uploaded_file = st.file_uploader("Upload a voice sample (.wav format only)", type=["wav"])

if uploaded_file is not None:
    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    st.audio(uploaded_file, format="audio/wav")

    st.info("⏳ Extracting features and predicting emotion...")
    features = extract_features(temp_path)
    os.remove(temp_path)  # Cleanup temp file

    # Scale features
    features_scaled = scaler.transform([features])

    # Predict emotion
    prediction = model.predict(features_scaled)[0]
    emotion = encoder.inverse_transform([prediction])[0]

    # Predict probabilities for confidence
    proba = model.predict_proba(features_scaled)[0]
    confidence = np.max(proba) * 100

    # Display results
    st.success(f"🎯 Predicted Emotion: **{emotion.upper()}**")
    st.progress(int(confidence))
    st.caption(f"Confidence: {confidence:.2f}%")

    # Add interpretation
    st.markdown("### 🧠 Interpretation")
    interpretations = {
        "angry": "The voice sounds intense and forceful — possibly anger or frustration.",
        "calm": "The voice is steady and relaxed — likely calmness or composure.",
        "disgust": "The voice carries an unpleasant tone — indicating disgust.",
        "fearful": "The tone is trembling or uncertain — possibly fear or anxiety.",
        "happy": "The tone is bright and lively — expressing happiness or excitement.",
        "neutral": "A flat and balanced tone — neutral emotion detected.",
        "sad": "A low, soft tone — expressing sadness or disappointment.",
        "surprised": "The tone rises suddenly — surprise or astonishment detected."
    }
    st.write(interpretations.get(emotion, "Emotion interpretation not available."))

else:
    st.warning("👆 Please upload a `.wav` file to start emotion analysis.")

# Footer
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit, Librosa, and Scikit-learn.")
