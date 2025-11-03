# 🎙️ Voice Emotion Recognition using Machine Learning

This project detects human emotions from **voice recordings** using **audio signal processing** and **machine learning**.

## 🚀 Features
- Upload a `.wav` file and instantly get emotion predictions  
- Displays predicted **emotion**, **confidence**, and **interpretation**  
- Built using `Streamlit`, `Librosa`, and `Scikit-learn`

## 🧠 Model Workflow
1. **Feature Extraction** – Extracts MFCCs, Chroma, Spectral Contrast, and ZCR from audio  
2. **Training** – MLP Neural Network trained on the **RAVDESS dataset**  
3. **Prediction** – Real-time inference from uploaded `.wav` samples  

## 🧩 Tech Stack
- Python 🐍  
- Streamlit 🎨  
- Librosa 🎵  
- Scikit-learn 🤖  
- Pandas & NumPy  

## 📊 Dataset
**RAVDESS Emotional Speech Audio Dataset**  
🎧 [Download from Kaggle](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)

## 💻 How to Run
```bash
# Clone the repo
git clone https://github.com/your-username/VoiceSentimentAnalysis.git
cd VoiceSentimentAnalysis

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py

---

## 🪶 Step 4 — Initialize Git Repository

Run these commands in your terminal:

```bash
cd C:\Users\govin\VoiceSentimentAnalysis
git init
git add .
git commit -m "Initial commit: Voice Emotion Recognition app with Streamlit"

