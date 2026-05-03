import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import joblib
import os
import torch
from transformers import Wav2Vec2FeatureExtractor, AutoModel

st.set_page_config(page_title="Music Genre Classifier", page_icon="🎶", layout="wide")

# ==========================================
# 1. Load BOTH ML Pipelines
# ==========================================
@st.cache_resource
def load_pipelines():
    # --- YAMNet Components ---
    yamnet_model = tf.keras.models.load_model('yamnet_music_model.keras', compile=False)
    # yamnet_model = tf.keras.models.load_model('yamnet_music_model.h5', compile=False)
    yamnet_scaler = joblib.load('yamnet_scaler.pkl')
    yamnet_le = joblib.load('yamnet_label_encoder.pkl')
    yamnet_extractor = hub.load('https://tfhub.dev/google/yamnet/1')
    
    # --- MERT Components ---
    mert_classifier = joblib.load('mert_mlp_model.pkl')
    mert_scaler = joblib.load('mert_scaler.pkl')
    mert_le = joblib.load('mert_label_encoder.pkl')
    
    # Load HuggingFace MERT feature extractor
    mert_processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
    mert_extractor = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
    
    return (yamnet_model, yamnet_scaler, yamnet_le, yamnet_extractor, 
            mert_classifier, mert_scaler, mert_le, mert_processor, mert_extractor)

(yamnet_model, yamnet_scaler, yamnet_le, yamnet_extractor, 
 mert_classifier, mert_scaler, mert_le, mert_processor, mert_extractor) = load_pipelines()

# ==========================================
# 2. Build the User Interface
# ==========================================
st.title("Music Genre Classifier")
st.write("Upload a `.wav` file to see how YAMNet (TensorFlow) compares against MERT (PyTorch)!")

uploaded_file = st.file_uploader("Choose an audio file...", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("Classify Genre", type="primary"):
        with st.spinner("Extracting acoustic features and analyzing (this may take a moment)..."):
            try:
                temp_filename = "temp_upload.wav"
                with open(temp_filename, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # ==========================================
                # YAMNET PREDICTION (Requires 16kHz)
                # ==========================================
                wav_16k, _ = librosa.load(temp_filename, sr=16000, mono=True)
                waveform_16k = tf.convert_to_tensor(wav_16k, dtype=tf.float32)
                _, yamnet_embs, _ = yamnet_extractor(waveform_16k)
                yamnet_embedding = tf.reduce_mean(yamnet_embs, axis=0).numpy().reshape(1, -1)
                
                yamnet_scaled = yamnet_scaler.transform(yamnet_embedding)
                
                # Extract full probability array [0]
                yamnet_pred_probs = yamnet_model.predict(yamnet_scaled)[0] 
                yamnet_pred_idx = np.argmax(yamnet_pred_probs)
                yamnet_genre = yamnet_le.inverse_transform([yamnet_pred_idx])[0]
                
                # Create a sorted DataFrame for the chart
                yamnet_df = pd.DataFrame({
                    'Genre': yamnet_le.classes_,
                    'Confidence (%)': yamnet_pred_probs * 100
                }).sort_values(by='Confidence (%)', ascending=False)

                # ==========================================
                # MERT PREDICTION (Requires 24kHz)
                # ==========================================
                wav_24k, _ = librosa.load(temp_filename, sr=24000, mono=True)
                inputs = mert_processor(wav_24k, sampling_rate=24000, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = mert_extractor(**inputs, output_hidden_states=True)
                
                mert_embedding = outputs.last_hidden_state.mean(dim=1).numpy()
                mert_scaled = mert_scaler.transform(mert_embedding)
                
                # Use predict_proba() to get the full probability array [0]
                mert_pred_probs = mert_classifier.predict_proba(mert_scaled)[0]
                mert_pred_idx = np.argmax(mert_pred_probs)
                mert_genre = mert_le.inverse_transform([mert_pred_idx])[0]
                
                # Create a sorted DataFrame for the chart
                mert_df = pd.DataFrame({
                    'Genre': mert_le.classes_,
                    'Confidence (%)': mert_pred_probs * 100
                }).sort_values(by='Confidence (%)', ascending=False)

                # ==========================================
                # SIDE-BY-SIDE DISPLAY
                # ==========================================
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.header("🧠 YAMNet Model")
                    st.success(f"Predicted Genre: **{yamnet_genre.upper()}**")
                    st.write("**Confidence Distribution:**")
                    st.bar_chart(yamnet_df.set_index('Genre'))
                
                with col2:
                    st.header("🎧 MERT Model")
                    st.info(f"Predicted Genre: **{mert_genre.upper()}**")
                    st.write("**Confidence Distribution:**")
                    st.bar_chart(mert_df.set_index('Genre'))

                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")
