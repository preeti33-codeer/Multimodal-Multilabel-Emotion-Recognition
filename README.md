# Multimodal-Multilabel-Emotion-Recognition
# A multimodal emotion recognition system that detects human emotions from text, image, and audio using pretrained deep learning models (DistilRoBERTa, ViT, Wav2Vec2). It runs in Google Colab, capturing webcam and microphone inputs, and fuses results for accurate real-time emotion detection.
# === Multimodal Emotion Recognition (Text + Webcam + Microphone) ===
# Google Colab + Gradio (UI + CSV saving + confidence graph)
# ✅ Works with Gradio >= 4.44 and Hugging Face Transformers >= 4.44

!pip install -q gradio transformers torch torchvision torchaudio librosa pandas matplotlib Pillow --upgrade

import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import io, time, os
from PIL import Image
from transformers import pipeline
import torch

# -------------------------------
# SETUP
# -------------------------------
CSV_FILE = "emotion_results.csv"
DEVICE = 0 if torch.cuda.is_available() else -1

# Create CSV if not exists
if not os.path.exists(CSV_FILE):
    pd.DataFrame(columns=[
        "timestamp","text","text_label","text_score",
        "image_label","image_score","audio_label","audio_score"
    ]).to_csv(CSV_FILE, index=False)

# -------------------------------
# Load models (once)
# -------------------------------
print("Loading models (please wait 1–2 mins)...")

text_pipe = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    device=DEVICE
)
image_pipe = pipeline(
    "image-classification",
    model="trpakov/vit-face-expression",
    top_k=5,
    device=DEVICE
)
audio_pipe = pipeline(
    "audio-classification",
    model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
    top_k=6,
    device=DEVICE
)

print("✅ Models loaded successfully.")

# -------------------------------
# Helpers
# -------------------------------
def analyze_text(text):
    try:
        res = text_pipe(text)
        if isinstance(res, list): res = res[0]
        return str(res["label"]).title(), float(res["score"])
    except:
        return "Neutral", 0.0

def analyze_image(img):
    try:
        res = image_pipe(img)
        if isinstance(res, list) and len(res) > 0:
            return str(res[0]["label"]).title(), float(res[0]["score"])
        return "Neutral", 0.0
    except:
        return "Neutral", 0.0

def analyze_audio(aud_path):
    try:
        res = audio_pipe(aud_path)
        if isinstance(res, list) and len(res) > 0:
            return str(res[0]["label"]).title(), float(res[0]["score"])
        return "Neutral", 0.0
    except:
        return "Neutral", 0.0

def save_result(row):
    df = pd.read_csv(CSV_FILE)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)

def create_chart(text_label, text_score, image_label, image_score, audio_label, audio_score):
    labels = ["Text", "Image", "Audio"]
    scores = [text_score, image_score, audio_score]
    emotions = [text_label, image_label, audio_label]

    fig, ax = plt.subplots(figsize=(6,3))
    bars = ax.bar(labels, scores)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Confidence")
    ax.set_title("Emotion Confidence per Modality")

    # Adding text labels on bars with better positioning and white color
    for b, emo, sc in zip(bars, emotions, scores):
        height = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, height / 2, f"{emo}\n{sc:.2f}",
                ha="center", va="center", fontsize=9, color='white') # Centered and white text


    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# -------------------------------
# Main Function
# -------------------------------
def process_all(user_text, user_image, user_audio):
    if not user_text or user_image is None or user_audio is None:
        return ("⚠️ Please provide all three inputs: text, webcam photo, and audio recording.",
                None, None, None)

    ts = time.strftime("%Y-%m-%d %H:%M:%S")

    t_label, t_score = analyze_text(user_text)
    i_label, i_score = analyze_image(user_image)
    a_label, a_score = analyze_audio(user_audio)

    # Save results
    row = {
        "timestamp": ts,
        "text": user_text,
        "text_label": t_label,
        "text_score": round(t_score, 3),
        "image_label": i_label,
        "image_score": round(i_score, 3),
        "audio_label": a_label,
        "audio_score": round(a_score, 3)
    }
    save_result(row)

    # Graph
    chart = create_chart(t_label, t_score, i_label, i_score, a_label, a_score)

    summary = f"""
### 🧠 Multimodal Emotion Results
🕒 **Time:** {ts}

**Text:** {t_label} ({t_score:.2f})
**Image:** {i_label} ({i_score:.2f})
**Audio:** {a_label} ({a_score:.2f})

✅ Saved to `{CSV_FILE}`
"""
    df = pd.read_csv(CSV_FILE).tail(10).reset_index(drop=True)
    return summary, chart, CSV_FILE, df

# -------------------------------
# Gradio Frontend (UI/UX)
# -------------------------------
title = "🎭 Multimodal Emotion Recognition"
desc = """
This demo captures:
- 📝 Text input (your feelings in words)
- 📷 Webcam photo (facial expression)
- 🎤 Microphone audio (tone of voice)

and predicts emotions from all three modalities using Hugging Face models.
"""

# Added some basic CSS for a slightly more creative look
css = """
.gradio-container {
    background: linear-gradient(to right, #ffecd2 0%, #fcb69f 100%);
    font-family: 'Arial', sans-serif;
}
h1, h3 {
    color: #333;
}
.gradio-h2 {
    color: #555;
}
.gr-button {
    background-color: #ff8c00;
    color: white;
    border-radius: 20px;
    padding: 10px 20px;
    font-size: 16px;
}
.gr-button:hover {
    background-color: #e07b00;
}
"""


with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo: # Added css here
    gr.Markdown(f"# {title}")
    gr.Markdown(desc)

    with gr.Row():
        with gr.Column(scale=5):
            with gr.Group(): # Using Group for a card-like effect
                gr.Markdown("## Inputs") # Added a heading for inputs
                txt = gr.Textbox(label="📝 Enter your feeling", placeholder="Type how you feel...", lines=2)
                img = gr.Image(label="📷 Capture photo", sources=["webcam"], type="pil")
                aud = gr.Audio(label="🎤 Record your voice", sources=["microphone"], type="filepath")
                btn = gr.Button("🔍 Analyze Emotions", variant="primary")

        with gr.Column(scale=5):
            with gr.Group(): # Using Group for a card-like effect
                gr.Markdown("## Results") # Added a heading for results
                out_md = gr.Markdown("Awaiting input...")
                out_chart = gr.Image(label="📊 Confidence Chart")
                out_csv = gr.File(label="⬇️ Download Results CSV")
                out_df = gr.Dataframe(label="Recent Predictions", interactive=False)

    btn.click(process_all, inputs=[txt, img, aud], outputs=[out_md, out_chart, out_csv, out_df])

demo.launch(share=True)
