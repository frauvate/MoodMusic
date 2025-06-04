import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from youtube_search import YoutubeSearch

# cache the model loading to speed up subsequent runs
@st.cache_resource
def load_emotion_model():
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, emotion_model = load_emotion_model()

# Load the model with a spinner to indicate loading
with st.spinner("Model loading, please wait..."):
    tokenizer, emotion_model = load_emotion_model()

# Function to detect emotion from text
def detect_emotion(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = emotion_model(**inputs)
    scores = outputs.logits[0].softmax(dim=0)
    labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    top_label = labels[torch.argmax(scores)]
    return top_label

if "video_offset" not in st.session_state:
    st.session_state.video_offset = 0

# Function to search YouTube for music based on mood
def search_youtube(mood):
    results = YoutubeSearch(f"{mood} music", max_results=5).to_dict()
    return results

# Streamlit app setup
st.title("🎧 Find a Song That Matches Your Mood")
st.write("How do you feel today?")

# User mood selection
seçim = st.radio("How do you want to choose your mood?", ["I want to write", "I'll choose from what's available"])

mood = None
user_input = None

if seçim == "I want to write":
    user_input = st.text_input("Write a short sentence about how you feel 🧠")
    if user_input:
        with st.spinner("Analyzing your mood..."):
            mood = detect_emotion(user_input)
            st.success(f"Detected Mood: **{mood}**")

elif seçim == "I'll choose from what's available":
    mood = st.selectbox("Choose your mood:", [
        "Happy 😊", "Romantic 💖", "Sad 😢", "Angry 😡",
        "Stressed 😰", "Energetic ⚡", "Unmotivated 😩", "Sleepy 😴", "In Love 💘"
    ])
    if mood:
        st.success(f"Selected mood: **{mood}**")


# Mood normalization and suggestion
if mood:

    genre = st.selectbox("What genre interest you the most?", [
        "Lo-fi", "Jazz", "Rock", "Classic", "Pop", "R&B", "Rap", "Instrumental", "Chill"
    ])

    st.subheader("🎵 Suggested From Youtube")
    query = f"{mood.lower()} {genre.lower()} music"
        
    videos = search_youtube(query)

    for v in videos:
        st.markdown(f"**{v['title']}**")
        st.markdown(f"[▶️ İzle](https://www.youtube.com{v['url_suffix']})")

    if st.button("🎲 You didn't like it? Try again"):
        st.session_state.video_offset += 5
        st.rerun()  # Rerun the app to show more videos

    st.write("Enjoy listening! 🎶")
