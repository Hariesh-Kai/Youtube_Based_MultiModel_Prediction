import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import yt_dlp
from spellchecker import SpellChecker
from hello import fetch_trending_songs

# Load Models
emotion_model = tf.keras.models.load_model("D:/Music_v3/Emotion_Based_Music_Player_with_Age_Preference/emotion_model.h5")
age_model = load_model("D:/Music_v3/Emotion_Based_Music_Player_with_Age_Preference/age_classifier_model.h5")
gender_model = load_model("D:/Music_v3/Emotion_Based_Music_Player_with_Age_Preference/gender_classifier_model.h5")
indoor_model = load_model("D:/Music_v3/Emotion_Based_Music_Player_with_Age_Preference/indoor_model.h5")  # Indoor scene classifier

# Label encoder for age categories
age_categories = ["Child", "Teenager", "Adult", "Older Adult"]
label_encoder = LabelEncoder()
label_encoder.fit(age_categories)

# Indoor scene labels (from MIT Indoor Scenes dataset)
indoor_labels = {
    'airport_inside': 0, 'artstudio': 1, 'auditorium': 2, 'bakery': 3, 'bar': 4, 'bathroom': 5,
    'bedroom': 6, 'bookstore': 7, 'bowling': 8, 'buffet': 9, 'casino': 10, 'children_room': 11,
    'church_inside': 12, 'classroom': 13, 'cloister': 14, 'closet': 15, 'clothingstore': 16,
    'computerroom': 17, 'concert_hall': 18, 'corridor': 19, 'deli': 20, 'dentaloffice': 21,
    'dining_room': 22, 'elevator': 23, 'fastfood_restaurant': 24, 'florist': 25, 'gameroom': 26,
    'garage': 27, 'greenhouse': 28, 'grocerystore': 29, 'gym': 30, 'hairsalon': 31, 'hospitalroom': 32,
    'inside_bus': 33, 'inside_subway': 34, 'jewelleryshop': 35, 'kindergarden': 36, 'kitchen': 37,
    'laboratorywet': 38, 'laundromat': 39, 'library': 40, 'livingroom': 41, 'lobby': 42,
    'locker_room': 43, 'mall': 44, 'meeting_room': 45, 'movietheater': 46, 'museum': 47,
    'nursery': 48, 'office': 49, 'operating_room': 50, 'pantry': 51, 'poolinside': 52, 'prisoncell': 53,
    'restaurant': 54, 'restaurant_kitchen': 55, 'shoeshop': 56, 'stairscase': 57, 'studiomusic': 58,
    'subway': 59, 'toystore': 60, 'trainstation': 61, 'tv_studio': 62, 'videostore': 63,
    'waitingroom': 64, 'warehouse': 65, 'winecellar': 66
}
# Reverse mapping: index to label name
indoor_labels_rev = {v: k for k, v in indoor_labels.items()}

# Dictionary mapping each indoor scene label to a recommended music genre
env_to_genre = {
    "airport_inside": "Ambient lounge",
    "artstudio": "Experimental indie pop",
    "auditorium": "Classical orchestral",
    "bakery": "Light acoustic pop",
    "bar": "Upbeat pop rock jazz",
    "bathroom": "Calming ambient",
    "bedroom": "Lo fi chillhop soft RnB",
    "bookstore": "Classical acoustic",
    "bowling": "Retro upbeat pop",
    "buffet": "Ambient chill",
    "casino": "Lounge electronic",
    "children_room": "Children music",
    "church_inside": "Gospel choral classical",
    "classroom": "Classical instrumental",
    "cloister": "Ambient meditative",
    "closet": "Lo fi",
    "clothingstore": "Trendy pop",
    "computerroom": "Electronic ambient",
    "concert_hall": "Classical orchestral",
    "corridor": "Ambient chill",
    "deli": "Light pop",
    "dentaloffice": "Calming ambient",
    "dining_room": "Jazz soft rock",
    "elevator": "Elevator music",
    "fastfood_restaurant": "Upbeat pop",
    "florist": "Acoustic pop",
    "gameroom": "Energetic electronic",
    "garage": "Rock",
    "greenhouse": "Nature sounds",
    "grocerystore": "Mainstream pop",
    "gym": "Dance hip hop energetic pop",
    "hairsalon": "Upbeat pop",
    "hospitalroom": "Minimal ambient",
    "inside_bus": "Pop rock",
    "inside_subway": "Ambient chill",
    "jewelleryshop": "Lounge electronic",
    "kindergarden": "Children music",
    "kitchen": "Light pop",
    "laboratorywet": "Experimental",
    "laundromat": "Chillout",
    "library": "Classical instrumental",
    "livingroom": "Lo fi",
    "lobby": "Ambient",
    "locker_room": "Energetic workout",
    "mall": "Mainstream pop",
    "meeting_room": "Soft instrumental",
    "movietheater": "Soundtrack orchestral",
    "museum": "Ambient classical",
    "nursery": "Soft lullabies",
    "office": "Instrumental focus",
    "operating_room": "Minimal ambient",
    "pantry": "Light pop",
    "poolinside": "Chillout",
    "prisoncell": "Dark ambient",
    "restaurant": "Upbeat pop",
    "restaurant_kitchen": "Instrumental cooking",
    "shoeshop": "Trendy pop",
    "stairscase": "Ambient",
    "studiomusic": "Experimental",
    "subway": "Ambient chill",
    "toystore": "Children music",
    "trainstation": "Ambient pop",
    "tv_studio": "Electronic",
    "videostore": "Retro",
    "waitingroom": "Easy listening",
    "warehouse": "Ambient industrial",
    "winecellar": "Classical wine bar"
}

def recommend_genre(environment_label):
    """
    Given an environment label, return the recommended music genre.
    If the label is not found, return a default recommendation.
    """
    return env_to_genre.get(environment_label, "Default ambient")

def preprocess_emotion_image(image):
    img = Image.fromarray(image).convert('L').resize((64, 64))
    img_rgb = Image.new('RGB', img.size)
    img_rgb.paste(img)
    return np.array(img_rgb).reshape(1, 64, 64, 3) / 255.0

def preprocess_age_image(frame):
    img = cv2.resize(frame, (64, 64)) / 255.0
    return np.expand_dims(img, axis=0)

def preprocess_gender_image(frame):
    img = cv2.resize(frame, (64, 64)) / 255.0
    return np.expand_dims(img, axis=0)

# Preprocessing for indoor scene model (assuming an input size of 224x224)
def preprocess_indoor_image(image, target_size=(224, 224)):
    img = Image.fromarray(image).resize(target_size)
    img_array = np.array(img).astype("float32") / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_age_category(frame):
    img = preprocess_age_image(frame)
    prediction = age_model.predict(img)
    return label_encoder.inverse_transform([np.argmax(prediction)])[0]

def predict_emotion(image):
    img_arr = preprocess_emotion_image(image)
    prediction = emotion_model.predict(img_arr)
    emotions = ["angry", "contempt", "disgust", "fear", "happiness", "neutrality", "sadness", "surprise"]
    return emotions[np.argmax(prediction)]

def predict_gender(frame):
    img = preprocess_gender_image(frame)
    prediction = gender_model.predict(img)
    return "Female" if prediction[0][0] > 0.5 else "Male"

def predict_indoor_scene(image):
    img_arr = preprocess_indoor_image(image)
    prediction = indoor_model.predict(img_arr)
    predicted_index = np.argmax(prediction)
    return indoor_labels_rev[predicted_index]

def get_audio_url(video_id):
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {'format': 'bestaudio/best', 'quiet': True, 'extractaudio': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=False)
        return info_dict.get('url', None)

st.header("Emotion, Age, Gender & Environment-Based Music Player 🎶")
spell = SpellChecker()

# Step 1: Age & Gender Detection
st.write("Detecting age and gender...")
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    age = predict_age_category(frame)
    gender = predict_gender(frame)
    st.success(f"Predicted Age Category: {age}")
    st.success(f"Predicted Gender: {gender}")
cap.release()

# Language Input (Dropdown Selection)
language_options = ["English", "Hindi", "Tamil", "Spanish", "French", "German", "Chinese", "Japanese"]
languages_selected = st.multiselect("Select your preferred languages:", language_options, default=["English"])

# Step 2: Image Upload for Emotion & Indoor Scene Detection
upload_choice = st.radio("Select an option to upload an image:", ("Use Camera", "Upload Photo"))
placeholder = st.empty()
uploaded_image = None
if upload_choice == "Use Camera":
    uploaded_image = placeholder.camera_input("Capture an image for emotion and environment detection")
elif upload_choice == "Upload Photo":
    uploaded_image = placeholder.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image and languages_selected:
    try:
        image = np.array(Image.open(uploaded_image))
        detected_emotion = predict_emotion(image)
        # Predict indoor scene but do not display it
        detected_indoor = predict_indoor_scene(image)
        recommended_genre = recommend_genre(detected_indoor)
        st.success(f"Detected Emotion: {detected_emotion}")
        st.success(f"Recommended Music Genre: {recommended_genre}")
        st.session_state.update({
            "emotion_detected": detected_emotion,
            "recommended_genre": recommended_genre,
            "languages": languages_selected,
            "age": age,
            "gender": gender,
            "process_completed": True
        })
    except Exception as e:
        st.error(f"Error processing the image: {e}")

# Step 3: Recommend & Play Songs
if st.session_state.get("emotion_detected"):
    st.subheader(f"Emotion Detected: {st.session_state['emotion_detected']}")
    st.write(f"Preferred Languages: {', '.join(st.session_state['languages'])}")
    st.write(f"User Age: {st.session_state['age']}")
    st.write(f"User Gender: {st.session_state['gender']}")
    st.write(f"Recommended Genre: {st.session_state.get('recommended_genre', 'Default ambient')}")

    if "songs" not in st.session_state:
        songs = fetch_trending_songs(
            st.session_state["emotion_detected"], 
            st.session_state["age"], 
            st.session_state["gender"],
            st.session_state["languages"],
            st.session_state["recommended_genre"],
            api_key='AIzaSyAYmxZ0Wg5ZsTTUJ3cijr_IrzjnJSQcy8U'
        )
        st.session_state["songs"] = songs

    st.subheader("Trending Songs:")
    for song in st.session_state["songs"]:
        st.write(f"**{song['title']}**")
        audio_url = get_audio_url(song['video_id'])
        if audio_url:
            st.audio(audio_url, format="audio/mp4")
        else:
            st.warning("Audio not available or duration is less than 2 minutes.")

# Optional: Example usage of the recommend_genre function
if __name__ == "__main__":
    test_labels = ["airport_inside", "bar", "library", "gym", "restaurant", "unknown_env"]
    for label in test_labels:
        genre = recommend_genre(label)
        print(f"Environment: {label} -> Recommended Genre: {genre}")
