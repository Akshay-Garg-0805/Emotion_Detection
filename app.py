import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image

# Load the trained model
model = load_model("new_model.h5")

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Emotion labels (as per FER 2013 dataset)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Ensure session state initialization
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

# Streamlit UI Configuration
st.set_page_config(page_title="Emotion Detection", page_icon="😀", layout="wide")

# Sidebar Navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Live Detection", "About"], index=["Home", "Live Detection", "About"].index(st.session_state["page"]))
st.session_state["page"] = page  # Update session state

# Home Page
if page == "Home":
    
    st.markdown("<h1 style='text-align: center;'>🧠 Real-Time Emotion Detection App</h1>", unsafe_allow_html=True)

    st.image("download.jpeg")  # Add an image for a better UI

    st.markdown(
        """
        ### 🎭 **Welcome to the Emotion Detection App!**
        This app uses **Artificial Intelligence (AI) and Deep Learning** to analyze facial expressions 
        in real-time and detect human emotions. Whether you're a researcher, developer, or just curious 
        about AI, this tool provides an interactive experience with real-time **facial emotion recognition**.

        ### 🔍 **How It Works**
        1. **Facial Detection** – The app uses OpenCV to detect faces in the live camera feed.
        2. **Emotion Analysis** – A trained **Deep Learning model** analyzes your facial expression.
        3. **Live Results** – The detected emotion is displayed on the screen with a confidence percentage.

        ### 🚀 **Get Started**
        - Click **"Live Detection"** from the sidebar.
        - Toggle the **"Start Camera"** button.
        - Watch as the app detects and labels your emotions in real-time!

        ---
        **🎯 Why Use This App?**
        ✅ AI-powered real-time emotion detection  
        ✅ Simple & user-friendly interface  
        ✅ Supports multiple emotions (Happy, Sad, Angry, etc.)  
        ✅ Ideal for research, psychology, and fun experiments  

        **👨‍💻 Built with:** Python, OpenCV, TensorFlow, and Streamlit.
        """
    )

# Live Detection Page
elif page == "Live Detection":
    st.markdown("<h3 style='text-align: center;'>🎥 Real-time Emotion Detection</h3>", unsafe_allow_html=True)
    st.markdown("""<div style='text-align: center;'>
        <img src='https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExd3RueTJqeDYzbXcwbHZiOGtqcjYycnBwMnBhbGFmeTlkY2szd3JoMyZlcD12MV9naWZzX3NlYXJjaCZjdD1n/26BRzozg4TCBXv6QU/giphy.gif' width='300'>
    </div> """, unsafe_allow_html=True)
    run = st.toggle("✅ Start Camera")

    if run:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("⚠️ Failed to capture video")
                break

            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48))  # Resize correctly

                # Convert grayscale to 3-channel format
                roi_rgb = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)  # Converts (48, 48) -> (48, 48, 3)

                # Normalize the pixel values
                roi_rgb = roi_rgb.astype("float32") / 255.0  

                # Ensure batch dimension
                roi_rgb = np.expand_dims(roi_rgb, axis=0)  # Shape: (1, 48, 48, 3)

                # Pass to model
                preds = model.predict(roi_rgb)[0]
                emotion_index = np.argmax(preds)
                emotion = emotion_labels[emotion_index]
                confidence = round(float(preds[emotion_index]) * 100, 2)  # Convert to percentage

                # Draw bounding box and emotion label with confidence
                cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame_rgb, f"{emotion} ({confidence}%)", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Display frame in Streamlit
            stframe.image(frame_rgb, channels="RGB")

        cap.release()

    else:
        st.session_state["run_camera"] = False

# About Page
elif page == "About":
    st.markdown("""
        <h1 style='text-align: center; color: #ff4b4b;'>ℹ️ About This App</h1>
        
        <h3 style='color: #4CAF50;'>🎯 What is Emotion Detection?</h3>
        <p style='text-align: justify;'>Emotion detection is a branch of artificial intelligence that analyzes human facial expressions 
        to determine emotions such as happiness, sadness, anger, surprise, and more. This technology is widely used in 
        applications like mental health monitoring, customer sentiment analysis, and human-computer interaction.</p>
        
        <h3 style='color: #4CAF50;'>🚀 How Does This App Work?</h3>
        <ol>
            <li><b>Face Detection:</b> The app uses OpenCV's pre-trained <i>Haarcascade classifier</i> to detect faces in real-time.</li>
            <li><b>Emotion Recognition:</b> The detected face is processed and passed through a <i>deep learning model</i> trained on 
            the <b>FER-2013</b> dataset.</li>
            <li><b>Classification:</b> The AI model predicts the most likely emotion from the detected face.</li>
            <li><b>Live Feedback:</b> The emotion label is displayed along with a confidence score, and the detected emotion is overlaid on the live feed.</li>
        </ol>
        
        <h3 style='color: #4CAF50;'>🛠️ Technologies Used</h3>
        <ul>
            <li><b>Deep Learning:</b> The app is powered by a <i>Convolutional Neural Network (CNN)</i> trained on the <b>FER-2013 dataset</b>.</li>
            <li><b>OpenCV:</b> Used for <i>real-time face detection</i> from a live video feed.</li>
            <li><b>TensorFlow/Keras:</b> The deep learning framework used to train and load the emotion classification model.</li>
            <li><b>Streamlit:</b> A Python-based framework for creating an interactive web application.</li>
            <li><b>NumPy & PIL:</b> Used for image processing and array manipulations.</li>
        </ul>

        <h3 style='color: #4CAF50;'>📊 Supported Emotions</h3>
        <p>This app can detect the following seven emotions based on facial expressions:</p>
        <ul>
            <li>😠 <b>Angry</b></li>
            <li>🤢 <b>Disgust</b></li>
            <li>😨 <b>Fear</b></li>
            <li>😀 <b>Happy</b></li>
            <li>😐 <b>Neutral</b></li>
            <li>😢 <b>Sad</b></li>
            <li>😲 <b>Surprise</b></li>
        </ul>

        <h3 style='color: #4CAF50;'>💡 Use Cases</h3>
        <ul>
            <li><b>👩‍⚕️ Mental Health:</b> AI-powered emotion detection can assist therapists in tracking patient moods.</li>
            <li><b>🎭 Human-Computer Interaction:</b> Improve AI assistant responses by detecting user emotions.</li>
            <li><b>📊 Customer Sentiment Analysis:</b> Analyze real-time reactions of customers in retail or customer support.</li>
            <li><b>🎮 Gaming:</b> Adjust game difficulty based on a player's emotions.</li>
        </ul>

        <h3 style='color: #4CAF50;'>📩 Contact & Support</h3>
        <p>If you have any issues, feedback, or suggestions, feel free to reach out:</p>
        <ul>
            <li>🔗 Github: <b>https://github.com/Akshay-Garg-0805</b></li>
        </ul>

        <h3 style='text-align: center; color: #ff4b4b;'>Thank You for Using Our App! 😊</h3>
    """, unsafe_allow_html=True)

