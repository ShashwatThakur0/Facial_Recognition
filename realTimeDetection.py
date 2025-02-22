import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = load_model('facial_emotion_recognition_model.h5')

# Define the emotion labels
emotion_labels = ["Anger", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprised"]

# Function to preprocess the video frames
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
    face = cv2.resize(rgb_frame, (224, 224))
    face = face.astype("float") / 255.0
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)
    return face

# Function to use the model for emotion detection
def detect_emotion(frame, model):
    face = preprocess_frame(frame)
    preds = model.predict(face)[0]
    emotion = np.argmax(preds)
    confidence = preds[emotion]
    return emotion, confidence

# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 for the default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect emotion
    emotion, confidence = detect_emotion(frame, model)
    emotion_label = f"{emotion_labels[emotion]} ({confidence:.2f})"
    
    # Display the label on the frame
    cv2.putText(frame, emotion_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Emotion Detector', frame)
    
    # Break the loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()