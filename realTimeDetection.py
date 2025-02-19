import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('emotionDetector.h5')

# Define emotions
EMOTIONS = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']

# Open the default camera (index 0)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convert frame to grayscale and resize
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (48, 48))
    
    # Make prediction
    prediction = model.predict(gray.reshape(1, 48, 48, 1))
    predicted_emotion = np.argmax(prediction)
    
    # Display result
    cv2.putText(frame, f"Emotion: {EMOTIONS[predicted_emotion]}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('frame', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()