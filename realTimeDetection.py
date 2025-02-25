import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
import json
import time

print("OpenCV version:", cv2.__version__)

# Load the model architecture from json
try:
    with open('model_architecture.json', 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    print("Model architecture loaded successfully")
except Exception as e:
    print(f"Error loading model architecture: {str(e)}")
    exit(1)

# Load the trained weights
try:
    model.load_weights('facial_emotion_recognition_weights.h5')
    print("Model weights loaded successfully")
except Exception as e:
    print(f"Error loading model weights: {str(e)}")
    exit(1)

# Load the face cascade classifier
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    print("Face cascade classifier loaded successfully")
except Exception as e:
    print(f"Error loading face cascade: {str(e)}")
    exit(1)

# Define the emotion labels
emotion_labels = ["Anger", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprised"]

def preprocess_face(face):
    try:
        face = cv2.resize(face, (224, 224))
        face = face.astype("float") / 255.0
        face = np.expand_dims(face, axis=0)
        return face
    except Exception as e:
        print(f"Error preprocessing face: {str(e)}")
        return None

def detect_emotion(frame, model):
    try:
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        results = []
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            # Preprocess face
            processed_face = preprocess_face(face_roi)
            if processed_face is None:
                continue
            
            # Predict emotion
            preds = model.predict(processed_face, verbose=0)[0]
            emotion = np.argmax(preds)
            confidence = preds[emotion]
            
            results.append((x, y, w, h, emotion, confidence))
        
        return results
    except Exception as e:
        print(f"Error in detect_emotion: {str(e)}")
        return []

def main():
    print("Attempting to open camera...")
    
    # Try different camera indices
    for camera_index in [0, 1]:
        try:
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # Try DirectShow
            if cap.isOpened():
                print(f"Successfully opened camera {camera_index}")
                break
        except Exception as e:
            print(f"Error with camera {camera_index}: {str(e)}")
            continue
    else:
        print("Could not open any camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Starting emotion detection... Press 'q' to quit")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Process every other frame to improve performance
            frame_count += 1
            if frame_count % 2 != 0:
                continue
                
            # Detect emotions
            results = detect_emotion(frame, model)
            
            # Draw results
            for (x, y, w, h, emotion, confidence) in results:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Display emotion and confidence
                label = f"{emotion_labels[emotion]}: {confidence*100:.1f}%"
                cv2.putText(frame, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Calculate and display FPS
            if frame_count % 30 == 0:
                fps = frame_count / (time.time() - start_time)
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Emotion Detection', frame)
            
            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print(f"Error in main loop: {str(e)}")
            break
    
    # Cleanup
    print("Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()