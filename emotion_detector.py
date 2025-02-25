import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
import tensorflow as tf

def load_model():
    print("Loading model...")
    # Load model architecture from JSON
    with open('model_architecture.json', 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    
    # Load weights
    model.load_weights('facial_emotion_recognition_weights.h5')
    print("Model loaded successfully!")
    return model

def preprocess_face(face_img):
    try:
        # Convert BGR to RGB
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Enhance contrast using CLAHE
        lab = cv2.cvtColor(face_img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        face_img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        # Resize to model's expected input size
        face_img = cv2.resize(face_img, (160, 160))
        
        # Normalize
        face_img = face_img.astype(np.float32) / 255.0
        
        # Add batch dimension
        face_img = np.expand_dims(face_img, axis=0)
        return face_img
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        return None

def get_emotion(prediction):
    emotions = ["Anger", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprised"]
    
    # Get probabilities for each emotion
    probs = prediction[0]
    
    # Apply some rules to make it more sensitive to happy expressions
    happy_idx = emotions.index("Happy")
    neutral_idx = emotions.index("Neutral")
    
    # If probability of happy is close to neutral, boost it slightly
    if probs[happy_idx] > 0.2 and probs[happy_idx] < probs[neutral_idx]:
        if probs[happy_idx] / probs[neutral_idx] > 0.7:  # If happy is at least 70% of neutral
            probs[happy_idx] *= 1.2  # Boost happy probability by 20%
    
    # Get the highest probability emotion
    max_idx = np.argmax(probs)
    emotion = emotions[max_idx]
    probability = probs[max_idx]
    
    return emotion, probability, probs

def get_emotion_color(emotion):
    color_map = {
        'Anger': (0, 0, 255),      # Red
        'Contempt': (255, 0, 255), # Magenta
        'Disgust': (0, 140, 255),  # Orange
        'Fear': (0, 69, 255),      # Light Red
        'Happy': (0, 255, 0),      # Green
        'Neutral': (255, 255, 0),  # Cyan
        'Sad': (255, 0, 0),        # Blue
        'Surprised': (255, 255, 0) # Yellow
    }
    return color_map.get(emotion, (255, 255, 255))

def main():
    try:
        model = load_model()
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        print("Opening camera...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print("Could not open camera! Trying backup method...")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Camera failed to open!")
                return
        
        print("Camera opened successfully!")
        print("Press 'q' to quit")
        
        # Set camera properties for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            frame_count += 1
            if frame_count % 2 != 0:
                continue
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces with adjusted parameters
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(60, 60),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            for (x, y, w, h) in faces:
                # Add padding to get more of the face
                padding_top = int(h * 0.2)
                padding_bottom = int(h * 0.1)
                padding_sides = int(w * 0.1)
                
                # Calculate padded coordinates
                y1 = max(0, y - padding_top)
                y2 = min(frame.shape[0], y + h + padding_bottom)
                x1 = max(0, x - padding_sides)
                x2 = min(frame.shape[1], x + w + padding_sides)
                
                # Extract and preprocess face
                face_roi = frame[y1:y2, x1:x2]
                preprocessed_face = preprocess_face(face_roi)
                
                if preprocessed_face is not None:
                    # Predict emotion
                    prediction = model.predict(preprocessed_face, verbose=0)
                    emotion, probability, all_probs = get_emotion(prediction)
                    
                    # Get color for emotion
                    color = get_emotion_color(emotion)
                    
                    # Draw rectangle around face
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Display emotion and probability
                    text = f"{emotion}: {probability:.2f}"
                    cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.9, color, 2)
                    
                    # Show top 2 emotions if they're close
                    sorted_indices = np.argsort(all_probs)[::-1]
                    emotions = ["Anger", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprised"]
                    if all_probs[sorted_indices[1]] > 0.3:  # If second emotion has >30% probability
                        second_emotion = emotions[sorted_indices[1]]
                        second_prob = all_probs[sorted_indices[1]]
                        text2 = f"{second_emotion}: {second_prob:.2f}"
                        cv2.putText(frame, text2, (x1, y2+25), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.7, get_emotion_color(second_emotion), 2)
            
            # Display frame
            cv2.imshow('Emotion Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        print("Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
