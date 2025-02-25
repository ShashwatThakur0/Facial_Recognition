import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import cv2
from tensorflow.keras.utils import to_categorical
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants
IMAGE_SIZE = 160
BATCH_SIZE = 8
EPOCHS = 100

# Simplified emotion mapping
EMOTION_MAPPING = {
    "Anger": "Negative",
    "Contempt": "Negative",
    "Disgust": "Negative",
    "Fear": "Negative",
    "Happy": "Positive",
    "Neutral": "Neutral",
    "Sad": "Negative",
    "Surprised": "Positive"
}

EMOTION_LABELS = ["Negative", "Neutral", "Positive"]
NUM_CLASSES = len(EMOTION_LABELS)

def load_and_preprocess_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to grayscale for better face detection
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Detect face
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            # Add padding
            padding = int(w * 0.4)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img.shape[1] - x, w + 2*padding)
            h = min(img.shape[0] - y, h + 2*padding)
            img = img[y:y+h, x:x+w]
            
            # Enhance contrast using CLAHE
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            enhanced = cv2.merge((cl,a,b))
            img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
            
            # Add edge detection for better feature extraction
            edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 100, 200)
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            img = cv2.addWeighted(img, 0.7, edges, 0.3, 0)
        
        # Resize
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        return img
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")
        return None

def prepare_dataset():
    images = []
    labels = []
    
    # Load original dataset
    base_dir = "images"
    logger.info("Loading original dataset...")
    
    for person_dir in os.listdir(base_dir):
        if not person_dir.isdigit():
            continue
            
        person_path = os.path.join(base_dir, person_dir)
        if not os.path.isdir(person_path):
            continue
            
        for emotion_file in os.listdir(person_path):
            emotion_name = os.path.splitext(emotion_file)[0]
            if emotion_name not in EMOTION_MAPPING:
                continue
                
            # Map to simplified emotion
            simplified_emotion = EMOTION_MAPPING[emotion_name]
            
            image_path = os.path.join(person_path, emotion_file)
            img = load_and_preprocess_image(image_path)
            
            if img is not None:
                images.append(img)
                labels.append(EMOTION_LABELS.index(simplified_emotion))
    
    # Load synthetic dataset
    synthetic_dir = "images/synthetic"
    logger.info("Loading synthetic dataset...")
    
    if os.path.exists(synthetic_dir):
        for emotion in os.listdir(synthetic_dir):
            emotion_path = os.path.join(synthetic_dir, emotion)
            if not os.path.isdir(emotion_path):
                continue
                
            # Map synthetic emotion to our simplified categories
            if emotion == "Anger":
                simplified_emotion = "Negative"
            elif emotion == "Happy":
                simplified_emotion = "Positive"
            elif emotion == "Sad":
                simplified_emotion = "Negative"
            elif emotion == "Neutral":
                simplified_emotion = "Neutral"
            elif emotion == "Surprise":
                simplified_emotion = "Positive"
            else:
                continue
            
            for img_file in os.listdir(emotion_path):
                image_path = os.path.join(emotion_path, img_file)
                img = cv2.imread(image_path)
                
                if img is not None:
                    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                    img = img.astype(np.float32) / 255.0
                    images.append(img)
                    labels.append(EMOTION_LABELS.index(simplified_emotion))
    
    return np.array(images), np.array(labels)

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

def main():
    try:
        # Prepare dataset
        logger.info("Loading and preprocessing images...")
        X, y = prepare_dataset()
        
        # Print class distribution
        unique, counts = np.unique(y, return_counts=True)
        logger.info("Class distribution:")
        for emotion_idx, count in zip(unique, counts):
            logger.info(f"{EMOTION_LABELS[emotion_idx]}: {count}")
        
        # Calculate class weights
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y),
            y=y
        )
        class_weight_dict = dict(enumerate(class_weights))
        
        y = to_categorical(y, NUM_CLASSES)
        
        # Split dataset
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")
        
        # Create data generator with more aggressive augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.3,
            height_shift_range=0.3,
            horizontal_flip=True,
            fill_mode='nearest',
            zoom_range=0.3,
            brightness_range=[0.7, 1.3],
            shear_range=0.2,
            channel_shift_range=0.2
        )
        
        # Create and compile model
        logger.info("Creating model...")
        model = create_model()
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Save model architecture
        model_json = model.to_json()
        with open("model_architecture.json", "w") as json_file:
            json_file.write(model_json)
        
        # Callbacks with reduced patience
        callbacks = [
            ModelCheckpoint(
                'facial_emotion_recognition_weights.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        # Train model with smaller batch size
        logger.info("Training model...")
        history = model.fit(
            train_datagen.flow(X_train, y_train, batch_size=8),
            validation_data=(X_val, y_val),
            epochs=100,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        # Evaluate model
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        logger.info(f"Final validation accuracy: {val_accuracy*100:.2f}%")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
