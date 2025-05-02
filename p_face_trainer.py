import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import cv2
import numpy as np
from PIL import Image
import os
import logging
import mediapipe as mp
from settings.settings import PATHS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.6)

def get_images_and_labels(path: str):

    try:
        imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        faceSamples = []
        ids = []
        
        for imagePath in imagePaths:
            if not os.path.exists(imagePath):
                logger.warning(f"Image file not found: {imagePath}")
                continue

            try:
                PIL_img = Image.open(imagePath).convert('RGB')
                img_numpy = np.array(PIL_img, 'uint8')
            except Exception as e:
                logger.error(f"Failed to load image {imagePath}: {e}")
                continue

            if img_numpy is None or img_numpy.size == 0:
                logger.warning(f"Skipping empty image: {imagePath}")
                continue

            id = int(os.path.split(imagePath)[-1].split("-")[1])

            # Convert to RGB for Mediapipe processing
            results = mp_face_detection.process(cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR))

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = img_numpy.shape
                    x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                    
                    face_roi = img_numpy[y:y+h, x:x+w]
                    if face_roi.size == 0:
                        logger.warning(f"Empty face region detected in {imagePath}, skipping.")
                        continue

                    face_gray = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
                    faceSamples.append(face_gray)
                    ids.append(id)

        return faceSamples, ids
    except Exception as e:
        logger.error(f"Error processing images: {e}")
        raise

if __name__ == "__main__":
    try:
        logger.info("Starting face recognition training...")
        
        # Initialize face recognizer
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Get training data
        faces, ids = get_images_and_labels(PATHS['image_dir'])
        
        if not faces or not ids:
            raise ValueError("No training data found")
            
        # Train the model
        logger.info("Training model...")
        recognizer.train(faces, np.array(ids))
        
        # Save the model
        recognizer.write(PATHS['trainer_file'])
        logger.info(f"Model trained with {len(np.unique(ids))} faces")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
