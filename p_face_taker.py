import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import json
import cv2
import os
from typing import Optional, Dict
import logging
from settings.settings import CAMERA, FACE_DETECTION, TRAINING, PATHS
from picamera2 import Picamera2
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directory(directory: str) -> None:
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
    except OSError as e:
        logger.error(f"Error creating directory {directory}: {e}")
        raise

def get_face_id(directory: str) -> int:
    try:
        if not os.path.exists(directory):
            return 1
            
        user_ids = []
        for filename in os.listdir(directory):
            if filename.startswith('Users-'):
                try:
                    number = int(filename.split('-')[1])
                    user_ids.append(number)
                except (IndexError, ValueError):
                    continue
                    
        return max(user_ids + [0]) + 1
    except Exception as e:
        logger.error(f"Error getting face ID: {e}")
        raise

def save_name(face_id: int, face_name: str, filename: str) -> None:
    try:
        names_json: Dict[str, str] = {}
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as fs:
                    content = fs.read().strip()
                    if content:  # Only try to load if file is not empty
                        names_json = json.loads(content)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in {filename}, starting fresh")
                names_json = {}
        
        names_json[str(face_id)] = face_name
        
        with open(filename, 'w') as fs:
            json.dump(names_json, fs, indent=4, ensure_ascii=False)
        logger.info(f"Saved name mapping for ID {face_id}")
    except Exception as e:
        logger.error(f"Error saving name mapping: {e}")
        raise

def initialize_camera() -> Optional[Picamera2]:
    try:
        camera = Picamera2()
        camera.configure(camera.create_still_configuration())
        camera.start()
        return camera
    except Exception as e:
        logger.error(f"Error initializing camera: {e}")
        return None

if __name__ == '__main__':
    try:
        # Initialize
        create_directory(PATHS['image_dir'])
        face_cascade = cv2.CascadeClassifier(PATHS['cascade_file'])
        if face_cascade.empty():
            raise ValueError("Error loading cascade classifier")
        
        camera = initialize_camera()
        if camera is None:
            raise ValueError("Failed to initialize Pi camera")
            
        # Get user info
        face_name = input('\nEnter user name and press <return> -->  ').strip()
        if not face_name:
            raise ValueError("Name cannot be empty")
            
        face_id = get_face_id(PATHS['image_dir'])
        save_name(face_id, face_name, PATHS['names_file'])
        
        logger.info(f"Initializing face capture for {face_name} (ID: {face_id})")
        logger.info("Look at the camera and wait...")
        
        count = 0
        while True:
            # Capture image from PiCamera2
            img = camera.capture_array()
            
            if img is None:
                logger.warning("Failed to grab frame")
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=FACE_DETECTION['scale_factor'],
                minNeighbors=FACE_DETECTION['min_neighbors'],
                minSize=FACE_DETECTION['min_size']
            )
            
            for (x, y, w, h) in faces:
                if count < TRAINING['samples_needed']:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    face_img = gray[y:y+h, x:x+w]
                    img_path = f'./{PATHS["image_dir"]}/Users-{face_id}-{count+1}.jpg'
                    cv2.imwrite(img_path, face_img)
                    
                    count += 1
                    
                    progress = f"Capturing: {count}/{TRAINING['samples_needed']}"
                    cv2.putText(img, progress, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    break
            
            cv2.imshow('Face Capture', img)
            
            if cv2.waitKey(100) & 0xFF == 27:  # ESC key
                break
            if count >= TRAINING['samples_needed']:
                break
                
        logger.info(f"Successfully captured {count} images")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        
    finally:
        if 'camera' in locals():
            camera.close()
        cv2.destroyAllWindows()
