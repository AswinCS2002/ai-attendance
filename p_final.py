import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import cv2
from picamera2 import Picamera2
import numpy as np
import time
import json
import os
import logging
import dlib
import pyttsx3
from datetime import datetime, timedelta
from openpyxl import Workbook, load_workbook
from scipy.spatial import distance as dist
from imutils import face_utils

from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account

# Constants
SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT", "account name.json")
FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "your-folder-id")
FILE_NAME = "attendance.xlsx"

# Authenticate and create the Drive API client
SCOPES = ["https://www.googleapis.com/auth/drive"]
creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
drive_service = build("drive", "v3", credentials=creds)


# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
ATTENDANCE_FILE = "attendance.xlsx"
TRAINER_FILE = "trainer.yml"
NAMES_FILE = "names.json"
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# Period Timing
PERIOD_DURATION = timedelta(minutes=1)
start_time = datetime.now()
logged_attendance = set()

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Drowsiness detection constants
EYE_AR_THRESH = 0.27
MOUTH_AR_THRESH = 0.35  
DROWSY_TIME_THRESHOLD = 2

drowsiness_count = {}
alert_given = {}
drowsy_start_time = {} 

# Exit tracking
last_seen_time = {}
exit_logged = set()
EXIT_THRESHOLD = 3  # Number of seconds before considering a student "exited"

def get_existing_file_id():
    """Check if the file already exists in the Drive folder."""
    query = f"name='{FILE_NAME}' and '{FOLDER_ID}' in parents and trashed=false"
    results = drive_service.files().list(q=query, fields="files(id)").execute()
    files = results.get("files", [])
    return files[0]["id"] if files else None

def upload_or_update_file():
    """Uploads or updates the Excel file in Google Drive."""
    file_id = get_existing_file_id()
    
    file_metadata = {"name": FILE_NAME, "parents": [FOLDER_ID]}
    media = MediaFileUpload(FILE_NAME, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    if file_id:
        # Update existing file
        drive_service.files().update(fileId=file_id, media_body=media).execute()
        print(f"Updated '{FILE_NAME}' in Google Drive.")
    else:
        # Upload new file
        drive_service.files().create(body=file_metadata, media_body=media, fields="id").execute()
        print(f"Uploaded '{FILE_NAME}' to Google Drive.")

# Functions
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[3], mouth[9])
    C = dist.euclidean(mouth[4], mouth[8])
    D = dist.euclidean(mouth[0], mouth[6])
    return (A + B + C) / (3 * D)

def create_attendance_file():
    if not os.path.exists(ATTENDANCE_FILE):
        wb = Workbook()
        ws = wb.active
        ws.title = "Attendance_Log"
        ws.append(["Name", "Entry Time", "Status", "Drowsiness Status", "Period", "Exit Time"])
        wb.save(ATTENDANCE_FILE)
        logger.info("Created attendance.xlsx file.")

def load_names():
    if os.path.exists(NAMES_FILE):
        with open(NAMES_FILE, 'r') as fs:
            content = fs.read().strip()
            return json.loads(content) if content else {}
    return {}

def log_attendance(name: str, status: str, drowsiness_count: int, period: int):
    if (name, period) in logged_attendance:
        return
    
    logged_attendance.add((name, period))
    current_time = datetime.now()
    try:
        create_attendance_file()
        wb = load_workbook(ATTENDANCE_FILE)
        date_str = current_time.strftime('%Y-%m-%d')
        
        if date_str not in wb.sheetnames:
            ws = wb.create_sheet(title=date_str)
            ws.append(["Name", "Entry Time", "Status", "Drowsiness Alerts", "Period", "Exit Time"])
        else:
            ws = wb[date_str]
        
        ws.append([name, current_time.strftime('%Y-%m-%d %H:%M:%S'), status, drowsiness_count, period, ""])
        wb.save(ATTENDANCE_FILE)
        logger.info(f"✅ Period {period} - Attendance logged for {name} - {status} - {drowsiness_count} alerts")
    except Exception as e:
        logger.error(f"❌ Error logging attendance: {e}")

def log_exit_time(name: str, period: int):
    """ Log the exit time for a student in the Excel sheet. """
    try:
        if not os.path.exists(ATTENDANCE_FILE):
            logger.warning("Attendance file does not exist!")
            return
        
        wb = load_workbook(ATTENDANCE_FILE)
        date_str = datetime.now().strftime('%Y-%m-%d')
        
        if date_str not in wb.sheetnames:
            logger.warning(f"No attendance logged for today ({date_str})")
            return
        
        ws = wb[date_str]
        
        for row in reversed(list(ws.iter_rows(min_row=2, values_only=False))):
            if row[0].value == name and not row[5].value:  # If exit time is empty
                row[5].value = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                wb.save(ATTENDANCE_FILE)
                logger.info(f"✅ Exit time logged for {name}")
                return

    except Exception as e:
        logger.error(f"❌ Error logging exit time: {e}")

def initialize_camera():
    picam2 = Picamera2()
    picam2.configure(picam2.create_still_configuration())  # Configures for still capture; change to video if needed
    picam2.start()
    
    # Give the camera some time to initialize
    time.sleep(2)
    
    return picam2

# Main script
if __name__ == "__main__":
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    if not os.path.exists(TRAINER_FILE):
        raise ValueError("Trainer file not found. Please train the model first.")
    recognizer.read(TRAINER_FILE)
    
    names = load_names()
    if not names:
        logger.warning("No names loaded, recognition will be limited.")
    
    picam2 = initialize_camera() 
    if picam2 is None:
        raise ValueError("Failed to initialize camera")
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    
    logger.info("Press 'ESC' to exit.")
    
    last_logged_period = 1  # Track last updated period
    
    while True:
        frame = picam2.capture_array()  # Capture a frame from the camera
        
        if frame is None:
            logger.warning("Failed to grab frame")
            continue
        
        elapsed_time = datetime.now() - start_time
        period = (elapsed_time // PERIOD_DURATION) + 1  # Increments every minute

        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for face detection
        rects = detector(gray, 0)
        
        detected_names = set()

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[36:42]
            rightEye = shape[42:48]
            mouth = shape[48:68]

            ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
            mar = mouth_aspect_ratio(mouth)
            
            #print(f"EAR for {detected_names}: {ear}")
            #print(f"MAR for {detected_names}: {mar}")


            id, confidence = recognizer.predict(gray[rect.top():rect.bottom(), rect.left():rect.right()]) if rect.width() > 0 and rect.height() > 0 else (None, 100)

            if id is not None and str(id) in names and confidence < 50:
                name = names[str(id)]
                detected_names.add(name)

                last_seen_time[name] = datetime.now()
                exit_logged.discard(name)

                if ear < EYE_AR_THRESH:  # Eyes closed
                    if name not in drowsy_start_time:
                        drowsy_start_time[name] = time.time()  # Start timing
                    elif time.time() - drowsy_start_time[name] >= DROWSY_TIME_THRESHOLD:
                        if name not in alert_given:
                            drowsiness_count[name] = 0  # Initialize count if not present
                        drowsiness_count[name] += 1  # Increment alert count
                        
                        # Print the alert count
                        print(f"⚠️ {name} has been given {drowsiness_count[name]} drowsy alerts.")

                        logger.warning(f"⚠️ {name} is drowsy! ALERT {drowsiness_count[name]} times")
                        engine.say(f"Excuse me, {name}, please be alert in the class")
                        engine.runAndWait()
                        
                        drowsy_start_time[name] = time.time()  # Reset timer after alert

                else:
                    drowsy_start_time.pop(name, None)  # Reset if eyes are open
                log_attendance(name, "Present", drowsiness_count.get(name, 0), period)

        # Log exit times
        current_time = datetime.now()
        for name in list(last_seen_time.keys()):
            if (current_time - last_seen_time[name]).seconds > EXIT_THRESHOLD and name not in detected_names:
                if name not in exit_logged:
                    log_exit_time(name, period)
                    exit_logged.add(name)
                    last_seen_time.pop(name, None)
                    
        if period != last_logged_period:
            logger.info(f"Period {period - 1} ended. Uploading attendance data to Google Drive.")
            upload_or_update_file()  # Call the upload function once per period
            last_logged_period = period  # Update the last logged period

        # Display frame
        cv2.imshow('Face Recognition & Drowsiness Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
            break
    upload_or_update_file()
    picam2.stop()  # Stop the camera when done
    cv2.destroyAllWindows()
