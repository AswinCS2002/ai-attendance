# 🎓 Attendance and Drowsiness Monitoring System

A smart real-time Python-based system that performs face recognition to mark attendance and monitors student alertness using eye and mouth aspect ratio detection. Designed for use in classrooms and integrated with IoT (Google Drive) for remote monitoring.

---

## 📌 Key Features

- ✅ Face recognition-based student identification
- ✅ Attendance marking with time constraints
- ✅ Drowsiness detection using eye and mouth aspect ratio (EAR & MAR)
- ✅ Audio and on-screen alerts for drowsy students
- ✅ Excel sheet logging of attendance and alertness
- ✅ Ubidots integration to send data to the cloud
- ✅ Identifies unknown subjects and prevents misclassification
- ✅ Real-time monitoring via a web dashboard

---

## 🧰 Technologies Used

- Python 3.x  
- OpenCV  
- Mediapipe  
- Dlib  
- NumPy  
- pyttsx3 (text-to-speech)  
- pandas / xlwt (Excel handling)  
- `pydrive` or `Google API` for Google Drive uploads 

---
## 🚀 How to Run the Project

1. Clone the repository or download the `.zip`:
 
   ```bash
   git clone https://github.com/AswinCS2002/ai_attendance.git
   cd ai_attendance

2. Install dependencies:
     
     pip install -r requirements.txt

3. Setup Google Drive API Access:
    
    Go to Google Cloud Console

    Create a project and enable the Google Drive API

    Download credentials.json and place it in your project folder

    Authenticate via the browser when prompted

3. Run the main script:
     
     p_final.py

📂 Project Structure

ai attendance/
│
├── p_final.py                  # Main script
├── requirements.txt         # Required Python libraries
├── README.md                # Project documentation
├── .gitignore               # Ignored files
├── attendance/              # Attendance logs
└── images/                  # Sample screenshots

📊 Data Saved and Uploaded
   
   Student name

   Entry and exit time

   Class period

   Drowsiness status

  Alert message (if drowsy)

All logs are stored in an Excel file and automatically uploaded to a linked Google Drive account.

📝 Future Enhancements

   Web interface to manage and visualize data

   Admin dashboard for professors

   OTP-based student login before session

   Mobile App integration

⚠️ Warning: This project requires a Google service account key. Never upload your .json key file to GitHub. Use .gitignore to exclude it and use environment variables to access credentials securely.


## 📸 Screenshots

### ✅ Drowsiness Detection Output
![Detection](Photos/drowsiness_detected.png)

### ✅ Prototype Model
![Prototype](Photos/prototype_model.png)


##👤 Author
   Aswin Santhosh Kumar
   Electronics and Communication Engineering
   Final Year Project | B. Tech in Electronics & Communication Engineering
   [www.linkedin.com/in/aswinsk27cs]
