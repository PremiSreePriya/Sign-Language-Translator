# Sign-Language-Translator 🖐️
This project detects and recognizes hand gestures (like alphabets or words such as "HELLO") using hand landmark data and trains a machine learning model with Scikit-learn (RandomForestClassifier) for real-time predictions.

📦 TECH STACK

🔹 Python – Main programming language

🔹 MediaPipe – Detects 21 hand landmarks from video

🔹 OpenCV – Captures real-time webcam video

🔹 Scikit-learn – Trains ML model (RandomForestClassifier)

🔹 NumPy & Pandas – Handle and process landmark data for ML

🔧 FUNCTIONS EXPLAINED
 
 🔹collect_data.py – Opens webcam, detects hand, saves landmarks + label to dataset.csv.

 🔹train_model.py – Trains a Random Forest model using landmark data, saves it as model.pkl.

 🔹app.py – Loads the model, reads webcam feed, predicts the gesture live on screen.

 🔹dataset.csv – Stores collected landmark data and labels.

 🔹model.pkl – Final trained model used for prediction.

🚀 HOW TO RUN

✅ Step 1: Install Requirements

pip install opencv-python mediapipe pandas numpy scikit-learn

📸 Step 2: Collect Landmark Data

python collect_data.py

🔹Enter a label (e.g., A, HELLO) in the terminal and press Enter

🔹The webcam will open, then
Show your gesture to the camera

🔹Press s to save a sample and
Press q or Ctrl+C to exit

🧠 Step 3: Train the Model

python train_model.py

🔹Trains using dataset.csv.

🔹Shows accuracy score and
Saves the model as model.pkl.

👁️ Step 4: Run Real-Time Detection

python app.py

🔹Starts webcam and it
Shows predicted gesture based on current hand position.

🔹Press q to quit (or Ctrl+C if it doesn’t respond).

📌 TIPS

✔️Supports letters (A, B, etc.) and full words (like HELLO).

✔️For best accuracy, collect multiple samples of each gesture.










