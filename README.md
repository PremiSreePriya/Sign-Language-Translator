# Sign-Language-Translator 🖐️
This project detects and recognizes hand gestures (like alphabets or words such as "HELLO") using hand landmark data and trains a machine learning model with Scikit-learn (RandomForestClassifier) for real-time predictions.

📦 TECH STACK

🔹 Python               -     	Main programming language

🔹 MediaPipe            -    	Extracts 21 hand landmarks from video frames

🔹 OpenCV               -    	Real-time webcam video capture

🔹 Scikit-learn	       -      ML model (RandomForestClassifier) for gesture classification

🔹 Pandas               -     	Reads and writes dataset to CSV (dataset.csv)

🔹 NumPy	               -      Converts landmark data to numeric format for ML processing

🔹Pickle               -     	Saves/loads trained ML model

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

🔹The webcam will open

🔹Show your gesture to the camera

🔹Press s to save a sample

🔹Press q or Ctrl+C to exit

🧠 Step 3: Train the Model

python train_model.py

🔹Trains using dataset.csv.

🔹Shows accuracy score and
Saves the model as model.pkl.

👁️ Step 4: Run Real-Time Detection

python app.py

🔹Starts webcam.

🔹Shows predicted gesture based on current hand position.

🔹Press q to quit (or Ctrl+C if it doesn’t respond).

📌 TIPS

✔️Supports letters (A, B, etc.) and full words (like HELLO).

✔️For best accuracy, collect multiple samples of each gesture.

✔️Use consistent hand position and lighting during collection.









