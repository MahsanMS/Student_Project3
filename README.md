# Student_Project3
Posture detection and correction using Python, OpenCV, and Mediapipe with a simple GUI.

# Posture Detection and Correction with Python

This project was developed during my high school years as an practice project. It aims to detect the user's sitting posture and provide feedback on how to sit correctly in order to prevent strain on the back, neck, and spine.  
It is implemented in Python and uses computer vision and machine learning techniques to analyze images and videos of the user.

---

## 🎯 Objectives
- Detect the quality of the user’s sitting posture.  
- Notify the user when sitting incorrectly and suggest the proper posture.  

---

## ⚙️ Methodology
1. **Programming Language**: Python  
2. **Input**: A file path (image or video). The program processes the file to detect posture.  
3. **Libraries Used**:  
   - `opencv` for image and video processing  
   - `mediapipe` (Pose model based on BlazePose) to detect 33 body landmarks in 3D  
   - `scikit-learn` (KNN classifier) to distinguish between side-view and front-view postures  
   - `numpy` and `pandas` for data handling  
   - `tkinter` for GUI development  
   - `json` for storing and managing logs of previous runs  

4. **Algorithm Workflow**:  
   - Detect body landmarks using **Mediapipe Pose**.  
   - Classify the perspective (side or front) using **KNN**.  
   - Apply a specific algorithm for each case (side-view vs front-view) to evaluate posture quality.  
   - Save results in the same folder as the input file.  

5. **Model Training**:  
   - **Training dataset**: Collected from internet images.  
   - **Evaluation dataset**: Self-captured photos.  
   - Features:  
     - Ratio of shoulder-to-lower-back distance  
     - Ratio of eye-to-ear distance  
   - Features stored as a CSV file and used to train the KNN model.  

6. **Evaluation**:  
   - Confusion matrix and evaluation metrics from `scikit-learn` were used to assess model performance.  

7. **Extra Features**:  
   - A history section in the GUI allows users to view previous runs.  
   - Run logs (file path, name, format) are stored in JSON format.  
   - Missing files are automatically checked and removed from the log.  

---

## 📊 Analysis
- The user’s **face must be visible** in the input image/video.  
- **Clothing style** may affect model accuracy.  
- **Video processing** is computationally expensive and puts pressure on the system.  

---

## 📝 Results
- The **Pose model** has limitations, especially in edge cases.  
- The **diagonal (angled) posture** — when the user is neither fully side-view nor fully front-view — is problematic.  
  - Algorithms designed for side or front views perform poorly in this case.  
  - Treating diagonal data as outliers was necessary, as adding them to the dataset decreased model accuracy.  

---

## 🚀 Future Improvements
- Real-time posture processing using a live camera.  
- Sending **alerts** to the user when sitting incorrectly.  
- Developing an algorithm for detecting posture quality in the **diagonal view**.  
- Expanding the dataset with diagonal samples or using a more robust classification method.  

---

## 💻 How to Run
1. Provide an input file (image or video).  
2. Run the program in Python.  
3. The processed output will be saved in the same directory as the input file.  
4. Use the GUI to interact with the program and review previous results.  

### Required Libraries
- `opencv-python`  
- `mediapipe`  
- `scikit-learn`  
- `numpy`  
- `pandas`  
- `tkinter`  
- `json` (built-in in Python)  

Install dependencies with:  
```bash
pip install opencv-python mediapipe scikit-learn numpy pandas
```

## 📌 Notes
- The current version of the program is in Persian. An English version will be added in the future.
- This project demonstrates how computer vision and machine learning can be applied to real-world health applications.
