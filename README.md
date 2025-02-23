# Emotion Detection using Streamlit

## 📌 Overview
This project is a **real-time Emotion Detection App** built with **Streamlit** and **OpenCV**. It uses a **pre-trained deep learning model** to recognize emotions from live camera feed.

---
<div align="center"> <img src="https://media.giphy.com/media/Z9kNOx4hrUx68/giphy.gif" width="60%" alt="Emotion Detection Animation"> </div>

## 🚀 Features
- 🎥 **Real-time webcam support**
- 🤖 **Deep learning model for emotion recognition**
- 📷 **Upload an image for analysis**
- 📊 **User-friendly UI powered by Streamlit**

---

## 📂 Project Structure
```
📦 emotion_detection
├── 📂 models              # Pre-trained deep learning models
│   ├── emotion_model.h5   # Trained emotion detection model
├── 📂 images              # Sample images
├── 📄 app.py              # Main Streamlit app
├── 📄 requirements.txt    # Dependencies
├── 📄 README.md           # Documentation
```

---

## 🔧 Installation & Setup
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-repo/emotion-detection.git
cd emotion-detection
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application
```bash
streamlit run app.py
```

---

## 📷 How to Use
1. **Allow camera permissions** when prompted.
2. The app will start detecting emotions in real time.
3. You can also **upload an image** for analysis.

---
## 📸 How It Works
🔹 The app accesses your webcam using OpenCV
🔹 A pre-trained Deep Learning model detects emotions
🔹 Face bounding box + Emotion label with confidence score is displayed
🔹 Streamlit provides a smooth user interface

---

###💡 Supported Emotions:
😃 Happy | 😢 Sad | 😠 Angry | 😮 Surprised | 😐 Neutral


## 🛠 Troubleshooting
### 🔴 Camera Not Working
- **Check Camera Index:** Update `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`.
- **Run without Streamlit:**
  ```python
  import cv2
  cap = cv2.VideoCapture(0)
  if not cap.isOpened():
      print("Cannot open camera")
      exit()
  ret, frame = cap.read()
  cv2.imshow("Test", frame)
  cv2.waitKey(0)
  cap.release()
  cv2.destroyAllWindows()
  ```
- **Linux Users:**
  ```bash
  sudo chmod 777 /dev/video0
  ```
- **Windows Users:** Ensure "Allow apps to access your camera" is enabled in Privacy Settings.

### 🔴 Error: `libGL.so.1: cannot open shared object file`
Run the following command:
```bash
sudo apt-get install libgl1-mesa-glx
```

### 🔴 Streamlit Deployment Issues
If deploying to a cloud platform:
- Cloud servers **do not have direct webcam access**.
- Use **image upload mode** instead.
- Check **log errors in the deployment dashboard**.

---

## 📜 License
This project is licensed under the MIT License.

---

## 👨‍💻 Author
[Akshay Garg](https://github.com/Akshay-Garg-0805)

Feel free to contribute! 🚀

