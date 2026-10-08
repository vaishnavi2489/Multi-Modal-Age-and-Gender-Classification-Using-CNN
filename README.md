# 👤 Multi-Modal Age & Gender Classification

An AI-powered face analysis application that estimates **age, gender, emotion, and skin type** from an uploaded face image using Python, OpenCV, DeepFace, TensorFlow, and Streamlit.

## 🚀 Live Demo

🌐 **Live Application:**  
https://multi-modal-age-and-gender-8qon.onrender.com

> **Note:** The application is hosted on Render's free instance. The first request after inactivity may take some time to start.

---

## ✨ Features

- 📷 Upload a face image
- 🧠 Age estimation
- 🚻 Gender classification
- 😊 Emotion detection
- 🧴 Skin type detection
  - Dry
  - Oily
  - Normal
  - Combination
- 💡 Personalized skincare recommendations
- 🖥️ Interactive Streamlit web interface
- 🔍 OpenCV-based face detection
- 🤖 DeepFace-based facial analysis

---

## 🧠 How It Works

### 1. Image Upload

The user uploads a JPG, JPEG, or PNG face image through the Streamlit web interface.

### 2. Face Detection and Analysis

The application uses OpenCV for face detection and DeepFace for facial analysis.

DeepFace provides:

- Estimated age
- Gender
- Dominant emotion

### 3. Skin Type Detection

The detected face image is converted into HSV color space.

The application uses:

- Brightness
- Saturation

to classify the skin type as:

- Dry
- Oily
- Normal
- Combination

### 4. Personalized Recommendations

The application generates skincare recommendations based on:

- Estimated age
- Detected emotion
- Detected skin type

---

## 🛠️ Technologies Used

- **Python**
- **Streamlit**
- **OpenCV**
- **NumPy**
- **DeepFace**
- **TensorFlow**
- **Keras**

---

## 📁 Project Structure

```text
Multi-Modal-Age-and-Gender-Classification-Using-CNN/
│
├── app.py
├── streamlit_app.py
├── requirements.txt
├── render.yaml
├── .python-version
└── README.md
▶️ How to Run Locally
**Clone the Repository**
git clone https://github.com/vaishnavi2489/Multi-Modal-Age-and-Gender-Classification-Using-CNN.git

**Navigate to the Project**
cd Multi-Modal-Age-and-Gender-Classification-Using-CNN
**
Install Dependencies**
pip install -r requirements.txt

**Run the Application**
python -m streamlit run streamlit_app.py

**📊 Sample Output**
Face analysis completed!

Estimated Age: 30
Gender: Man
Emotion: happy
Skin Type: Combination

💡 Recommendations
The application can provide recommendations such as:
- Use a lightweight moisturizer for combination skin.
- Use a gentle cleanser and lightweight moisturizer.
- Apply a calming serum for redness.
- Maintain hydration with a balanced skincare routine.
- Use an anti-aging night cream for higher age estimates.
⚠️ Limitations
- Age and gender results are AI-based estimates and may not always be accurate.
- Results can vary depending on image quality and lighting.
- Skin type detection is based on image brightness and saturation.
- The application is not intended for medical diagnosis.
🛡️ Privacy & Ethical Considerations
- Facial images are used for analysis.
- Avoid uploading sensitive biometric information without appropriate consent.
- AI predictions should be treated as estimates and not as definitive facts.
**🌐 Deployment**
The application is deployed on Render using:
- Python 3.10.11
- Streamlit
- TensorFlow
- DeepFace
- OpenCV
**👩‍💻 Author**
Vaishnavi
🔗 GitHub:
https://github.com/vaishnavi2489/Multi-Modal-Age-and-Gender-Classification-Using-CNN
