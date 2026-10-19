
import os
import tempfile

import cv2
import numpy as np
import streamlit as st
from deepface import DeepFace


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Age & Gender Classification",
    page_icon="👤",
    layout="centered"
)

st.title("👤 Multi-Modal Age & Gender Classification")

st.write(
    "Analyze your face using your camera or by uploading "
    "an image to estimate age, gender, emotion, and skin type."
)


# --------------------------------------------------
# Cache DeepFace Reference
# --------------------------------------------------

@st.cache_resource
def load_deepface():
    return DeepFace


# --------------------------------------------------
# Skin Type Detection
# --------------------------------------------------

def detect_skin_type(face):
    try:
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)

        brightness = float(np.mean(hsv[:, :, 2]))
        saturation = float(np.mean(hsv[:, :, 1]))

        if brightness < 50:
            return "Dry"
        elif saturation > 100:
            return "Oily"
        elif 50 <= brightness <= 150 and 50 <= saturation <= 100:
            return "Normal"
        else:
            return "Combination"

    except Exception:
        return "Unknown"


# --------------------------------------------------
# Face Analysis
# --------------------------------------------------

def analyze_face(face):
    temp_path = None

    try:
        if face is None or face.size == 0:
            st.error("No valid image was provided.")
            return None

        # Resize very large images to reduce processing overhead.
        height, width = face.shape[:2]
        max_dimension = 1024

        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)

            face = cv2.resize(
                face,
                (
                    int(width * scale),
                    int(height * scale)
                ),
                interpolation=cv2.INTER_AREA
            )

        # Save image temporarily for DeepFace.
        with tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False
        ) as temp:
            temp_path = temp.name

        if not cv2.imwrite(temp_path, face):
            st.error("Could not prepare the image for analysis.")
            return None

        deepface = load_deepface()

        # Use OpenCV detector to avoid the heavier RetinaFace detector.
        analysis = deepface.analyze(
            img_path=temp_path,
            actions=["age", "gender", "emotion"],
            detector_backend="opencv",
            align=False,
            enforce_detection=False,
            silent=True
        )

        if isinstance(analysis, list) and analysis:
            analysis_data = analysis[0]
        elif isinstance(analysis, dict):
            analysis_data = analysis
        else:
            st.error("The analysis model returned no results.")
            return None

        analysis_data["skin_type"] = detect_skin_type(face)

        return analysis_data

    except Exception:
        st.error(
            "Face analysis failed. Please try a clear, well-lit "
            "image with one visible face. If the problem continues, "
            "the hosting server may be running out of resources."
        )

        return None

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


# --------------------------------------------------
# Skincare Recommendations
# --------------------------------------------------

def suggest_skincare(analysis):
    suggestions = []

    if not analysis:
        return ["Face analysis could not be completed."]

    emotions = analysis.get("emotion", {})
    age = analysis.get("age", 0)
    skin_type = analysis.get("skin_type", "Unknown")

    # Emotion-based suggestions.
    if emotions.get("sad", 0) > 30:
        suggestions.append(
            "Consider a gentle, hydrating skincare routine."
        )

    if emotions.get("angry", 0) > 30:
        suggestions.append(
            "Consider gentle skincare products if your skin is irritated."
        )

    # Age-based suggestions.
    try:
        age = float(age)

        if age > 40:
            suggestions.append(
                "Consider a moisturizing night cream."
            )
        elif age < 20:
            suggestions.append(
                "Consider a gentle cleanser and lightweight moisturizer."
            )
    except (ValueError, TypeError):
        pass

    # Skin-type suggestions.
    if skin_type == "Oily":
        suggestions.append(
            "Consider a gentle cleanser and oil-free moisturizer."
        )
    elif skin_type == "Dry":
        suggestions.append(
            "Consider a hydrating moisturizer."
        )
    elif skin_type == "Combination":
        suggestions.append(
            "Consider a lightweight moisturizer."
        )
    elif skin_type == "Normal":
        suggestions.append(
            "Maintain a balanced skincare routine."
        )

    if not suggestions:
        suggestions.append(
            "Maintain a simple, balanced skincare routine."
        )

    return suggestions


# --------------------------------------------------
# Image Input: Camera or Upload
# --------------------------------------------------

st.subheader("📸 Choose Your Image Method")

input_method = st.radio(
    "Select an option:",
    ["📷 Use Camera", "🖼️ Upload Image"],
    horizontal=True
)

image = None
image_source = None

if input_method == "📷 Use Camera":

    st.info(
        "Allow camera access in your browser and take a picture."
    )

    camera_image = st.camera_input("Take a picture")

    if camera_image is not None:
        file_bytes = np.frombuffer(
            camera_image.getvalue(),
            dtype=np.uint8
        )

        image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        image_source = "Camera Image"

else:

    uploaded_file = st.file_uploader(
        "Choose a face image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        file_bytes = np.frombuffer(
            uploaded_file.getvalue(),
            dtype=np.uint8
        )

        image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        image_source = "Uploaded Image"


# --------------------------------------------------
# Preview and Analyze
# --------------------------------------------------

if image is not None:

    st.image(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        caption=image_source,
        use_container_width=True
    )

    if st.button(
        "🔍 Analyze Face",
        type="primary"
    ):

        with st.spinner(
            "Analyzing face. The first analysis may take longer "
            "while the models initialize."
        ):
            analysis = analyze_face(image)

        if analysis:

            st.success("Face analysis completed!")

            st.subheader("📊 Analysis Results")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Estimated Age",
                    str(analysis.get("age", "Unknown"))
                )

                st.metric(
                    "Gender",
                    analysis.get("dominant_gender", "Unknown")
                )

            with col2:
                st.metric(
                    "Emotion",
                    analysis.get("dominant_emotion", "Unknown")
                )

                st.metric(
                    "Skin Type",
                    analysis.get("skin_type", "Unknown")
                )

            st.subheader("💡 Skincare Recommendations")

            for suggestion in suggest_skincare(analysis):
                st.write(f"✔️ {suggestion}")

            st.caption(
                "Age, gender, and emotion are AI-generated estimates. "
                "Skin type detection is experimental and is not a "
                "medical diagnosis."
            )

        else:
            st.warning(
                "Try another clear image. If the issue continues, "
                "check the server logs."
            )
