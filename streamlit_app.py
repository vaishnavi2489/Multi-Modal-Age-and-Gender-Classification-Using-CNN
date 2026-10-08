import streamlit as st
import cv2
import numpy as np
import tempfile
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
    "Upload a face image to analyze estimated age, gender, "
    "emotion, and skin type."
)


# --------------------------------------------------
# Skin Type Detection
# --------------------------------------------------

def detect_skin_type(face):
    try:
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)

        brightness = np.mean(hsv[:, :, 2])
        saturation = np.mean(hsv[:, :, 1])

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

    try:

        # Save uploaded image temporarily
        with tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False
        ) as temp:

            temp_path = temp.name

            cv2.imwrite(
                temp_path,
                face
            )

        # DeepFace analysis
        analysis = DeepFace.analyze(
            img_path=temp_path,
            actions=[
                "age",
                "gender",
                "emotion"
            ],
            detector_backend="opencv",
            align=True,
            enforce_detection=False
        )

        # DeepFace may return a list or dictionary
        if isinstance(analysis, list) and analysis:

            analysis_data = analysis[0]

        elif isinstance(analysis, dict):

            analysis_data = analysis

        else:

            return None

        # Add skin type
        analysis_data["skin_type"] = detect_skin_type(face)

        return analysis_data

    except Exception as e:

        st.error(
            f"Error in face analysis: {e}"
        )

        return None


# --------------------------------------------------
# Skincare Recommendations
# --------------------------------------------------

def suggest_skincare(analysis):

    suggestions = []

    if not analysis:

        return [
            "Face analysis failed."
        ]

    emotions = analysis.get(
        "emotion",
        {}
    )

    age = analysis.get(
        "age",
        0
    )

    skin_type = analysis.get(
        "skin_type",
        "Unknown"
    )


    # Emotion based
    if emotions.get("sad", 0) > 30:

        suggestions.append(
            "Use a hydrating moisturizer for dry skin."
        )


    if emotions.get("angry", 0) > 30:

        suggestions.append(
            "Apply a calming serum to reduce redness."
        )


    # Age based
    if age > 40:

        suggestions.append(
            "Use an anti-aging night cream."
        )

    elif age < 20:

        suggestions.append(
            "Use a gentle cleanser and lightweight moisturizer."
        )


    # Skin type based
    if skin_type == "Oily":

        suggestions.append(
            "Use an oil-free cleanser and lightweight moisturizer."
        )

    elif skin_type == "Dry":

        suggestions.append(
            "Apply a hydrating face cream with hyaluronic acid."
        )

    elif skin_type == "Combination":

        suggestions.append(
            "Use a lightweight moisturizer for balancing hydration."
        )

    elif skin_type == "Normal":

        suggestions.append(
            "Maintain hydration with a balanced skincare routine."
        )


    if not suggestions:

        suggestions.append(
            "No specific recommendations needed."
        )

    return suggestions


# --------------------------------------------------
# Image Upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "📷 Upload a face image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# --------------------------------------------------
# Process Image
# --------------------------------------------------

if uploaded_file:

    file_bytes = np.asarray(
        bytearray(
            uploaded_file.read()
        ),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )


    if image is None:

        st.error(
            "Could not read the uploaded image."
        )


    else:

        # Display uploaded image
        st.image(
            cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            ),
            caption="Uploaded Image",
            use_container_width=True
        )


        # Analyze button
        if st.button(
            "🔍 Analyze Face",
            type="primary"
        ):

            with st.spinner(
                "Analyzing face... This may take a little while on first run."
            ):

                analysis = analyze_face(
                    image
                )


            # --------------------------------------------------
            # Display Results
            # --------------------------------------------------

            if analysis:

                st.success(
                    "Face analysis completed!"
                )


                col1, col2 = st.columns(2)


                with col1:

                    st.metric(
                        "Estimated Age",
                        str(
                            analysis.get(
                                "age",
                                "Unknown"
                            )
                        )
                    )


                    st.metric(
                        "Gender",
                        analysis.get(
                            "dominant_gender",
                            "Unknown"
                        )
                    )


                with col2:

                    st.metric(
                        "Emotion",
                        analysis.get(
                            "dominant_emotion",
                            "Unknown"
                        )
                    )


                    st.metric(
                        "Skin Type",
                        analysis.get(
                            "skin_type",
                            "Unknown"
                        )
                    )


                # --------------------------------------------------
                # Skincare Recommendations
                # --------------------------------------------------

                st.subheader(
                    "💡 Skincare Recommendations"
                )


                recommendations = suggest_skincare(
                    analysis
                )


                for suggestion in recommendations:

                    st.write(
                        f"✔️ {suggestion}"
                    )