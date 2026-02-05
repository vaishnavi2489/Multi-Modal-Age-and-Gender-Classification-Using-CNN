import cv2
import numpy as np
from deepface import DeepFace

def capture_face():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    face = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to access webcam.")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        largest_face = None
        max_area = 0
        
        for (x, y, w, h) in faces:
            area = w * h
            if area > max_area:
                max_area = area
                largest_face = frame[y:y+h, x:x+w]
                
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        if largest_face is not None:
            face = largest_face
        
        cv2.imshow('Face Capture - Press Q to Analyze', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q') and face is not None:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return face

def analyze_face(face):
    try:
        if face is None:
            print("❌ No face detected.")
            return None
        
        temp_image_path = "temp_face.jpg"
        cv2.imwrite(temp_image_path, face)
        
        analysis = DeepFace.analyze(img_path=temp_image_path, actions=['age', 'gender', 'emotion'], enforce_detection=False)
        
        # Handle both dict and list return types
        if isinstance(analysis, list) and len(analysis) > 0:
            analysis_data = analysis[0]
        elif isinstance(analysis, dict):
            analysis_data = analysis
        else:
            print("❌ DeepFace returned unexpected format.")
            return None
        
        analysis_data['skin_type'] = detect_skin_type(face)
        return analysis_data
    except Exception as e:
        print("❌ Error in face analysis:", str(e))
        return None

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
    except Exception as e:
        print("❌ Error in skin type detection:", str(e))
        return "Unknown"

def suggest_skincare(analysis):
    suggestions = []
    
    if not analysis:
        return ["❌ Face analysis failed."]
    
    emotions = analysis.get('emotion', {})
    age = analysis.get('age', 0)
    gender = analysis.get('dominant_gender', "Unknown")
    skin_type = analysis.get('skin_type', "Unknown")
    
    # Emotion-based
    if emotions.get('sad', 0) > 30:
        suggestions.append("Use a hydrating moisturizer for dry skin.")
    if emotions.get('angry', 0) > 30:
        suggestions.append("Apply a calming serum to reduce redness.")
    
    # Age-based
    if age > 40:
        suggestions.append("Use an anti-aging night cream.")
    elif age < 20:
        suggestions.append("Use a gentle cleanser and lightweight moisturizer.")
    
    # Skin-type-based
    if skin_type == "Oily":
        suggestions.append("Use an oil-free cleanser and lightweight moisturizer.")
    elif skin_type == "Dry":
        suggestions.append("Apply a hydrating face cream with hyaluronic acid.")
    elif skin_type == "Combination":
        suggestions.append("Use a lightweight moisturizer for balancing hydration.")
    elif skin_type == "Normal":
        suggestions.append("Maintain hydration with a balanced skincare routine.")
    
    return suggestions if suggestions else ["✅ No specific recommendations needed."]

if __name__ == "__main__":
    print("📷 Capturing face...")
    face = capture_face()
    
    if face is not None:
        print("🔍 Analyzing face...")
        analysis = analyze_face(face)
        
        if analysis:
            print("\n🎯 **Face Analysis Results:**")
            print(f"👤 Age: {analysis.get('age', 'Unknown')}")
            print(f"🚻 Gender: {analysis.get('dominant_gender', 'Unknown')}")
            print(f"😊 Emotion: {analysis.get('dominant_emotion', 'Unknown')}")
            print(f"🧴 Skin Type: {analysis.get('skin_type', 'Unknown')}")
            
            print("\n💡 **Skincare Recommendations:**")
            for suggestion in suggest_skincare(analysis):
                print(f"✔️ {suggestion}")
        else:
            print("❌ Error analyzing face.")
    else:
        print("❌ No face detected!")