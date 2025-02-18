from flask import Flask, Response, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp

app = Flask(__name__)
CORS(app)

checkbox_states = {"faceTracking": True, "handTracking": True}

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

@app.route("/update_options", methods=["POST"])

def update_options():
    global checkbox_states
    checkbox_states = request.json
    print("Updated options: ",checkbox_states)
    return jsonify({"message": "options Updated", "checkbox_states": checkbox_states})


# def get_options():
#     return jsonify(checkbox_states)

def generate_frames():
    cap = cv2.VideoCapture(0)  # Capture from webcam
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# Updated options:  {'faceTracking': True, 'handTracking': False}
            # Process with MediaPipe
            results_hand = hands.process(frame_rgb)
            results_face = face_detection.process(frame_rgb)

            # Draw face landmarks
            if results_face.detections and checkbox_states["faceTracking"]:
                for detection in results_face.detections:
                    mp_drawing.draw_detection(frame, detection)

            # Draw hand landmarks
            if results_hand.multi_hand_landmarks and checkbox_states["handTracking"]:
                for hand_landmarks in results_hand.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Encode frame to JPEG format
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # Stream frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
