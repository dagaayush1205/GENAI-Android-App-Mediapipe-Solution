
from flask import Flask, Response
import cv2
import mediapipe as mp

app = Flask(__name__)

# Initialize MediaPipe solutions
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def generate_frames():
    cap = cv2.VideoCapture(0)  # Try changing to 1 or -1 if this fails
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Convert frame to RGB for MediaPipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_hand = hands.process(frame_rgb)
        results_face = face_detection.process(frame_rgb)

        # Draw face detections
        if results_face.detections:
            for detection in results_face.detections:
                mp_drawing.draw_detection(frame, detection)

        # Draw hand landmarks
        if results_hand.multi_hand_landmarks:
            for hand_landmarks in results_hand.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Encode frame in JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield frame as a response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


