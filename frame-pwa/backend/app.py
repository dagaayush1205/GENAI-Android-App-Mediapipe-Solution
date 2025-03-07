from flask import Flask, Response, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
from peft import PeftModel, LoraConfig
import os
from transformers import AutoModel
from transformers.pipelines import base
base_model = AutoModel.from_pretrained("facebook/detr-resnet-50")
current_model = base_model
mode = "left"
left_lora_path = "./lora_left_hand"
right_lora_path = "./lora_right_hand"
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
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

def switch_lora(hand_label, base_model):
    if hand_label == "Left":
        print("Switching to right Hand LoRa")
        lora_path = os.path.abspath("lora_left_hand")
        # model = PeftModel.from_pretrained(base_model, left_lora_path)
    elif hand_label == "Right":
        print("Switching to left Hand LoRa")
        lora_path = os.path.abspath("lora_right_hand")
        # model = PeftModel.from_pretrained(base_model, right_lora_path)
    else:
        print("No hand detected")
        return base_model
    model = PeftModel.from_pretrained(base_model, lora_path)
    return model


def generate_frames():
    cap = cv2.VideoCapture(0)  # Capture from webcam
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
        
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
# Updated options:  {'faceTracking': True, 'handTracking': False}
            # Process with MediaPipe
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_label = handedness.classification[0].label
                    switch_lora(hand_label, base_model)
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
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
