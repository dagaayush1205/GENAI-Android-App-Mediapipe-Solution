import cv2
from peft import PeftModel, LoraConfig
import mediapipe as mp
import os
from transformers import AutoModel
from transformers.pipelines import base
base_model = AutoModel.from_pretrained("facebook/detr-resnet-50")
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
current_model = base_model
mode = "left"
left_lora_path = "./lora_left_hand"
right_lora_path = "./lora_right_hand"

def switch_lora(hand_label, base_model):
    if hand_label == "Left":
        print("Switching to Left Hand LoRa")
        lora_path = os.path.abspath("lora_left_hand")
        # model = PeftModel.from_pretrained(base_model, left_lora_path)
    elif hand_label == "Right":
        print("Switching to Right Hand LoRa")
        lora_path = os.path.abspath("lora_right_hand")
        # model = PeftModel.from_pretrained(base_model, right_lora_path)
    else:
        print("No hand detected")
        return base_model
    model = PeftModel.from_pretrained(base_model, lora_path)
    return model



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label

            current_model = switch_lora(hand_label, base_model)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow("Hand Tracking", frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('l'):
            mode = "Left"
        elif key & 0xFF == ord('r'):
            mode = "Right"
        elif key & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows
