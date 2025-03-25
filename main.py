
# smart_trial_room.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('mediapipe').setLevel(logging.ERROR)

import cv2
import mediapipe as mp
from PIL import Image
import numpy as np
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# --------------------------
# Haar Cascade Face Detector (for face detection as described in the paper)
face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# --------------------------
# Initialize MediaPipe Pose for skeleton (body landmark) detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8,
    model_complexity=2
)
mp_drawing = mp.solutions.drawing_utils

# --------------------------
# Load garment image (T-shirt) and convert to RGBA for transparency
garment_image = Image.open("rb_80999.png").convert("RGBA")

def calculate_body_dimensions(landmarks, frame_shape):
    """
    Calculate key body measurements based on MediaPipe pose landmarks.
    This mirrors the research paper's approach to sizing the garment.
    """
    h, w = frame_shape[:2]
    ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    rh = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    
    shoulder_width = abs(ls.x - rs.x) * w
    torso_height = abs((ls.y + rs.y)/2 - (lh.y + rh.y)/2) * h
    
    return shoulder_width, torso_height

def overlay_garment(frame, results, garment_img):
    """
    Resize and overlay the garment image on the frame based on the user's body.
    Uses alpha blending (via PIL) to composite the garment.
    """
    if not results.pose_landmarks:
        return frame

    landmarks = results.pose_landmarks.landmark

    # Ensure that the key landmarks are sufficiently visible.
    required_landmarks = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP
    ]
    if any(landmarks[lm].visibility < 0.7 for lm in required_landmarks):
        return frame

    try:
        # Calculate dimensions (with additional padding as per the paper)
        shoulder_width, torso_height = calculate_body_dimensions(landmarks, frame.shape)
        width_scale = (shoulder_width * 1.25) / garment_img.width   # 25% extra width
        height_scale = (torso_height * 1.4) / garment_img.height     # 40% extra height
        scale = max(width_scale, height_scale)
        
        new_width = int(garment_img.width * scale)
        new_height = int(garment_img.height * scale)
        resized_garment = garment_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Calculate position based on shoulder midpoint (with an offset for the neckline)
        h, w = frame.shape[:2]
        mid_shoulder_x = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x +
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 2 * w
        mid_shoulder_y = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y +
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2 * h
        
        x_pos = int(mid_shoulder_x - new_width / 2)
        y_pos = int(mid_shoulder_y - new_height * 0.2)  # Adjust to align the garment's neckline
        
        # Composite the garment using alpha blending.
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
        frame_pil.alpha_composite(resized_garment, (x_pos, y_pos))
        return cv2.cvtColor(np.array(frame_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Overlay error: {e}")
        return frame

def apply_edge_detection(frame):
    """
    Apply Gaussian blur and Canny edge detection to the frame.
    This step emulates the edge detection discussed in the research paper.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return edges_colored

def process_frame(frame, garment_img):
    """
    Process each frame:
      1. Mirror the frame.
      2. Detect the face using Haar Cascade.
      3. Detect body landmarks using MediaPipe Pose.
      4. Optionally draw the detected face and pose landmarks.
      5. Overlay the selected garment.
    """
    # Mirror frame for natural interaction.
    mirrored_frame = cv2.flip(frame, 1)

    # --- Face Detection ---
    gray_frame = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # Draw rectangles around detected faces for visual confirmation.
    for (x, y, w, h) in faces:
        cv2.rectangle(mirrored_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # --- Pose Detection ---
    results = pose.process(cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2RGB))
    # Optionally draw the pose landmarks.
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(mirrored_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # --- Garment Overlay ---
    output_frame = overlay_garment(mirrored_frame, results, garment_img)
    return output_frame

def main():
    """
    Main loop: capture video from the webcam, process each frame, and display the output.
    Press 'e' to toggle edge detection overlay (mimicking additional image processing steps from the paper).
    Press 'q' to quit.
    """
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Press 'e' to toggle edge detection overlay. Press 'q' to exit.")
    edge_mode = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        output_frame = process_frame(frame, garment_image)
        
        # Optionally blend edge detection result.
        if edge_mode:
            edges = apply_edge_detection(frame)
            output_frame = cv2.addWeighted(output_frame, 0.8, edges, 0.2, 0)

        # Flip back for display so it appears natural.
        display_frame = cv2.flip(output_frame, 1)
        cv2.imshow('Smart Trial Room', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e'):
            edge_mode = not edge_mode

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
