import cv2
import mediapipe as mp
import numpy as np
import joblib
import time

# --- 1. CONFIGURATION ---
MODEL_PATH = "syncope_v2_model.pkl"
FRAME_THRESHOLD = 45   # ~1.5 seconds. Adjust based on your camera speed.
CALIBRATION_TIME = 5   # Look at the camera normally for 5 seconds at start.

# --- 2. INITIALIZE ---
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except:
    print(f"Error: {MODEL_PATH} not found.")
    exit()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5, # Lowered slightly to catch fast movement
    min_tracking_confidence=0.5
)

# --- 3. REFINED FEATURE EXTRACTION ---
def get_detailed_features(landmarks):
    L_EYE = [33, 160, 158, 133, 153, 144]
    R_EYE = [362, 385, 387, 263, 373, 380]
    
    def calc_ear(pts):
        p = [np.array([landmarks[i].x, landmarks[i].y]) for i in pts]
        return (np.linalg.norm(p[1]-p[5]) + np.linalg.norm(p[2]-p[4])) / (2.0 * np.linalg.norm(p[0]-p[3]))
    
    l_ear, r_ear = calc_ear(L_EYE), calc_ear(R_EYE)
    avg_ear = (l_ear + r_ear) / 2.0
    
    # Mouth & Pitch
    upper = np.linalg.norm(np.array([landmarks[1].x, landmarks[1].y]) - np.array([landmarks[10].x, landmarks[10].y]))
    lower = np.linalg.norm(np.array([landmarks[1].x, landmarks[1].y]) - np.array([landmarks[152].x, landmarks[152].y]))
    pitch = upper / lower
    
    mar = np.linalg.norm(np.array([landmarks[13].x, landmarks[13].y]) - np.array([landmarks[14].x, landmarks[14].y]))
    
    return l_ear, r_ear, avg_ear, mar, pitch

# --- 4. EXECUTION LOOP ---
cap = cv2.VideoCapture(0)
counter = 0
start_time = time.time()
calibrated = False
base_ear, base_pitch = 0.28, 1.0

print("System Online. Calibrating... Sit upright.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    status, color = "Status: Monitoring", (0, 255, 0) # Default
    debug_info = ""

    # A. FACE DETECTED
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        l_ear, r_ear, avg_ear, mar, pitch = get_detailed_features(landmarks)

        if not calibrated:
            elapsed = time.time() - start_time
            status, color = f"CALIBRATING: {int(CALIBRATION_TIME - elapsed)}s", (255, 255, 0)
            if elapsed >= CALIBRATION_TIME:
                base_ear, base_pitch = avg_ear, pitch
                calibrated = True
                print("Calibration Complete.")
        else:
            # AI Logic
            prob = model.predict_proba([[avg_ear, mar, pitch]])[0][1]
            
            # Triple Lock Gates
            eyes_closed = (l_ear < base_ear * 0.75) and (r_ear < base_ear * 0.75)
            head_slumped = (pitch > base_pitch * 1.15)
            
            if eyes_closed and (head_slumped or prob > 0.45):
                counter += 1
            else:
                counter = max(0, counter - 5)
            
            debug_info = f"E:{avg_ear:.2f} P:{pitch:.2f} Prob:{prob:.2f}"

    # B. TRACKING LOST (The Fail-Safe)
    else:
        if calibrated:
            if counter > 5: # We were already suspicious
                counter += 1
                status, color = "!!! TRACKING LOST (POSSIBLE SLUMP) !!!", (0, 165, 255)
            else:
                counter = max(0, counter - 1)
                status, color = "Searching for Face...", (200, 200, 200)

    # C. FINAL STATUS OVERRIDE
    if counter >= FRAME_THRESHOLD:
        status, color = "!!! CRITICAL: SYNCOPE !!!", (0, 0, 255)
    elif counter > 10 and "TRACKING LOST" not in status:
        status, color = "WARNING: Signs Detected", (0, 165, 255)

    # DRAWING
    cv2.putText(frame, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    if debug_info:
        cv2.putText(frame, debug_info, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Counter: {counter}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Syncope Detection System', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()