import cv2
import numpy as np
import random
import time
from pathlib import Path
import sys


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DETECTOR_MODEL = str(
    BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx"
)

RECOGNIZER_MODEL = str(
    BASE_DIR / "models" / "face_recognition_sface_2021dec.onnx"
)

EMBEDDING_FILE = BASE_DIR / "data" / "face_embedding.npy"

CAMERA = "/dev/video0"


# ============================================================
# Security settings
# ============================================================

RECOGNITION_THRESHOLD = 0.40
MOVEMENT_THRESHOLD = 0.18

LIVENESS_TIMEOUT = 5.0


# ============================================================
# Load enrolled face
# ============================================================

if not EMBEDDING_FILE.exists():
    print("ERROR: No enrolled face")
    sys.exit(1)

known_embedding = np.load(
    EMBEDDING_FILE
).astype(np.float32)

known_embedding = known_embedding.flatten()

known_embedding /= (
    np.linalg.norm(known_embedding) + 1e-10
)


# ============================================================
# Load AI models
# ============================================================

detector = cv2.FaceDetectorYN.create(
    DETECTOR_MODEL,
    "",
    (320, 320),
    0.6,
    0.3,
    5000
)

recognizer = cv2.FaceRecognizerSF.create(
    RECOGNIZER_MODEL,
    ""
)


# ============================================================
# Open camera
# ============================================================

cap = cv2.VideoCapture(
    CAMERA,
    cv2.CAP_V4L2
)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1920
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    1080
)

cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*"MJPG")
)

if not cap.isOpened():
    print("ERROR: Cannot open camera")
    sys.exit(1)


# ============================================================
# Liveness challenge
# ============================================================

challenge = random.choice([
    "LEFT",
    "RIGHT"
])

print(f"Challenge: {challenge}", flush=True)

baseline = None
liveness_passed = False

start_time = time.time()


# ============================================================
# Liveness phase
# ============================================================

while time.time() - start_time < LIVENESS_TIMEOUT:

    ret, frame = cap.read()

    if not ret:
        continue

    height, width = frame.shape[:2]

    detector.setInputSize(
        (width, height)
    )

    _, faces = detector.detect(frame)

    if faces is None or len(faces) == 0:
        continue

    face = max(
        faces,
        key=lambda f: f[2] * f[3]
    )

    left_eye = np.array([
        face[4],
        face[5]
    ])

    right_eye = np.array([
        face[6],
        face[7]
    ])

    nose = np.array([
        face[8],
        face[9]
    ])

    eye_center = (
        left_eye + right_eye
    ) / 2.0

    eye_distance = np.linalg.norm(
        right_eye - left_eye
    )

    if eye_distance <= 1:
        continue

    nose_position = (
        nose[0] - eye_center[0]
    ) / eye_distance

    if baseline is None:
        baseline = nose_position
        continue

    movement = nose_position - baseline

    if challenge == "LEFT":

        if movement < -MOVEMENT_THRESHOLD:
            liveness_passed = True
            break

    elif challenge == "RIGHT":

        if movement > MOVEMENT_THRESHOLD:
            liveness_passed = True
            break


if not liveness_passed:

    print("LIVENESS FAILED", flush=True)

    cap.release()

    sys.exit(1)


print("LIVENESS PASSED", flush=True)


# ============================================================
# Recognition phase
# ============================================================

best_similarity = -1.0

recognition_start = time.time()

while time.time() - recognition_start < 2.0:

    ret, frame = cap.read()

    if not ret:
        continue

    height, width = frame.shape[:2]

    detector.setInputSize(
        (width, height)
    )

    _, faces = detector.detect(frame)

    if faces is None or len(faces) == 0:
        continue

    face = max(
        faces,
        key=lambda f: f[2] * f[3]
    )

    aligned_face = recognizer.alignCrop(
        frame,
        face
    )

    feature = recognizer.feature(
        aligned_face
    )

    feature = feature.flatten().astype(
        np.float32
    )

    feature /= (
        np.linalg.norm(feature) + 1e-10
    )

    similarity = float(
        np.dot(
            known_embedding,
            feature
        )
    )

    best_similarity = max(
        best_similarity,
        similarity
    )


# ============================================================
# Final authentication decision
# ============================================================

print(
    f"Similarity: {best_similarity:.4f}",
    flush=True
)

cap.release()


if best_similarity >= RECOGNITION_THRESHOLD:

    print(
        "AUTHENTICATION SUCCESS",
        flush=True
    )

    sys.exit(0)

else:

    print(
        "AUTHENTICATION FAILED",
        flush=True
    )

    sys.exit(1)

