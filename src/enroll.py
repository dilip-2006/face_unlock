import cv2
import numpy as np
from pathlib import Path
import time

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

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

EMBEDDING_FILE = DATA_DIR / "face_embedding.npy"

CAMERA = "/dev/video0"

# ============================================================
# Settings
# ============================================================

TARGET_SAMPLES = 40

# ============================================================
# Load models
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
# Camera
# ============================================================

cap = cv2.VideoCapture(CAMERA, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*"MJPG")
)

if not cap.isOpened():
    raise RuntimeError("Could not open /dev/video0")

print()
print("========================================")
print("        FACE ENROLLMENT")
print("========================================")
print()
print("Look directly at the camera.")
print("Keep your face inside the frame.")
print()
print("Move your head slowly:")
print("  - Look straight")
print("  - Slightly left")
print("  - Slightly right")
print("  - Slightly up")
print("  - Slightly down")
print()
print("Press Q to cancel.")
print()

time.sleep(2)

embeddings = []

# ============================================================
# Enrollment loop
# ============================================================

while len(embeddings) < TARGET_SAMPLES:

    ret, frame = cap.read()

    if not ret:
        print("Camera frame failed")
        continue

    height, width = frame.shape[:2]

    detector.setInputSize((width, height))

    _, faces = detector.detect(frame)

    status = "Searching for face..."

    if faces is not None and len(faces) > 0:

        # ----------------------------------------------------
        # Select largest face
        # ----------------------------------------------------

        face = max(
            faces,
            key=lambda f: f[2] * f[3]
        )

        x, y, w, h = face[:4].astype(int)

        # ----------------------------------------------------
        # Basic quality check
        # ----------------------------------------------------

        if w >= 120 and h >= 120:

            # ------------------------------------------------
            # Align face
            # ------------------------------------------------

            aligned_face = recognizer.alignCrop(
                frame,
                face
            )

            # ------------------------------------------------
            # Generate embedding
            # ------------------------------------------------

            feature = recognizer.feature(
                aligned_face
            )

            # ------------------------------------------------
            # Normalize embedding
            # ------------------------------------------------

            feature = feature.flatten().astype(np.float32)

            feature = feature / (
                np.linalg.norm(feature) + 1e-10
            )

            embeddings.append(feature)

            status = (
                f"Captured {len(embeddings)}"
                f"/{TARGET_SAMPLES}"
            )

            print(status)

        else:

            status = "Move closer to camera"

        # ----------------------------------------------------
        # Draw bounding box
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    else:

        status = "No face detected"

    # ========================================================
    # Display
    # ========================================================

    cv2.putText(
        frame,
        status,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Move head slowly",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Face Enrollment",
        frame
    )

    key = cv2.waitKey(100) & 0xFF

    if key == ord("q"):
        print("Enrollment cancelled.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

# ============================================================
# Create final face template
# ============================================================

embeddings = np.array(embeddings)

print()
print("Captured embeddings:", embeddings.shape)

# Average all embeddings
mean_embedding = np.mean(
    embeddings,
    axis=0
)

# Normalize final template
mean_embedding = mean_embedding / (
    np.linalg.norm(mean_embedding) + 1e-10
)

# Save
np.save(
    EMBEDDING_FILE,
    mean_embedding
)

print()
print("========================================")
print("       ENROLLMENT COMPLETE")
print("========================================")
print()
print("Embedding shape:", mean_embedding.shape)
print("Saved to:")
print(EMBEDDING_FILE)
print()
print("You can now test recognition.")
print()

cap.release()
cv2.destroyAllWindows()
