import cv2
import numpy as np
from pathlib import Path

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
# Recognition threshold
# ============================================================
# We will calibrate this later.
# For the first test, use 0.40.

THRESHOLD = 0.40

# ============================================================
# Load enrolled face
# ============================================================

if not EMBEDDING_FILE.exists():
    raise RuntimeError(
        "No enrolled face found. Run enroll.py first."
    )

known_embedding = np.load(
    EMBEDDING_FILE
).astype(np.float32)

known_embedding = known_embedding.flatten()

known_embedding = known_embedding / (
    np.linalg.norm(known_embedding) + 1e-10
)

print("Loaded enrolled face")
print("Embedding shape:", known_embedding.shape)

# ============================================================
# Load YuNet
# ============================================================

detector = cv2.FaceDetectorYN.create(
    DETECTOR_MODEL,
    "",
    (320, 320),
    0.6,
    0.3,
    5000
)

# ============================================================
# Load SFace
# ============================================================

recognizer = cv2.FaceRecognizerSF.create(
    RECOGNIZER_MODEL,
    ""
)

# ============================================================
# Camera
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
    raise RuntimeError(
        "Could not open /dev/video0"
    )

print("Camera started")
print("Press Q to quit")
print()

# ============================================================
# Main loop
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera frame failed")
        continue

    height, width = frame.shape[:2]

    detector.setInputSize(
        (width, height)
    )

    _, faces = detector.detect(frame)

    if faces is not None:

        for face in faces:

            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            x, y, w, h = face[:4].astype(int)

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

            feature = feature.flatten().astype(
                np.float32
            )

            # ------------------------------------------------
            # Normalize
            # ------------------------------------------------

            feature = feature / (
                np.linalg.norm(feature) + 1e-10
            )

            # ------------------------------------------------
            # Cosine similarity
            # ------------------------------------------------

            similarity = float(
                np.dot(
                    known_embedding,
                    feature
                )
            )

            # ------------------------------------------------
            # Recognition decision
            # ------------------------------------------------

            if similarity >= THRESHOLD:

                label = "DILIP - RECOGNIZED"

            else:

                label = "UNKNOWN"

            # ------------------------------------------------
            # Draw
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x, y - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Similarity: {similarity:.3f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            # ------------------------------------------------
            # Terminal output
            # ------------------------------------------------

            print(
                f"Similarity: {similarity:.4f} -> {label}"
            )

    else:

        cv2.putText(
            frame,
            "NO FACE",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2
        )

    # ========================================================
    # Display
    # ========================================================

    cv2.imshow(
        "Ubuntu Face Authentication",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# ============================================================
# Cleanup
# ============================================================

cap.release()
cv2.destroyAllWindows()

