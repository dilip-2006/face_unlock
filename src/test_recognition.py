import cv2
import numpy as np
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DETECTOR_MODEL = str(
    BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx"
)

RECOGNIZER_MODEL = str(
    BASE_DIR / "models" / "face_recognition_sface_2021dec.onnx"
)

CAMERA = "/dev/video0"

# --------------------------------------------------
# Load YuNet face detector
# --------------------------------------------------

detector = cv2.FaceDetectorYN.create(
    DETECTOR_MODEL,
    "",
    (320, 320),
    0.6,
    0.3,
    5000
)

# --------------------------------------------------
# Load SFace recognizer
# --------------------------------------------------

recognizer = cv2.FaceRecognizerSF.create(
    RECOGNIZER_MODEL,
    ""
)

# --------------------------------------------------
# Camera
# --------------------------------------------------

cap = cv2.VideoCapture(CAMERA, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*"MJPG")
)

if not cap.isOpened():
    raise RuntimeError("Could not open /dev/video0")

print("Camera started")
print("Press Q to quit")

# --------------------------------------------------
# Main loop
# --------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read camera")
        break

    height, width = frame.shape[:2]

    detector.setInputSize((width, height))

    _, faces = detector.detect(frame)

    if faces is not None:

        for face in faces:

            # --------------------------------------
            # Draw face bounding box
            # --------------------------------------

            x, y, w, h = face[:4].astype(int)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # --------------------------------------
            # Align face for SFace
            # --------------------------------------

            aligned_face = recognizer.alignCrop(
                frame,
                face
            )

            # --------------------------------------
            # Extract face feature
            # --------------------------------------

            feature = recognizer.feature(
                aligned_face
            )

            # --------------------------------------
            # Display embedding information
            # --------------------------------------

            cv2.putText(
                frame,
                "FACE FEATURE GENERATED",
                (x, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # Print only once-ish information
            print(
                "Embedding shape:",
                feature.shape,
                "Norm:",
                np.linalg.norm(feature)
            )

    cv2.imshow(
        "Face Recognition Test",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
