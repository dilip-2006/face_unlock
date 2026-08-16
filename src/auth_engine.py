import cv2
import numpy as np
import random
import time
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
# Authentication settings
# ============================================================

RECOGNITION_THRESHOLD = 0.40

MOVEMENT_THRESHOLD = 0.18

LIVENESS_TIMEOUT = 5.0


# ============================================================
# Load enrolled face
# ============================================================

if not EMBEDDING_FILE.exists():

    raise RuntimeError(
        "Face not enrolled. Run enroll.py first."
    )

known_embedding = np.load(
    EMBEDDING_FILE
).astype(np.float32)

known_embedding = known_embedding.flatten()

known_embedding /= (
    np.linalg.norm(known_embedding) + 1e-10
)


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
        "Could not open camera"
    )


# ============================================================
# Liveness challenge
# ============================================================

challenge = random.choice([
    "LEFT",
    "RIGHT"
])

print()
print("========================================")
print("       FACE AUTHENTICATION")
print("========================================")
print()
print("Challenge:", challenge)
print()


baseline = None
liveness_passed = False

start_time = time.time()


# ============================================================
# Authentication loop
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        continue

    height, width = frame.shape[:2]

    detector.setInputSize(
        (width, height)
    )

    _, faces = detector.detect(frame)


    # ========================================================
    # Face detected
    # ========================================================

    if faces is not None and len(faces) > 0:

        # Largest face

        face = max(
            faces,
            key=lambda f: f[2] * f[3]
        )

        x, y, w, h = face[:4].astype(int)


        # ----------------------------------------------------
        # Liveness
        # ----------------------------------------------------

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


        if eye_distance > 1:

            nose_position = (
                nose[0] - eye_center[0]
            ) / eye_distance


            if baseline is None:

                baseline = nose_position


            movement = (
                nose_position - baseline
            )


            if challenge == "LEFT":

                if movement < -MOVEMENT_THRESHOLD:

                    liveness_passed = True


            elif challenge == "RIGHT":

                if movement > MOVEMENT_THRESHOLD:

                    liveness_passed = True


        # ----------------------------------------------------
        # Draw
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


    # ========================================================
    # Display liveness state
    # ========================================================

    if not liveness_passed:

        message = (
            f"TURN {challenge}"
        )

    else:

        message = (
            "LIVENESS OK - VERIFYING..."
        )


    cv2.putText(
        frame,
        message,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )


    # ========================================================
    # Recognition
    # ========================================================

    if liveness_passed and faces is not None:

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


        print(
            f"Similarity: {similarity:.4f}"
        )


        # ----------------------------------------------------
        # Recognition decision
        # ----------------------------------------------------

        if similarity >= RECOGNITION_THRESHOLD:

            print()
            print("========================================")
            print("       AUTHENTICATION SUCCESS")
            print("       USER: DILIP")
            print("========================================")
            print()

            cv2.putText(
                frame,
                "AUTHENTICATED",
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                3
            )

            cv2.imshow(
                "Face Authentication",
                frame
            )

            cv2.waitKey(1500)

            break

        else:

            print()
            print("========================================")
            print("       AUTHENTICATION FAILED")
            print("========================================")
            print()

            cv2.putText(
                frame,
                "UNKNOWN USER",
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3
            )

            cv2.imshow(
                "Face Authentication",
                frame
            )

            cv2.waitKey(1500)

            break


    # ========================================================
    # Timeout
    # ========================================================

    if time.time() - start_time > LIVENESS_TIMEOUT:

        print()
        print("LIVENESS TIMEOUT")
        print()

        break


    # ========================================================
    # Display
    # ========================================================

    cv2.imshow(
        "Face Authentication",
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

