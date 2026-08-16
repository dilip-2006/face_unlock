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

CAMERA = "/dev/video0"


# ============================================================
# Liveness settings
# ============================================================

CHALLENGES = [
    "LEFT",
    "RIGHT",
]

CHALLENGE_TIME = 5.0

# How much the nose must move relative to the eyes.
# We will calibrate this based on your camera.
MOVEMENT_THRESHOLD = 0.18


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


# ============================================================
# Challenge
# ============================================================

challenge = random.choice(CHALLENGES)

print()
print("========================================")
print("          LIVENESS TEST")
print("========================================")
print()
print("Challenge:", challenge)
print()
print("Follow the instruction shown on screen.")
print()


# ============================================================
# Variables
# ============================================================

start_time = time.time()

baseline = None

liveness_passed = False

movement_values = []


# ============================================================
# Main loop
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

    status = "NO FACE"

    if faces is not None and len(faces) > 0:

        # ----------------------------------------------------
        # Use largest detected face
        # ----------------------------------------------------

        face = max(
            faces,
            key=lambda f: f[2] * f[3]
        )

        x, y, w, h = face[:4].astype(int)

        # ----------------------------------------------------
        # YuNet landmarks
        #
        # face[4:14]
        #
        # 4,5   = left eye
        # 6,7   = right eye
        # 8,9   = nose
        # 10,11 = left mouth
        # 12,13 = right mouth
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

        # ----------------------------------------------------
        # Calculate eye midpoint
        # ----------------------------------------------------

        eye_center = (
            left_eye + right_eye
        ) / 2.0

        # Distance between eyes
        eye_distance = np.linalg.norm(
            right_eye - left_eye
        )

        if eye_distance > 1:

            # ------------------------------------------------
            # Normalize nose position
            #
            # This makes movement less dependent on
            # how close you are to the camera.
            # ------------------------------------------------

            normalized_nose_x = (
                nose[0] - eye_center[0]
            ) / eye_distance

            movement_values.append(
                normalized_nose_x
            )

            # ------------------------------------------------
            # Establish center baseline
            # ------------------------------------------------

            if baseline is None:

                baseline = normalized_nose_x

            # ------------------------------------------------
            # Relative movement
            # ------------------------------------------------

            movement = (
                normalized_nose_x - baseline
            )

            # ------------------------------------------------
            # Challenge detection
            # ------------------------------------------------

            if challenge == "LEFT":

                if movement < -MOVEMENT_THRESHOLD:

                    liveness_passed = True

            elif challenge == "RIGHT":

                if movement > MOVEMENT_THRESHOLD:

                    liveness_passed = True

            # ------------------------------------------------
            # Status
            # ------------------------------------------------

            if liveness_passed:

                status = "LIVENESS CONFIRMED"

            else:

                status = (
                    f"Turn {challenge}"
                )

        # ----------------------------------------------------
        # Draw face
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # Draw landmarks
        # ----------------------------------------------------

        for point in [
            left_eye,
            right_eye,
            nose
        ]:

            px = int(point[0])
            py = int(point[1])

            cv2.circle(
                frame,
                (px, py),
                5,
                (255, 0, 0),
                -1
            )

    # ========================================================
    # Timeout
    # ========================================================

    elapsed = time.time() - start_time

    if elapsed > CHALLENGE_TIME:

        if not liveness_passed:

            print()
            print("LIVENESS FAILED")
            print()

            status = "LIVENESS FAILED"

    # ========================================================
    # Display challenge
    # ========================================================

    cv2.putText(
        frame,
        f"Challenge: TURN {challenge}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        status,
        (30, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    if baseline is not None:

        cv2.putText(
            frame,
            f"Movement: {movement:.3f}",
            (30, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    cv2.imshow(
        "Face Liveness Test",
        frame
    )

    # --------------------------------------------------------
    # Exit conditions
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break

    if liveness_passed:

        print()
        print("========================================")
        print("       LIVENESS CONFIRMED")
        print("========================================")
        print()

        time.sleep(1)

        break

    if elapsed > CHALLENGE_TIME:

        break


# ============================================================
# Cleanup
# ============================================================

cap.release()
cv2.destroyAllWindows()
