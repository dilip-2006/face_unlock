import cv2
import numpy as np
import socket
import os
import time
import signal
import sys
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path("/home/dk/face_unlock")

DETECTOR_MODEL = str(
    BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx"
)

RECOGNIZER_MODEL = str(
    BASE_DIR / "models" / "face_recognition_sface_2021dec.onnx"
)

EMBEDDING_FILE = (
    BASE_DIR / "data" / "face_embedding.npy"
)

SOCKET_PATH = "/run/face-unlock/face-unlock.sock"
UI_STATE_FILE = "/run/face-unlock/ui-state"

CAMERA = "/dev/video0"

RECOGNITION_THRESHOLD = 0.40

LIVENESS_TIMEOUT = 5.0

RECOGNITION_TIMEOUT = 2.0

# Minimum normalized head-position variation
MOVEMENT_THRESHOLD = 0.08

# Number of useful movement samples required
MIN_MOVEMENT_SAMPLES = 4


# ============================================================
# Camera
# ============================================================

camera = None


# ============================================================
# Cleanup
# ============================================================

def cleanup(signum=None, frame=None):

    global camera

    if camera is not None:

        camera.release()
        camera = None

    if os.path.exists(SOCKET_PATH):

        try:
            os.unlink(SOCKET_PATH)
        except OSError:
            pass

    sys.exit(0)


signal.signal(
    signal.SIGTERM,
    cleanup
)

signal.signal(
    signal.SIGINT,
    cleanup
)


# ============================================================
# UI state control
# ============================================================

def set_ui_state(state):

    try:

        with open(UI_STATE_FILE, "w") as f:
            f.write(state + "\n")

        print(
            f"UI STATE: {state}",
            flush=True
        )

    except Exception as e:

        print(
            f"UI state error: {e}",
            flush=True
        )


# ============================================================
# Camera control
# ============================================================

def open_camera():

    global camera

    print(
        "Opening camera...",
        flush=True
    )

    camera = cv2.VideoCapture(
        CAMERA,
        cv2.CAP_V4L2
    )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )

    camera.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )

    if not camera.isOpened():

        print(
            "ERROR: Cannot open camera",
            flush=True
        )

        camera.release()
        camera = None

        return False

    print(
        "Camera active",
        flush=True
    )

    return True


def close_camera():

    global camera

    if camera is not None:

        camera.release()
        camera = None

        print(
            "Camera released",
            flush=True
        )


# ============================================================
# Load enrolled face
# ============================================================

if not EMBEDDING_FILE.exists():

    print(
        "ERROR: face_embedding.npy not found",
        flush=True
    )

    sys.exit(1)


known_embedding = np.load(
    EMBEDDING_FILE
).astype(np.float32)

known_embedding = known_embedding.flatten()

known_embedding /= (
    np.linalg.norm(known_embedding)
    + 1e-10
)


# ============================================================
# Load AI models
# ============================================================

print(
    "Loading YuNet...",
    flush=True
)

detector = cv2.FaceDetectorYN.create(
    DETECTOR_MODEL,
    "",
    (320, 320),
    0.6,
    0.3,
    5000
)


print(
    "Loading SFace...",
    flush=True
)

recognizer = cv2.FaceRecognizerSF.create(
    RECOGNIZER_MODEL,
    ""
)


# ============================================================
# Face detection
# ============================================================

def get_face(frame):

    height, width = frame.shape[:2]

    detector.setInputSize(
        (width, height)
    )

    _, faces = detector.detect(
        frame
    )

    if faces is None:

        return None

    if len(faces) == 0:

        return None

    # Select largest face

    return max(
        faces,
        key=lambda f: f[2] * f[3]
    )


# ============================================================
# Passive liveness
# ============================================================

def perform_liveness():

    print(
        "Starting passive liveness...",
        flush=True
    )

    start = time.time()

    positions = []

    stable_frames = 0

    while (
        time.time() - start
        < LIVENESS_TIMEOUT
    ):

        ret, frame = camera.read()

        if not ret:

            continue

        face = get_face(frame)

        if face is None:

            stable_frames = 0

            continue


        # ----------------------------------------------------
        # Face landmarks
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
        # Eye center
        # ----------------------------------------------------

        eye_center = (
            left_eye +
            right_eye
        ) / 2.0


        # ----------------------------------------------------
        # Eye distance
        # ----------------------------------------------------

        eye_distance = np.linalg.norm(
            right_eye -
            left_eye
        )

        if eye_distance <= 1:

            continue


        # ----------------------------------------------------
        # Normalize nose position
        # ----------------------------------------------------

        normalized_x = (
            nose[0] -
            eye_center[0]
        ) / eye_distance

        normalized_y = (
            nose[1] -
            eye_center[1]
        ) / eye_distance


        position = np.array([
            normalized_x,
            normalized_y
        ])


        positions.append(
            position
        )

        stable_frames += 1


        # ----------------------------------------------------
        # Need enough samples
        # ----------------------------------------------------

        if len(positions) < 8:

            continue


        recent = np.array(
            positions[-8:]
        )


        # ----------------------------------------------------
        # Calculate movement
        # ----------------------------------------------------

        movement_x = (
            np.max(recent[:, 0])
            -
            np.min(recent[:, 0])
        )

        movement_y = (
            np.max(recent[:, 1])
            -
            np.min(recent[:, 1])
        )


        total_movement = max(
            movement_x,
            movement_y
        )


        # ----------------------------------------------------
        # Liveness decision
        # ----------------------------------------------------

        if (
            total_movement
            >= MOVEMENT_THRESHOLD
            and
            stable_frames
            >= MIN_MOVEMENT_SAMPLES
        ):

            print(
                f"Liveness movement: "
                f"{total_movement:.3f}",
                flush=True
            )

            print(
                "LIVENESS PASSED",
                flush=True
            )

            return True


        # Keep memory bounded

        if len(positions) > 30:

            positions = positions[-20:]


    print(
        "LIVENESS FAILED",
        flush=True
    )

    return False


# ============================================================
# Face recognition
# ============================================================

def recognize_face():

    best_similarity = -1.0

    start = time.time()

    while (
        time.time() - start
        < RECOGNITION_TIMEOUT
    ):

        ret, frame = camera.read()

        if not ret:

            continue


        face = get_face(frame)

        if face is None:

            continue


        # Align face

        aligned = recognizer.alignCrop(
            frame,
            face
        )


        # Generate SFace embedding

        feature = recognizer.feature(
            aligned
        )


        feature = feature.flatten().astype(
            np.float32
        )


        feature /= (
            np.linalg.norm(feature)
            + 1e-10
        )


        # Cosine similarity

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


    print(
        f"Best similarity: "
        f"{best_similarity:.4f}",
        flush=True
    )


    return (
        best_similarity
        >= RECOGNITION_THRESHOLD
    )


# ============================================================
# Authentication
# ============================================================

def authenticate():

    print(
        "Authentication request",
        flush=True
    )

    # Tell the GNOME lock-screen UI that authentication started.
    set_ui_state("AUTHENTICATING")

    # Camera is OFF until authentication is requested.
    if not open_camera():

        set_ui_state("FAIL")
        return False

    try:

        # ----------------------------------------------------
        # Liveness
        # ----------------------------------------------------

        if not perform_liveness():

            set_ui_state("FAIL")
            return False

        # ----------------------------------------------------
        # Face recognition
        # ----------------------------------------------------

        result = recognize_face()

        if result:

            print(
                "AUTHENTICATION SUCCESS",
                flush=True
            )

            set_ui_state("SUCCESS")
            return True

        print(
            "AUTHENTICATION FAILED",
            flush=True
        )

        set_ui_state("FAIL")
        return False

    except Exception as e:

        print(
            f"Authentication error: {e}",
            flush=True
        )

        set_ui_state("FAIL")
        return False

    finally:

        # Always release the webcam after authentication.
        close_camera()


# ============================================================
# Socket setup
# ============================================================

socket_dir = os.path.dirname(
    SOCKET_PATH
)


os.makedirs(
    socket_dir,
    exist_ok=True
)


if os.path.exists(
    SOCKET_PATH
):

    os.unlink(
        SOCKET_PATH
    )


server = socket.socket(
    socket.AF_UNIX,
    socket.SOCK_STREAM
)


server.bind(
    SOCKET_PATH
)


os.chmod(
    SOCKET_PATH,
    0o660
)


server.listen(5)


# ============================================================
# Service startup
# ============================================================

print(
    "========================================",
    flush=True
)

print(
    "     FACE AUTHENTICATION SERVICE",
    flush=True
)

print(
    "========================================",
    flush=True
)

print(
    f"Socket: {SOCKET_PATH}",
    flush=True
)

print(
    "Passive liveness mode",
    flush=True
)

print(
    "Camera mode: ON-DEMAND",
    flush=True
)

print(
    "Service ready",
    flush=True
)


# ============================================================
# Server loop
# ============================================================

# Service starts idle: UI hidden and camera closed.
set_ui_state("IDLE")

while True:

    connection, _ = server.accept()


    try:

        request = connection.recv(
            1024
        ).decode().strip()


        print(
            f"Request: {request}",
            flush=True
        )


        if request == "AUTHENTICATE":

            success = authenticate()


            if success:

                response = "SUCCESS\n"

            else:

                response = "FAIL\n"


        else:

            response = "INVALID\n"


        connection.sendall(
            response.encode()
        )


    except Exception as e:

        print(
            f"Service error: {e}",
            flush=True
        )


        try:

            connection.sendall(
                b"FAIL\n"
            )

        except Exception:

            pass


    finally:

        connection.close()