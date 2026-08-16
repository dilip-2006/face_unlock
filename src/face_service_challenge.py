#!/usr/bin/env python3

import cv2
import numpy as np
import socket
import os
import time
import random
import signal
import sys
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path("/home/dk/face_unlock")

DETECTOR_MODEL = str(
    BASE_DIR /
    "models" /
    "face_detection_yunet_2023mar.onnx"
)

RECOGNIZER_MODEL = str(
    BASE_DIR /
    "models" /
    "face_recognition_sface_2021dec.onnx"
)

EMBEDDING_FILE = (
    BASE_DIR /
    "data" /
    "face_embedding.npy"
)
SOCKET_PATH = "/run/face-unlock/face-unlock.sock"

CAMERA = "/dev/video0"

RECOGNITION_THRESHOLD = 0.40

MOVEMENT_THRESHOLD = 0.18

LIVENESS_TIMEOUT = 5.0

RECOGNITION_TIMEOUT = 2.0


# ============================================================
# Global camera
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
# Load YuNet
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


# ============================================================
# Load SFace
# ============================================================

print(
    "Loading SFace...",
    flush=True
)

recognizer = cv2.FaceRecognizerSF.create(
    RECOGNIZER_MODEL,
    ""
)


# ============================================================
# Open camera
# ============================================================

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
    1920
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    1080
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

    sys.exit(1)


print(
    "Camera ready",
    flush=True
)


# ============================================================
# Detect face
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

    # Largest face

    return max(
        faces,
        key=lambda f: f[2] * f[3]
    )


# ============================================================
# Liveness
# ============================================================

def perform_liveness():

    challenge = random.choice(
        ["LEFT", "RIGHT"]
    )

    print(
        f"Liveness challenge: {challenge}",
        flush=True
    )

    baseline = None

    start = time.time()

    while (
        time.time() - start
        < LIVENESS_TIMEOUT
    ):

        ret, frame = camera.read()

        if not ret:

            continue

        face = get_face(frame)

        if face is None:

            continue

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
            left_eye +
            right_eye
        ) / 2.0

        eye_distance = np.linalg.norm(
            right_eye -
            left_eye
        )

        if eye_distance <= 1:

            continue

        nose_position = (
            nose[0] -
            eye_center[0]
        ) / eye_distance

        if baseline is None:

            baseline = nose_position

            continue

        movement = (
            nose_position -
            baseline
        )

        if challenge == "LEFT":

            if movement < -MOVEMENT_THRESHOLD:

                print(
                    "Liveness passed",
                    flush=True
                )

                return True

        if challenge == "RIGHT":

            if movement > MOVEMENT_THRESHOLD:

                print(
                    "Liveness passed",
                    flush=True
                )

                return True

    print(
        "Liveness failed",
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

        aligned = recognizer.alignCrop(
            frame,
            face
        )

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
        best_similarity >=
        RECOGNITION_THRESHOLD
    )


# ============================================================
# Authentication
# ============================================================

def authenticate():

    print(
        "Authentication request",
        flush=True
    )

    if not perform_liveness():

        return False

    result = recognize_face()

    if result:

        print(
            "AUTHENTICATION SUCCESS",
            flush=True
        )

        return True

    print(
        "AUTHENTICATION FAILED",
        flush=True
    )

    return False


# ============================================================
# Unix socket
# ============================================================

if os.path.exists(SOCKET_PATH):

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

# Temporary permissions.
# We'll tighten this after testing.

os.chmod(
    SOCKET_PATH,
    0o660
)

server.listen(5)

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
    "Service ready",
    flush=True
)


# ============================================================
# Server loop
# ============================================================

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
