import cv2


from pathlib import Path
CAMERA = "/dev/video0"
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL = str(BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx")

detector = cv2.FaceDetectorYN.create(
    MODEL,
    "",
    (320, 320),
    0.6,
    0.3,
    5000
)

cap = cv2.VideoCapture(CAMERA, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

if not cap.isOpened():
    raise RuntimeError("Could not open /dev/video0")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera frame failed")
        break

    h, w = frame.shape[:2]

    detector.setInputSize((w, h))

    _, faces = detector.detect(frame)

    if faces is not None:

        for face in faces:

            x, y, fw, fh = face[:4].astype(int)

            cv2.rectangle(
                frame,
                (x, y),
                (x + fw, y + fh),
                (0, 255, 0),
                2
            )

            confidence = face[-1]

            cv2.putText(
                frame,
                f"Face {confidence:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # YuNet provides 5 facial landmarks
            for i in range(5):
                lx = int(face[4 + i * 2])
                ly = int(face[5 + i * 2])

                cv2.circle(
                    frame,
                    (lx, ly),
                    3,
                    (255, 0, 0),
                    -1
                )

    cv2.imshow("Face Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
