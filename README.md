# Face Unlock for Ubuntu 22.04

A custom AI-powered face authentication system for Ubuntu 22.04 LTS using OpenCV YuNet, SFace, camera-based similarity matching, PAM integration, a background systemd service, Unix domain socket IPC, and a GNOME Shell glassmorphism interface.

The system is designed to provide a native-feeling face-unlock experience for Ubuntu while preserving traditional password authentication as a safe fallback.

---

## Features

- **Real-time Face Detection**: Powered by OpenCV YuNet (`face_detection_yunet_2023mar.onnx`) for fast and accurate face detection and 5-point facial landmarking.
- **128-Dimensional Face Embeddings**: Feature extraction using OpenCV SFace (`face_recognition_sface_2021dec.onnx`).
- **Multiple-Frame Face Enrollment**: Captures up to 40 frames during enrollment to generate a robust reference embedding.
- **Cosine-Similarity Matching**: High-precision vector similarity comparison with configurable confidence thresholds.
- **Similarity-Based Identity Matching**: Live camera embeddings are compared with the enrolled SFace embedding.
- **PAM Integration**: Pluggable Authentication Module wrapper (`pam_exec.so`) bridging system authentication with the face service.
- **GNOME Login Authentication**: Integrated with GDM (`/etc/pam.d/gdm-password`).
- **GNOME Lock-Screen Authentication**: Unlocks the lock screen automatically upon face verification.
- **Sudo Face Authentication**: Passwordless sudo verification using `auth sufficient` in `/etc/pam.d/sudo`.
- **Password Fallback**: Seamless fallback to password entry if face authentication fails, times out, or is cancelled.
- **Dedicated Background Service**: Systemd-managed daemon (`face_service.py`) running persistent AI models in memory.
- **Unix Domain Socket API**: High-performance, secure IPC via `/run/face-unlock/face-unlock.sock`.
- **Automatic Camera Activation**: On-demand camera startup only during active authentication requests.
- **Camera Auto-Release**: Instant release of `/dev/video0` upon completion, preserving privacy and hardware availability.
- **GNOME Shell Glassmorphism UI**: Custom GNOME extension (`face-unlock-glass@dilip`) providing a translucent frosted-glass visual indicator.
- **White Face Icon & Breathing Animation**: Smooth sine-wave light pulsing during face search.
- **Eight-Dot Scanning Animation**: Sequential dot movement indicator with fixed element spacing.
- **Bottom-Center UI Placement**: Non-intrusive placement on the primary monitor lock screen.
- **Keyboard Shortcut Support**: Trigger manual authentication via `Super + F` or socket client commands.
- **Systemd Integration**: Persistent service lifecycle management with automatic restart and system boot integration.
- **Authentication Logging**: Detailed logging to `~/face_unlock/logs/pam_auth.log` and system journal logs.
- **Headless Authentication Engine**: GUI-independent engine (`auth_engine_headless.py`) for display-agnostic PAM contexts.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      GNOME / Ubuntu                      │
│                                                          │
│               Login  /  Lock Screen  /  sudo             │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
                     ┌───────────────┐
                     │      PAM      │
                     └───────┬───────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │     pam_face_check      │
                │                         │
                │ PAM authentication      │
                │ wrapper                 │
                └────────────┬────────────┘
                             │
                             ▼
                    Unix Domain Socket
                             │
              /run/face-unlock/face-unlock.sock
                             │
                             ▼
           ┌───────────────────────────────────┐
           │          face_service.py          │
           │                                   │
           │ Background authentication service │
           └─────────────────┬─────────────────┘
                             │
                             ▼
                        Camera ON
                             │
                             ▼
                ┌─────────────────────────┐
                │     YuNet Detection     │
                └────────────┬────────────┘
                             │
                             ▼
                       Face Alignment
                             │
                             ▼
                ┌─────────────────────────┐
                │     SFace Embedding     │
                │       128-D vector      │
                └────────────┬────────────┘
                             │
                             ▼
                      Face Similarity
                             │
                             ▼
                ┌─────────────────────────┐
                │ Similarity Threshold    │
                └────────────┬────────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
             SUCCESS                    FAIL
                │                         │
                ▼                         ▼
           PAM SUCCESS               PAM FAILURE
                │                         │
                ▼                         ▼
           Unlock/Login                Password
```

---

## Technology Stack

| Component | Technology |
| :--- | :--- |
| **Operating System** | Ubuntu 22.04.5 LTS |
| **Desktop Environment** | GNOME 42.9 |
| **Windowing System** | X11 |
| **Camera Hardware** | Integrated UVC Webcam (`/dev/video0`) |
| **Camera Interface** | V4L2 (Video for Linux 2) |
| **Face Detection** | YuNet (`face_detection_yunet_2023mar.onnx`) |
| **Face Recognition** | SFace (`face_recognition_sface_2021dec.onnx`) |
| **Computer Vision Engine**| OpenCV (Python 3) |
| **Language** | Python 3.10 |
| **Authentication Subsystem**| PAM (`pam_exec.so`) |
| **Service Manager** | systemd |
| **IPC Infrastructure** | Unix Domain Socket |
| **User Interface** | GNOME Shell Extension |
| **UI Styling** | Vanilla CSS (Glassmorphism effect) |
| **Embedding Size** | 128-dimensional floating-point vector |

---

## Hardware

The system utilizes a standard UVC-compatible web camera.

### Example Hardware Profile:
- **Driver**: `uvcvideo`
- **Camera Device**: `/dev/video0`
- **Resolution**: 1920 × 1080 (supports 1280 × 720)
- **Pixel Format**: `MJPG`
- **Frame Rate**: 30 FPS

### Camera Diagnostic Commands:
```bash
# Check primary camera attributes
v4l2-ctl -d /dev/video0 --all

# List all connected video capture devices
v4l2-ctl --list-devices
```

---

## Project Structure

```
/home/dk/face_unlock/
├── bin/
│   └── pam_face_check                    # Executable Python wrapper for PAM
├── data/
│   └── face_embedding.npy                # Enrolled 128-D numpy reference vector
├── logs/
│   └── pam_auth.log                      # System authentication attempt logs
├── models/
│   ├── face_detection_yunet_2023mar.onnx  # YuNet face detector model
│   └── face_recognition_sface_2021dec.onnx# SFace face recognition model
├── src/
│   ├── auth_engine.py                    # Graphical test authentication script
│   ├── auth_engine_headless.py           # Headless authentication core
│   ├── enroll.py                         # Interactive face enrollment tool
│   ├── face_service.py                   # Main systemd background daemon
│   ├── face_service_challenge.py         # Socket service with active challenge
│   ├── liveness.py                       # Standalone liveness verification test
│   ├── recognize.py                      # Standalone face recognition test
│   ├── test_detection.py                 # YuNet camera detection test
│   └── test_recognition.py               # SFace feature extraction test
└── README.md                             # System documentation

# System Integration Files:
/etc/pam.d/face-unlock-test               # Dedicated test PAM stack configuration
/etc/pam.d/gdm-password                   # GNOME Display Manager PAM config
/etc/pam.d/sudo                           # System sudo PAM config
/etc/systemd/system/face-unlock.service   # Systemd unit service definition

# GNOME Extension Integration:
~/.local/share/gnome-shell/extensions/face-unlock-glass@dilip/
├── extension.js                          # Extension lifecycle & state listener
├── face-icon.svg                         # Vector face icon asset
├── metadata.json                         # GNOME Shell extension metadata
├── stylesheet.css                        # Glassmorphism styling & dot classes
└── README.md                             # Extension overview

# Runtime State Files:
/run/face-unlock/face-unlock.sock         # Unix domain socket endpoint
/run/face-unlock/ui-state                 # Shared UI state pipe file
```

---

## 1. Install Ubuntu Dependencies

Update system package repositories:
```bash
sudo apt update
sudo apt upgrade -y
```

Install mandatory system packages:
```bash
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-opencv \
    v4l-utils \
    pamtester \
    systemd \
    gnome-shell \
    gnome-extensions-app
```

Verify Python installation:
```bash
python3 --version
```

Verify system OpenCV installation:
```bash
python3 -c "import cv2; print(cv2.__version__)"
```

---

## 2. Create the Project Directory

```bash
mkdir -p ~/face_unlock
cd ~/face_unlock

mkdir -p data logs models src bin
```

---

## 3. Python Virtual Environment

Create the isolated virtual environment:
```bash
cd ~/face_unlock
python3 -m venv .venv
```

Activate the virtual environment:
```bash
source .venv/bin/activate
```

Upgrade package manager and install dependencies:
```bash
pip install --upgrade pip
pip install numpy opencv-python
```

Verify OpenCV inside the virtual environment:
```bash
python -c "import cv2; print(cv2.__version__)"
```

---

## 4. AI Models Setup

The authentication pipeline depends on two ONNX neural net models:

1. **YuNet Face Detector** (`face_detection_yunet_2023mar.onnx`)
   - **Input**: Raw camera video frame (`1920x1080` / `1280x720`)
   - **Output**: Bounding box `[x, y, w, h]`, confidence score, and 5 facial landmarks (left eye, right eye, nose tip, left mouth corner, right mouth corner).

2. **SFace Face Recognizer** (`face_recognition_sface_2021dec.onnx`)
   - **Input**: Aligned face image cropped by YuNet (`112x112`).
   - **Output**: L2-normalized 128-dimensional floating-point feature embedding vector.

Ensure models are located in `~/face_unlock/models/`:
```bash
ls -lh ~/face_unlock/models/
```
Expected output:
```
face_detection_yunet_2023mar.onnx   (~232 KB)
face_recognition_sface_2021dec.onnx (~38.7 MB)
```

---

## 5. Face Detection Verification

Test real-time detection, bounding boxes, and 5-point facial landmarking using YuNet:

```bash
cd ~/face_unlock
source .venv/bin/activate
python src/test_detection.py
```

*Press `Q` in the preview window to exit.*

---

## 6. Face Recognition Verification

Test feature vector extraction using SFace:

```bash
cd ~/face_unlock
source .venv/bin/activate
python src/test_recognition.py
```

The output embedding vector shape is `(128,)` with an L2 norm near `1.0`.

---

## 7. Face Enrollment

Enrollment captures 40 face frames across slightly varied head angles (straight, left, right, up, down) to form a robust averaged template.

Run the interactive enrollment script:
```bash
cd ~/face_unlock
source .venv/bin/activate
python src/enroll.py
```

### Process:
```
40 face sample frames
       │
       ▼
40 × 128 embedding matrices
       │
       ▼
Mean vector aggregation
       │
       ▼
L2 Normalization
       │
       ▼
Saved to: ~/face_unlock/data/face_embedding.npy
```

---

## 8. Face Matching

During verification, live face embeddings are compared against the stored template using cosine similarity (dot product of L2-normalized vectors):

$$\text{Similarity} = \mathbf{v}_{\text{live}} \cdot \mathbf{v}_{\text{stored}}$$

- **Match Threshold**: `0.40`
- Typical legitimate match scores: `0.65` – `0.92`
- Typical imposter/unknown match scores: `< 0.25`

Test recognition matching:
```bash
python src/recognize.py
```

---

## 9. Identity Matching

Unlock authentication uses SFace embeddings and cosine similarity against the enrolled `face_embedding.npy` template. Directional LEFT/RIGHT challenges are not part of the unlock decision.

---

## 10. Headless Authentication Engine

PAM modules execute outside the graphical user session. `auth_engine_headless.py` executes face verification without calling `cv2.imshow()` or requiring X11 DISPLAY access.

Test headless authentication directly:
```bash
python src/auth_engine_headless.py
```
*Returns exit code `0` on success, `1` on failure.*

---

## 11. PAM Wrapper Script

The executable PAM wrapper `~/face_unlock/bin/pam_face_check` connects to the running face service Unix socket, sends `AUTHENTICATE`, waits for response, and logs details to `~/face_unlock/logs/pam_auth.log`. It is intended for the enrolled `dk` account; other users continue through normal password authentication.

Verify script permissions:
```bash
chmod +x ~/face_unlock/bin/pam_face_check
```

---

## 12. Dedicated PAM Test Service

A dedicated test PAM configuration allows testing without risking desktop lockout.

File location: `/etc/pam.d/face-unlock-test`
```pam
auth    requisite    pam_exec.so    quiet    /home/dk/face_unlock/bin/pam_face_check
account required     pam_permit.so
```

Test PAM authentication using `pamtester`:
```bash
pamtester face-unlock-test dk authenticate
```
Expected output:
```
pamtester: successfully authenticated
```

Verbose testing:
```bash
pamtester -v face-unlock-test dk authenticate
```

---

## 13. GNOME Login & Lock-Screen Integration

Face unlock is installed in `/etc/pam.d/common-auth`, the shared PAM
authentication stack. This makes it run first for GDM, lock-screen unlock,
console login, `su`, and other services that include `common-auth`.

Because GDM uses `gdm-password` for both the power-on login screen and the
GNOME lock screen, face unlock is attempted in both places. If the match
fails, the regular password prompt remains available.

If face authentication succeeds, GDM unlocks immediately. If it fails or times out, PAM falls back seamlessly to standard password input.

---

## 14. Sudo Authentication Integration

`/etc/pam.d/gdm-password` and `/etc/pam.d/sudo` use the shared stack and do
not need separate face-authentication entries.

Test sudo authentication:
```bash
# Clear sudo credential cache
sudo -k

# Execute sudo command
sudo whoami
```
Expected output:
```
root
```

---

## 15. Why `sufficient` Control Flag is Used

- `requisite`: If the module fails, authentication fails **immediately** and terminates the stack without requesting password fallback.
- `sufficient`: If the module succeeds, authentication succeeds **immediately**. If it fails, PAM **continues** down the stack to the standard password authentication module (`common-auth`).

Using `sufficient` ensures password authentication remains available when face recognition fails or camera is covered.

---

## 16. Background Face Service

The daemon `src/face_service.py` runs as a persistent service, keeping YuNet and SFace models loaded in memory for instant response times.

Features:
- Models loaded once at daemon startup.
- Unix domain socket server at `/run/face-unlock/face-unlock.sock`.
- Manages state file `/run/face-unlock/ui-state`.
- Keeps camera closed until `AUTHENTICATE` request arrives.
- Automatically closes camera upon completion.

---

## 17. Unix Domain Socket API

- **Socket Path**: `/run/face-unlock/face-unlock.sock`
- **Permissions**: `0660` (owned by user and `faceauth` group)
- **Protocol**: Simple text-based commands over TCP stream.

### Request Payload:
```
AUTHENTICATE\n
```

### Response Payloads:
- `SUCCESS\n`: Face recognized above the similarity threshold.
- `FAIL\n`: Face unverified or recognition timed out.
- `INVALID\n`: Unrecognized socket command.

---

## 18. Service Communication Architecture

```
┌──────────────────┐               ┌─────────────────────────────────┐
│   PAM Module /   │  AUTHENTICATE │       face_service.py           │
│  Keyboard Trigger│───────────────> (Socket Daemon in background)   │
└──────────────────┘               └────────────────┬────────────────┘
         ▲                                          │
         │             SUCCESS / FAIL               │ Updates state
         └──────────────────────────────────────────┘
                                                    │
                                                    ▼
                                       /run/face-unlock/ui-state
                                                    │
                                                    ▼
                                       ┌─────────────────────────┐
                                       │   GNOME Shell Extension │
                                       │ (face-unlock-glass@dilip│
                                       └─────────────────────────┘
```

---

## 19. Systemd Service Setup

Systemd service template: [`systemd/face-unlock.service`](systemd/face-unlock.service), installed as `/etc/systemd/system/face-unlock.service`.

```ini
[Unit]
Description=Face Unlock Authentication Service
After=network.target

[Service]
Type=simple
User=dk
Group=faceauth
SupplementaryGroups=video
WorkingDirectory=/home/dk/face_unlock
ExecStart=/home/dk/face_unlock/.venv/bin/python /home/dk/face_unlock/src/face_service.py
Restart=on-failure
RestartSec=3
RuntimeDirectory=face-unlock
RuntimeDirectoryMode=0750
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

### Management Commands:
```bash
# Install/update the boot service, reload systemd, and start it now
sudo install -m 0644 systemd/face-unlock.service /etc/systemd/system/face-unlock.service
sudo systemctl daemon-reload
sudo systemctl enable --now face-unlock.service

# Check service status
sudo systemctl status face-unlock.service

# Stop the service
sudo systemctl stop face-unlock.service

# Restart the service
sudo systemctl restart face-unlock.service

# View live service log stream
sudo journalctl -u face-unlock.service -f
```

---

## 20. Camera Lifecycle

To guarantee hardware privacy and low energy consumption:

```
[ Idle State ]  ────> Camera OFF (LED Off)
       │
       │  Receive "AUTHENTICATE" via Socket
       ▼
[ Opening Camera ] ──> Acquire /dev/video0 (LED On)
       │
       │  Execute YuNet + SFace similarity matching
       ▼
[ Complete ]   ──────> Release /dev/video0 (LED Off)
       │
       ▼
[ Idle State ]  ────> Camera OFF
```

---

## 21. UI State Protocol

The service broadcasts visual state changes via `/run/face-unlock/ui-state`:

- `IDLE`: UI overlay hidden.
- `AUTHENTICATING`: UI overlay appears with scanning animation.
- `SUCCESS`: Visual success confirmation, overlay fades out.
- `FAIL`: Authentication failed, overlay fades out.

Manual test state command:
```bash
echo AUTHENTICATING | sudo tee /run/face-unlock/ui-state
echo SUCCESS | sudo tee /run/face-unlock/ui-state
echo IDLE | sudo tee /run/face-unlock/ui-state
```

---

## 22. GNOME Glassmorphism UI Extension

Custom extension located at:
`~/.local/share/gnome-shell/extensions/face-unlock-glass@dilip/`

### Features:
- Translucent frosted glass effect using CSS backdrop properties (`rgba(255, 255, 255, 0.075)`).
- Rounded pill container with white accent borders and drop-shadow glow.
- Renders vector SVG face icon (`face-icon.svg`).
- Positioned dynamically at the bottom-center of the primary display monitor.

---

## 23. Eight-Dot Scanner Animation

- Displays 8 fixed-size white dots (`8px x 8px`).
- Scanning dot receives `.active` class (`opacity: 1.0`, glow shadow).
- Neighboring dots receive `.near` class (`opacity: 0.55`).
- Moving window updates every `120ms` without triggering layout reflow or moving dot containers.

---

## 24. UI Animations & Transitions

- **Entrance**: Scale `0.80 -> 1.0`, opacity `0 -> 255`, hover translation upward `20px -> 0px` (cubic ease-out curve).
- **Breathing**: Smooth sine-wave opacity and scale oscillation (`1.00 -> 1.06`) applied to the central SVG icon during active detection.
- **Exit**: Smooth scale-down and fade-out animation when transitioning to `SUCCESS`, `FAIL`, or `IDLE`.

---

## 25. GNOME Extension Management

```bash
# List installed extensions
gnome-extensions list

# Inspect extension metadata
gnome-extensions info face-unlock-glass@dilip

# Enable extension
gnome-extensions enable face-unlock-glass@dilip

# Disable extension
gnome-extensions disable face-unlock-glass@dilip
```

### Reload GNOME Shell (X11):
Press `Alt + F2`, type `r`, and press `Enter`.

---

## 26. Keyboard Shortcut Support

You can configure a desktop keyboard shortcut to trigger face authentication inside a logged-in GNOME session.

1. Open **GNOME Settings** -> **Keyboard** -> **View and Customize Shortcuts** -> **Custom Shortcuts**.
2. Add a new shortcut:
   - **Name**: Face Unlock Trigger
   - **Command**: `python3 -c "import socket; s=socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.connect('/run/face-unlock/face-unlock.sock'); s.sendall(b'AUTHENTICATE\n'); print(s.recv(1024).decode())"`
   - **Shortcut**: `Super + F`

---

## 27. Manual Socket Test

You can manually send authentication requests via terminal socket tools or Python:

```bash
python3 -c "
import socket
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
    s.connect('/run/face-unlock/face-unlock.sock')
    s.sendall(b'AUTHENTICATE\n')
    print('Response:', s.recv(1024).decode().strip())
"
```

---

## 28. Complete Authentication Flow

1. User requests authentication (lock screen unlock attempt, GDM login, or `sudo` prompt).
2. PAM invokes `/home/dk/face_unlock/bin/pam_face_check`.
3. Wrapper writes `AUTHENTICATING` to `/run/face-unlock/ui-state` and opens socket connection.
4. GNOME extension detects `AUTHENTICATING` state and animates glassmorphism UI onto screen.
5. `face_service.py` powers ON camera (`/dev/video0`).
6. YuNet detects face and crops aligned bounding box.
7. SFace extracts 128-dimensional embedding vector.
8. Cosine similarity score calculated against `face_embedding.npy`.
9. SFace cosine similarity is checked against the enrolled template.
10. On success: socket returns `SUCCESS`, UI transitions to `SUCCESS`, camera powers OFF, PAM returns `0` (User authenticated).
11. On failure/timeout: socket returns `FAIL`, UI transitions to `FAIL`, camera powers OFF, PAM returns `1` (Falls back to password).

---

## 29. Authentication Output Examples

### Successful Terminal Log:
```
========================================
       FACE AUTHENTICATION
========================================
UI STATE: AUTHENTICATING
Opening camera...
Camera active
Similarity recognition mode
Best similarity: 0.8897
AUTHENTICATION SUCCESS
UI STATE: SUCCESS
Camera released
Exit code: 0
```

### Failed Terminal Log:
```
========================================
       FACE AUTHENTICATION
========================================
UI STATE: AUTHENTICATING
Opening camera...
Camera active
Best similarity: 0.3210
AUTHENTICATION FAILED
UI STATE: FAIL
Camera released
Exit code: 1
```

---

## 30. Logging & Monitoring

### PAM Authentication Log:
```bash
cat ~/face_unlock/logs/pam_auth.log
```

### Systemd Daemon Journal:
```bash
sudo journalctl -u face-unlock.service -n 50 --no-pager
```

### GNOME Shell Extension Journal Logs:
```bash
journalctl -b /usr/bin/gnome-shell --no-pager | grep -Ei 'face-unlock|extension'
```

---

## 31. Step-by-Step Testing Procedure

Follow this checklist to verify each subsystem in order:

1. **Test 1 — Camera Hardware Access**:
   ```bash
   v4l2-ctl -d /dev/video0 --all
   ```
2. **Test 2 — YuNet Face Detection**:
   ```bash
   python src/test_detection.py
   ```
3. **Test 3 — SFace Recognition Embedding**:
   ```bash
   python src/test_recognition.py
   ```
4. **Test 4 — User Face Enrollment**:
   ```bash
   python src/enroll.py
   ```
   *Generates `data/face_embedding.npy`.*
5. **Test 5 — Headless Engine Test**:
   ```bash
   python src/auth_engine_headless.py
   ```
6. **Test 6 — PAM Tester Service**:
   ```bash
   pamtester face-unlock-test dk authenticate
   ```
7. **Test 7 — Systemd Service**:
   ```bash
   sudo systemctl restart face-unlock.service
   sudo systemctl status face-unlock.service
   ```
8. **Test 8 — GNOME Extension State Overlay**:
   ```bash
   echo AUTHENTICATING | sudo tee /run/face-unlock/ui-state
   sleep 2
   echo SUCCESS | sudo tee /run/face-unlock/ui-state
   ```
9. **Test 9 — Sudo Authentication**:
   ```bash
   sudo -k
   sudo whoami
   ```
10. **Test 10 — Lock Screen Authentication**:
    Press `Super + L` to lock screen and verify instant face unlock.

---

## 32. Troubleshooting Guide

### 1. Camera Not Detected (`Cannot open /dev/video0`)
- Verify physical USB connection and status: `ls -la /dev/video*`
- Check user group permissions: `groups` (ensure user belongs to `video` group: `sudo usermod -aG video dk`).
- Ensure no other application (e.g. Cheese, Zoom, browser) is locking `/dev/video0`.

### 2. ONNX Model Loading Errors (`Can't read ONNX file`)
- Verify files exist in `~/face_unlock/models/`:
  - `face_detection_yunet_2023mar.onnx`
  - `face_recognition_sface_2021dec.onnx`
- Ensure correct read permissions: `chmod 644 ~/face_unlock/models/*.onnx`.

### 3. Qt / Display Display Errors (`could not connect to display`)
- Occurs when running graphical preview scripts (`cv2.imshow()`) inside SSH sessions or PAM hooks.
- Use `auth_engine_headless.py` and `face_service.py` for headless service contexts.

### 4. PAM System Error / Permission Denied
- Check socket permissions: `ls -la /run/face-unlock/face-unlock.sock`. Socket should have `0660` permissions owned by group `faceauth`.
- Inspect PAM execution log: `cat ~/face_unlock/logs/pam_auth.log`.
- Verify path to `pam_face_check` in `/etc/pam.d/gdm-password` and `/etc/pam.d/sudo`.

### 5. GNOME Extension Not Showing
- Verify GNOME Shell version compatibility: `gnome-shell --version` (Targeted for GNOME 42).
- Ensure extension is enabled: `gnome-extensions enable face-unlock-glass@dilip`.
- Restart GNOME Shell under X11: `Alt + F2` -> `r` -> `Enter`.
