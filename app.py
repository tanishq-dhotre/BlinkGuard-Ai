import os
import wave
import time
import sqlite3
import secrets
from contextlib import contextmanager
from flask import Flask, Response, abort, jsonify, redirect, request, send_file, session, url_for
import cv2
import numpy as np
from scipy.spatial import distance as dist
from PIL import Image, ImageDraw, ImageFont
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from werkzeug.security import check_password_hash, generate_password_hash

app = Flask(__name__)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUTH_DB_PATH = os.environ.get("AUTH_DB_PATH", os.path.join(SCRIPT_DIR, "app_auth.sqlite3"))

@contextmanager
def auth_connection():
    connection = sqlite3.connect(AUTH_DB_PATH)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

def initialize_auth_store():
    with auth_connection() as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS accounts ("
            "username TEXT PRIMARY KEY COLLATE NOCASE, password_hash TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS app_settings ("
            "setting_key TEXT PRIMARY KEY, setting_value TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT OR IGNORE INTO app_settings (setting_key, setting_value) VALUES (?, ?)",
            ("session_secret", secrets.token_hex(32)),
        )
        row = connection.execute(
            "SELECT setting_value FROM app_settings WHERE setting_key = ?",
            ("session_secret",),
        ).fetchone()
        return row[0]

app.secret_key = initialize_auth_store()
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("COOKIE_SECURE") == "1",
    MAX_CONTENT_LENGTH=2 * 1024 * 1024,
)
if os.environ.get("SESSION_SECRET"):
    app.secret_key = os.environ["SESSION_SECRET"]

@app.before_request
def require_login():
    if request.endpoint not in {"login", "signup", "logout"} and not session.get("authenticated"):
        if request.endpoint in {"alarm_status", "emergency_siren", "video_feed", "process_frame"}:
            abort(401)
        return redirect(url_for("login"))

# --- Generate Emergency Siren WAV ---
def create_emergency_siren(filename="emergency_siren.wav"):
    if os.path.exists(filename):
        return

    sample_rate = 44100
    duration = 1.0
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, False)

    freq_sweep = 900 + 300 * np.sin(2 * np.pi * 4 * t)
    phase = 2 * np.pi * np.cumsum(freq_sweep) / sample_rate
    siren = 0.7 * np.sin(phase) + 0.3 * np.sign(np.sin(phase * 2))

    audio_data = (np.clip(siren, -1.0, 1.0) * 32767).astype(np.int16)

    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())

# Dynamic script folder pathing
SIREN_PATH = os.path.join(SCRIPT_DIR, "emergency_siren.wav")
create_emergency_siren(SIREN_PATH)

alarm_playing = False

def start_buzzer():
    global alarm_playing
    alarm_playing = True

def stop_buzzer():
    global alarm_playing
    alarm_playing = False

# --- UI Drawing Helpers ---
def draw_rounded_rect(img, pt1, pt2, color, radius=12, thickness=-1):
    x1, y1 = pt1
    x2, y2 = pt2
    if thickness == -1:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
    else:
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

def draw_overlay_panel(img, pt1, pt2, bg_color=(15, 15, 15), alpha=0.85):
    overlay = img.copy()
    draw_rounded_rect(overlay, pt1, pt2, bg_color, radius=12, thickness=-1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def draw_unicode_text(img, text, position, font_size, color_bgr):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    font_candidates = ["Nirmala.ttf", "mangall.ttf", "arial.ttf"]
    font = None
    for f_path in font_candidates:
        try:
            font = ImageFont.truetype(f_path, font_size)
            break
        except IOError:
            continue

    if font is None:
        font = ImageFont.load_default()

    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# --- Car Interior Renderer ---
def render_full_car_interior(frame, w, h, avg_ear, elapsed_time, status_txt, status_color):
    shade = frame.copy()
    cv2.rectangle(shade, (0, 0), (w, int(h * 0.12)), (10, 10, 10), -1)
    cv2.addWeighted(shade, 0.45, frame, 0.55, 0, frame)

    left_pillar = np.array([[0, 0], [int(w * 0.12), 0], [0, int(h * 0.65)]], np.int32)
    right_pillar = np.array([[w, 0], [int(w * 0.88), 0], [w, int(h * 0.65)]], np.int32)
    cv2.fillPoly(frame, [left_pillar], (18, 18, 20))
    cv2.fillPoly(frame, [right_pillar], (18, 18, 20))

    draw_overlay_panel(frame, (w//2 - 90, 8), (w//2 + 90, 50), bg_color=(25, 25, 25), alpha=0.9)
    cv2.line(frame, (w//2, 0), (w//2, 8), (40, 40, 40), 6)

    dash_h = int(h * 0.28)
    draw_overlay_panel(frame, (0, h - dash_h), (w, h), bg_color=(12, 14, 18), alpha=0.95)
    cv2.line(frame, (0, h - dash_h), (w, h - dash_h), (0, 215, 255), 2)

    draw_overlay_panel(frame, (25, h - dash_h + 15), (170, h - 25), bg_color=(22, 27, 34), alpha=0.9)
    draw_rounded_rect(frame, (25, h - dash_h + 15), (170, h - 25), (0, 215, 255), radius=10, thickness=1)
    cv2.putText(frame, "85", (52, h - dash_h + 65), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, "KM/H", (68, h - dash_h + 88), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "GEAR: D", (50, h - dash_h + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 127), 1, cv2.LINE_AA)

    draw_overlay_panel(frame, (w // 2 - 170, h - dash_h + 15), (w // 2 + 170, h - 25), bg_color=(22, 27, 34), alpha=0.9)
    draw_rounded_rect(frame, (w // 2 - 170, h - dash_h + 15), (w // 2 + 170, h - 25), status_color, radius=10, thickness=1)

    cv2.putText(frame, f"EAR: {avg_ear:.2f}", (w // 2 - 145, h - dash_h + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Closed: {elapsed_time:.1f}s", (w // 2 + 15, h - dash_h + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2, cv2.LINE_AA)

    frame = draw_unicode_text(frame, status_txt, (w // 2 - 140, h - dash_h + 68), 18, status_color)

    draw_overlay_panel(frame, (w - 170, h - dash_h + 15), (w - 25, h - 25), bg_color=(22, 27, 34), alpha=0.9)
    draw_rounded_rect(frame, (w - 170, h - dash_h + 15), (w - 25, h - 25), (0, 215, 255), radius=10, thickness=1)
    cv2.putText(frame, "MODE: SPORT", (w - 158, h - dash_h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "FUEL: 82%", (w - 158, h - dash_h + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 127), 1, cv2.LINE_AA)

    cv2.rectangle(frame, (w - 158, h - dash_h + 85), (w - 40, h - dash_h + 95), (50, 50, 50), -1)
    cv2.rectangle(frame, (w - 158, h - dash_h + 85), (w - 70, h - dash_h + 95), (0, 215, 255), -1)

    steering_center = (w // 2, h + int(dash_h * 0.15))
    steering_radius = int(w * 0.24)

    cv2.circle(frame, steering_center, steering_radius, (20, 20, 22), 26, cv2.LINE_AA)
    cv2.circle(frame, steering_center, steering_radius, (60, 60, 65), 3, cv2.LINE_AA)

    hub_center = (w // 2, h - int(dash_h * 0.25))
    cv2.circle(frame, hub_center, 40, (30, 30, 35), -1, cv2.LINE_AA)
    cv2.circle(frame, hub_center, 40, (0, 215, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "AI", (hub_center[0] - 12, hub_center[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.line(frame, (hub_center[0] - 40, hub_center[1]), (hub_center[0] - steering_radius + 15, hub_center[1]), (35, 35, 40), 16, cv2.LINE_AA)
    cv2.line(frame, (hub_center[0] + 40, hub_center[1]), (hub_center[0] + steering_radius - 15, hub_center[1]), (35, 35, 40), 16, cv2.LINE_AA)
    cv2.line(frame, (hub_center[0], hub_center[1] + 40), (hub_center[0], steering_center[1] - steering_radius + 10), (35, 35, 40), 16, cv2.LINE_AA)

    return frame

# --- MediaPipe Setup ---
MODEL_PATH = os.path.join(SCRIPT_DIR, 'face_landmarker.task')
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def calculate_ear(eye_landmarks, landmarks, img_w, img_h):
    pts = np.array([(int(landmarks[idx].x * img_w), int(landmarks[idx].y * img_h)) for idx in eye_landmarks])
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3])
    ear = (A + B) / (2.0 * C)
    return ear, pts

EAR_THRESHOLD = 0.23
STAGE_1_LIMIT = 1.5
STAGE_2_LIMIT = 3.0
def generate_frames(source_frame=None, state=None):
    cap = cv2.VideoCapture(0) if source_frame is None else None
    state = state if state is not None else {}
    closed_start_time = state.get("closed_start_time")
    elapsed_time = state.get("elapsed_time", 0.0)
    alarm_active = state.get("alarm_active", False)

    while True:
        if source_frame is None:
            ret, frame = cap.read()
        else:
            ret, frame = True, source_frame.copy()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)

        avg_ear = 0.0
        status_txt = "AWAKE (सतर्क)"
        status_color = (0, 255, 127)

        if not detection_result.face_landmarks:
            closed_start_time = None
            elapsed_time = 0.0
            alarm_active = False
            stop_buzzer()

        if detection_result.face_landmarks:
            for face_landmarks in detection_result.face_landmarks:
                left_ear, left_pts = calculate_ear(LEFT_EYE, face_landmarks, w, h)
                right_ear, right_pts = calculate_ear(RIGHT_EYE, face_landmarks, w, h)
                avg_ear = (left_ear + right_ear) / 2.0

                if elapsed_time >= STAGE_2_LIMIT:
                    eye_color = (0, 0, 255)
                elif elapsed_time >= STAGE_1_LIMIT:
                    eye_color = (0, 215, 255)
                else:
                    eye_color = (0, 255, 127)

                cv2.polylines(frame, [left_pts], True, eye_color, 2, cv2.LINE_AA)
                cv2.polylines(frame, [right_pts], True, eye_color, 2, cv2.LINE_AA)

                if avg_ear < EAR_THRESHOLD:
                    if closed_start_time is None:
                        closed_start_time = time.time()

                    elapsed_time = time.time() - closed_start_time

                    if elapsed_time >= STAGE_2_LIMIT:
                        status_txt = "CRITICAL (झोपेत)"
                        status_color = (0, 0, 255)
                        draw_rounded_rect(frame, (5, 5), (w - 5, h - 5), (0, 0, 255), radius=10, thickness=12)

                        box_w, box_h = 500, 180
                        cx, cy = w // 2, h // 2 - 40
                        draw_overlay_panel(frame, (cx - box_w//2, cy - box_h//2), (cx + box_w//2, cy + box_h//2), bg_color=(0, 0, 40), alpha=0.9)
                        draw_rounded_rect(frame, (cx - box_w//2, cy - box_h//2), (cx + box_w//2, cy + box_h//2), (0, 0, 255), radius=15, thickness=2)

                        frame = draw_unicode_text(frame, "DROWSINESS DETECTED!", (cx - 180, cy - 70), 26, (0, 0, 255))
                        frame = draw_unicode_text(frame, "हिंदी: चालक को नींद आ रही है!", (cx - 160, cy - 20), 22, (255, 255, 255))
                        frame = draw_unicode_text(frame, "मराठी: चालकाला झोप येत आहे!", (cx - 160, cy + 25), 22, (255, 255, 255))
                        start_buzzer()
                        alarm_active = True

                    elif elapsed_time >= STAGE_1_LIMIT:
                        status_txt = "WARNING (सावधान)"
                        status_color = (0, 215, 255)
                        draw_rounded_rect(frame, (5, 5), (w - 5, h - 5), (0, 215, 255), radius=10, thickness=8)

                        box_w, box_h = 520, 130
                        cx, cy = w // 2, h // 2 - 40
                        draw_overlay_panel(frame, (cx - box_w//2, cy - box_h//2), (cx + box_w//2, cy + box_h//2), bg_color=(20, 30, 40), alpha=0.9)
                        draw_rounded_rect(frame, (cx - box_w//2, cy - box_h//2), (cx + box_w//2, cy + box_h//2), (0, 215, 255), radius=15, thickness=2)

                        frame = draw_unicode_text(frame, "WARNING: DROWSINESS APPROACHING!", (cx - 230, cy - 45), 22, (0, 215, 255))
                        frame = draw_unicode_text(frame, "सावधान: सतर्क रहें / सावध राहा!", (cx - 150, cy + 5), 20, (255, 255, 255))

                        stop_buzzer()
                        alarm_active = False

                else:
                    closed_start_time = None
                    elapsed_time = 0.0
                    stop_buzzer()
                    alarm_active = False

        frame = render_full_car_interior(frame, w, h, avg_ear, elapsed_time, status_txt, status_color)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        state["closed_start_time"] = closed_start_time
        state["elapsed_time"] = elapsed_time
        state["alarm_active"] = alarm_active

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        if source_frame is not None:
            break

    if cap is not None:
        cap.release()

@app.route('/')
def index():
    return '''<!doctype html>
<html>
<head>
    <title>Drowsiness Detector Cockpit</title>
    <style>
        body { margin: 0; background: #000; color: #fff; font: 16px Arial, sans-serif; }
        #sound-controls { position: fixed; z-index: 1; top: 12px; left: 12px; }
        #enable-sound { padding: 10px 14px; cursor: pointer; }
        #sound-status { margin-left: 8px; }
        #video { display: block; max-width: 100%; max-height: 100vh; margin: auto; }
        #location-controls { position: fixed; z-index: 1; top: 56px; left: 12px; }
        #location-status { margin-left: 8px; }
        #camera-status { position: fixed; z-index: 1; top: 100px; left: 12px; }
        #logout { margin-left: 8px; }
    </style>
</head>
<body>
    <div id="sound-controls">
        <button id="enable-sound">Enable buzzer sound</button>
        <span id="sound-status">Click to allow alarm audio.</span>
        <form action="/logout" method="post" style="display:inline">
            <button id="logout" type="submit">Log out</button>
        </form>
    </div>
    <div id="location-controls">
        <button id="enable-location">Share my location</button>
        <span id="location-status">Location is not requested unless you click.</span>
    </div>
    <div id="camera-status">Allow camera access to start the detector.</div>
    <video id="camera" autoplay playsinline muted hidden></video>
    <canvas id="frame-canvas" hidden></canvas>
    <audio id="buzzer" src="/emergency_siren.wav" loop></audio>
    <img id="video" alt="Drowsiness detector camera feed">
    <script>
        const buzzer = document.getElementById("buzzer");
        const enableButton = document.getElementById("enable-sound");
        const soundStatus = document.getElementById("sound-status");
        const locationButton = document.getElementById("enable-location");
        const locationStatus = document.getElementById("location-status");
        const cameraStatus = document.getElementById("camera-status");
        const camera = document.getElementById("camera");
        const canvas = document.getElementById("frame-canvas");
        const context = canvas.getContext("2d");
        const output = document.getElementById("video");
        let soundEnabled = false;
        let alarmActive = false;
        let cameraStream = null;
        let previousImageUrl = null;

        locationButton.addEventListener("click", () => {
            if (!navigator.geolocation) {
                locationStatus.textContent = "Location is not supported by this browser.";
                return;
            }
            locationStatus.textContent = "Waiting for location permission...";
            navigator.geolocation.getCurrentPosition(
                position => {
                    const latitude = position.coords.latitude.toFixed(5);
                    const longitude = position.coords.longitude.toFixed(5);
                    locationStatus.textContent = `Location: ${latitude}, ${longitude}`;
                },
                error => {
                    const messages = {
                        1: "Location permission was denied.",
                        2: "Your location is unavailable.",
                        3: "Location request timed out."
                    };
                    locationStatus.textContent = messages[error.code] || "Could not get your location.";
                },
                { enableHighAccuracy: true, timeout: 15000, maximumAge: 0 }
            );
        });

        enableButton.addEventListener("click", async () => {
            soundEnabled = true;
            enableButton.textContent = "Buzzer sound enabled";
            soundStatus.textContent = alarmActive ? "Alarm active" : "Sound ready.";
            if (alarmActive) {
                try {
                    await buzzer.play();
                } catch (error) {
                    soundStatus.textContent = "Could not play buzzer audio.";
                    console.error("Buzzer playback failed:", error);
                }
            }
        });

        async function checkAlarm() {
            try {
                const response = await fetch("/alarm_status", { cache: "no-store" });
                if (!response.ok) throw new Error(`Alarm status returned ${response.status}`);
                const data = await response.json();
                const alarmStarted = data.active && !alarmActive;
                alarmActive = data.active;

                if (alarmActive && soundEnabled && buzzer.paused) {
                    await buzzer.play();
                } else if (!alarmActive && !buzzer.paused) {
                    buzzer.pause();
                    buzzer.currentTime = 0;
                }
                if (alarmStarted && "speechSynthesis" in window) {
                    window.speechSynthesis.cancel();
                    window.speechSynthesis.speak(new SpeechSynthesisUtterance(
                        "Alert! Driver is sleeping. Please wake up."
                    ));
                }
                soundStatus.textContent = alarmActive
                    ? (soundEnabled ? "Alarm active" : "Click to allow alarm audio.")
                    : (soundEnabled ? "Sound ready." : "Click to allow alarm audio.");
            } catch (error) {
                soundStatus.textContent = "Unable to check alarm status.";
                console.error("Alarm status check failed:", error);
            }
        }

        async function captureLoop() {
            if (!cameraStream) return;
            const width = Math.min(camera.videoWidth, 640);
            const height = Math.round(camera.videoHeight * width / camera.videoWidth);
            if (!width || !height) {
                window.setTimeout(captureLoop, 200);
                return;
            }

            canvas.width = width;
            canvas.height = height;
            context.drawImage(camera, 0, 0, width, height);
            const frame = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", 0.75));
            if (frame && cameraStream) {
                try {
                    const response = await fetch("/process_frame", {
                        method: "POST",
                        headers: { "Content-Type": "image/jpeg" },
                        body: frame
                    });
                    if (!response.ok) throw new Error(`Frame processing returned ${response.status}`);
                    const imageUrl = URL.createObjectURL(await response.blob());
                    output.src = imageUrl;
                    if (previousImageUrl) URL.revokeObjectURL(previousImageUrl);
                    previousImageUrl = imageUrl;
                    cameraStatus.textContent = "Camera is running.";
                } catch (error) {
                    cameraStatus.textContent = "Unable to process camera frames.";
                    console.error("Camera frame processing failed:", error);
                }
            }
            window.setTimeout(captureLoop, 120);
        }

        async function startCamera() {
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                cameraStatus.textContent = "Camera access requires HTTPS and a supported browser.";
                return;
            }
            try {
                cameraStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
                camera.srcObject = cameraStream;
                await camera.play();
                cameraStatus.textContent = "Starting camera...";
                captureLoop();
            } catch (error) {
                cameraStatus.textContent = "Camera access was denied or is unavailable.";
                console.error("Camera startup failed:", error);
            }
        }

        window.addEventListener("pagehide", () => {
            if (cameraStream) cameraStream.getTracks().forEach(track => track.stop());
            if (previousImageUrl) URL.revokeObjectURL(previousImageUrl);
        });

        setInterval(checkAlarm, 500);
        checkAlarm();
        startCamera();
    </script>
</body>
</html>'''

@app.route('/login', methods=['GET', 'POST'])
def login():
    with auth_connection() as connection:
        account_exists = connection.execute("SELECT 1 FROM accounts LIMIT 1").fetchone()
    if not account_exists:
        return redirect(url_for("signup"))

    error = ""
    if request.method == 'POST':
        submitted_username = request.form.get("username", "")
        submitted_password = request.form.get("password", "")
        with auth_connection() as connection:
            account = connection.execute(
                "SELECT username, password_hash FROM accounts WHERE username = ?",
                (submitted_username,),
            ).fetchone()
        if account and check_password_hash(account[1], submitted_password):
            session.clear()
            session["authenticated"] = True
            session["username"] = account[0]
            return redirect(url_for("index"))
        error = "Incorrect username or password."

    return f'''<!doctype html>
<html>
<head><title>Sign in - Drowsiness Detector</title>
<style>
    body {{ margin: 0; min-height: 100vh; display: grid; place-items: center; background: #111; color: #fff; font: 16px Arial, sans-serif; }}
    form {{ display: grid; gap: 12px; width: min(320px, 85vw); padding: 24px; background: #222; border-radius: 8px; }}
    input, button {{ box-sizing: border-box; padding: 10px; font: inherit; }}
    .error {{ color: #ff7777; }}
</style></head>
<body>
    <form method="post">
        <h2>Sign in</h2>
        <label>Username <input name="username" autocomplete="username" required></label>
        <label>Password <input type="password" name="password" autocomplete="current-password" required></label>
        <button type="submit">Sign in</button>
        <span class="error">{error}</span>
        <a href="{url_for('signup')}" style="color:#9cf">Create a new account</a>
    </form>
</body>
</html>'''

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error = ""
    if request.method == 'POST':
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username or len(username) > 80:
            error = "Enter a username between 1 and 80 characters."
        elif len(password) < 8:
            error = "Password must be at least 8 characters."
        elif password != confirm_password:
            error = "Passwords do not match."
        else:
            password_hash = generate_password_hash(password)
            try:
                with auth_connection() as connection:
                    connection.execute(
                        "INSERT INTO accounts (username, password_hash) VALUES (?, ?)",
                        (username, password_hash),
                    )
            except sqlite3.IntegrityError:
                error = "That username already exists. Please choose another."
            else:
                session.clear()
                session["authenticated"] = True
                session["username"] = username
                return redirect(url_for("index"))

    return f'''<!doctype html>
<html>
<head><title>Create account - Drowsiness Detector</title>
<style>
    body {{ margin: 0; min-height: 100vh; display: grid; place-items: center; background: #111; color: #fff; font: 16px Arial, sans-serif; }}
    form {{ display: grid; gap: 12px; width: min(320px, 85vw); padding: 24px; background: #222; border-radius: 8px; }}
    input, button {{ box-sizing: border-box; padding: 10px; font: inherit; }}
    .error {{ color: #ff7777; }}
</style></head>
<body>
    <form method="post">
        <h2>Create a new account</h2>
        <p>Create a username and password to use this app.</p>
        <label>Username <input name="username" autocomplete="username" maxlength="80" required></label>
        <label>Password <input type="password" name="password" autocomplete="new-password" minlength="8" required></label>
        <label>Confirm password <input type="password" name="confirm_password" autocomplete="new-password" minlength="8" required></label>
        <button type="submit">Create account</button>
        <span class="error">{error}</span>
        <a href="{url_for('login')}" style="color:#9cf">Already have an account? Sign in</a>
    </form>
</body>
</html>'''

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route('/alarm_status')
def alarm_status():
    return jsonify(active=session.get("alarm_active", False))

@app.route('/emergency_siren.wav')
def emergency_siren():
    return send_file(SIREN_PATH, mimetype='audio/wav')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    if request.mimetype != "image/jpeg":
        abort(415)
    frame = cv2.imdecode(np.frombuffer(request.get_data(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        abort(400, description="Could not decode the submitted camera frame.")

    state = {
        "closed_start_time": session.get("closed_start_time"),
        "elapsed_time": session.get("elapsed_time", 0.0),
    }
    processed_frame = next(generate_frames(source_frame=frame, state=state), None)
    if processed_frame is None:
        abort(500, description="Could not process the submitted camera frame.")

    session["closed_start_time"] = state["closed_start_time"]
    session["elapsed_time"] = state["elapsed_time"]
    session["alarm_active"] = state["alarm_active"]
    image = processed_frame.split(b"\r\n\r\n", 1)[1].rsplit(b"\r\n", 1)[0]
    return Response(image, mimetype="image/jpeg")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", "5000")))