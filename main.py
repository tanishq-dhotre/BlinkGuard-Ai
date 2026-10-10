import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import platform

# BlinkGuard: webcam-based driver drowsiness detection prototype.
# This is for controlled testing only, not a certified vehicle safety system.

# MediaPipe Face Mesh eye landmark indices (six points per eye).
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.21       # Tune for your face/camera if needed
CLOSED_SECONDS = 1.5       # Eye closure duration before warning
ALARM_REPEAT_SECONDS = 2.0

alarm_stop = threading.Event()
alarm_thread = None


def eye_aspect_ratio(eye_points):
    """Calculate Eye Aspect Ratio (EAR) from 6 ordered eye points."""
    p1, p2, p3, p4, p5, p6 = eye_points
    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def alarm_worker():
    """Run a repeating Windows beep without freezing the camera loop."""
    if platform.system() == "Windows":
        try:
            import winsound
            while not alarm_stop.is_set():
                winsound.Beep(1500, 250)
                if alarm_stop.wait(0.15):
                    break
                winsound.Beep(1000, 250)
                alarm_stop.wait(0.35)
        except Exception:
            print("\a", end="", flush=True)
    else:
        while not alarm_stop.is_set():
            print("\a", end="", flush=True)
            alarm_stop.wait(1.0)


def start_alarm():
    global alarm_thread
    if alarm_thread is None or not alarm_thread.is_alive():
        alarm_stop.clear()
        alarm_thread = threading.Thread(target=alarm_worker, daemon=True)
        alarm_thread.start()


def stop_alarm():
    alarm_stop.set()


def draw_eye_points(frame, points):
    for point in points:
        cv2.circle(frame, tuple(point.astype(int)), 2, (0, 255, 255), -1)


def main():
    mp_face_mesh = mp.solutions.face_mesh

    # Try the default webcam first.
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if platform.system() == "Windows" else 0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Check camera permissions or try another camera index.")
        return

    # Request a modest resolution for smoother performance.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    eyes_closed_since = None
    drowsy = False
    last_alarm_time = 0.0
    ear_value = 0.0

    print("BlinkGuard started. Press Q or ESC in the camera window to quit.")
    print("Keep your face visible and look toward the camera during testing.")

    try:
        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("ERROR: Could not read from webcam.")
                    break

                # Mirror view so it behaves like a normal webcam preview.
                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = face_mesh.process(rgb)
                rgb.flags.writeable = True

                now = time.time()
                face_found = bool(results.multi_face_landmarks)

                if face_found:
                    landmarks = results.multi_face_landmarks[0].landmark

                    def to_pixels(indices):
                        return np.array([
                            [landmarks[i].x * width, landmarks[i].y * height]
                            for i in indices
                        ], dtype=np.float32)

                    left_points = to_pixels(LEFT_EYE)
                    right_points = to_pixels(RIGHT_EYE)
                    draw_eye_points(frame, left_points)
                    draw_eye_points(frame, right_points)

                    left_ear = eye_aspect_ratio(left_points)
                    right_ear = eye_aspect_ratio(right_points)
                    ear_value = (left_ear + right_ear) / 2.0

                    if ear_value < EAR_THRESHOLD:
                        if eyes_closed_since is None:
                            eyes_closed_since = now
                        closed_duration = now - eyes_closed_since
                        if closed_duration >= CLOSED_SECONDS:
                            drowsy = True
                    else:
                        eyes_closed_since = None
                        drowsy = False

                    cv2.putText(frame, f"EAR: {ear_value:.3f}", (20, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                    cv2.putText(frame, f"Threshold: {EAR_THRESHOLD:.2f}", (20, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

                    if eyes_closed_since is not None and not drowsy:
                        remaining = max(0.0, CLOSED_SECONDS - (now - eyes_closed_since))
                        cv2.putText(frame, f"Eyes closed: {now - eyes_closed_since:.1f}s",
                                    (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 200, 255), 2)
                        cv2.putText(frame, f"Alert in: {remaining:.1f}s", (20, 122),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

                    if drowsy:
                        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 8)
                        cv2.putText(frame, "DROWSINESS ALERT!", (max(15, width // 2 - 190), 75),
                                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 3)
                        cv2.putText(frame, "Open your eyes / take a break", (20, height - 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                        if now - last_alarm_time >= ALARM_REPEAT_SECONDS:
                            start_alarm()
                            last_alarm_time = now
                    else:
                        stop_alarm()
                        last_alarm_time = 0.0
                        cv2.putText(frame, "STATUS: ALERT", (20, height - 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2)
                else:
                    eyes_closed_since = None
                    drowsy = False
                    stop_alarm()
                    last_alarm_time = 0.0
                    cv2.putText(frame, "Face not detected - face the camera", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)

                cv2.putText(frame, "BlinkGuard | Q/ESC: Quit", (20, height - 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("BlinkGuard - AI Drowsiness Detection", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        stop_alarm()
        cap.release()
        cv2.destroyAllWindows()
        print("BlinkGuard closed.")


if __name__ == "__main__":
    main()
