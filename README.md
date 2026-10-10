# BlinkGuard — OpenCV Software Test

A webcam-based drowsiness-detection prototype using OpenCV + MediaPipe Face Mesh.
It calculates the Eye Aspect Ratio (EAR) from eye landmarks and warns when the eyes
appear closed for longer than a configurable duration.

## Requirements
- Windows 10/11 recommended
- Python 3.11 (recommended for the pinned MediaPipe version)
- A working webcam
- Good front lighting; keep your face visible

## Setup (Windows / PowerShell)
1. Install Python 3.11 from https://www.python.org/downloads/release/python-3119/
   During installation, select **Add Python to PATH**.
2. Extract this folder, then open PowerShell inside it.
3. Create a virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

   If PowerShell blocks activation, use:
   `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
   and activate again.

4. Run:

   ```powershell
   python main.py
   ```

5. Press **Q** or **Esc** while the camera window is selected to quit.

## Testing it
- Sit in front of the webcam with your face well lit.
- Normal open eyes should show `STATUS: ALERT` (meaning awake/attentive).
- Close your eyes gently for about 1.5 seconds; the red `DROWSINESS ALERT!`
  and audible warning should activate.
- Do not test while driving. Test while seated safely at a desk.

## Tuning
Open `main.py` and adjust:
- `EAR_THRESHOLD = 0.21`: lower it if normal open eyes trigger warnings; raise it if
  eye closure is not detected. Typical values vary by person and camera.
- `CLOSED_SECONDS = 1.5`: how long eyes must remain below the threshold before alerting.

## Troubleshooting
- **No camera:** close Zoom/Teams/browser tabs using the camera; check Windows
  Settings → Privacy & security → Camera. If needed, change `VideoCapture(0)` to
  `VideoCapture(1)` in `main.py`.
- **MediaPipe install error:** make sure you are using 64-bit Python 3.11, then recreate
  the virtual environment and install the requirements again.
- **PowerShell activation error:** see the activation command above.
- **No sound:** check system volume. The on-screen warning should still work.

## Important limitations
This is a student prototype, not a certified driver-safety device. Eye closure alone
does not reliably identify every form of drowsiness and can be affected by glasses,
lighting, camera angle, and individual facial features. Never rely on it as the sole
safety measure in a real vehicle.
