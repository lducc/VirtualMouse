# VirtualMouse

This project is a hand-gesture-controlled virtual mouse, This project uses **Computer Vision** to track your hand in real time and translate finger patterns into Windows mouse actions.

## Files

*   `virtual_mouse.py` – The main file for the application
*   `requirements.txt` – Needed library dependencies
*   `README.md` – This file you are reading
*   `config.json` - Customizable configs in JSON


## Setup Guide

### 1. Create an Environment (Optional)

**Option A: Using Conda**

```bash
conda create -n <env-name> python=3.10 -y
conda activate <env-name>
```

**Option B: Using Venv**

```bash
python -m venv venv
# Windows:
    venv\Scripts\activate
# Mac/Linux:
    source venv/bin/activate
```
---
### 2. Setup Files and Dependencies
1. **Clone the project** or download the source code.
2. **Open your terminal** in the project folder.
3. **Install dependencies** using the requirements file:
   ```bash
   pip install -r requirements.txt
   ```


## How to Use

1. Ensure your webcam is connected.
2. Run the script:
   ```bash
   python main.py
   ```
3. **Controls:**
   - Press **'q'** in the webcam window to quit.
   - Set `"debug": false` in `config.json` to run the app in the background.

---

## Gesture Cheat Sheet
The mouse uses **Bitmasking** (binary math) to detect fingers.

*Values: Thumb=1, Index=2, Middle=4, Ring=8, Pinky=16*

| Action | Pattern (config.json) | Physical Gesture |
| :--- | :--- | :--- |
| **Move** | `6` | Index + Middle fingers up |
| **Left Click** | `4` | Middle finger only up |
| **Right Click** | `2` | Index finger only up |
| **Scroll** | `7` | Thumb + Index + Middle up |
| **Drag & Drop** | `31` | All 5 fingers up |

---

## Customization (`config.json`)
You can tweak the behavior of the mouse without touching any code:

- **`smoothing`**: Higher values = smoother but slower. Lower = snappier.
- **`reduction`**: Increase this if you want to reach screen corners with less hand movement.
- **`pointer_landmark`**: Use `8` for index finger-tip movement or `9` for knuckle-based stability (or customize it however you want based on MediaPipe hand landmarks notation)
- **`frames`**: How many frames to hold a gesture before it triggers (prevents accidental clicks).

---

>**Pro Tip:** Performance
If the mouse feels slow, open `config.json` and ensure `"debug"` is set to `false`. This disables the video window and the drawing logic, allowing the CPU to focus entirely on the hand-tracking math.
```

