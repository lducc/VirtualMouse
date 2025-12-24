import os, cv2, json, time
import datetime
import numpy as np
import mediapipe as mp
import ctypes 
import psutil
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple
from threading import Thread

# Direct Win32 Setup
user32 = ctypes.windll.user32
user32.SetProcessDPIAware() # Fixes scaling issues on 1080p/4K screens ? Allegedly

@dataclass
class Gesture:
    name: str
    pattern: int
    on_start: Optional[Callable] = None
    on_hold: Optional[Callable] = None
    on_exit: Optional[Callable] = None
    is_active: bool = False
    required_frames: int = 3
    counter : int = 0

class Finger:
    THUMB  = 1   # 00001
    INDEX  = 2   # 00010
    MIDDLE = 4   # 00100
    RING   = 8   # 01000
    PINKY  = 16  # 10000

class Mouse:
    def __init__(self, config: dict):
            self.screen_w , self.screen_h= user32.GetSystemMetrics(0),  user32.GetSystemMetrics(1) #1920, 1080
            self.cam_config = config["camera"]
            self.mouse_config = config["mouse"]
            
            self.prev_x = self.screen_w // 2
            self.prev_y = self.screen_h // 2
            self.curr_x, self.curr_y = 0, 0
            
            self.frame_reduction = 120 

    def move(self, x: int, y: int):
            frame_reduction = self.mouse_config["reduction"]
            # Mapping from [reduction, cam_w - reduction] to [0, screen_w]
            x_mapped = np.interp(x, 
                                (frame_reduction, self.cam_config["width"] - frame_reduction), 
                                (0, self.screen_w))
            y_mapped = np.interp(y, 
                                (frame_reduction, self.cam_config["height"] - frame_reduction), 
                                (0, self.screen_h))

            alpha = 1 / self.mouse_config["smoothing"]
            self.curr_x = (1 - alpha) * self.prev_x + alpha * x_mapped
            self.curr_y = (1 - alpha) * self.prev_y + alpha * y_mapped

            # Direct Win32 SetCursorPos (Instant)
            user32.SetCursorPos(int(self.curr_x), int(self.curr_y))
            
            self.prev_x, self.prev_y = self.curr_x, self.curr_y

    def left_click(self): 
        user32.mouse_event(2, 0, 0, 0, 0) # Left Down
        user32.mouse_event(4, 0, 0, 0, 0) # Left Up
    
    def right_click(self):
        user32.mouse_event(8, 0, 0, 0, 0)  # Right Down
        user32.mouse_event(16, 0, 0, 0, 0) # Right Up

    def drag_start(self): 
        user32.mouse_event(2, 0, 0, 0, 0)

    def drag_stop(self): 
        user32.mouse_event(4, 0, 0, 0, 0)

    def scroll(self, x: int, y: int):
        speed = self.mouse_config["scroll_speed"]
        if y < (self.cam_config["height"] // 2):
            direction = speed  
        else:
            direction = -speed
        user32.mouse_event(0x0800, 0, 0, direction, 0) # Wheel Scroll

    def double_click(self):
        self.left_click()
        self.left_click()
    
    def media_key(self, key_code):
        # volume up = 0xAF, volume down = 0xAE, play/pause = 0xB3
        user32.keybd_event(key_code, 0, 0, 0) # Key Down
        user32.keybd_event(key_code, 0, 2, 0) # Key Up


class WebcamStream:
    def __init__(self, config: dict):
        c = config["camera"]

        self.cap = cv2.VideoCapture(c["device_index"], cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FPS, c["fps"])
        self.cap.set(3, c["width"])
        self.cap.set(4, c["height"])
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.success, self.frame = self.cap.read()
        self.running = True

    def start(self):
        #Making a new thread just for capturing a new camera frame
        #so that the program can just continue processing without having to wait to cap.read()
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while self.running:
            self.success, self.frame = self.cap.read()
            if not self.success:
                self.running = False

    def read(self) -> np.ndarray:
        return self.frame

    def stop(self):
        self.running = False
        self.cap.release()

class GestureEngine:
    def __init__(self, mouse: Mouse, config: dict):
        self.mouse = mouse
        self.gestures: List[Gesture] = []
        self._load_config(config["gestures"])

    def _load_config(self, g_list: list):
        # Maps the string name in JSON to the mouse methods
        action_map = {
            "Move": {"on_hold": lambda x, y: self.mouse.move(x, y)},
            "Left Click": {"on_start": lambda x, y: self.mouse.left_click()},
            "Right Click": {"on_start": lambda x, y: self.mouse.right_click()},
            "Drag": {
                "on_start": lambda x, y: self.mouse.drag_start(),
                "on_hold": lambda x, y: self.mouse.move(x, y),
                "on_exit": lambda x, y: self.mouse.drag_stop()
            },
            "Scroll": {"on_hold": lambda x, y: self.mouse.scroll(x, y)},
            "Double Click": {"on_start": lambda x, y: self.mouse.double_click()},
            "Volumn Up": {"on_hold": lambda x, y: self.mouse.media_key(0xAF)},
            "Volumn Down": {"on_hold": lambda x, y: self.mouse.media_key(0xAE)},
            "Play/Pause": {"on_start": lambda x, y: self.mouse.media_key(0xB3)}
        }

        for g in g_list:
            act = action_map.get(g["name"], {})
            self.gestures.append(Gesture(
                name=g["name"], pattern=g["pattern"], 
                required_frames=g["frames"], **act
            ))

    def update(self, pattern: int, x: int, y: int):
        for gesture in self.gestures:
            if pattern == gesture.pattern:
                #Check if gesture is active before, if active then it is holding, if not then it is starting
                gesture.counter += 1
                if gesture.counter >= gesture.required_frames:
                    if not gesture.is_active:
                        if gesture.on_start:
                            gesture.on_start(x, y)
                        gesture.is_active = True
                    
                    if gesture.on_hold:
                        gesture.on_hold(x, y)
                
            else:
                #Check if gesture is active before, if yes then it is exitting the gesture
                if gesture.is_active:
                    if gesture.on_exit:
                        gesture.on_exit(x, y)
                    gesture.is_active = False
                gesture.counter = 0

class HandTracker():
    TIPS_IDS = [8, 12, 16, 20]
    PIPS_IDS = [6, 10, 14, 18]

    def __init__(self, config: dict):
        self.cam_config = config["camera"]
        self.debug = config["debug"]
        self.MP_DRAW = mp.solutions.drawing_utils
        self.MP_HANDS = mp.solutions.hands
        self.hands = self.MP_HANDS.Hands(static_image_mode=False,
                                        max_num_hands=1,
                                        model_complexity=0, # Speed improvement
                                        min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5)
        self.hand_label = None
        self.result = None

    def find_hands(self, image: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        self.result = self.hands.process(image_rgb)
        image_rgb.flags.writeable = True

        if self.result.multi_hand_landmarks:
            self.hand_label = self.result.multi_handedness[0].classification[0].label
            for hand in self.result.multi_hand_landmarks:
                if self.debug:
                    self.MP_DRAW.draw_landmarks(image, hand, self.MP_HANDS.HAND_CONNECTIONS)
                
        return image

    def get_point(self, landmark_id: int) -> Tuple[int, int]:
        if not self.result.multi_hand_landmarks:
            return 0, 0
        lm = self.result.multi_hand_landmarks[0].landmark[landmark_id]
        return int(lm.x * self.cam_config["width"]), int(lm.y * self.cam_config["height"])
    
    def find_fingers_up(self) -> int:
        mask = 0

        if not self.result.multi_hand_landmarks:
            return mask
        
        lms = self.result.multi_hand_landmarks[0].landmark

        # Determine if palm is facing camera or not + if thumb is open or closed
        pinky_tip, thumb_tip, thumb_ip = lms[20], lms[4], lms[3]
        if self.hand_label == "Right":
            is_palm_facing = True if pinky_tip.x > thumb_tip.x else False
            thumb_up = True if (thumb_tip.x < thumb_ip.x if is_palm_facing else thumb_tip.x > thumb_ip.x) else False
        else:
            is_palm_facing = True if pinky_tip.x < thumb_tip.x else False
            thumb_up = True if (thumb_tip.x > thumb_ip.x if is_palm_facing else thumb_tip.x < thumb_ip.x) else False

        if thumb_up:
            mask |= (1 << 0)

        # Check for the remaining 4 fingers
        for i, (tip_id, pip_id) in enumerate(zip(self.TIPS_IDS, self.PIPS_IDS)):
            if lms[tip_id].y < lms[pip_id].y:
                mask |= (1 << (i + 1))

        return mask

def main():
    # Set high process priority for more performance
    try:
        p = psutil.Process(os.getpid())
        p.nice(psutil.HIGH_PRIORITY_CLASS)
    except: 
        pass
    
    with open("config.json", "r") as f:
        config = json.load(f)

    is_debug = config.get("debug", True)
    stream = WebcamStream(config).start() 
    mouse = Mouse(config)
    gesture_engine = GestureEngine(mouse, config)
    hand_tracker = HandTracker(config)

    p_time = time.time()

    if not is_debug:
        print("Press Ctrl+C in this terminal to exit.")

    while stream.running:
        frame = stream.read()

        if frame is None:
            continue
        frame = cv2.flip(frame, 1)
        frame = hand_tracker.find_hands(frame)
        
        if hand_tracker.result and hand_tracker.result.multi_hand_landmarks:
            pattern = hand_tracker.find_fingers_up()            
            # if pattern > 0:
            #     print(f"Current Bitmask: {pattern} | Hand: {hand_tracker.hand_label}")
            x, y = hand_tracker.get_point(config["mouse"]["pointer_landmark"])
            gesture_engine.update(pattern, x, y)
       
        if is_debug:
            curr_time = time.time()
            fps = 1 / (curr_time - p_time + 0.001)
            p_time = curr_time
            cv2.putText(frame, f'FPS: {np.around(fps, 1)}', (20, 50), 1, cv2.FONT_HERSHEY_PLAIN, (0, 255, 0), 2)
            # print(fps)
            cv2.imshow("Webcam", frame)

            if cv2.waitKey(1) == ord("q"):
                stream.stop()
                break
        else:
            time.sleep(0.00005)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()