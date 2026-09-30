from picamera2 import Picamera2
import threading
import cv2
import time

class CameraThread:
    def __init__(self):
        print("[INFO] Đang khởi động Picamera2 (IMX219: 3280x2464 -> ISP Resize)...")
        self.picam2 = Picamera2()
        
        # Mở full 8MP ở luồng main, bóp xuống 640x480 YUV420 cho AI xử lý mượt
        self.config = self.picam2.create_video_configuration(
            main={"size": (3280, 2464), "format": "RGB888"},
            lores={"size": (640, 480), "format": "YUV420"},
            controls={"FrameRate": 30} 
        )
        
        self.picam2.configure(self.config)
        self.picam2.start()
        
        self.stopped = False
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        global latest_frame, cam_fps
        prev_time = time.time()
        frames_count = 0
        
        while not self.stopped:
            try:
                frame_yuv = self.picam2.capture_array("lores")
                frame_bgr = cv2.cvtColor(frame_yuv, cv2.COLOR_YUV2BGR_I420)
                
                with frame_lock:
                    latest_frame = frame_bgr
                
                frames_count += 1
                now = time.time()
                if now - prev_time >= 1.0:
                    cam_fps = frames_count / (now - prev_time)
                    frames_count = 0
                    prev_time = now
            except Exception:
                time.sleep(0.01)

    def stop(self):
        self.stopped = True
        self.picam2.stop()