import cv2
import numpy as np
import ncnn
import time
from picam2 import Picam2 # Thư viện chính thức của Pi

# 1. KHỞI TẠO CAMERA BẰNG PICAM2
picam2 = Picam2()
config = picam2.configure("main")
picam2.start()

# 2. KHỞI TẠO YOLO
yolo = YoloFastestV2("yolo-fastestv2.param", "yolo-fastestv2.bin")

print("--- BẮT ĐẦU TEST (PICAM2) ---")
while True:
    # Lấy ảnh trực tiếp từ hardware buffer (Không lỗi reshape)
    frame = picam2.capture_array() 
    
    # Picam2 trả về RGB, cần chuyển sang BGR cho OpenCV hiển thị
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Gọi AI
    results = yolo.detect(frame)
    
    cv2.putText(frame, f"Found: {len(results)}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("TEST AI", frame)
    
    if cv2.waitKey(1) == 27: break

picam2.stop()
cv2.destroyAllWindows()