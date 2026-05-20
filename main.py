import cv2
import numpy as np
import sys
import threading
import time
from collections import deque

# =====================================================================
# 1. CẤU HÌNH AI (MOBILENET-SSD) VÀ HIỆU NĂNG
# =====================================================================
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

PROTOTXT = "MobileNetSSD_deploy.prototxt"
MODEL = "MobileNetSSD_deploy.caffemodel"

print("[INFO] Đang tải mô hình AI MobileNet-SSD...")
try:
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
except Exception as e:
    print("[ERROR] Không tìm thấy file mô hình! Vui lòng tải file .prototxt và .caffemodel")
    sys.exit(1)

CONFIDENCE_THRESHOLD = 0.5 

# Kích thước khung hình hiển thị và xử lý (Phần mềm)
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
TARGET_FPS = 8  

# =====================================================================
# 2. CẤU HÌNH 1 VẠCH DỌC VÀ BIẾN ĐẾM
# =====================================================================
LINE_X = int(FRAME_WIDTH / 2)

TOTAL_IN = 0
TOTAL_OUT = 0
PEOPLE_IN_ROOM = 0

MAX_DISAPPEARED = 30     

trackable_objects = {}
next_object_id = 0
WINDOW_NAME = "Dem nguoi AI (Ban Sieu Nhe)"

# =====================================================================
# 3. HÀM BẮT SỰ KIỆN CHUỘT 
# =====================================================================
def adjust_counters(event, x, y, flags, param):
    global TOTAL_IN, TOTAL_OUT, PEOPLE_IN_ROOM
    if event == cv2.EVENT_LBUTTONDOWN:
        if 190 <= y <= 202: 
            if 70 <= x <= 85: TOTAL_IN = max(0, TOTAL_IN - 1)
            elif 95 <= x <= 110: TOTAL_IN += 1
        elif 205 <= y <= 217: 
            if 70 <= x <= 85: TOTAL_OUT = max(0, TOTAL_OUT - 1)
            elif 95 <= x <= 110: TOTAL_OUT += 1
        elif 220 <= y <= 232: 
            if 70 <= x <= 85: PEOPLE_IN_ROOM = max(0, PEOPLE_IN_ROOM - 1)
            elif 95 <= x <= 110: PEOPLE_IN_ROOM += 1

# =====================================================================
# 4. HÀM XÁC ĐỊNH VỊ TRÍ 
# =====================================================================
def get_zone(cx):
    if cx < LINE_X:
        return "LEFT"
    else:
        return "RIGHT"

# =====================================================================
# 5. KHỞI TẠO CAMERA (Đã sửa lỗi sọc hình)
# =====================================================================
class FrameGrabber:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.stopped = False
        self.frame = None
        self.lock = threading.Lock()
        
        # ĐỂ PHẦN CỨNG CHẠY Ở 640x480 ĐỂ KHÔNG BỊ LỖI SỌC ĐEN TRẮNG
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30) 
        
        threading.Thread(target=self._reader, daemon=True).start()

    def isOpened(self): return self.cap.isOpened()
    
    def _reader(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def release(self):
        self.stopped = True
        time.sleep(0.05)
        self.cap.release()

vs = FrameGrabber(0)
if not vs.isOpened():
    print("[ERROR] Không thể kết nối camera. Nhớ dùng lệnh libcamerify.")
    sys.exit(1)

cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, adjust_counters)

print(f"[INFO] Hệ thống chạy ở chế độ Siêu Nhẹ ({FRAME_WIDTH}x{FRAME_HEIGHT} @ {TARGET_FPS} FPS)...")

# =====================================================================
# VÒNG LẶP XỬ LÝ CHÍNH
# =====================================================================
prev_time = 0

while True:
    frame = vs.read()
    if frame is None:
        time.sleep(0.01)
        continue

    # --- BỘ LỌC FPS ---
    current_time = time.time()
    if (current_time - prev_time) < (1.0 / TARGET_FPS):
        time.sleep(0.005)
        continue
    
    prev_time = current_time 

    # --- ÉP ĐỘ PHÂN GIẢI BẰNG PHẦN MỀM ĐỂ LÀM NHẸ PI ---
    # Ảnh gốc 640x480 đẹp, sạch sẽ được bóp lại thành 320x240 ở đây
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    (h, w) = frame.shape[:2]
    current_centroids = []

    # --- CHẠY AI MOBILENET-SSD ---
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > CONFIDENCE_THRESHOLD:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] != "person":
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            current_centroids.append((cX, cY, startX, startY, endX, endY))

    # --- LOGIC TRACKING (1 Vạch) ---
    updated_trackable_objects = dict(trackable_objects)
    seen_ids = set()

    for (cX, cY, startX, startY, endX, endY) in current_centroids:
        matched_id = None
        min_distance = 80  

        for obj_id, (old_cX, old_cY, zone_history, disappeared) in trackable_objects.items():
            d = np.hypot(cX - old_cX, cY - old_cY)
            if d < min_distance:
                min_distance = d
                matched_id = obj_id

        if matched_id is None:
            matched_id = next_object_id
            next_object_id += 1
            zone_history = deque([get_zone(cX)], maxlen=10) 
            disappeared = 0
        else:
            old_cX, old_cY, zone_history, disappeared = trackable_objects[matched_id]
            current_zone = get_zone(cX)
            if len(zone_history) == 0 or zone_history[-1] != current_zone:
                zone_history.append(current_zone)
            disappeared = 0

        final_compressed = []
        for z in zone_history:
            if not final_compressed or final_compressed[-1] != z:
                final_compressed.append(z)

        if "LEFT" in final_compressed and "RIGHT" in final_compressed:
            idx_left = final_compressed.index("LEFT")
            idx_right = final_compressed.index("RIGHT")
            
            if idx_left < idx_right:
                TOTAL_IN += 1
                PEOPLE_IN_ROOM += 1
                zone_history = deque(["RIGHT"], maxlen=10)
                
            elif idx_right < idx_left:
                TOTAL_OUT += 1
                PEOPLE_IN_ROOM = max(0, PEOPLE_IN_ROOM - 1) 
                zone_history = deque(["LEFT"], maxlen=10)

        updated_trackable_objects[matched_id] = (cX, cY, zone_history, disappeared)
        seen_ids.add(matched_id)

        cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 150, 0), 2)
        cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
        
        current_zone_str = get_zone(cX)
        text = f"ID:{matched_id} | {current_zone_str}"
        cv2.putText(frame, text, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 150, 0), 1)

    for obj_id in list(updated_trackable_objects.keys()):
        if obj_id not in seen_ids:
            cX, cY, zone_history, disappeared = updated_trackable_objects[obj_id]
            disappeared += 1
            if disappeared > MAX_DISAPPEARED:
                del updated_trackable_objects[obj_id]
            else:
                updated_trackable_objects[obj_id] = (cX, cY, zone_history, disappeared)

    trackable_objects = updated_trackable_objects

    # =================================================================
    # 6. VẼ GIAO DIỆN 
    # =================================================================
    cv2.line(frame, (LINE_X, 0), (LINE_X, h), (0, 255, 255), 2)
    
    cv2.putText(frame, "RA (<--)", (LINE_X - 65, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.putText(frame, "(-->) VAO", (LINE_X + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    cv2.rectangle(frame, (0, 180), (120, 240), (0, 0, 0), -1)

    cv2.putText(frame, f"Vao: {TOTAL_IN}", (5, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(frame, f"Ra: {TOTAL_OUT}", (5, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.putText(frame, f"Trg: {PEOPLE_IN_ROOM}", (5, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    for y_btn in [190, 205, 220]:
        cv2.rectangle(frame, (70, y_btn), (85, y_btn + 12), (80, 80, 80), -1)
        cv2.putText(frame, "-", (73, y_btn + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.rectangle(frame, (95, y_btn), (110, y_btn + 12), (80, 80, 80), -1)
        cv2.putText(frame, "+", (98, y_btn + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.imshow(WINDOW_NAME, frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vs.release()
cv2.destroyAllWindows()