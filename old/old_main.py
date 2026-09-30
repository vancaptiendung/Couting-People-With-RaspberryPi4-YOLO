import cv2
import numpy as np
import sys
import threading
import time
import os
import glob
import json
import queue
from datetime import datetime
from collections import deque

# =====================================================================
# 1. KHỞI TẠO HỆ THỐNG LƯU TRỮ (JSON)
# =====================================================================
DATA_FILE = "counter_data.json"

def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                data = json.load(f)
                print(f"[INFO] Đã khôi phục số liệu cũ: {data}")
                return data.get("in", 0), data.get("out", 0), data.get("room", 0)
        except: pass
    return 0, 0, 0

def save_data(t_in, t_out, t_room):
    with open(DATA_FILE, "w") as f:
        json.dump({"in": t_in, "out": t_out, "room": t_room}, f)

TOTAL_IN, TOTAL_OUT, PEOPLE_IN_ROOM = load_data()

# =====================================================================
# 2. CẤU HÌNH AI & HIỆU NĂNG 
# =====================================================================
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

PROTOTXT = "MobileNetSSD_deploy.prototxt"
MODEL = "MobileNetSSD_deploy.caffemodel"

try:
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
except Exception as e:
    print("[ERROR] Không tìm thấy file mô hình AI!")
    sys.exit(1)

CONFIDENCE_THRESHOLD = 0.5 

# Kích thước siêu nhẹ cho phần mềm và AI xử lý
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
TARGET_FPS = 6  # KHÓA CỨNG MƯỢT MÀ 6 FPS

# =====================================================================
# 3. CẤU HÌNH VẠCH, ĐẾM & GHI HÌNH XOAY VÒNG
# =====================================================================
LINE_X = int(FRAME_WIDTH / 2)
is_dragging_line = False 
MAX_DISAPPEARED = 30     
trackable_objects = {}
next_object_id = 0
WINDOW_NAME = "CCTV AI - Pi Cam V2.1 (Real Max FOV)"

VIDEO_DIR = "videos"
MAX_VIDEOS = 3
CHUNK_DURATION = 30 * 60  
recording_enabled = False
video_writer = None
chunk_start_time = 0

if not os.path.exists(VIDEO_DIR):
    os.makedirs(VIDEO_DIR)

class ThreadedVideoWriter:
    def __init__(self, filename, fourcc, fps, frame_size):
        self.writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
        self.q = queue.Queue(maxsize=128)
        self.stopped = False
        threading.Thread(target=self._write, daemon=True).start()

    def write(self, frame):
        if not self.q.full(): self.q.put(frame.copy())
            
    def _write(self):
        while not self.stopped:
            if not self.q.empty(): self.writer.write(self.q.get())
            else: time.sleep(0.01) 
                
    def release(self):
        self.stopped = True
        while not self.q.empty(): self.writer.write(self.q.get())
        self.writer.release()

# =====================================================================
# 4. HÀM BẮT SỰ KIỆN CHUỘT
# =====================================================================
def adjust_counters(event, x, y, flags, param):
    global TOTAL_IN, TOTAL_OUT, PEOPLE_IN_ROOM, LINE_X
    global is_dragging_line, recording_enabled

    if event == cv2.EVENT_LBUTTONDOWN:
        if 245 <= x <= 315 and 5 <= y <= 25:
            recording_enabled = not recording_enabled
            return

        changed = False
        if 190 <= y <= 202: 
            if 70 <= x <= 85: TOTAL_IN = max(0, TOTAL_IN - 1); changed = True
            elif 95 <= x <= 110: TOTAL_IN += 1; changed = True
        elif 205 <= y <= 217: 
            if 70 <= x <= 85: TOTAL_OUT = max(0, TOTAL_OUT - 1); changed = True
            elif 95 <= x <= 110: TOTAL_OUT += 1; changed = True
        elif 220 <= y <= 232: 
            if 70 <= x <= 85: PEOPLE_IN_ROOM = max(0, PEOPLE_IN_ROOM - 1); changed = True
            elif 95 <= x <= 110: PEOPLE_IN_ROOM += 1; changed = True

        if changed: save_data(TOTAL_IN, TOTAL_OUT, PEOPLE_IN_ROOM)

        if abs(x - LINE_X) < 15: is_dragging_line = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_dragging_line: LINE_X = max(20, min(FRAME_WIDTH - 20, x))
    elif event == cv2.EVENT_LBUTTONUP:
        is_dragging_line = False

def get_zone(cx): return "INSIDE" if cx < LINE_X else "OUTSIDE"

# =====================================================================
# 5. KHỞI TẠO CAMERA (GÓC RỘNG THẬT FOV & CHỐNG SILENT CROP)
# =====================================================================
class FrameGrabber:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.stopped = False
        self.frame = None
        self.lock = threading.Lock()
        
        # SỬ DỤNG ĐÚNG ĐỘ PHÂN GIẢI CHUẨN CỦA IMX219 (1280x720)
        # Hệ thống sẽ lấy toàn bộ góc rộng theo chiều ngang và CÂN BẰNG TÂM TUYỆT ĐỐI
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        self.cap.set(cv2.CAP_PROP_FPS, 10) 
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
        
        threading.Thread(target=self._reader, daemon=True).start()

    def isOpened(self): return self.cap.isOpened()
    
    def _reader(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
                
            # Vẫn nén về siêu nhẹ để AI mượt mà 6 FPS
            small_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            
            with self.lock:
                self.frame = small_frame

    def read(self):
        with self.lock:
            if self.frame is None: return None
            f = self.frame.copy()
            self.frame = None 
            return f

    def release(self):
        self.stopped = True
        time.sleep(0.05)
        self.cap.release()

vs = FrameGrabber(0)
if not vs.isOpened():
    print("[ERROR] Lỗi Camera. Hãy kiểm tra kết nối!")
    sys.exit(1)

cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, adjust_counters)

print(f"[INFO] CCTV AI kích hoạt góc rộng thật FOV | Mượt 6 FPS...")

# =====================================================================
# VÒNG LẶP XỬ LÝ CHÍNH
# =====================================================================
prev_time = 0

while True:
    frame = vs.read()
    if frame is None:
        time.sleep(0.01)
        continue

    # --- KHÓA CHẶT TỐC ĐỘ 6 FPS ---
    current_time = time.time()
    if (current_time - prev_time) < (1.0 / TARGET_FPS):
        time.sleep(0.005) 
        continue
    prev_time = current_time 

    (h, w) = frame.shape[:2]
    current_centroids = []

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONFIDENCE_THRESHOLD:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] != "person": continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            current_centroids.append((cX, cY, startX, startY, endX, endY))

    # --- LOGIC ĐẾM & THEO DÕI ---
    updated_trackable_objects = dict(trackable_objects)
    seen_ids = set()
    counters_changed_this_frame = False

    for (cX, cY, startX, startY, endX, endY) in current_centroids:
        matched_id = None
        min_distance = 100 

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

        if "OUTSIDE" in final_compressed and "INSIDE" in final_compressed:
            idx_out = final_compressed.index("OUTSIDE")
            idx_in = final_compressed.index("INSIDE")
            
            if idx_out < idx_in:
                TOTAL_IN += 1
                PEOPLE_IN_ROOM += 1
                zone_history = deque(["INSIDE"], maxlen=10)
                counters_changed_this_frame = True
            elif idx_in < idx_out:
                TOTAL_OUT += 1
                PEOPLE_IN_ROOM = max(0, PEOPLE_IN_ROOM - 1) 
                zone_history = deque(["OUTSIDE"], maxlen=10)
                counters_changed_this_frame = True

        updated_trackable_objects[matched_id] = (cX, cY, zone_history, disappeared)
        seen_ids.add(matched_id)

        cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 150, 0), 2)
        cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
        text = f"ID:{matched_id} | {get_zone(cX)}"
        cv2.putText(frame, text, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 150, 0), 1)

    if counters_changed_this_frame:
        save_data(TOTAL_IN, TOTAL_OUT, PEOPLE_IN_ROOM)

    for obj_id in list(updated_trackable_objects.keys()):
        if obj_id not in seen_ids:
            cX, cY, zone_history, disappeared = updated_trackable_objects[obj_id]
            disappeared += 1
            if disappeared > MAX_DISAPPEARED: del updated_trackable_objects[obj_id]
            else: updated_trackable_objects[obj_id] = (cX, cY, zone_history, disappeared)

    trackable_objects = updated_trackable_objects

    # =================================================================
    # 7. VẼ GIAO DIỆN ĐỒ HỌA
    # =================================================================
    line_color = (0, 0, 255) if is_dragging_line else (0, 255, 255)
    line_thick = 3 if is_dragging_line else 2
    cv2.line(frame, (LINE_X, 0), (LINE_X, h), line_color, line_thick)
    
    cv2.putText(frame, "TRONG (<-- Vao)", (LINE_X - 110, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(frame, "(Ra -->) NGOAI", (LINE_X + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    cv2.rectangle(frame, (0, 180), (120, 240), (0, 0, 0), -1)
    cv2.putText(frame, f"Vao: {TOTAL_IN}", (5, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(frame, f"Ra: {TOTAL_OUT}", (5, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.putText(frame, f"Trg: {PEOPLE_IN_ROOM}", (5, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    for y_btn in [190, 205, 220]:
        cv2.rectangle(frame, (70, y_btn), (85, y_btn + 12), (80, 80, 80), -1)
        cv2.putText(frame, "-", (73, y_btn + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.rectangle(frame, (95, y_btn), (110, y_btn + 12), (80, 80, 80), -1)
        cv2.putText(frame, "+", (98, y_btn + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    rec_color = (0, 0, 255) if recording_enabled else (100, 100, 100)
    cv2.rectangle(frame, (245, 5), (315, 25), rec_color, -1)
    rec_text = "REC: ON" if recording_enabled else "REC: OFF"
    cv2.putText(frame, rec_text, (250, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    if recording_enabled:
        if video_writer is None or (time.time() - chunk_start_time) >= CHUNK_DURATION:
            if video_writer is not None: video_writer.release()
            
            existing_files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.avi")))
            while len(existing_files) >= MAX_VIDEOS:
                os.remove(existing_files[0])
                existing_files.pop(0)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(VIDEO_DIR, f"cctv_{timestamp}.avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = ThreadedVideoWriter(filename, fourcc, TARGET_FPS, (FRAME_WIDTH, FRAME_HEIGHT))
            chunk_start_time = time.time()
        video_writer.write(frame)
    else:
        if video_writer is not None:
            video_writer.release()
            video_writer = None

    cv2.imshow(WINDOW_NAME, frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        save_data(TOTAL_IN, TOTAL_OUT, PEOPLE_IN_ROOM)
        break

if video_writer is not None: video_writer.release()
vs.release()
cv2.destroyAllWindows()