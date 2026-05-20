import cv2
import numpy as np
import sys
import threading
import time
import os
import glob
from datetime import datetime
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

FRAME_WIDTH = 320
FRAME_HEIGHT = 240
TARGET_FPS = 8  

# =====================================================================
# 2. CẤU HÌNH VẠCH DỌC, ĐẾM & GHI HÌNH
# =====================================================================
LINE_X = int(FRAME_WIDTH / 2)
is_dragging_line = False # Biến trạng thái kéo vạch

TOTAL_IN = 0
TOTAL_OUT = 0
PEOPLE_IN_ROOM = 0

MAX_DISAPPEARED = 30     

trackable_objects = {}
next_object_id = 0
WINDOW_NAME = "Dem nguoi AI (Ban Sieu Nhe)"

# Cấu hình Ghi hình
VIDEO_DIR = "videos"
MAX_VIDEOS = 3
CHUNK_DURATION = 30 * 60  # 30 phút = 1800 giây
recording_enabled = False
video_writer = None
chunk_start_time = 0

# Tạo thư mục chứa video nếu chưa có
if not os.path.exists(VIDEO_DIR):
    os.makedirs(VIDEO_DIR)

# =====================================================================
# 3. HÀM BẮT SỰ KIỆN CHUỘT (THÊM KÉO VẠCH VÀ NÚT REC)
# =====================================================================
def adjust_counters(event, x, y, flags, param):
    global TOTAL_IN, TOTAL_OUT, PEOPLE_IN_ROOM, LINE_X
    global is_dragging_line, recording_enabled

    # Khi NHẤN chuột
    if event == cv2.EVENT_LBUTTONDOWN:
        # Nút BẬT/TẮT Quay Video (Góc trên phải)
        if 245 <= x <= 315 and 5 <= y <= 25:
            recording_enabled = not recording_enabled
            return

        # Nút Trừ/Cộng số đếm (Góc dưới trái)
        if 190 <= y <= 202: 
            if 70 <= x <= 85: TOTAL_IN = max(0, TOTAL_IN - 1)
            elif 95 <= x <= 110: TOTAL_IN += 1
        elif 205 <= y <= 217: 
            if 70 <= x <= 85: TOTAL_OUT = max(0, TOTAL_OUT - 1)
            elif 95 <= x <= 110: TOTAL_OUT += 1
        elif 220 <= y <= 232: 
            if 70 <= x <= 85: PEOPLE_IN_ROOM = max(0, PEOPLE_IN_ROOM - 1)
            elif 95 <= x <= 110: PEOPLE_IN_ROOM += 1

        # Kiểm tra xem có bấm trúng Vạch dọc để kéo không (sai số 15 pixel)
        if abs(x - LINE_X) < 15:
            is_dragging_line = True

    # Khi DI CHUYỂN chuột (Đang giữ vạch)
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_dragging_line:
            # Ép vạch không được chạy ra khỏi màn hình
            LINE_X = max(20, min(FRAME_WIDTH - 20, x))

    # Khi NHẢ chuột
    elif event == cv2.EVENT_LBUTTONUP:
        is_dragging_line = False

# =====================================================================
# 4. HÀM XÁC ĐỊNH VỊ TRÍ 
# =====================================================================
def get_zone(cx):
    if cx < LINE_X: return "INSIDE"
    else: return "OUTSIDE"

# =====================================================================
# 5. KHỞI TẠO CAMERA
# =====================================================================
class FrameGrabber:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.stopped = False
        self.frame = None
        self.lock = threading.Lock()
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

print(f"[INFO] Hệ thống Siêu Nhẹ | Có hỗ trợ Kéo Vạch & Ghi hình 30p/File...")

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

    # --- ÉP ĐỘ PHÂN GIẢI BẰNG PHẦN MỀM LÀM NHẸ PI ---
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

    # --- LOGIC TRACKING ---
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

        if "OUTSIDE" in final_compressed and "INSIDE" in final_compressed:
            idx_out = final_compressed.index("OUTSIDE")
            idx_in = final_compressed.index("INSIDE")
            
            if idx_out < idx_in:
                TOTAL_IN += 1
                PEOPLE_IN_ROOM += 1
                zone_history = deque(["INSIDE"], maxlen=10)
            elif idx_in < idx_out:
                TOTAL_OUT += 1
                PEOPLE_IN_ROOM = max(0, PEOPLE_IN_ROOM - 1) 
                zone_history = deque(["OUTSIDE"], maxlen=10)

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
    # 6. VẼ GIAO DIỆN (LINE, TEXT, NÚT BẤM)
    # =================================================================
    # Màu vạch dọc thay đổi khi đang dùng chuột kéo
    line_color = (0, 0, 255) if is_dragging_line else (0, 255, 255)
    line_thick = 3 if is_dragging_line else 2
    cv2.line(frame, (LINE_X, 0), (LINE_X, h), line_color, line_thick)
    
    cv2.putText(frame, "TRONG (<--)", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(frame, "(-->) NGOAI", (FRAME_WIDTH - 85, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Khung đen hiển thị số
    cv2.rectangle(frame, (0, 180), (120, 240), (0, 0, 0), -1)
    cv2.putText(frame, f"Vao: {TOTAL_IN}", (5, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(frame, f"Ra: {TOTAL_OUT}", (5, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.putText(frame, f"Trg: {PEOPLE_IN_ROOM}", (5, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Nút bấm số đếm
    for y_btn in [190, 205, 220]:
        cv2.rectangle(frame, (70, y_btn), (85, y_btn + 12), (80, 80, 80), -1)
        cv2.putText(frame, "-", (73, y_btn + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.rectangle(frame, (95, y_btn), (110, y_btn + 12), (80, 80, 80), -1)
        cv2.putText(frame, "+", (98, y_btn + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Vẽ nút bấm REC
    rec_color = (0, 0, 255) if recording_enabled else (100, 100, 100)
    cv2.rectangle(frame, (245, 5), (315, 25), rec_color, -1)
    rec_text = "REC: ON" if recording_enabled else "REC: OFF"
    cv2.putText(frame, rec_text, (250, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # =================================================================
    # 7. LOGIC QUẢN LÝ GHI HÌNH (LƯU VIDEO 30 PHÚT)
    # =================================================================
    if recording_enabled:
        # Nếu chưa tạo file video HOẶC đã trôi qua 30 phút (1800 giây)
        if video_writer is None or (time.time() - chunk_start_time) >= CHUNK_DURATION:
            if video_writer is not None:
                video_writer.release() # Đóng file cũ

            # Quét tìm các video đang có, xóa file cũ nhất nếu quá 3 file
            existing_files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.avi")))
            while len(existing_files) >= MAX_VIDEOS:
                os.remove(existing_files[0])
                existing_files.pop(0)

            # Khởi tạo file mới với tên theo thời gian thực
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(VIDEO_DIR, f"cctv_{timestamp}.avi")
            
            # Khởi tạo thuật toán nén Video XVID
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(filename, fourcc, TARGET_FPS, (FRAME_WIDTH, FRAME_HEIGHT))
            chunk_start_time = time.time()
            print(f"[REC] Bắt đầu ghi file mới: {filename}")

        # Ghi khung hình có chứa các vạch vuông AI vào file
        video_writer.write(frame)
    else:
        # Nếu đang quay mà bấm OFF thì đóng file lại
        if video_writer is not None:
            video_writer.release()
            video_writer = None
            print("[REC] Đã dừng ghi hình.")

    cv2.imshow(WINDOW_NAME, frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Dọn dẹp trước khi thoát
if video_writer is not None:
    video_writer.release()
vs.release()
cv2.destroyAllWindows()