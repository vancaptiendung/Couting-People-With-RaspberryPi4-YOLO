import cv2
import numpy as np
import sys
import threading
import time
from collections import deque

# =====================================================================
# CẤU HÌNH ĐƯỜNG DẪN VÀ THAM SỐ ĐẾM VÙNG (CHỐNG SÓT DO LAG)
# =====================================================================
PROTO = "/home/yguy/Documents/pi_counter/MobileNetSSD_deploy.prototxt"
MODEL = "/home/yguy/Documents/pi_counter/MobileNetSSD_deploy.caffemodel"
CONFIDENCE_THRESHOLD = 0.45 # Giảm nhẹ để nhạy hơn trên Pi

# Kẻ 2 vạch đứng chia màn hình làm 3 vùng (Trái - Giữa - Phải)
LINE_A_X = 240  
LINE_B_X = 400  

TOTAL_IN = 0
TOTAL_OUT = 0
PEOPLE_IN_ROOM = 0

# trackable_objects: id -> (cX, cY, zone_history_deque, disappeared_count)
trackable_objects = {}
next_object_id = 0
frame_count = 0 
SKIP_FRAMES = 3 # TỐI ƯU: Cứ 3 frame mới chạy AI 1 lần để GIẢM LAG
MAX_DISAPPEARED = 30  # Số frame tối đa để chờ object xuất hiện lại

# Hàm xác định vị trí vùng dựa trên tọa độ X
def get_zone(x):
    if x < LINE_A_X:
        return "OUTSIDE" # Bên ngoài (Trái)
    elif x > LINE_B_X:
        return "INSIDE"  # Bên trong (Phải)
    else:
        return "MIDDLE"  # Vùng đệm ở giữa

# =====================================================================
# KHỞI TẠO HỆ THỐNG
# =====================================================================
print("[INFO] Đang tải mô hình...")
net = cv2.dnn.readNetFromCaffe(PROTO, MODEL)

print("[INFO] Đang mở Camera...")
# Dùng thread để đọc frame liên tục, giảm lag do I/O
class FrameGrabber:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.stopped = False
        self.frame = None
        self.lock = threading.Lock()
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        t = threading.Thread(target=self._reader, daemon=True)
        t.start()

    def isOpened(self):
        return self.cap.isOpened()

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

# Luồng inference: lấy frame mới nhất từ FrameGrabber và chạy net.forward liên tục
class DetectorThread:
    def __init__(self, net, frame_grabber, confidence=0.45, input_size=(300,300), skip_frames=3):
        self.net = net
        self.vs = frame_grabber
        self.confidence = confidence
        self.input_size = input_size
        self.skip_frames = skip_frames
        self.current_centroids = []
        self.lock = threading.Lock()
        self.stopped = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._frame_count = 0

    def start(self):
        self._thread.start()
        return self

    def _run(self):
        while not self.stopped:
            frame = self.vs.read()
            if frame is None:
                time.sleep(0.01)
                continue
            self._frame_count += 1
            if self._frame_count % self.skip_frames != 0:
                # do not run inference every frame
                time.sleep(0.001)
                continue

            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, self.input_size), 0.007843, self.input_size, 127.5)
            self.net.setInput(blob)
            detections = self.net.forward()

            centroids = []
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > self.confidence:
                    if int(detections[0, 0, i, 1]) == 15:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        cX = int((startX + endX) / 2.0)
                        cY = int((startY + endY) / 2.0)
                        centroids.append((cX, cY, startX, startY, endX, endY))

            with self.lock:
                self.current_centroids = centroids

    def get_centroids(self):
        with self.lock:
            return list(self.current_centroids)

    def stop(self):
        self.stopped = True

detector = DetectorThread(net, vs, confidence=CONFIDENCE_THRESHOLD, skip_frames=SKIP_FRAMES).start()

if not vs.isOpened():
    print("[ERROR] Không thể kết nối camera.")
    sys.exit(1)

while True:
    frame = vs.read()
    if frame is None:
        time.sleep(0.01)
        continue

    (h, w) = frame.shape[:2]
    current_centroids = []

    # Lấy kết quả inference từ luồng detector (nó chạy ở chế độ skip_frames)
    current_centroids = detector.get_centroids()
    
    frame_count += 1

    # --- LOGIC TRACKING VÀ ĐẾM THEO VÙNG (KHÔNG SỢ LAG NHẢY BƯỚC) ---
    # Khởi tạo updated dựa trên object cũ để không mất track khi 1 số object không được phát hiện
    updated_trackable_objects = dict(trackable_objects)
    seen_ids = set()

    for (cX, cY, startX, startY, endX, endY) in current_centroids:
        matched_id = None
        min_distance = 80 # Tăng bán kính tìm kiếm vật thể giữa các frame bị lag

        for obj_id, (old_cX, old_cY, zone_history, disappeared) in trackable_objects.items():
            d = np.hypot(cX - old_cX, cY - old_cY)
            if d < min_distance:
                min_distance = d
                matched_id = obj_id

        if matched_id is None:
            matched_id = next_object_id
            next_object_id += 1
            zone_history = deque([get_zone(cX)], maxlen=10) # Khởi tạo lịch sử vùng ban đầu
            disappeared = 0
        else:
            old_cX, old_cY, zone_history, disappeared = trackable_objects[matched_id]
            current_zone = get_zone(cX)
            # Chỉ ghi lại lịch sử nếu vùng thay đổi để tiết kiệm bộ nhớ
            if len(zone_history) == 0 or zone_history[-1] != current_zone:
                zone_history.append(current_zone)
            disappeared = 0

        # LOGIC ĐẾM: Nén lịch sử loại bỏ vùng trùng lặp liên tiếp
        compressed = []
        for z in zone_history:
            if not compressed or compressed[-1] != z:
                compressed.append(z)

        if "OUTSIDE" in compressed and "INSIDE" in compressed:
            idx_out = compressed.index("OUTSIDE")
            idx_in = compressed.index("INSIDE")
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

        # Vẽ giao diện bao quanh người
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"ID {matched_id}", (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Tăng disappeared cho những object không thấy trong frame AI hiện tại
    for obj_id in list(updated_trackable_objects.keys()):
        if obj_id not in seen_ids:
            cX, cY, zone_history, disappeared = updated_trackable_objects[obj_id]
            disappeared = disappeared + 1
            if disappeared > MAX_DISAPPEARED:
                del updated_trackable_objects[obj_id]
            else:
                updated_trackable_objects[obj_id] = (cX, cY, zone_history, disappeared)

    # Nếu không phải frame AI, giữ nguyên trackable_objects (skip)
    if frame_count % SKIP_FRAMES != 0 and len(current_centroids) == 0:
        # Không thay đổi tracking
        pass

    trackable_objects = updated_trackable_objects

    # --- VẼ GIAO DIỆN MÀN HÌNH ---
    cv2.line(frame, (LINE_A_X, 0), (LINE_A_X, h), (0, 255, 255), 2)
    cv2.line(frame, (LINE_B_X, 0), (LINE_B_X, h), (255, 255, 0), 2)
    
    cv2.putText(frame, "NGOAI", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, "TRONG", (w - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Bảng số liệu
    cv2.rectangle(frame, (5, 400), (200, 475), (0, 0, 0), -1)
    cv2.putText(frame, f"Vao: {TOTAL_IN}", (15, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Ra: {TOTAL_OUT}", (15, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f"Trong: {PEOPLE_IN_ROOM}", (15, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("He thong dem nguoi toi uu CPU", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vs.release()
cv2.destroyAllWindows()
try:
    detector.stop()
except Exception:
    pass
