import cv2
import numpy as np
import sys
import threading
import time
from collections import deque

# =====================================================================
# CẤU HÌNH THAM SỐ VÀ VẠCH CHÉO (HÀNH LANG ĐẾM)
# =====================================================================
L1_PT1 = tuple([313, 204])
L1_PT2 = tuple([524, 4])
L2_PT1 = tuple([201, 481])
L2_PT2 = tuple([575, -2])


TOTAL_IN = 0
TOTAL_OUT = 0
PEOPLE_IN_ROOM = 0

MIN_CONTOUR_AREA = 2500  
MAX_DISAPPEARED = 30     

trackable_objects = {}
next_object_id = 0
WINDOW_NAME = "He thong dem nguoi (Vach Cheo)"

# =====================================================================
# HÀM BẮT SỰ KIỆN CHUỘT (ĐIỀU CHỈNH SỐ LƯỢNG)
# =====================================================================
def adjust_counters(event, x, y, flags, param):
    global TOTAL_IN, TOTAL_OUT, PEOPLE_IN_ROOM
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Hàng 1: VÀO (Tọa độ Y từ 400 đến 420)
        if 400 <= y <= 420:
            if 120 <= x <= 150:   # Bấm nút [-]
                TOTAL_IN = max(0, TOTAL_IN - 1)
            elif 160 <= x <= 190: # Bấm nút [+]
                TOTAL_IN += 1
                
        # Hàng 2: RA (Tọa độ Y từ 430 đến 450)
        elif 430 <= y <= 450:
            if 120 <= x <= 150:
                TOTAL_OUT = max(0, TOTAL_OUT - 1)
            elif 160 <= x <= 190:
                TOTAL_OUT += 1
                
        # Hàng 3: TRONG (Tọa độ Y từ 460 đến 480)
        elif 460 <= y <= 480:
            if 120 <= x <= 150:
                PEOPLE_IN_ROOM = max(0, PEOPLE_IN_ROOM - 1)
            elif 160 <= x <= 190:
                PEOPLE_IN_ROOM += 1

# =====================================================================
# CÁC HÀM TOÁN HỌC VÀ XÁC ĐỊNH VÙNG
# =====================================================================
def get_position(cx, cy, pt1, pt2):
    cross_product = (pt2[0] - pt1[0]) * (cy - pt1[1]) - (pt2[1] - pt1[1]) * (cx - pt1[0])
    if cross_product > 0:
        return "BELOW"
    else:
        return "ABOVE"

def get_zone(cx, cy):
    min_x = min(L1_PT1[0], L1_PT2[0], L2_PT1[0], L2_PT2[0]) - 20
    max_x = max(L1_PT1[0], L1_PT2[0], L2_PT1[0], L2_PT2[0]) + 20
    
    # Nếu đi ra khỏi hành lang ngang (rẽ vào ngõ tủ) -> Bỏ qua
    if cx < min_x or cx > max_x:
        return "IGNORE"

    pos_line1 = get_position(cx, cy, L1_PT1, L1_PT2)
    pos_line2 = get_position(cx, cy, L2_PT1, L2_PT2)

    if pos_line1 == "ABOVE":
        return "OUTSIDE"
    elif pos_line2 == "BELOW":
        return "INSIDE"
    else:
        return "MIDDLE"

# =====================================================================
# KHỞI TẠO CAMERA VÀ BỘ TÁCH NỀN (MOG2)
# =====================================================================
print("[INFO] Khởi tạo bộ tách nền MOG2...")
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=150, detectShadows=True)

class FrameGrabber:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.stopped = False
        self.frame = None
        self.lock = threading.Lock()
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 15)
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

if not vs.isOpened():
    print("[ERROR] Không thể kết nối camera.")
    sys.exit(1)

# Khởi tạo cửa sổ và gắn hàm chuột
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, adjust_counters)

print("[INFO] Hệ thống bắt đầu chạy...")

# =====================================================================
# VÒNG LẶP XỬ LÝ CHÍNH
# =====================================================================
while True:
    frame = vs.read()
    if frame is None:
        time.sleep(0.01)
        continue

    (h, w) = frame.shape[:2]
    current_centroids = []

    # 1. TIỀN XỬ LÝ ẢNH (GIẢM NHIỄU SIÊU MẠNH)
    blur = cv2.GaussianBlur(frame, (9, 9), 0)
    fgmask = fgbg.apply(blur, learningRate=0.005)
    _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel_open)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 51))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel_close)
    fgmask = cv2.dilate(fgmask, None, iterations=3)

    # 2. TÌM KIẾM CONTOUR VÀ LỌC VẬT THỂ
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_rects = []
    
    for contour in contours:
        hull = cv2.convexHull(contour)
        if cv2.contourArea(hull) < MIN_CONTOUR_AREA:
            continue
        (startX, startY, w_box, h_box) = cv2.boundingRect(hull)
        aspect_ratio = w_box / float(h_box)
        if aspect_ratio > 2.0: 
            continue
        valid_rects.append([startX, startY, w_box, h_box])

    # Gộp các bounding box bị đè lên nhau
    grouped_rects, _ = cv2.groupRectangles(valid_rects, groupThreshold=1, eps=0.3)
    final_rects = valid_rects if (len(grouped_rects) == 0 and len(valid_rects) > 0) else grouped_rects

    for (x, y, w_box, h_box) in final_rects:
        endX = x + w_box
        endY = y + h_box
        cX = int(x + w_box / 2.0)
        cY = int(y + h_box / 2.0)
        current_centroids.append((cX, cY, x, y, endX, endY))

    # 3. LOGIC TRACKING VÀ THEO DÕI VÙNG
    updated_trackable_objects = dict(trackable_objects)
    seen_ids = set()

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
            zone_history = deque([get_zone(cX, cY)], maxlen=10) 
            disappeared = 0
        else:
            old_cX, old_cY, zone_history, disappeared = trackable_objects[matched_id]
            current_zone = get_zone(cX, cY)
            if len(zone_history) == 0 or zone_history[-1] != current_zone:
                zone_history.append(current_zone)
            disappeared = 0

        # Lọc trạng thái "IGNORE"
        compressed = []
        for z in zone_history:
            if z != "IGNORE":
                if not compressed or compressed[-1] != z:
                    compressed.append(z)

        # 4. KIỂM TRA ĐIỀU KIỆN ĐỂ ĐẾM SỐ
        if "OUTSIDE" in compressed and "INSIDE" in compressed:
            idx_out = compressed.index("OUTSIDE")
            idx_in = compressed.index("INSIDE")
            
            # Khách đi VÀO (TỪ NGOÀI VÀO TRONG)
            if idx_out < idx_in:
                TOTAL_IN += 1
                PEOPLE_IN_ROOM += 1
                zone_history = deque(["INSIDE"], maxlen=10)
                
            # Khách đi RA (TỪ TRONG RA NGOÀI)
            elif idx_in < idx_out:
                TOTAL_OUT += 1
                PEOPLE_IN_ROOM = max(0, PEOPLE_IN_ROOM - 1) 
                zone_history = deque(["OUTSIDE"], maxlen=10)

        updated_trackable_objects[matched_id] = (cX, cY, zone_history, disappeared)
        seen_ids.add(matched_id)

        # Vẽ object và hiển thị trạng thái Vùng
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
        current_zone_str = get_zone(cX, cY)
        cv2.putText(frame, f"ID {matched_id} | {current_zone_str}", (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 5. XÓA OBJECT KHỎI BỘ NHỚ NẾU MẤT DẤU QUÁ LÂU
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
    # 6. VẼ GIAO DIỆN (NÚT BẤM VÀ BẢNG THÔNG KÊ)
    # =================================================================
    # Vẽ 2 Vạch chéo
    cv2.line(frame, L1_PT1, L1_PT2, (0, 0, 255), 2)
    cv2.line(frame, L2_PT1, L2_PT2, (0, 0, 255), 2)
    
    cv2.putText(frame, "NGOAI", (w - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, "TRONG", (w - 150, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Khung đen hiển thị số liệu
    cv2.rectangle(frame, (5, 385), (205, 485), (0, 0, 0), -1)

    # Text hiển thị số liệu đếm
    cv2.putText(frame, f"Vao: {TOTAL_IN}", (10, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Ra: {TOTAL_OUT}", (10, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f"Trong: {PEOPLE_IN_ROOM}", (10, 475), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Nút bấm VÀO
    cv2.rectangle(frame, (120, 400), (150, 420), (80, 80, 80), -1)
    cv2.putText(frame, "-", (130, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.rectangle(frame, (160, 400), (190, 420), (80, 80, 80), -1)
    cv2.putText(frame, "+", (168, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Nút bấm RA
    cv2.rectangle(frame, (120, 430), (150, 450), (80, 80, 80), -1)
    cv2.putText(frame, "-", (130, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.rectangle(frame, (160, 430), (190, 450), (80, 80, 80), -1)
    cv2.putText(frame, "+", (168, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Nút bấm TRONG
    cv2.rectangle(frame, (120, 460), (150, 480), (80, 80, 80), -1)
    cv2.putText(frame, "-", (130, 475), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.rectangle(frame, (160, 460), (190, 480), (80, 80, 80), -1)
    cv2.putText(frame, "+", (168, 475), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Hiển thị cửa sổ
    cv2.imshow(WINDOW_NAME, frame)
    cv2.imshow("Mask Debug (Den/Trang)", fgmask) 
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vs.release()
cv2.destroyAllWindows()
