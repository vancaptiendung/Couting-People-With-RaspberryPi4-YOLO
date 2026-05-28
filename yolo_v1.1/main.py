import os
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.fonts.warning=false"

import cv2
import ncnn
import numpy as np
import threading
import time
import json
import glob
import queue
from collections import deque
from datetime import datetime
from flask import Flask, Response, render_template, jsonify, request
from picamera2 import Picamera2

# =====================================================================
# 1. KHỞI TẠO HỆ THỐNG LƯU TRỮ VÀ BIẾN TOÀN CỤC
# =====================================================================
import sqlite3
import requests 

DB_FILE = "inoutlog.db"

def init_db():
    # Tạo bảng nếu chưa tồn tại
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                object_id INTEGER,
                details TEXT,
                synced INTEGER DEFAULT 0
            )
        ''')
        conn.commit()

def log_event(event_type, object_id=None, details=""):
    """Ghi lịch sử vào SQLite"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO system_logs (timestamp, event_type, object_id, details) VALUES (?, ?, ?, ?)",
                (timestamp, event_type, object_id, details)
            )
            conn.commit()
    except Exception as e:
        print(f"[DB ERROR] Không thể ghi log: {e}")

init_db()
log_event("SYSTEM_START", details="Hệ thống Camera AI Raspberry Pi 5 khởi động")
# load data
DATA_FILE = "counter_data.json"

def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                data = json.load(f)
                return data.get("in", 0), data.get("out", 0), data.get("room", 0)
        except: pass
    return 0, 0, 0

def save_data():
    with open(DATA_FILE, "w") as f:
        json.dump({"in": TOTAL_IN, "out": TOTAL_OUT, "room": PEOPLE_IN_ROOM}, f)

TOTAL_IN, TOTAL_OUT, PEOPLE_IN_ROOM = load_data()

latest_frame = None
output_frame_bgr = None
frame_lock = threading.Lock()
recording_enabled = False

cam_fps = 0
ai_fps = 0

# =====================================================================
# 2. HỆ THỐNG GHI HÌNH
# =====================================================================
VIDEO_DIR = "videos"
MAX_VIDEOS = 3
CHUNK_DURATION = 30 * 60 
if not os.path.exists(VIDEO_DIR): os.makedirs(VIDEO_DIR)

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
# 3. KIẾN TRÚC CAMERA (SỬ DỤNG HARDWARE BINNING 1640x1232)
# =====================================================================
class CameraThread:
    def __init__(self):
        print("[INFO] Đang khởi động Picamera2 ở chế độ Binned 1640x1232...")
        self.picam2 = Picamera2()
        
        # Sử dụng 1640x1232 (2x2 Binning) để giữ nguyên FOV rộng 
        # nhưng giảm tải CPU 4 lần, mở khóa giới hạn FPS lên 30.
        self.config = self.picam2.create_video_configuration(
            main={"size": (1640, 1232), "format": "RGB888"},
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
                frame_raw = self.picam2.capture_array()
                
                with frame_lock:
                    latest_frame = frame_raw.copy()
                
                frames_count += 1
                now = time.time()
                if now - prev_time >= 1.0:
                    cam_fps = frames_count / (now - prev_time)
                    frames_count = 0
                    prev_time = now
            except Exception as e:
                time.sleep(0.01)

    def stop(self):
        self.stopped = True
        self.picam2.stop()

# =====================================================================
# 4. MÁY CHỦ WEB API
# =====================================================================
app = Flask(__name__)

def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return round(float(f.read()) / 1000.0, 1)
    except: return 0.0

@app.route("/")
def index():
    return render_template("index.html")

def generate_stream():
    global output_frame_bgr
    while True:
        with frame_lock:
            if output_frame_bgr is None: 
                continue
            flag, encodedImage = cv2.imencode(".jpg", output_frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not flag: continue
            
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
        time.sleep(0.04) 

@app.route("/video_feed")
def video_feed():
    return Response(generate_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/data")
def api_data():
    return jsonify({
        "in": TOTAL_IN, "out": TOTAL_OUT, "room": PEOPLE_IN_ROOM, 
        "recording": recording_enabled,
        "cam_fps": round(cam_fps, 1), "ai_fps": round(ai_fps, 1),
        "cpu_temp": get_cpu_temp()
    })

@app.route("/api/action", methods=["POST"])
def api_action():
    global TOTAL_IN, TOTAL_OUT, PEOPLE_IN_ROOM, recording_enabled
    action = request.json.get("action")
    if action == "in_plus": TOTAL_IN += 1
    elif action == "in_minus": TOTAL_IN = max(0, TOTAL_IN - 1)
    elif action == "out_plus": TOTAL_OUT += 1
    elif action == "out_minus": TOTAL_OUT = max(0, TOTAL_OUT - 1)
    elif action == "room_plus": PEOPLE_IN_ROOM += 1
    elif action == "room_minus": PEOPLE_IN_ROOM = max(0, PEOPLE_IN_ROOM - 1)
    elif action == "toggle_record": recording_enabled = not recording_enabled
    save_data()
    return jsonify({"status": "success"})

# =====================================================================
# 5. LUỒNG AI CHÍNH (XỬ LÝ ẢNH BẰNG PHẦN MỀM)
# =====================================================================
def main():
    global latest_frame, output_frame_bgr, ai_fps
    global TOTAL_IN, TOTAL_OUT, PEOPLE_IN_ROOM
    
    cam_thread = CameraThread()
    
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR) 
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False), daemon=True).start()
    
    print("[INFO] Đang load model YOLO-Fastest v1.1...")
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = 4 
    net.load_param("yolo-fastest-1.1-xl.param")
    net.load_model("yolo-fastest-1.1-xl.bin")
    
    AI_SIZE = 320
    FRAME_SIZE = 640 
    LINE_X = FRAME_SIZE // 2 
    
    trackable_objects = {}
    next_object_id = 0
    MAX_DISAPPEARED = 30
    
    video_writer = None
    chunk_start_time = 0
    
    prev_ai_time = time.time()
    frames_ai = 0

    print("\n" + "="*50)
    print("[HỆ THỐNG ĐÃ SẴN SÀNG]")
    print("Truy cập Web Dashboard tại: http://<IP_CỦA_PI>:5000")
    print("="*50 + "\n")

    try:
        while True:
            with frame_lock:
                if latest_frame is None:
                    time.sleep(0.01)
                    continue
                # Nhận ảnh 1640x1232 đã binned
                frame_raw = latest_frame.copy()
                latest_frame = None 

            # ==================================================================
            # QUY TRÌNH: CẮT CHÍNH GIỮA -> THU NHỎ -> XỬ LÝ
            # ==================================================================
            # 1. Cắt lấy hình vuông 1232x1232 chính giữa (Bỏ 204 pixel hai bên rìa)
            square_raw = frame_raw[:, 204:1436] 
            
            # 2. Bóp hình vuông xuống 320x320 cho AI (CPU bây giờ xử lý rất nhẹ)
            resized_ai = cv2.resize(square_raw, (AI_SIZE, AI_SIZE), interpolation=cv2.INTER_LINEAR)
            in_mat = ncnn.Mat.from_pixels(resized_ai, ncnn.Mat.PixelType.PIXEL_RGB, AI_SIZE, AI_SIZE)
            
            # 3. Bóp hình vuông đó xuống 640x640 để làm mảng vẽ hiển thị Web
            display_frame = cv2.resize(square_raw, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_LINEAR)
            # ==================================================================

            in_mat.substract_mean_normalize([0.0, 0.0, 0.0], [1/255.0, 1/255.0, 1/255.0])

            ex = net.create_extractor()
            ex.input("data", in_mat) 
            ret, out_mat = ex.extract("output") 

            current_centroids = []
            
            if out_mat:
                for i in range(out_mat.h):
                    values = out_mat.row(i)
                    class_id = int(values[0])
                    score = values[1]

                    if score > 0.30 and class_id in [0, 1]: 
                        if values[2] <= 1.5: 
                            x1 = int(values[2] * FRAME_SIZE)
                            y1 = int(values[3] * FRAME_SIZE)
                            x2 = int(values[4] * FRAME_SIZE)
                            y2 = int(values[5] * FRAME_SIZE)
                        else:
                            x1 = int((values[2] / AI_SIZE) * FRAME_SIZE)
                            y1 = int((values[3] / AI_SIZE) * FRAME_SIZE)
                            x2 = int((values[4] / AI_SIZE) * FRAME_SIZE)
                            y2 = int((values[5] / AI_SIZE) * FRAME_SIZE)
                        
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(FRAME_SIZE, x2), min(FRAME_SIZE, y2)
                        
                        cX = int((x1 + x2) / 2.0)
                        cY = int((y1 + y2) / 2.0)
                        current_centroids.append((cX, cY, x1, y1, x2, y2))
                        
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"Nguoi: {score*100:.1f}%"
                        cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # --- THEO DÕI & ĐẾM ---
            matches = []
            
            # 1. Dự đoán vị trí tương lai và đo khoảng cách (Greedy Matching)
            for i, (cX, cY, startX, startY, endX, endY) in enumerate(current_centroids):
                for obj_id, obj_data in trackable_objects.items():
                    # Đảm bảo cấu trúc dữ liệu mới: (cX, cY, dx, dy, zone_history, disappeared)
                    if len(obj_data) == 4: # Khởi tạo mặc định nếu là data đời cũ
                        old_cX, old_cY, zone_history, disappeared = obj_data
                        dx, dy = 0, 0
                    else:
                        old_cX, old_cY, dx, dy, zone_history, disappeared = obj_data
                    
                    # DỰ ĐOÁN QUÁN TÍNH: Lấy vị trí cũ + (hướng đi * số frame đã biến mất)
                    pred_cX = old_cX + (dx * (disappeared + 1))
                    pred_cY = old_cY + (dy * (disappeared + 1))
                    
                    # So sánh vị trí thực tế với vị trí "dự đoán"
                    d = np.hypot(cX - pred_cX, cY - pred_cY)
                    matches.append((d, i, obj_id))

            # 2. Sắp xếp ưu tiên ghép các cặp ở gần nhau nhất
            matches.sort(key=lambda x: x[0])
            
            used_centroids = set()
            used_ids = set()
            updated_trackable_objects = {}
            counters_changed = False
            
            # Bán kính tìm kiếm được tăng vọt lên 250 vì ta đã có dự đoán hướng đi cực chuẩn
            MAX_DISTANCE = 250 

            # 3. Tiến hành ghép nối ID cũ với Box mới
            for d, i, obj_id in matches:
                if d > MAX_DISTANCE: continue
                if i in used_centroids or obj_id in used_ids: continue

                cX, cY, startX, startY, endX, endY = current_centroids[i]
                obj_data = trackable_objects[obj_id]
                
                old_cX = obj_data[0]
                old_cY = obj_data[1]

                # Tính VẬN TỐC (hướng đi hiện tại) để dành cho lần sau lỡ bị mất dấu
                dx = cX - old_cX
                dy = cY - old_cY

                # Cập nhật vùng (INSIDE/OUTSIDE)
                if len(obj_data) == 4: zone_history = obj_data[2]
                else: zone_history = obj_data[4]

                zone = "INSIDE" if cX < LINE_X else "OUTSIDE"
                if len(zone_history) == 0 or zone_history[-1] != zone:
                    zone_history.append(zone)

                updated_trackable_objects[obj_id] = (cX, cY, dx, dy, zone_history, 0)
                used_centroids.add(i)
                used_ids.add(obj_id)

            # 4. Cấp ID cho đối tượng mới toanh
            for i, (cX, cY, startX, startY, endX, endY) in enumerate(current_centroids):
                if i not in used_centroids:
                    zone = "INSIDE" if cX < LINE_X else "OUTSIDE"
                    zone_history = deque([zone], maxlen=10)
                    updated_trackable_objects[next_object_id] = (cX, cY, 0, 0, zone_history, 0)
                    used_ids.add(next_object_id)
                    next_object_id += 1

            # 5. Logic băng qua vạch (Chỉ xử lý đếm khi ID đang hiển thị thực tế)
            for obj_id, data in updated_trackable_objects.items():
                cX, cY, dx, dy, zone_history, disappeared = data
                if disappeared == 0: 
                    final_comp = []
                    for z in zone_history:
                        if not final_comp or final_comp[-1] != z: 
                            final_comp.append(z)

                    if "OUTSIDE" in final_comp and "INSIDE" in final_comp:
                        idx_out = final_comp.index("OUTSIDE")
                        idx_in = final_comp.index("INSIDE")
                        
                        if idx_out < idx_in: # Đi TỪ NGOÀI VÀO TRONG
                            TOTAL_IN += 1
                            PEOPLE_IN_ROOM += 1
                            updated_trackable_objects[obj_id] = (cX, cY, dx, dy, deque(["INSIDE"], maxlen=10), 0)
                            counters_changed = True
                            
                        elif idx_in < idx_out: # Đi TỪ TRONG RA NGOÀI
                            TOTAL_OUT += 1
                            PEOPLE_IN_ROOM = max(0, PEOPLE_IN_ROOM - 1)
                            updated_trackable_objects[obj_id] = (cX, cY, dx, dy, deque(["OUTSIDE"], maxlen=10), 0)
                            counters_changed = True

            if counters_changed: save_data()

            # 6. Kéo dài tuổi thọ cho các ID lỡ bị mất dấu (Disappeared)
            for obj_id, obj_data in trackable_objects.items():
                if obj_id not in used_ids:
                    if len(obj_data) == 4:
                        old_cX, old_cY, zone_history, disappeared = obj_data
                        dx, dy = 0, 0
                    else:
                        old_cX, old_cY, dx, dy, zone_history, disappeared = obj_data
                    
                    disappeared += 1
                    if disappeared <= MAX_DISAPPEARED:
                        # Vẫn giữ trong bộ nhớ, chờ người này "hiện hình" lại
                        updated_trackable_objects[obj_id] = (old_cX, old_cY, dx, dy, zone_history, disappeared)

            trackable_objects = updated_trackable_objects

            # Vẽ vạch kẻ ngay giữa hình vuông
            cv2.line(display_frame, (LINE_X, 0), (LINE_X, FRAME_SIZE), (0, 255, 255), 2)
            
            frames_ai += 1
            now = time.time()
            if now - prev_ai_time >= 1.0:
                ai_fps = frames_ai / (now - prev_ai_time)
                frames_ai = 0
                prev_ai_time = now

            display_frame_corrected = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

            if recording_enabled:
                if video_writer is None or (time.time() - chunk_start_time) >= CHUNK_DURATION:
                    if video_writer is not None: video_writer.release()
                    existing_files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.avi")))
                    while len(existing_files) >= MAX_VIDEOS:
                        os.remove(existing_files[0])
                        existing_files.pop(0)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    rec_fps = ai_fps if ai_fps > 0 else 10
                    video_writer = ThreadedVideoWriter(os.path.join(VIDEO_DIR, f"cctv_{timestamp}.avi"), cv2.VideoWriter_fourcc(*'XVID'), int(rec_fps), (FRAME_SIZE, FRAME_SIZE))
                    chunk_start_time = time.time()
                video_writer.write(display_frame)
            else:
                if video_writer is not None: 
                    video_writer.release()
                    video_writer = None

            with frame_lock:
                output_frame_bgr = display_frame.copy()

    except KeyboardInterrupt:
        print("\n[INFO] Đang lưu dữ liệu và tắt hệ thống...")
        save_data()
    finally:
        cam_thread.stop()
        if video_writer is not None: video_writer.release()
        print("[INFO] Đã tắt Camera an toàn!")

if __name__ == "__main__":
    main()