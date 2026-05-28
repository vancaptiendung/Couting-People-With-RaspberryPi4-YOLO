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
# 2. HỆ THỐNG GHI HÌNH (Chống Lag AI & Chống Sập Nguồn)
# =====================================================================
VIDEO_DIR = "videos"
MAX_VIDEOS = 6
CHUNK_DURATION = 30 * 60 
if not os.path.exists(VIDEO_DIR): os.makedirs(VIDEO_DIR)

class ThreadedVideoWriter:
    def __init__(self, filename, fps, frame_size):
        # Dùng chuẩn nén MJPG trâu bò, chống lỗi PTS, tương thích mọi thiết bị
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
        self.q = queue.Queue(maxsize=128)
        self.stopped = False
        
        # Giao toàn quyền ghi file cho luồng phụ
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()
        
    def write(self, frame):
        # Nếu thẻ nhớ ghi chậm, tự động bỏ qua khung hình để cứu sống AI
        if not self.stopped and not self.q.full(): 
            self.q.put(frame.copy())
            
    def _write(self):
        # Luồng phụ âm thầm ghi file
        while not self.stopped or not self.q.empty():
            if not self.q.empty(): 
                self.writer.write(self.q.get())
            else: 
                time.sleep(0.01) 
                
        # Duy nhất luồng phụ này được đóng file
        self.writer.release()
                
    def release(self):
        self.stopped = True
        # Luồng chính đứng đợi luồng phụ chốt sổ an toàn
        if self.thread.is_alive():
            self.thread.join()

# =====================================================================
# 3. KIẾN TRÚC CAMERA (HARDWARE DUAL-STREAM CHO V1.3)
# =====================================================================
class CameraThread:
    def __init__(self):
        print("[INFO] Đang khởi động Picamera2 (Lấy FULL góc -> Hardware Resize)...")
        self.picam2 = Picamera2()
        
        # Cấu hình ISP bóp ảnh từ cảm biến xuống thẳng 640x480 định dạng YUV420
        self.config = self.picam2.create_video_configuration(
            main={"size": (1296, 972), "format": "RGB888"},
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
                # Nhận YUV thô từ luồng lores của phần cứng
                frame_yuv = self.picam2.capture_array("lores")
                # Dịch YUV sang BGR chuẩn OpenCV siêu tốc
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

# =====================================================================
# 4. MÁY CHỦ WEB API (BẢO VỆ CHỐNG SẬP MẠNG)
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
    try:
        while True:
            with frame_lock:
                if output_frame_bgr is None: 
                    time.sleep(0.01)
                    continue
                flag, encodedImage = cv2.imencode(".jpg", output_frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not flag: continue
                
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
            time.sleep(0.04) 
    except (OSError, GeneratorExit):
        pass # Thoát êm ái khi tắt tab web hoặc rớt VPN

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
# 5. LUỒNG AI CHÍNH (YOLO-FASTEST V1.1 XL)
# =====================================================================

def main():
    global latest_frame, output_frame_bgr, ai_fps
    global TOTAL_IN, TOTAL_OUT, PEOPLE_IN_ROOM
    
    cam_thread = CameraThread()
    
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR) 
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False), daemon=True).start()
    
    print("[INFO] Đang load model YOLO-Fastest v1.1 XL...")
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = 4 
    
    # Nạp đúng file V1.1 XL (nhớ chắc chắn tên file đã đúng trong thư mục)
    net.load_param("yolo-fastest-1.1-xl.param")
    net.load_model("yolo-fastest-1.1-xl.bin")
    
    AI_SIZE = 320 
    DISPLAY_W, DISPLAY_H = 640, 480 
    LINE_X = DISPLAY_W // 2 
    
    trackable_objects = {}
    next_object_id = 0
    MAX_DISAPPEARED = 30
    
    video_writer = None
    chunk_start_time = 0
    
    prev_ai_time = time.time()
    frames_ai = 0

    print("\n" + "="*50)
    print("[HỆ THỐNG ĐÃ SẴN SÀNG - YOLO V1.1 XL + CAMERA V1.3]")
    print("Truy cập Web Dashboard tại: http://<IP_CỦA_PI>:5000")
    print("="*50 + "\n")

    try:
        while True:
            with frame_lock:
                if latest_frame is None:
                    time.sleep(0.01)
                    continue
                # Lấy khung hình trực tiếp 640x480 từ Camera
                frame_bgr = latest_frame.copy()
                latest_frame = None 

            display_frame = frame_bgr.copy()
            
            # Sử dụng C++ NEON Resize của NCNN + Chuyển BGR sang RGB cho AI
            in_mat = ncnn.Mat.from_pixels_resize(
                display_frame, 
                ncnn.Mat.PixelType.PIXEL_BGR2RGB, 
                DISPLAY_W, DISPLAY_H, 
                AI_SIZE, AI_SIZE
            )

            in_mat.substract_mean_normalize([0.0, 0.0, 0.0], [1/255.0, 1/255.0, 1/255.0])

            # Quy trình AI
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
                            x1 = int(values[2] * DISPLAY_W)
                            y1 = int(values[3] * DISPLAY_H)
                            x2 = int(values[4] * DISPLAY_W)
                            y2 = int(values[5] * DISPLAY_H)
                        else:
                            x1 = int((values[2] / AI_SIZE) * DISPLAY_W)
                            y1 = int((values[3] / AI_SIZE) * DISPLAY_H)
                            x2 = int((values[4] / AI_SIZE) * DISPLAY_W)
                            y2 = int((values[5] / AI_SIZE) * DISPLAY_H)
                        
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(DISPLAY_W, x2), min(DISPLAY_H, y2)
                        
                        cX = int((x1 + x2) / 2.0)
                        cY = int((y1 + y2) / 2.0)
                        current_centroids.append((cX, cY, x1, y1, x2, y2))
                        
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(display_frame, f"Nguoi: {score*100:.1f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # --- THEO DÕI & ĐẾM QUÁN TÍNH ---
            matches = []
            for i, (cX, cY, startX, startY, endX, endY) in enumerate(current_centroids):
                for obj_id, obj_data in trackable_objects.items():
                    if len(obj_data) == 4:
                        old_cX, old_cY, zone_history, disappeared = obj_data
                        dx, dy = 0, 0
                    else:
                        old_cX, old_cY, dx, dy, zone_history, disappeared = obj_data
                    
                    pred_cX = old_cX + (dx * (disappeared + 1))
                    pred_cY = old_cY + (dy * (disappeared + 1))
                    d = np.hypot(cX - pred_cX, cY - pred_cY)
                    matches.append((d, i, obj_id))

            matches.sort(key=lambda x: x[0])
            
            used_centroids, used_ids = set(), set()
            updated_trackable_objects = {}
            counters_changed = False
            MAX_DISTANCE = 250 

            for d, i, obj_id in matches:
                if d > MAX_DISTANCE or i in used_centroids or obj_id in used_ids: continue

                cX, cY, startX, startY, endX, endY = current_centroids[i]
                obj_data = trackable_objects[obj_id]
                old_cX, old_cY = obj_data[0], obj_data[1]
                dx, dy = cX - old_cX, cY - old_cY

                zone_history = obj_data[2] if len(obj_data) == 4 else obj_data[4]
                zone = "INSIDE" if cX < LINE_X else "OUTSIDE"
                
                if len(zone_history) == 0 or zone_history[-1] != zone:
                    zone_history.append(zone)

                updated_trackable_objects[obj_id] = (cX, cY, dx, dy, zone_history, 0)
                used_centroids.add(i)
                used_ids.add(obj_id)

            for i, (cX, cY, startX, startY, endX, endY) in enumerate(current_centroids):
                if i not in used_centroids:
                    zone = "INSIDE" if cX < LINE_X else "OUTSIDE"
                    updated_trackable_objects[next_object_id] = (cX, cY, 0, 0, deque([zone], maxlen=10), 0)
                    used_ids.add(next_object_id)
                    next_object_id += 1

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
                        
                        if idx_out < idx_in: 
                            TOTAL_IN += 1
                            PEOPLE_IN_ROOM += 1
                            updated_trackable_objects[obj_id] = (cX, cY, dx, dy, deque(["INSIDE"], maxlen=10), 0)
                            counters_changed = True
                            
                        elif idx_in < idx_out: 
                            TOTAL_OUT += 1
                            PEOPLE_IN_ROOM = max(0, PEOPLE_IN_ROOM - 1)
                            updated_trackable_objects[obj_id] = (cX, cY, dx, dy, deque(["OUTSIDE"], maxlen=10), 0)
                            counters_changed = True

            if counters_changed: save_data()

            for obj_id, obj_data in trackable_objects.items():
                if obj_id not in used_ids:
                    if len(obj_data) == 4:
                        old_cX, old_cY, zone_history, disappeared = obj_data
                        dx, dy = 0, 0
                    else:
                        old_cX, old_cY, dx, dy, zone_history, disappeared = obj_data
                    
                    disappeared += 1
                    if disappeared <= MAX_DISAPPEARED:
                        updated_trackable_objects[obj_id] = (old_cX, old_cY, dx, dy, zone_history, disappeared)

            trackable_objects = updated_trackable_objects

            # Vẽ vạch kẻ ngay giữa hình
            cv2.line(display_frame, (LINE_X, 0), (LINE_X, DISPLAY_H), (0, 255, 255), 2)
            
            # Tính toán AI FPS
            frames_ai += 1
            now = time.time()
            if now - prev_ai_time >= 1.0:
                ai_fps = frames_ai / (now - prev_ai_time)
                frames_ai = 0
                prev_ai_time = now

            # Đóng dấu thông tin lên video
            timestamp_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            info_str = f"IN: {TOTAL_IN} | OUT: {TOTAL_OUT} | ROOM: {PEOPLE_IN_ROOM}"
            
            cv2.rectangle(display_frame, (5, 5), (420, 65), (0, 0, 0), -1)
            cv2.putText(display_frame, timestamp_str, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display_frame, info_str, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if recording_enabled:
                if video_writer is None or (time.time() - chunk_start_time) >= CHUNK_DURATION:
                    if video_writer is not None: video_writer.release()
                    
                    # QUAN TRỌNG: Lưu file chuẩn .avi
                    existing_files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.avi")))
                    while len(existing_files) >= MAX_VIDEOS:
                        os.remove(existing_files[0])
                        existing_files.pop(0)
                        
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    rec_fps = ai_fps if ai_fps > 0 else 15
                    
                    video_writer = ThreadedVideoWriter(
                        os.path.join(VIDEO_DIR, f"cctv_{timestamp}.avi"), 
                        int(rec_fps), 
                        (DISPLAY_W, DISPLAY_H)
                    )
                    chunk_start_time = time.time()
                
                # Ghi ảnh thẳng vào luồng phụ
                video_writer.write(display_frame)
            else:
                if video_writer is not None: 
                    video_writer.release()
                    video_writer = None

            # Cập nhật ảnh cho Web Dashboard (Giữ nguyên chuẩn màu BGR cho Web)
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
