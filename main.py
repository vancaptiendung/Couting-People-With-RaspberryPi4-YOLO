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
# 3. KIẾN TRÚC CAMERA (CHỤP FULL ĐỘ PHÂN GIẢI CẢM BIẾN)
# =====================================================================
class CameraThread:
    def __init__(self):
        print("[INFO] Đang khởi động Picamera2 ở FULL độ phân giải 3280x2464...")
        self.picam2 = Picamera2()
        
        # Mở toang cảm biến lấy 3280x2464 để đảm bảo không một phần mềm nào tự ý cắt lệch góc
        # Giới hạn 10 FPS để CPU có thời gian bóp mảng dữ liệu khổng lồ này
        self.config = self.picam2.create_video_configuration(
            main={"size": (3280, 2464), "format": "RGB888"},
            controls={"FrameRate": 10} 
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
                # Bức ảnh thô khổng lồ 3280x2464
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
    net.load_param("yolo-fastest-1.1.param")
    net.load_model("yolo-fastest-1.1.bin")
    
    AI_SIZE = 320
    # Đặt khung web là hình vuông 640x640 cho đồng bộ với thuật toán cắt ảnh
    FRAME_SIZE = 640 
    LINE_X = FRAME_SIZE // 2 
    
    trackable_objects = {}
    next_object_id = 0
    MAX_DISAPPEARED = 15
    
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
                # Nhận ảnh 3280x2464 gốc
                frame_raw = latest_frame.copy()
                latest_frame = None 

            # ==================================================================
            # QUY TRÌNH: CẮT CHÍNH GIỮA -> THU NHỎ -> XỬ LÝ -> ĐỔI MÀU CUỐI CÙNG
            # ==================================================================
            # 1. Cắt lấy hình vuông 2464x2464 chính giữa (Bỏ 408 pixel hai bên rìa)
            # Dòng này đảm bảo ảnh luôn nằm chuẩn ở giữa, không bị lệch trái/phải.
            square_raw = frame_raw[:, 408:2872] 
            
            # 2. Bóp hình vuông khổng lồ đó xuống 320x320 cho AI
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

                    if score > 0.45 and class_id in [0, 1]: 
                        
                        # Ánh xạ tọa độ siêu chuẩn (Tất cả đều là vuông 1:1)
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
                        
                        # Vẽ khung lên mảng hình hiện tại
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"Nguoi: {score*100:.1f}%"
                        cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # --- THEO DÕI & ĐẾM ---
            updated_trackable_objects = dict(trackable_objects)
            seen_ids = set()
            counters_changed = False

            for (cX, cY, startX, startY, endX, endY) in current_centroids:
                matched_id = None
                min_distance = 80 
                for obj_id, (old_cX, old_cY, zone_history, disappeared) in trackable_objects.items():
                    d = np.hypot(cX - old_cX, cY - old_cY)
                    if d < min_distance:
                        min_distance = d
                        matched_id = obj_id

                zone = "INSIDE" if cX < LINE_X else "OUTSIDE"

                if matched_id is None:
                    matched_id = next_object_id
                    next_object_id += 1
                    zone_history = deque([zone], maxlen=10)
                    disappeared = 0
                else:
                    old_cX, old_cY, zone_history, disappeared = trackable_objects[matched_id]
                    if len(zone_history) == 0 or zone_history[-1] != zone:
                        zone_history.append(zone)
                    disappeared = 0

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
                        zone_history = deque(["INSIDE"], maxlen=10)
                        counters_changed = True
                    elif idx_in < idx_out:
                        TOTAL_OUT += 1
                        PEOPLE_IN_ROOM = max(0, PEOPLE_IN_ROOM - 1)
                        zone_history = deque(["OUTSIDE"], maxlen=10)
                        counters_changed = True

                updated_trackable_objects[matched_id] = (cX, cY, zone_history, disappeared)
                seen_ids.add(matched_id)
                
                cv2.circle(display_frame, (cX, cY), 4, (0, 0, 255), -1)

            if counters_changed: save_data()

            for obj_id in list(updated_trackable_objects.keys()):
                if obj_id not in seen_ids:
                    cX, cY, zone_history, disappeared = updated_trackable_objects[obj_id]
                    disappeared += 1
                    if disappeared > MAX_DISAPPEARED:
                        del updated_trackable_objects[obj_id]
                    else:
                        updated_trackable_objects[obj_id] = (cX, cY, zone_history, disappeared)
            
            trackable_objects = updated_trackable_objects

            # Vẽ vạch kẻ ngay giữa hình vuông
            cv2.line(display_frame, (LINE_X, 0), (LINE_X, FRAME_SIZE), (0, 255, 255), 2)
            
            frames_ai += 1
            now = time.time()
            if now - prev_ai_time >= 1.0:
                ai_fps = frames_ai / (now - prev_ai_time)
                frames_ai = 0
                prev_ai_time = now

            # ==================================================================
            # BƯỚC CUỐI CÙNG: ĐẢO MÀU LẠI NHƯ BẠN YÊU CẦU TRƯỚC KHI XUẤT 
            # ==================================================================
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

            # CHỐT KHUNG HÌNH WEB
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