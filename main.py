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
from flask import Flask, Response, render_template, jsonify, request
import ncnn # Thư viện AI siêu tốc của Tencent

# =====================================================================
# 1. KHỞI TẠO HỆ THỐNG LƯU TRỮ (JSON)
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

# =====================================================================
# 2. CẤU HÌNH YOLO-FASTEST V2 & NCNN
# =====================================================================
class YoloFastestV2:
    def __init__(self, param_path, bin_path):
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False # Ép 100% CPU NEON
        self.net.opt.num_threads = 4            # Bung hết 4 nhân của Pi
        
        self.net.load_param(param_path)
        self.net.load_model(bin_path)
        
        self.target_size = 352 # Kích thước chuẩn Max mAP của Yolo-Fastest V2
        self.mean_vals = [0.0, 0.0, 0.0]
        self.norm_vals = [1/255.0, 1/255.0, 1/255.0]
        
        # Sửa lại tên cổng cho đúng với file .param
        self.input_name = "input.1"
        self.output_names = ["794", "796"]

    def detect(self, img, conf_thresh=0.45):
        img_h, img_w = img.shape[:2]
        
        # Nén ảnh siêu tốc 
        mat_in = ncnn.Mat.from_pixels_resize(img, ncnn.Mat.PixelType.PIXEL_BGR2RGB, img_w, img_h, self.target_size, self.target_size)
        mat_in.substract_mean_normalize(self.mean_vals, self.norm_vals)

        ex = self.net.create_extractor()
        ex.input(self.input_name, mat_in)
        
        boxes = []
        confidences = []
        
        scale_w = img_w / self.target_size
        scale_h = img_h / self.target_size
        
        # Giải mã 2 ma trận đầu ra "794" và "796"
        for out_name in self.output_names:
            ret, mat_out = ex.extract(out_name)
            if not ret: continue
            
            # Chuyển NCNN Mat sang Numpy Array (Shape: 95 channels, H, W)
            out = np.array(mat_out) 
            
            # Kênh 15 chứa điểm tự tin (Class Score) của riêng class 0 (Person)
            cls_score = out[15, :, :] 
            
            # Chạy qua 3 Anchors
            for b in range(3):
                # Kênh 12, 13, 14 chứa điểm phát hiện vật thể (Objectness)
                obj_score = out[12 + b, :, :] 
                
                # Nhân 2 điểm số để ra độ tin cậy cuối cùng
                conf = obj_score * cls_score 
                
                # BÍ QUYẾT NUMPY: Lọc ma trận song song để lấy các pixel > threshold (cực nhanh)
                mask = conf > conf_thresh
                grid_y, grid_x = np.where(mask)
                
                for gy, gx in zip(grid_y, grid_x):
                    c = conf[gy, gx]
                    
                    # Trích xuất 4 kênh tọa độ Box (Đã được mô hình hóa sẵn)
                    bcx = out[b * 4 + 0, gy, gx]
                    bcy = out[b * 4 + 1, gy, gx]
                    bw = out[b * 4 + 2, gy, gx]
                    bh = out[b * 4 + 3, gy, gx]
                    
                    # Phóng tọa độ về đúng tỷ lệ khung hình thật
                    width = bw * scale_w
                    height = bh * scale_h
                    x1 = int((bcx * scale_w) - width / 2.0)
                    y1 = int((bcy * scale_h) - height / 2.0)
                    
                    boxes.append([x1, y1, int(width), int(height)])
                    confidences.append(float(c))
                    
        # NMS: Khử các khung hình đè lên nhau
        final_boxes = []
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, 0.45)
            for idx in indices:
                idx = idx if isinstance(idx, int) else idx[0]
                x, y, w, h = boxes[idx]
                final_boxes.append((x, y, x + w, y + h, confidences[idx]))
                
        return final_boxes

print("[INFO] Đang nạp lõi NCNN YOLO-Fastest V2...")
try:
    yolo_net = YoloFastestV2("yolo-fastestv2.param", "yolo-fastestv2.bin")
except Exception as e:
    print("[ERROR] Không tìm thấy file model YOLO NCNN! Hãy kiểm tra lại.")
    sys.exit(1)

# =====================================================================
# 3. CẤU HÌNH THÔNG SỐ HỆ THỐNG
# =====================================================================
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
LINE_X = int(FRAME_WIDTH / 2)
MAX_DISAPPEARED = 30     
trackable_objects = {}
next_object_id = 0

VIDEO_DIR = "videos"
MAX_VIDEOS = 3
CHUNK_DURATION = 30 * 60  
recording_enabled = True 
video_writer = None
chunk_start_time = 0
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

def get_zone(cx): return "INSIDE" if cx < LINE_X else "OUTSIDE"

def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp = float(f.read()) / 1000.0
        return round(temp, 1)
    except: return 0.0

# =====================================================================
# 4. KHỞI TẠO CAMERA (CHUẨN HD 1280x720 - CÂN TÂM IMX219)
# =====================================================================
class FrameGrabber:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.stopped = False
        self.frame = None
        self.lock = threading.Lock()
        
        self.capture_fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # ---> XÓA HOẶC COMMENT DÒNG NÀY LẠI <---
        # self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        self.cap.set(cv2.CAP_PROP_FPS, 25) 
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
        threading.Thread(target=self._reader, daemon=True).start()

    def isOpened(self): return self.cap.isOpened()
    
    def _reader(self):
        while not self.stopped:
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue
                
                # Đo FPS Camera
                self.frame_count += 1
                elapsed = time.time() - self.start_time
                if elapsed >= 1.0:
                    self.capture_fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.start_time = time.time()

                # Nén ảnh về 320x240
                small_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                with self.lock: 
                    self.frame = small_frame
                    
            except cv2.error as e:
                # Nếu gặp lỗi Reshape do buffer rác, bỏ qua khung hình này!
                # print(f"[CẢNH BÁO] Lỗi buffer camera: {e}")
                time.sleep(0.01)
                continue    

    def read(self):
        with self.lock:
            if self.frame is None: return None
            f = self.frame.copy(); self.frame = None; return f

    def release(self):
        self.stopped = True
        time.sleep(0.05); self.cap.release()

vs = FrameGrabber(0)
if not vs.isOpened():
    print("[ERROR] Lỗi Camera. Hãy kiểm tra kết nối!")
    sys.exit(1)

# =====================================================================
# 5. MÁY CHỦ WEB API (FLASK SERVER)
# =====================================================================
app = Flask(__name__)
outputFrame = None
lock = threading.Lock()
processing_fps = 0

@app.route("/")
def index():
    return render_template("index.html")

def generate():
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None: continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if not flag: continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/data", methods=["GET"])
def api_data():
    return jsonify({
        "in": TOTAL_IN, 
        "out": TOTAL_OUT, 
        "room": PEOPLE_IN_ROOM, 
        "recording": recording_enabled,
        "cam_fps": round(vs.capture_fps, 1),
        "ai_fps": round(processing_fps, 1),
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

def start_flask():
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR) 
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, use_reloader=False)

threading.Thread(target=start_flask, daemon=True).start()
print("\n" + "="*50)
print("[HỆ THỐNG ĐÃ SẴN SÀNG]")
print(f"[ĐỊA CHỈ WEB] Truy cập: http://<ĐỊA_CHỈ_IP_CỦA_PI>:5000")
print("[TẮT MÁY] Bấm 'Ctrl + C' trên Terminal để tắt chương trình.")
print("="*50 + "\n")

# =====================================================================
# VÒNG LẶP XỬ LÝ CHÍNH (AI CORE)
# =====================================================================
proc_frame_count = 0
proc_start_time = time.time()

try:
    while True:
        frame = vs.read()
        if frame is None:
            time.sleep(0.01); continue

        # Đo FPS của AI
        proc_frame_count += 1
        elapsed = time.time() - proc_start_time
        if elapsed >= 1.0:
            processing_fps = proc_frame_count / elapsed
            proc_frame_count = 0
            proc_start_time = time.time()

        current_centroids = []

        # Chạy AI Hủy Diệt YOLO-Fastest V2 (NCNN)
        detections = yolo_net.detect(frame, conf_thresh=0.45)

        for (startX, startY, endX, endY, conf) in detections:
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            current_centroids.append((cX, cY, startX, startY, endX, endY))

        updated_trackable_objects = dict(trackable_objects)
        seen_ids = set()
        counters_changed = False

        for (cX, cY, startX, startY, endX, endY) in current_centroids:
            matched_id = None; min_distance = 100 
            for obj_id, (old_cX, old_cY, zone_history, disappeared) in trackable_objects.items():
                d = np.hypot(cX - old_cX, cY - old_cY)
                if d < min_distance:
                    min_distance = d; matched_id = obj_id

            if matched_id is None:
                matched_id = next_object_id; next_object_id += 1
                zone_history = deque([get_zone(cX)], maxlen=10); disappeared = 0
            else:
                old_cX, old_cY, zone_history, disappeared = trackable_objects[matched_id]
                current_zone = get_zone(cX)
                if len(zone_history) == 0 or zone_history[-1] != current_zone:
                    zone_history.append(current_zone)
                disappeared = 0

            final_comp = []
            for z in zone_history:
                if not final_comp or final_comp[-1] != z: final_comp.append(z)

            if "OUTSIDE" in final_comp and "INSIDE" in final_comp:
                idx_out = final_comp.index("OUTSIDE"); idx_in = final_comp.index("INSIDE")
                if idx_out < idx_in:
                    TOTAL_IN += 1; PEOPLE_IN_ROOM += 1; zone_history = deque(["INSIDE"], maxlen=10); counters_changed = True
                elif idx_in < idx_out:
                    TOTAL_OUT += 1; PEOPLE_IN_ROOM = max(0, PEOPLE_IN_ROOM - 1); zone_history = deque(["OUTSIDE"], maxlen=10); counters_changed = True

            updated_trackable_objects[matched_id] = (cX, cY, zone_history, disappeared)
            seen_ids.add(matched_id)
            
            # Vẽ Box
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 150, 0), 2)
            cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)

        if counters_changed: save_data()

        for obj_id in list(updated_trackable_objects.keys()):
            if obj_id not in seen_ids:
                cX, cY, zone_history, disappeared = updated_trackable_objects[obj_id]
                disappeared += 1
                if disappeared > MAX_DISAPPEARED: del updated_trackable_objects[obj_id]
                else: updated_trackable_objects[obj_id] = (cX, cY, zone_history, disappeared)
        trackable_objects = updated_trackable_objects

        # Vẽ Line vạch kẻ
        cv2.line(frame, (LINE_X, 0), (LINE_X, frame.shape[0]), (0, 255, 255), 2)

        # Chốt hình ảnh
        with lock:
            outputFrame = frame.copy()

        # Quay Video
        if recording_enabled:
            if video_writer is None or (time.time() - chunk_start_time) >= CHUNK_DURATION:
                if video_writer is not None: video_writer.release()
                existing_files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.avi")))
                while len(existing_files) >= MAX_VIDEOS:
                    os.remove(existing_files[0]); existing_files.pop(0)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                rec_fps = processing_fps if processing_fps > 0 else 10
                video_writer = ThreadedVideoWriter(os.path.join(VIDEO_DIR, f"cctv_{timestamp}.avi"), cv2.VideoWriter_fourcc(*'XVID'), rec_fps, (FRAME_WIDTH, FRAME_HEIGHT))
                chunk_start_time = time.time()
            video_writer.write(frame)
        else:
            if video_writer is not None: video_writer.release(); video_writer = None

except KeyboardInterrupt:
    print("\n[INFO] Đang lưu dữ liệu và tắt hệ thống...")
    save_data()

if video_writer is not None: video_writer.release()
vs.release()
print("[INFO] Đã tắt Camera an toàn!")