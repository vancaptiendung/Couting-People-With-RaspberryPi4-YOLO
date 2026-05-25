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

# =====================================================================
# 1. KHỞI TẠO HỆ THỐNG LƯU TRỮ
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
# 2. CẤU HÌNH AI, HIỆU NĂNG & GHI HÌNH
# =====================================================================
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

try:
    net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")
except:
    print("[ERROR] Không tìm thấy file mô hình AI!")
    sys.exit(1)

CONFIDENCE_THRESHOLD = 0.5 
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

# =====================================================================
# 3. KHỞI TẠO CAMERA (NÂNG CẤP FPS CHỤP HÌNH)
# =====================================================================
class FrameGrabber:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.stopped = False
        self.frame = None
        self.lock = threading.Lock()
        
        # Biến đếm FPS của riêng camera
        self.capture_fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # Ép camera chụp cực nhanh ở 25 FPS
        self.cap.set(cv2.CAP_PROP_FPS, 25) 
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
        threading.Thread(target=self._reader, daemon=True).start()

    def isOpened(self): return self.cap.isOpened()
    
    def _reader(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01); continue
                
            # Đếm số khung hình Camera bắt được mỗi giây
            self.frame_count += 1
            elapsed = time.time() - self.start_time
            if elapsed >= 1.0:
                self.capture_fps = self.frame_count / elapsed
                self.frame_count = 0
                self.start_time = time.time()

            small_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            with self.lock: self.frame = small_frame

    def read(self):
        with self.lock:
            if self.frame is None: return None
            f = self.frame.copy(); self.frame = None; return f

    def release(self):
        self.stopped = True
        time.sleep(0.05); self.cap.release()

vs = FrameGrabber(0)
if not vs.isOpened(): sys.exit(1)

# =====================================================================
# 4. MÁY CHỦ WEB API (FLASK SERVER)
# =====================================================================
app = Flask(__name__)
outputFrame = None
lock = threading.Lock()

# Biến đếm FPS của CPU/AI
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

# API GỬI DATA VỀ DASHBOARD WEB (Đã thêm 2 tham số FPS)
@app.route("/api/data", methods=["GET"])
def api_data():
    return jsonify({
        "in": TOTAL_IN, 
        "out": TOTAL_OUT, 
        "room": PEOPLE_IN_ROOM, 
        "recording": recording_enabled,
        "cam_fps": round(vs.capture_fps, 1),
        "ai_fps": round(processing_fps, 1)
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
print("\n[HỆ THỐNG ĐÃ SẴN SÀNG] Truy cập Dashboard tại: http://<IP_CỦA_PI>:5000\n")

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

        # Tính toán tốc độ xử lý của CPU cho AI
        proc_frame_count += 1
        elapsed = time.time() - proc_start_time
        if elapsed >= 1.0:
            processing_fps = proc_frame_count / elapsed
            proc_frame_count = 0
            proc_start_time = time.time()

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
                cX = int((startX + endX) / 2.0); cY = int((startY + endY) / 2.0)
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

        cv2.line(frame, (LINE_X, 0), (LINE_X, h), (0, 255, 255), 2)

        with lock:
            outputFrame = frame.copy()

        if recording_enabled:
            if video_writer is None or (time.time() - chunk_start_time) >= CHUNK_DURATION:
                if video_writer is not None: video_writer.release()
                existing_files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.avi")))
                while len(existing_files) >= MAX_VIDEOS:
                    os.remove(existing_files[0]); existing_files.pop(0)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Video sẽ được ghi với tốc độ thực tế của AI để tránh video bị tua nhanh
                video_writer = ThreadedVideoWriter(os.path.join(VIDEO_DIR, f"cctv_{timestamp}.avi"), cv2.VideoWriter_fourcc(*'XVID'), processing_fps if processing_fps > 0 else 5, (FRAME_WIDTH, FRAME_HEIGHT))
                chunk_start_time = time.time()
            video_writer.write(frame)
        else:
            if video_writer is not None: video_writer.release(); video_writer = None

except KeyboardInterrupt:
    save_data()

if video_writer is not None: video_writer.release()
vs.release()