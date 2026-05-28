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
# 2. HỆ THỐNG GHI HÌNH (Lưu 6 Video x 30 Phút)
# =====================================================================
VIDEO_DIR = "videos"
MAX_VIDEOS = 6
CHUNK_DURATION = 30 * 60 
if not os.path.exists(VIDEO_DIR): os.makedirs(VIDEO_DIR)

class ThreadedVideoWriter:
    def __init__(self, filename, fps, frame_size):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
        self.q = queue.Queue(maxsize=256)
        self.stopped = False
        threading.Thread(target=self._write, daemon=True).start()
        
    def write(self, frame):
        if not self.q.full(): 
            self.q.put(frame.copy())
            
    def _write(self):
        while not self.stopped:
            if not self.q.empty(): 
                self.writer.write(self.q.get())
            else: 
                time.sleep(0.01) 
                
    def release(self):
        self.stopped = True
        while not self.q.empty(): 
            self.writer.write(self.q.get())
        self.writer.release()

# =====================================================================
# 3. KIẾN TRÚC CAMERA (HARDWARE DUAL-STREAM CHO V1.3)
# =====================================================================
# =====================================================================
# 3. KIẾN TRÚC CAMERA (HARDWARE DUAL-STREAM CHO V1.3 - FIX YUV)
# =====================================================================
class CameraThread:
    def __init__(self):
        print("[INFO] Đang khởi động Picamera2 (Lấy FULL góc -> Hardware Resize)...")
        self.picam2 = Picamera2()
        
        # SỬA LỖI TẠI ĐÂY: Chuyển format của lores thành YUV420 theo quy định phần cứng
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
                # 1. Nhận mảng YUV thô từ luồng lores của phần cứng
                frame_yuv = self.picam2.capture_array("lores")
                
                # 2. Dịch mảng YUV đó sang mảng RGB bằng OpenCV (cực nhanh, không ngốn CPU)
                frame_rgb = cv2.cvtColor(frame_yuv, cv2.COLOR_YUV2RGB_I420)
                
                with frame_lock:
                    latest_frame = frame_rgb
                
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
# 4. MÁY CHỦ WEB API (BẢO VỆ CHỐNG RỚT KẾT NỐI)
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
        # Client đóng web hoặc mất mạng sẽ rơi vào đây, ngắt êm ái
        pass

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
# 5. LUỒNG AI CHÍNH VÀ BỘ GIẢI MÃ YOLO-FASTEST V2
# =====================================================================
def get_proposals(feat_mat, stride, anchors, prob_threshold, display_w, display_h, ai_size):
    grid_h = ai_size // stride
    grid_w = ai_size // stride
    
    # 1. Đập phẳng và tái tạo mảng chuẩn H x W x Channels (như C++)
    feat_flat = np.array(feat_mat).flatten()
    feat = feat_flat.reshape((grid_h, grid_w, -1)) 

    # 2. Cắt mảng y hệt Index C++
    reg = feat[:, :, 0:12].reshape((grid_h, grid_w, 3, 4)) # Tọa độ dx, dy, dw, dh
    obj = feat[:, :, 12:15] # Điểm vật thể
    cls = feat[:, :, 15:]   # Điểm phân loại 80 class

    boxes, scores, class_ids = [], [], []
    scale_x = display_w / float(ai_size)
    scale_y = display_h / float(ai_size)

    # 3. Lặp qua từng ô lưới
    for y in range(grid_h):
        for x in range(grid_w):
            # Điểm class được tính chung cho cả ô lưới này
            cls_vals = cls[y, x, :]
            cls_id = np.argmax(cls_vals)
            cls_score_val = cls_vals[cls_id]

            # Lặp qua 3 anchor
            for b in range(3):
                # Tuyệt đối KHÔNG dùng sigmoid vì model đã tự tính
                score = obj[y, x, b] * cls_score_val

                # Chỉ lấy Người (0) hoặc Xe đạp/máy (1)
                if score > prob_threshold and cls_id in [0, 1]:
                    dx = reg[y, x, b, 0]
                    dy = reg[y, x, b, 1]
                    dw = reg[y, x, b, 2]
                    dh = reg[y, x, b, 3]

                    # Công thức giải mã tọa độ y hệt C++
                    pb_cx = ((dx * 2.0 - 0.5) + x) * stride
                    pb_cy = ((dy * 2.0 - 0.5) + y) * stride

                    anchor_w, anchor_h = anchors[b][0], anchors[b][1]

                    pb_w = ((dw * 2.0) ** 2) * anchor_w
                    pb_h = ((dh * 2.0) ** 2) * anchor_h

                    x1 = pb_cx - pb_w * 0.5
                    y1 = pb_cy - pb_h * 0.5

                    boxes.append([int(x1 * scale_x), int(y1 * scale_y), int(pb_w * scale_x), int(pb_h * scale_y)])
                    scores.append(float(score))
                    class_ids.append(int(cls_id))

    return boxes, scores, class_ids


def main():
    global latest_frame, output_frame_bgr, ai_fps
    global TOTAL_IN, TOTAL_OUT, PEOPLE_IN_ROOM
    
    cam_thread = CameraThread()
    
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR) 
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False), daemon=True).start()
    
    print("[INFO] Đang load model YOLO-Fastest v2...")
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = 4 
    
    net.load_param("yolo-fastestv2-opt.param")
    net.load_model("yolo-fastestv2-opt.bin")
    
    AI_SIZE = 352
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
    print("[HỆ THỐNG ĐÃ SẴN SÀNG - TỐI ƯU PI4 + V1.3]")
    print("Truy cập Web Dashboard tại: http://<IP_CỦA_PI>:5000")
    print("="*50 + "\n")

    try:
        while True:
            with frame_lock:
                if latest_frame is None:
                    time.sleep(0.01)
                    continue
                frame_raw = latest_frame.copy()
                latest_frame = None 

            display_frame = frame_raw.copy()
            
            # Sử dụng C++ NEON Resize của NCNN thay cho cv2.resize của Python
            in_mat = ncnn.Mat.from_pixels_resize(
                display_frame, 
                ncnn.Mat.PixelType.PIXEL_RGB, 
                DISPLAY_W, DISPLAY_H, 
                AI_SIZE, AI_SIZE
            )

            in_mat.substract_mean_normalize([0.0, 0.0, 0.0], [1/255.0, 1/255.0, 1/255.0])

            ex = net.create_extractor()
            ex.input("input.1", in_mat) 
            
            ret1, out_mat1 = ex.extract("794") 
            ret2, out_mat2 = ex.extract("796") 

            boxes, scores, class_ids = [], [], []
            CONFIDENCE_THRESHOLD = 0.4

            if out_mat1:
                # Lấy 3 cặp Anchor đầu tiên từ C++
                anchors_16 = [[12.64, 19.39], [37.88, 51.48], [55.71, 138.31]]
                b, s, c = get_proposals(out_mat1, 16, anchors_16, CONFIDENCE_THRESHOLD, DISPLAY_W, DISPLAY_H, AI_SIZE)
                boxes.extend(b)
                scores.extend(s)
                class_ids.extend(c)
                
            if out_mat2:
                # Lấy 3 cặp Anchor sau từ C++
                anchors_32 = [[126.91, 78.23], [131.57, 214.55], [279.92, 258.87]]
                b, s, c = get_proposals(out_mat2, 32, anchors_32, CONFIDENCE_THRESHOLD, DISPLAY_W, DISPLAY_H, AI_SIZE)
                boxes.extend(b)
                scores.extend(s)
                class_ids.extend(c)

            current_centroids = []
            
            if len(boxes) > 0:
                indices = cv2.dnn.NMSBoxes(boxes, scores, CONFIDENCE_THRESHOLD, 0.45)
                if len(indices) > 0:
                    for i in np.array(indices).flatten():
                        x, y, w, h = boxes[i]
                        x1, y1 = max(0, x), max(0, y)
                        x2, y2 = min(DISPLAY_W, x + w), min(DISPLAY_H, y + h)
                        
                        cX = int((x1 + x2) / 2.0)
                        cY = int((y1 + y2) / 2.0)
                        current_centroids.append((cX, cY, x1, y1, x2, y2))
                        
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(display_frame, f"Nguoi: {scores[i]*100:.1f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # --- THEO DÕI & ĐẾM BẰNG QUÁN TÍNH ---
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

            # Đóng dấu ngày giờ và số liệu lên video
            timestamp_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            info_str = f"IN: {TOTAL_IN} | OUT: {TOTAL_OUT} | ROOM: {PEOPLE_IN_ROOM}"
            
            cv2.rectangle(display_frame, (5, 5), (420, 65), (0, 0, 0), -1)
            cv2.putText(display_frame, timestamp_str, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display_frame, info_str, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            display_frame_corrected = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

            if recording_enabled:
                if video_writer is None or (time.time() - chunk_start_time) >= CHUNK_DURATION:
                    if video_writer is not None: video_writer.release()
                    
                    existing_files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))
                    while len(existing_files) >= MAX_VIDEOS:
                        os.remove(existing_files[0])
                        existing_files.pop(0)
                        
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    rec_fps = ai_fps if ai_fps > 0 else 15
                    
                    video_writer = ThreadedVideoWriter(
                        os.path.join(VIDEO_DIR, f"cctv_{timestamp}.mp4"), 
                        int(rec_fps), 
                        (DISPLAY_W, DISPLAY_H)
                    )
                    chunk_start_time = time.time()
                
                video_writer.write(display_frame_corrected)
            else:
                if video_writer is not None: 
                    video_writer.release()
                    video_writer = None

            with frame_lock:
                output_frame_bgr = display_frame_corrected.copy()

    except KeyboardInterrupt:
        print("\n[INFO] Đang lưu dữ liệu và tắt hệ thống...")
        save_data()
    finally:
        cam_thread.stop()
        if video_writer is not None: video_writer.release()
        print("[INFO] Đã tắt Camera an toàn!")

if __name__ == "__main__":
    main()