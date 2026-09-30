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
# 3. KIẾN TRÚC CAMERA (SỬ DỤNG HARDWARE BINNING 1640x1232)
# =====================================================================
class CameraThread:
    def __init__(self):
        print("[INFO] Đang khởi động Picamera2 ở chế độ Binned 1640x1232...")
        self.picam2 = Picamera2()
        
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

def get_proposals(feat_mat, stride, anchors, prob_threshold, frame_size, ai_size):
    """
    Hàm giải mã (Decoder) chuyên dụng cho YOLO-FastestV2.
    Sử dụng kỹ thuật Flatten để ép Numpy đọc đúng ma trận của NCNN.
    """
    # 1. Tự tính toán số lượng ô lưới (Grid) chuẩn thay vì dựa vào NCNN
    grid_h = ai_size // stride
    grid_w = ai_size // stride

    # 2. Đập phẳng toàn bộ bộ nhớ thành 1 chiều để loại bỏ lỗi sắp xếp sai
    feat_flat = np.array(feat_mat).flatten()

    # 3. Tính toán linh hoạt số Kênh (Channels). Dù model bạn train 80 class hay 2 class đều chạy được.
    c = len(feat_flat) // (grid_h * grid_w)

    # 4. Dựng lại mảng theo đúng thứ tự vật lý do lớp Permute tạo ra: (Cao, Rộng, Kênh)
    feat = feat_flat.reshape((grid_h, grid_w, c))

    # 5. Đảo ngược lại thành (Kênh, Cao, Rộng) để Code Python dễ cắt lớp
    feat = feat.transpose(2, 0, 1)

    num_anchors = 3

    # Cắt 3 cụm thông tin riêng biệt của Decoupled Head
    reg = feat[0:12, :, :].reshape((num_anchors, 4, grid_h, grid_w)) # Tọa độ x,y,w,h
    obj = feat[12:15, :, :]                                          # Độ tự tin có vật thể
    cls = feat[15:, :, :]                                            # Phân loại đối tượng

    # Hàm đưa giá trị thô về tỷ lệ phần trăm (0.0 -> 1.0)
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    # Tính điểm tự tin và lọc nhanh các điểm nghi ngờ có người
    obj_score = sigmoid(obj) 
    anch_idx, y_idx, x_idx = np.where(obj_score > prob_threshold)
    
    boxes, scores, class_ids = [], [], []
    scale = frame_size / float(ai_size)
    
    for i in range(len(anch_idx)):
        a = anch_idx[i]
        y = y_idx[i]
        x = x_idx[i]
        
        # Kiểm tra đây là người hay xe hay vật khác
        cls_vals = cls[:, y, x]
        cls_id = np.argmax(cls_vals)
        cls_score_val = sigmoid(cls_vals[cls_id]) # Phải sigmoid để ra tỷ lệ chuẩn
        
        # Tổng điểm = (Độ tự tin có vật) * (Độ tự tin đó là con người)
        score = obj_score[a, y, x] * cls_score_val
        
        # Chỉ nhận ID 0 (Người) và ID 1 (Xe đạp/máy)
        if score > prob_threshold and cls_id in [0, 1]:
            dx = sigmoid(reg[a, 0, y, x])
            dy = sigmoid(reg[a, 1, y, x])
            dw = sigmoid(reg[a, 2, y, x])
            dh = sigmoid(reg[a, 3, y, x])
            
            # Công thức giải mã tọa độ bản lề của YOLOv5 / FastestV2
            pb_cx = (x + dx * 2.0 - 0.5) * stride
            pb_cy = (y + dy * 2.0 - 0.5) * stride
            
            anchor_w = anchors[a][0]
            anchor_h = anchors[a][1]
            
            pb_w = ((dw * 2.0)**2) * anchor_w
            pb_h = ((dh * 2.0)**2) * anchor_h
            
            x1 = pb_cx - pb_w * 0.5
            y1 = pb_cy - pb_h * 0.5
            
            # Lưu lại vào mảng để đưa qua bộ lọc trùng lặp NMS
            boxes.append([int(x1 * scale), int(y1 * scale), int(pb_w * scale), int(pb_h * scale)])
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
    
    # SỬA TÊN FILE VÀO ĐÂY
    net.load_param("yolo-fastestv2-opt.param")
    net.load_model("yolo-fastestv2-opt.bin")
    
    AI_SIZE = 352 # YOLOv2 thường train ở chuẩn 352x352, có thể hạ 320 nếu cần chạy nhanh
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
                frame_raw = latest_frame.copy()
                latest_frame = None 

            # Cắt lấy hình vuông 1232x1232 chính giữa
            square_raw = frame_raw[:, 204:1436] 
            
            resized_ai = cv2.resize(square_raw, (AI_SIZE, AI_SIZE), interpolation=cv2.INTER_LINEAR)
            in_mat = ncnn.Mat.from_pixels(resized_ai, ncnn.Mat.PixelType.PIXEL_RGB, AI_SIZE, AI_SIZE)
            
            display_frame = cv2.resize(square_raw, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_LINEAR)

            in_mat.substract_mean_normalize([0.0, 0.0, 0.0], [1/255.0, 1/255.0, 1/255.0])

            ex = net.create_extractor()
            ex.input("input.1", in_mat) # Tên cổng vào của bản v2
            
            # Trích xuất 2 nhánh Output của Decoupled Head
            ret1, out_mat1 = ex.extract("794") 
            ret2, out_mat2 = ex.extract("796") 

            boxes = []
            scores = []
            class_ids = []
            
            CONFIDENCE_THRESHOLD = 0.30

            # Xử lý nhánh 1 (Độ phân giải cao, bắt người nhỏ)
            if out_mat1:
                anchors_16 = [[12, 18], [37, 49], [52, 132]]
                b, s, c = get_proposals(out_mat1, 16, anchors_16, CONFIDENCE_THRESHOLD, FRAME_SIZE, AI_SIZE)
                boxes.extend(b)
                scores.extend(s)
                class_ids.extend(c)
                
            # Xử lý nhánh 2 (Độ phân giải thấp, bắt người to/gần)
            if out_mat2:
                anchors_32 = [[115, 73], [119, 199], [242, 238]]
                b, s, c = get_proposals(out_mat2, 32, anchors_32, CONFIDENCE_THRESHOLD, FRAME_SIZE, AI_SIZE)
                boxes.extend(b)
                scores.extend(s)
                class_ids.extend(c)

            current_centroids = []
            
            # Khử nhiễu các khung hình trùng lặp bằng thuật toán NMS của OpenCV
            if len(boxes) > 0:
                indices = cv2.dnn.NMSBoxes(boxes, scores, CONFIDENCE_THRESHOLD, 0.45)
                if len(indices) > 0:
                    for i in np.array(indices).flatten():
                        x, y, w, h = boxes[i]
                        x1, y1 = x, y
                        x2, y2 = x + w, y + h
                        
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(FRAME_SIZE, x2), min(FRAME_SIZE, y2)
                        
                        cX = int((x1 + x2) / 2.0)
                        cY = int((y1 + y2) / 2.0)
                        current_centroids.append((cX, cY, x1, y1, x2, y2))
                        
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"Nguoi: {scores[i]*100:.1f}%"
                        cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # --- THEO DÕI & ĐẾM ---
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
            
            used_centroids = set()
            used_ids = set()
            updated_trackable_objects = {}
            counters_changed = False
            
            MAX_DISTANCE = 250 

            for d, i, obj_id in matches:
                if d > MAX_DISTANCE: continue
                if i in used_centroids or obj_id in used_ids: continue

                cX, cY, startX, startY, endX, endY = current_centroids[i]
                obj_data = trackable_objects[obj_id]
                
                old_cX = obj_data[0]
                old_cY = obj_data[1]

                dx = cX - old_cX
                dy = cY - old_cY

                if len(obj_data) == 4: zone_history = obj_data[2]
                else: zone_history = obj_data[4]

                zone = "INSIDE" if cX < LINE_X else "OUTSIDE"
                if len(zone_history) == 0 or zone_history[-1] != zone:
                    zone_history.append(zone)

                updated_trackable_objects[obj_id] = (cX, cY, dx, dy, zone_history, 0)
                used_centroids.add(i)
                used_ids.add(obj_id)

            for i, (cX, cY, startX, startY, endX, endY) in enumerate(current_centroids):
                if i not in used_centroids:
                    zone = "INSIDE" if cX < LINE_X else "OUTSIDE"
                    zone_history = deque([zone], maxlen=10)
                    updated_trackable_objects[next_object_id] = (cX, cY, 0, 0, zone_history, 0)
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