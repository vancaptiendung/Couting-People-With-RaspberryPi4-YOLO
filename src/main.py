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
import re
import shutil
import sqlite3
import requests
from collections import deque
from datetime import datetime
from flask import Flask, Response, render_template, jsonify, request
from picamera2 import Picamera2
import logging

from log_handle import *
from const import *
from frame_handle import *
from camera import *

#system infomation
TOTAL_IN, TOTAL_OUT, PEOPLE_IN_ROOM = load_data()

latest_frame = None
output_frame_bgr = None
frame_lock = threading.Lock()
recording_enabled = False
cam_fps = 0
ai_fps = 0

VIDEO_DIR = "videos"
MAX_VIDEOS = 24
CHUNK_DURATION = 30 * 60 
if not os.path.exists(VIDEO_DIR): os.makedirs(VIDEO_DIR)

#setup system
init_db()
log_event("SYSTEM_START", details="Hệ thống Camera AI Pi 5 khởi động", data=[TOTAL_IN, TOTAL_OUT, PEOPLE_IN_ROOM])
cam_thread = CameraThread()

# web backend
app = Flask(__name__)

def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return round(float(f.read()) / 1000.0, 1)
    except: return 0.0

def parse_current_people(details):
    if not details:
        return None
    match = re.search(r"Phòng:\s*(\d+)", details)
    if match:
        return int(match.group(1))
    match = re.search(r"Số người hiện tại:\s*(\d+)", details)
    if match:
        return int(match.group(1))
    return None

def format_log_entry(row):
    log_id, timestamp, event_type, object_id, details, synced = row
    event_key = (event_type or "").upper()
    current_people = parse_current_people(details)

    if event_key in ("IN", "MANUAL_IN"):
        title = "Thay đổi thủ công" if event_key == "MANUAL_IN" else "Có người vào"
        message = f"Số người hiện tại: {current_people}" if current_people is not None else "Có người vào"
        tone = "manual" if event_key == "MANUAL_IN" else "success"
    elif event_key in ("OUT", "MANUAL_OUT"):
        title = "Thay đổi thủ công" if event_key == "MANUAL_OUT" else "Có người ra"
        message = f"Số người hiện tại: {current_people}" if current_people is not None else "Có người ra"
        tone = "manual" if event_key == "MANUAL_OUT" else "danger"
    elif event_key == "SYS_RECORDING":
        title = "Ghi hình"
        message = details or "Trạng thái ghi hình đã thay đổi"
        tone = "info"
    elif event_key == "SYSTEM_START":
        title = "Hệ thống khởi động"
        message = details or "Hệ thống đã sẵn sàng"
        tone = "info"
    elif event_key == "SYSTEM_STOP":
        title = "Hệ thống dừng"
        message = details or "Người dùng đã tắt hệ thống"
        tone = "warning"
    else:
        title = event_type or "Sự kiện"
        message = details or "Không có chi tiết"
        tone = "neutral"

    return {
        "id": log_id,
        "timestamp": timestamp,
        "title": title,
        "message": message,
        "tone": tone,
        "event_type": event_type,
        "synced": synced,
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/logs")
def logs_page():
    initial_logs, has_more, next_before_id = fetch_log_rows(limit=30)
    return render_template(
        "logs.html",
        initial_logs=initial_logs,
        has_more=has_more,
        next_before_id=next_before_id,
    )

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
        "cpu_temp": get_cpu_temp(),
        "storage_free_gb": round(shutil.disk_usage("/").free / (1024 ** 3), 1)
    })

@app.route("/api/logs")
def api_logs():
    before_id = request.args.get("before_id", type=int)
    limit = request.args.get("limit", default=30, type=int)
    items, has_more, next_before_id = fetch_log_rows(before_id=before_id, limit=limit)
    return jsonify({
        "items": items,
        "has_more": has_more,
        "next_before_id": next_before_id,
    })

@app.route("/api/action", methods=["POST"])
def api_action():
    global TOTAL_IN, TOTAL_OUT, PEOPLE_IN_ROOM, recording_enabled
    action = request.json.get("action")
    if action == "in_plus": 
        TOTAL_IN += 1
        PEOPLE_IN_ROOM = TOTAL_IN - TOTAL_OUT
        log_event("MANUAL_IN", "User manually added a person", details=f"Tổng IN: {TOTAL_IN} | Phòng: {PEOPLE_IN_ROOM}", data=[TOTAL_IN, TOTAL_OUT, PEOPLE_IN_ROOM])
    
    elif action == "in_minus" and TOTAL_IN > TOTAL_OUT: 
        TOTAL_IN = TOTAL_IN - 1
        PEOPLE_IN_ROOM = TOTAL_IN - TOTAL_OUT
        log_event("MANUAL_IN", "User manually removed a person", details=f"Tổng IN: {TOTAL_IN} | Phòng: {PEOPLE_IN_ROOM}", data=[TOTAL_IN, TOTAL_OUT, PEOPLE_IN_ROOM])
    
    elif action == "out_plus" and TOTAL_OUT < TOTAL_IN: 
        TOTAL_OUT += 1
        PEOPLE_IN_ROOM = TOTAL_IN - TOTAL_OUT
        log_event("MANUAL_OUT", "User manually added a person", details=f"Tổng OUT: {TOTAL_OUT} | Phòng: {PEOPLE_IN_ROOM}", data=[TOTAL_IN, TOTAL_OUT, PEOPLE_IN_ROOM])
    
    elif action == "out_minus" and TOTAL_OUT > 0: 
        TOTAL_OUT = TOTAL_OUT - 1
        PEOPLE_IN_ROOM = TOTAL_IN - TOTAL_OUT
        log_event("MANUAL_OUT", "User manually removed a person", details=f"Tổng OUT: {TOTAL_OUT} | Phòng: {PEOPLE_IN_ROOM}", data=[TOTAL_IN, TOTAL_OUT, PEOPLE_IN_ROOM])

    # elif action == "room_plus": PEOPLE_IN_ROOM += 1
    # elif action == "room_minus": PEOPLE_IN_ROOM = max(0, PEOPLE_IN_ROOM - 1)
    elif action == "toggle_record": 
        recording_enabled = not recording_enabled
        log_event("SYS_RECORDING", details=f"Ghi hình: {'Bật' if recording_enabled else 'Tắt'}", data=[TOTAL_IN, TOTAL_OUT, PEOPLE_IN_ROOM])
        
    save_data([TOTAL_IN, TOTAL_OUT, PEOPLE_IN_ROOM])
    return jsonify({"status": "success"})

# main function

def main():
    global latest_frame, output_frame_bgr, ai_fps
    global TOTAL_IN, TOTAL_OUT, PEOPLE_IN_ROOM
    
    # threading.Thread(target=nightly_sync_routine, daemon=True).start()

    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR) 
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False), daemon=True).start()
    
    print("[INFO] Đang load model YOLO-Fastest v1.1 XL...")
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = 4 
    
    net.load_param(PARAM_FILE_PATH)
    net.load_model(BIN_FILE_PATH)
    
    trackable_objects = {}
    next_object_id = 0
    
    video_writer = None
    chunk_start_time = 0
    
    prev_ai_time = time.time()
    frames_ai = 0

    print("\n" + "="*50)
    print("[HỆ THỐNG ĐÃ SẴN SÀNG - RASPBERRY PI 5 + YOLO V1.1 XL]")
    print("Truy cập Web Dashboard tại: http://<IP_CỦA_PI>:5000")
    print("="*50 + "\n")

    try:
        while True:
            with frame_lock:
                if latest_frame is None:
                    time.sleep(0.01)
                    continue
                frame_bgr = latest_frame.copy()
                latest_frame = None 

            display_frame = frame_bgr.copy()
            
            in_mat = ncnn.Mat.from_pixels_resize(
                display_frame, 
                ncnn.Mat.PixelType.PIXEL_BGR2RGB, 
                cosnt.DISPLAY_W, cosnt.DISPLAY_H, 
                cosnt.AI_SIZE, cosnt.AI_SIZE
            )

            in_mat.substract_mean_normalize([0.0, 0.0, 0.0], [1/255.0, 1/255.0, 1/255.0])

            ex = net.create_extractor()
            ex.input("data", in_mat) 
            ret, out_mat = ex.extract("output") 

            current_centroids = []
            
            # Tạo các mảng tạm để chứa dữ liệu trước khi lọc
            boxes = []
            scores = []
            class_ids = []
            
            if out_mat:
                for i in range(out_mat.h):
                    values = out_mat.row(i)
                    class_id = int(values[0])
                    score = values[1]

                    if score > 0.40 and class_id in [0, 1]: 
                        
                        if values[2] <= 1.5: 
                            x1 = int(values[2] * cosnt.DISPLAY_W)
                            y1 = int(values[3] * cosnt.DISPLAY_H)
                            x2 = int(values[4] * cosnt.DISPLAY_W)
                            y2 = int(values[5] * cosnt.DISPLAY_H)
                        else:
                            x1 = int((values[2] / cosnt.AI_SIZE) * cosnt.DISPLAY_W)
                            y1 = int((values[3] / cosnt.AI_SIZE) * cosnt.DISPLAY_H)
                            x2 = int((values[4] / cosnt.AI_SIZE) * cosnt.DISPLAY_W)
                            y2 = int((values[5] / cosnt.AI_SIZE) * cosnt.DISPLAY_H)
                        
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(cosnt.DISPLAY_W, x2), min(cosnt.DISPLAY_H, y2)
                        
                        # Chuyển đổi tọa độ sang dạng [x, y, chiều_rộng, chiều_cao] cho hàm NMS
                        boxes.append([x1, y1, x2 - x1, y2 - y1])
                        scores.append(float(score))
                        class_ids.append(class_id)

            # ==============================================================
            # BỘ LỌC NMS: TIÊU DIỆT CÁC HỘP TRÒNG LÊN NHAU
            # ==============================================================
            if len(boxes) > 0:
                # Tham số 0.4 cuối cùng là độ nhạy đè lấp. 
                # Nếu 2 hộp đè lên nhau > 40%, hộp điểm thấp sẽ bị xóa.
                indices = cv2.dnn.NMSBoxes(boxes, scores, 0.40, 0.40)
                
                if len(indices) > 0:
                    for i in indices.flatten():
                        x, y, w, h = boxes[i]
                        score = scores[i]
                        
                        # Tái tạo lại tọa độ x1, y1, x2, y2
                        x1, y1 = x, y
                        x2, y2 = x + w, y + h
                        
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
                zone = "INSIDE" if cX < cosnt.LINE_X else "OUTSIDE"
                
                if len(zone_history) == 0 or zone_history[-1] != zone:
                    zone_history.append(zone)

                updated_trackable_objects[obj_id] = (cX, cY, dx, dy, zone_history, 0)
                used_centroids.add(i)
                used_ids.add(obj_id)

            for i, (cX, cY, startX, startY, endX, endY) in enumerate(current_centroids):
                if i not in used_centroids:
                    zone = "INSIDE" if cX < cosnt.LINE_X else "OUTSIDE"
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
                            PEOPLE_IN_ROOM = TOTAL_IN - TOTAL_OUT
                            updated_trackable_objects[obj_id] = (cX, cY, dx, dy, deque(["INSIDE"], maxlen=10), 0)
                            counters_changed = True
                            # Ghi log SQL
                            log_event("IN", obj_id, f"Tổng IN: {TOTAL_IN} | Phòng: {PEOPLE_IN_ROOM}", data=[TOTAL_IN, TOTAL_OUT, PEOPLE_IN_ROOM])
                            
                        elif idx_in < idx_out and PEOPLE_IN_ROOM > 0: 
                            TOTAL_OUT += 1
                            PEOPLE_IN_ROOM = TOTAL_IN - TOTAL_OUT
                            updated_trackable_objects[obj_id] = (cX, cY, dx, dy, deque(["OUTSIDE"], maxlen=10), 0)
                            counters_changed = True
                            # Ghi log SQL
                            log_event("OUT", obj_id, f"Tổng OUT: {TOTAL_OUT} | Phòng: {PEOPLE_IN_ROOM}", data=[TOTAL_IN, TOTAL_OUT, PEOPLE_IN_ROOM])

            if counters_changed: save_data([TOTAL_IN, TOTAL_OUT, PEOPLE_IN_ROOM])

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

            cv2.line(display_frame, (cosnt.LINE_X, 0), (cosnt.LINE_X, cosnt.DISPLAY_H), (0, 255, 255), 2)
            
            frames_ai += 1
            now = time.time()
            if now - prev_ai_time >= 1.0:
                ai_fps = frames_ai / (now - prev_ai_time)
                frames_ai = 0
                prev_ai_time = now

            timestamp_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            info_str = f"IN: {TOTAL_IN} | OUT: {TOTAL_OUT} | ROOM: {PEOPLE_IN_ROOM}"
            
            # cv2.rectangle(display_frame, (5, 5), (200, 30), (0, 0, 0), -1)
            cv2.putText(display_frame, timestamp_str, (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(display_frame, info_str, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if recording_enabled:
                if video_writer is None or (time.time() - chunk_start_time) >= CHUNK_DURATION:
                    if video_writer is not None: video_writer.release()
                    
                    existing_files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.avi")))
                    while len(existing_files) >= MAX_VIDEOS:
                        os.remove(existing_files[0])
                        existing_files.pop(0)
                        
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    rec_fps = ai_fps if ai_fps > 0 else 15
                    
                    video_writer = ThreadedVideoWriter(
                        os.path.join(VIDEO_DIR, f"cctv_{timestamp}.avi"), 
                        int(rec_fps), 
                        (cosnt.DISPLAY_W, cosnt.DISPLAY_H)
                    )
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
        save_data([TOTAL_IN, TOTAL_OUT, PEOPLE_IN_ROOM])
        log_event("SYSTEM_STOP", details="Người dùng tắt hệ thống an toàn", data=[TOTAL_IN, TOTAL_OUT, PEOPLE_IN_ROOM])
    finally:
        cam_thread.stop()
        if video_writer is not None: video_writer.release()
        print("[INFO] Đã tắt Camera an toàn!")

if __name__ == "__main__":
    main()