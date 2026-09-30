from flask import Flask, Response, render_template, jsonify, request
import sqlite3
import re
import log_handle
import os

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
        log_event("MANUAL_IN", "User manually added a person", details=f"Tổng IN: {TOTAL_IN} | Phòng: {PEOPLE_IN_ROOM}")
    
    elif action == "in_minus" and TOTAL_IN > TOTAL_OUT: 
        TOTAL_IN = TOTAL_IN - 1
        PEOPLE_IN_ROOM = TOTAL_IN - TOTAL_OUT
        log_event("MANUAL_IN", "User manually removed a person", details=f"Tổng IN: {TOTAL_IN} | Phòng: {PEOPLE_IN_ROOM}")
    
    elif action == "out_plus" and TOTAL_OUT < TOTAL_IN: 
        TOTAL_OUT += 1
        PEOPLE_IN_ROOM = TOTAL_IN - TOTAL_OUT
        log_event("MANUAL_OUT", "User manually added a person", details=f"Tổng OUT: {TOTAL_OUT} | Phòng: {PEOPLE_IN_ROOM}")
    
    elif action == "out_minus" and TOTAL_OUT > 0: 
        TOTAL_OUT = TOTAL_OUT - 1
        PEOPLE_IN_ROOM = TOTAL_IN - TOTAL_OUT
        log_event("MANUAL_OUT", "User manually removed a person", details=f"Tổng OUT: {TOTAL_OUT} | Phòng: {PEOPLE_IN_ROOM}")

    # elif action == "room_plus": PEOPLE_IN_ROOM += 1
    # elif action == "room_minus": PEOPLE_IN_ROOM = max(0, PEOPLE_IN_ROOM - 1)
    elif action == "toggle_record": 
        recording_enabled = not recording_enabled
        log_event("SYS_RECORDING", details=f"Ghi hình: {'Bật' if recording_enabled else 'Tắt'}")
        
    save_data()
    return jsonify({"status": "success"})