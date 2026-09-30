import cv2
import numpy as np
import math

# =====================================================================
# KHỞI TẠO TỌA ĐỘ MẶC ĐỊNH
# =====================================================================
points = {
    "L1_PT1": [270, 280],
    "L1_PT2": [480, 150],
    "L2_PT1": [270, 420],
    "L2_PT2": [480, 290]
}

dragging_point = None
DRAG_RADIUS = 15  # Bán kính nhận diện nhấp chuột (pixel)

# =====================================================================
# HÀM XỬ LÝ SỰ KIỆN CHUỘT
# =====================================================================
def mouse_callback(event, x, y, flags, param):
    global dragging_point

    # Khi nhấn chuột trái: Kiểm tra xem có bấm trúng điểm nào không
    if event == cv2.EVENT_LBUTTONDOWN:
        for name, pt in points.items():
            # Tính khoảng cách từ chuột đến điểm
            distance = math.hypot(x - pt[0], y - pt[1])
            if distance < DRAG_RADIUS:
                dragging_point = name
                break
                
    # Khi di chuyển chuột (và đang giữ một điểm): Cập nhật tọa độ điểm
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging_point is not None:
            points[dragging_point] = [x, y]
            
    # Khi nhả chuột trái: Dừng kéo
    elif event == cv2.EVENT_LBUTTONUP:
        dragging_point = None

# =====================================================================
# KHỞI TẠO CAMERA VÀ GIAO DIỆN
# =====================================================================
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("[ERROR] Không thể mở camera.")
    exit()

cv2.namedWindow("Tool Hieu Chinh Vach")
cv2.setMouseCallback("Tool Hieu Chinh Vach", mouse_callback)

print("[INFO] Tool đang chạy. Kéo thả các chấm xanh để chỉnh vạch.")
print("[INFO] Nhấn ENTER hoặc 'q' để lưu tọa độ và thoát.")

# =====================================================================
# VÒNG LẶP LIVE STREAM
# =====================================================================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Vẽ Vạch 1 (NGOÀI)
    cv2.line(frame, tuple(points["L1_PT1"]), tuple(points["L1_PT2"]), (0, 0, 255), 2)
    cv2.putText(frame, "VACH NGOAI", (points["L1_PT2"][0] + 10, points["L1_PT2"][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Vẽ Vạch 2 (TRONG)
    cv2.line(frame, tuple(points["L2_PT1"]), tuple(points["L2_PT2"]), (0, 255, 255), 2)
    cv2.putText(frame, "VACH TRONG", (points["L2_PT2"][0] + 10, points["L2_PT2"][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Vẽ các "Tay cầm" (Chấm xanh) để kéo thả
    for name, pt in points.items():
        cv2.circle(frame, tuple(pt), 6, (0, 255, 0), -1)
        # Tạo hiệu ứng phát sáng nhẹ khi đang kéo điểm đó
        if dragging_point == name:
            cv2.circle(frame, tuple(pt), 10, (255, 255, 255), 2)

    # Hướng dẫn trên màn hình
    cv2.putText(frame, "Keo tha cham XANH LACA de chinh", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(frame, "Nhan ENTER de luu va thoat", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Tool Hieu Chinh Vach", frame)

    # Chờ phím bấm (13 = Phím Enter)
    key = cv2.waitKey(1) & 0xFF
    if key == 13 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# =====================================================================
# IN KẾT QUẢ ĐỂ COPY VÀO CODE CHÍNH
# =====================================================================
print("\n" + "="*50)
print("DA LUU THANH CONG! HAY COPY DOAN SAU VAO CODE CHINH:")
print("="*50 + "\n")

print(f"L1_PT1 = tuple({points['L1_PT1']})")
print(f"L1_PT2 = tuple({points['L1_PT2']})")
print(f"L2_PT1 = tuple({points['L2_PT1']})")
print(f"L2_PT2 = tuple({points['L2_PT2']})\n")
print("="*50)
