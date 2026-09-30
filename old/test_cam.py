import cv2
import sys

print("[INFO] Đang kết nối với Camera CSI qua cổng V4L2...")

# Khởi tạo camera tại cổng 0 (để kết hợp với libcamerify)
vs = cv2.VideoCapture(0, cv2.CAP_V4L2)
vs.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not vs.isOpened():
    print("[ERROR] Không thể mở camera. Vui lòng kiểm tra lại lệnh chạy!")
    sys.exit(1)

print("[INFO] Mở camera THÀNH CÔNG! Đang chuẩn bị hiển thị...")
print("[INFO] Mẹo: Vẫy tay trước camera để kiểm tra nhận diện chuyển động. Nhấn 'q' để THOÁT.")

# Biến lưu trữ khung hình nền đầu tiên để so sánh chuyển động
bg_frame = None

while True:
    ret, frame = vs.read()
    if not ret:
        continue  # Bỏ qua nếu có khung hình lỗi ban đầu

    # --- THUẬT TOÁN NHẬN DIỆN VẬT THỂ DI CHUYỂN (KHÔNG DÙNG AI) ---
    # 1. Chuyển ảnh sang màu xám và làm mịn để giảm nhiễu
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # 2. Nếu là khung hình đầu tiên, lưu nó làm nền (background)
    if bg_frame is None:
        bg_frame = gray
        continue

    # 3. Tính toán sự khác biệt giữa khung hình hiện tại và ảnh nền
    frame_delta = cv2.absdiff(bg_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    # 4. Tìm các đường viền (contours) của vật thể đang chuyển động
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. Vẽ khung vuông xung quanh các vật thể chuyển động đó
    for contour in contours:
        if cv2.contourArea(contour) < 500:  # Bỏ qua các vật thể quá nhỏ (nhiễu)
            continue
        
        # Lấy tọa độ và vẽ khung xanh
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Vat the di chuyen", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Cập nhật lại nền liên tục để thích ứng với ánh sáng thay đổi
    bg_frame = gray

    # --- HIỂN THỊ MÀN HÌNH ---
    cv2.imshow("Kiem tra Camera CSI - Raspberry Pi 4", frame)

    # Nhấn phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Giải phóng bộ nhớ
vs.release()
cv2.destroyAllWindows()
print("[INFO] Đã đóng file test an toàn.")
