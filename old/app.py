import os
# Tắt các dòng cảnh báo font chữ vô hại của Qt
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.fonts.warning=false"

import cv2
import ncnn
import numpy as np
from picamera2 import Picamera2
import time  # Thêm thư viện thời gian để tính FPS

def main():
    # 1. Khởi tạo mạng NCNN
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False  # Chạy bằng CPU

    print("Đang load model YOLO-Fastest v1.1...")
    net.load_param("yolo-fastest-1.1.param")
    net.load_model("yolo-fastest-1.1.bin")

    # 2. Khởi tạo Picamera2 
    print("Đang khởi động camera...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "BGR888"})
    picam2.configure(config)
    picam2.start()

    print("Hệ thống đã sẵn sàng. Nhấn phím 'q' để thoát.")

    # Khởi tạo các biến tính FPS
    prev_frame_time = 0
    new_frame_time = 0

    try:
        while True:
            # Lấy ảnh gốc từ camera (Mặc định là RGB)
            frame_rgb = picam2.capture_array()
            h_orig, w_orig, _ = frame_rgb.shape 
            
            # Chuyển sang BGR để hiển thị chuẩn màu bằng OpenCV
            display_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Resize ảnh về 320x320 chuẩn đầu vào model
            resized_rgb = cv2.resize(frame_rgb, (320, 320))

            # 3. Nạp ma trận ảnh RGB vào NCNN
            in_mat = ncnn.Mat.from_pixels(resized_rgb, ncnn.Mat.PixelType.PIXEL_RGB, 320, 320)
            
            # Chuẩn hóa ma trận điểm ảnh (chia 255)
            mean_vals = [0.0, 0.0, 0.0]
            norm_vals = [1/255.0, 1/255.0, 1/255.0]
            in_mat.substract_mean_normalize(mean_vals, norm_vals)

            # 4. Chạy suy luận (Inference)
            ex = net.create_extractor()
            ex.input("data", in_mat) 
            ret, out_mat = ex.extract("output") 

            # 5. Hậu xử lý dữ liệu và vẽ khung cho TẤT CẢ các vật thể tìm thấy
            if out_mat:
                for i in range(out_mat.h):
                    values = out_mat.row(i)
                    class_id = int(values[0])
                    score = values[1]

                    # Nhận diện tất cả vật thể có độ tự tin > 30% để bạn dễ test
                    if score > 0.30:
                        # Tính toán tọa độ hộp bao (bounding box) nhân ngược với ảnh gốc (640x480)
                        x1 = int(values[2] * w_orig)
                        y1 = int(values[3] * h_orig)
                        x2 = int(values[4] * w_orig)
                        y2 = int(values[5] * h_orig)

                        # Giới hạn tọa độ nằm trong rìa màn hình
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(w_orig, x2)
                        y2 = min(h_orig, y2)

                        # Vẽ khung màu xanh lá cây (0, 255, 0) dày 2 pixel quanh vật thể
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Hiển thị nhãn gồm ID của Class và phần trăm tự tin
                        label = f"ID {class_id}: {score*100:.1f}%"
                        cv2.putText(display_frame, label, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 6. TÍNH VÀ HIỂN THỊ FPS
            new_frame_time = time.time()
            # Công thức tính FPS: 1 / (Thời gian hiện tại - Thời gian khung hình trước)
            fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
            prev_frame_time = new_frame_time
            
            # Định dạng chữ FPS và vẽ lên góc trái màn hình (Màu vàng rực rỡ để dễ nhìn)
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(display_frame, fps_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

            # 7. Hiển thị kết quả ra màn hình
            cv2.imshow("YOLO-Fastest - Raspberry Pi 4", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()