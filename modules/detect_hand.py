import os
import sys

# --- SETUP MÔI TRƯỜNG CHẶN LOG ---
# Phải đặt trước tất cả các lệnh import thư viện nặng
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'       # Tắt oneDNN custom ops
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'        # Tắt log C++ của TensorFlow (0=ALL, 1=WARN, 2=ERR, 3=FATAL)
os.environ['GLOG_minloglevel'] = '2'            # Tắt log C++ của MediaPipe

import warnings
import logging

# Tắt cảnh báo Python (FutureWarning, DeprecationWarning...)
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Tắt log của Abseil (thư viện log Google dùng cho MediaPipe)
logging.getLogger('absl').setLevel(logging.ERROR)

import cv2
import numpy as np
import time
import tensorflow.lite as tflite
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FraudDetector:
    def __init__(self, tflite_model_path, hand_model_path, drawer_roi, pos_roi):
        # 1. Cấu hình & Load Model
        self.drawer_roi = drawer_roi
        self.pos_roi = pos_roi
        self.hand_model_path = hand_model_path
        
        # Load Model TFLite (Két tiền)
        self.interpreter = tflite.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']

        # Load Model MediaPipe (Tay)
        self._init_mediapipe()

        # 2. Logic Variables (FSM)
        self.state = "IDLE"
        self.last_pos_time = 0
        self.pos_timeout = 30.0     # Thời gian chờ từ lúc bấm POS đến lúc mở két
        self.drawer_buffer = deque(maxlen=5) 
        self.frame_count = 0 
        self.last_drawer_status = "CLOSED"
        self.close_confirm_counter = 0 
        self.CLOSE_THRESHOLD = 60  # Số frame liên tiếp két đóng để xác nhận kết đóng

        # Variables cho Dwell Time & Refund
        self.pos_enter_time = None     # Thời điểm tay bắt đầu vào vùng POS  
        self.POS_DWELL_THRESHOLD = 3.0   # Phải giữ tay 3s mới tính là bấm (chống lướt qua)
        self.is_pressing_pos = False     # Trạng thái xác nhận "Đang bấm thật"

        # --- MOTION GATE SETUP (Tối ưu hóa) ---
        # Sử dụng Background Subtractor để phát hiện chuyển động thô của pixel
        # history=500: Học nền trong 500 frame
        # varThreshold=50: Độ nhạy (cao hơn thì ít nhiễu hơn)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=False
        )
        
        self.MOTION_THRESHOLD = 0.05 # 5% diện tích vùng ROI thay đổi là có chuyển động
        self.ai_cooldown = 0         # Bộ đếm lùi (frames) để giữ AI chạy thêm
        self.is_sleeping = False     # Trạng thái hiện tại của hệ thống
        self.EDGE_THRESHOLD = 0.03  # Ngưỡng mật độ biên để xác định két mở
        
        self.last_transaction_end_time = 0 # Thời điểm kết thúc giao dịch gần nhất
        self.POST_TRANSACTION_COOLDOWN = 5.0 # Giây (Thời gian "nguội" sau khi đóng két)
    
    def _init_mediapipe(self):
        base_options = python.BaseOptions(model_asset_path=self.hand_model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(options)
        
    def reset(self):
        """Hàm reset trạng thái khi video chạy lại từ đầu"""
        # 1. Reset Logic Variables
        self.state = "IDLE"
        self.frame_count = 0
        self.drawer_buffer.clear()
        self.pos_enter_time = None
        self.is_pressing_pos = False
        self.ai_cooldown = 0
        
        # 2. Reset MediaPipe
        if hasattr(self, 'hand_detector'):
            self.hand_detector.close() # Đóng instance cũ
        self._init_mediapipe()         # Tạo instance mới
        
    def is_inside_roi(self, x, y, roi):
        x1, y1, x2, y2 = roi
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def calculate_edge_density(self, frame, roi):
        x1, y1, x2, y2 = roi
        h, w, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Cắt vùng ảnh ROI
        roi_img = frame[y1:y2, x1:x2]
        if roi_img.size == 0: return 0.0

        # 1. Chuyển sang ảnh xám
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        
        # 2. Làm mờ nhẹ để loại bỏ nhiễu (sàn nhà bẩn, hạt sạn)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. Dùng Canny để tìm biên
        # Ngưỡng 50, 150 là tiêu chuẩn, có thể chỉnh nếu cần
        edges = cv2.Canny(blurred, 50, 150)
        
        # 4. Tính mật độ: Số điểm trắng / Tổng số điểm ảnh
        density = np.count_nonzero(edges) / edges.size
        return density

    # --- HÀM KIỂM TRA CHUYỂN ĐỘNG (TẦNG 1) ---
    def _check_motion(self, frame_gray):
        """
        Kiểm tra xem có chuyển động đáng kể trong các vùng ROI hay không.
        """
        # 1. Trừ nền
        mask = self.bg_subtractor.apply(frame_gray)
        
        # 2. Lọc nhiễu (Morphology) - Loại bỏ hạt sạn, bụi
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 3. Kiểm tra chuyển động tại các vùng quan trọng
        has_motion = False
        for roi in [self.pos_roi, self.drawer_roi]:
            x1, y1, x2, y2 = roi
            # Cắt mask theo vùng
            # Đảm bảo tọa độ không vượt quá kích thước ảnh
            h, w = mask.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            roi_mask = mask[y1:y2, x1:x2]
            if roi_mask.size == 0: continue
            
            # Tính tỷ lệ điểm trắng (chuyển động)
            ratio = cv2.countNonZero(roi_mask) / roi_mask.size
            if ratio > self.MOTION_THRESHOLD:
                has_motion = True
                break
        return has_motion

    def classify_drawer(self, frame):
        x1, y1, x2, y2 = self.drawer_roi
        h, w, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0: return "CLOSED"

        # Resize chuẩn bị cho TFLite
        target_h, target_w = self.input_shape[1], self.input_shape[2]
        img = cv2.resize(roi, (target_w, target_h))
        input_data = (np.float32(img) / 127.5) - 1.0
        input_data = np.expand_dims(input_data, axis=0)

        # Chạy Inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Index 0 = OPEN, Index 1 = CLOSED
        open_score = output_data[0][0]
        
        # Kiểm tra mật độ biên để tăng độ chính xác
        edge_density = self.calculate_edge_density(frame, self.drawer_roi)

        if open_score > 0.5 or (open_score > 0.2 and edge_density > self.EDGE_THRESHOLD):
            is_open = True
        else:
            is_open = False

        # Logic Buffer để làm mượt kết quả
        self.drawer_buffer.append(is_open)
        # print(f'DEBUG BUFFER: {sum(self.drawer_buffer) >= (self.drawer_buffer.maxlen * 0.4)}')
        
        # Quyết định cuối cùng dựa trên buffer
        if sum(self.drawer_buffer) >= (self.drawer_buffer.maxlen * 0.4):
            return "OPEN"
        else:
            return "CLOSED"

    def update_pos_dwell_logic(self, hand_in_pos, current_time):
        """
        Logic kiểm tra thời gian lưu trú (Dwell Time)
        Trả về True nếu tay đã giữ đủ lâu ở vùng POS.
        """
        valid_click = False
        if hand_in_pos:
            if self.pos_enter_time is None:
                self.pos_enter_time = current_time  # Bắt đầu tính giờ

            # Tính thời gian đã giữ
            elapsed = current_time - self.pos_enter_time
            if elapsed >= self.POS_DWELL_THRESHOLD:
                self.is_pressing_pos = True
                valid_click = True  # Đã xác nhận bấm POS
        else:
            # Tay rời vùng POS -> Reset thời gian
            self.pos_enter_time = None
            self.is_pressing_pos = False
        return valid_click

    def update_fsm(self, drawer_status, hand_in_pos, hand_in_drawer):
        """
        LOGIC TUẦN TỰ (STRICT SEQUENTIAL FLOW):
        IDLE -> POS_INTERACTED -> DRAWER_OPENED -> MONEY_ACCESSED -> IDLE
        """
        event = None
        current_time = time.time()
        
        # 1. CẬP NHẬT LOGIC DWELL TIME TRƯỚC KHI BẤM POS
        in_cooldown = (current_time - self.last_transaction_end_time) < self.POST_TRANSACTION_COOLDOWN
        
        if in_cooldown:
            # Ép buộc không nhận POS dù tay có để đó
            is_valid_pos_action = False 
            # Reset luôn bộ đếm dwell time để tránh cộng dồn
            self.pos_enter_time = None 
            self.is_pressing_pos = False
        else:
            # Chạy logic bình thường
            is_valid_pos_action = self.update_pos_dwell_logic(hand_in_pos, current_time)

        # 1. CẬP NHẬT LOGIC DWELL TIME TRƯỚC
        # --- TRẠNG THÁI: IDLE (Chờ khách) ---
        if self.state == "IDLE":
            if is_valid_pos_action:
                self.state = "POS_INTERACTED"
                self.last_pos_time = current_time
                event = "1️⃣ STEP 1: Staff Inputting Order (Verified)"
                
            elif drawer_status == "OPEN":
                # Nếu két mở mà không bấm POS -> Ăn trộm
                self.state = "SUSPICIOUS"
                event = "🚨 ALARM: Drawer Opened without POS (Theft Detected!)"

        # --- TRẠNG THÁI: POS INTERACTED (Đã bấm máy, chờ mở két) ---
        elif self.state == "POS_INTERACTED":
            if is_valid_pos_action:
                self.last_pos_time = current_time   # Reset timeout
                
            if drawer_status == "OPEN":
                if current_time - self.last_pos_time <= self.pos_timeout:
                    self.state = "DRAWER_OPENED"
                    event = "2️⃣ STEP 2: Drawer Opened (Valid)"
                else:
                    self.state = "SUSPICIOUS"
                    event = "🚨 ALARM: Drawer Opened too late (Timeout)"
                    
            elif (current_time - self.last_pos_time) > self.pos_timeout:
                self.state = "IDLE"

        # --- TRẠNG THÁI: DRAWER OPENED (Két đã mở, chờ lấy tiền) ---
        elif self.state == "DRAWER_OPENED":
            if drawer_status == "CLOSED":
                self.close_confirm_counter += 1
                if self.close_confirm_counter > self.CLOSE_THRESHOLD:
                    self.state = "IDLE"
                    event = "✅ Transaction Ended (No money access detected)"
                    self.last_transaction_end_time = current_time
                    self.close_confirm_counter = 0
            else:
                self.close_confirm_counter = 0
                if hand_in_drawer:
                    self.state = "MONEY_ACCESSED"
                    event = "3️⃣ STEP 3: Money Access / Change Given"

        # --- TRẠNG THÁI: MONEY ACCESSED (Đang lấy tiền) ---
        elif self.state == "MONEY_ACCESSED":
            if drawer_status == "CLOSED":
                self.close_confirm_counter += 1
                if self.close_confirm_counter > self.CLOSE_THRESHOLD:
                    self.state = "IDLE"
                    event = "✅ STEP 4: Cycle Complete - Drawer Closed"
                    self.close_confirm_counter = 0
            else:
                self.close_confirm_counter = 0

        # --- TRẠNG THÁI: SUSPICIOUS (Cảnh báo) ---
        elif self.state == "SUSPICIOUS":
            if drawer_status == "CLOSED" and is_valid_pos_action:
                self.state = "POS_INTERACTED"
                self.last_pos_time = current_time
                event = "🔄 Info: System Reset - New Transaction"
            elif drawer_status == "CLOSED":
                self.state = "IDLE"

        return event

    def process_frame(self, frame, timestamp_ms):
        self.frame_count += 1
        
        # --- BƯỚC 1: MOTION GATING ---
        # Chuyển xám và làm mờ nhẹ để tối ưu tốc độ
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        
        has_motion = self._check_motion(blurred)
        
        # Logic quán tính cho AI: 
        # Nếu có chuyển động pixel -> Reset cooldown về 60 (2 giây)
        # Nếu không -> Giảm dần
        if has_motion:
            self.ai_cooldown = 60
        elif self.ai_cooldown > 0:
            self.ai_cooldown -= 1
            
        # QUYẾT ĐỊNH: Nếu Hết quán tính + Đang rảnh (IDLE) + Drawer CLOSED -> NGỦ ĐÔNG
        is_drawer_safe_to_sleep = (self.last_drawer_status == "CLOSED")
        if self.ai_cooldown == 0 and self.state == "IDLE" and is_drawer_safe_to_sleep:
            self.is_sleeping = True
            # TRẢ VỀ None: Báo hiệu cho app.py biết là AI đang ngủ
            return None, None, self.last_drawer_status

        # --- NẾU CÓ CHUYỂN ĐỘNG PIXEL: CHẠY TIẾP LOGIC AI ---
        self.is_sleeping = False
        
        # 1. Check Drawer (TFLite): Nếu vừa bấm POS trong vòng 5 giây, thì check két LIÊN TỤC (skip=1)
        # Nếu đang rảnh (IDLE), thì check thưa ra (skip=2) để đỡ nóng máy
        is_urgent = (time.time() - self.last_pos_time < 5.0) and (self.state == "POS_INTERACTED")
        if is_urgent or (self.frame_count % 2 == 0):
            drawer_status = self.classify_drawer(frame)
            self.last_drawer_status = drawer_status
        else:
            drawer_status = self.last_drawer_status

        # 2. Check Hand (MediaPipe)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = self.hand_detector.detect_for_video(mp_image, timestamp_ms)
        
        hand_in_pos = False
        hand_in_drawer = False
        h, w, _ = frame.shape

        if detection_result.hand_landmarks:
            for landmarks in detection_result.hand_landmarks:
                # Danh sách các điểm quan trọng: Cổ tay, Ngón cái, Ngón trỏ, Ngón giữa
                important_points = [landmarks[0], landmarks[4], landmarks[8], landmarks[12]]
                
                # Check vùng POS
                if any(self.is_inside_roi(pt.x * w, pt.y * h, self.pos_roi) for pt in important_points):
                    hand_in_pos = True
                    
                # Check vùng DRAWER
                if any(self.is_inside_roi(pt.x * w, pt.y * h, self.drawer_roi) for pt in important_points):
                    hand_in_drawer = True
                    
                if hand_in_pos or hand_in_drawer: break
        
        # 3. Cập nhật FSM Logic
        event = self.update_fsm(drawer_status, hand_in_pos, hand_in_drawer)
        
        return detection_result, event, drawer_status