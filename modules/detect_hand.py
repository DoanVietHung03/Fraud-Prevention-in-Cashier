import os
# --- SETUP MÔI TRƯỜNG ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'

import warnings
import logging

warnings.filterwarnings('ignore')
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
        
        # Load Model TFLite (Két tiền)
        self.interpreter = tflite.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']

        # Load Model MediaPipe (Tay)
        base_options = python.BaseOptions(model_asset_path=hand_model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(options)

        # 2. Logic Variables
        self.state = "IDLE"
        self.last_pos_time = 0
        self.pos_timeout = 30.0
        self.drawer_buffer = deque(maxlen=5) 
        self.frame_count = 0 
        self.last_drawer_status = "CLOSED"
        self.close_confirm_counter = 0 
        self.CLOSE_THRESHOLD = 50 

        self.pos_enter_time = None     
        self.POS_DWELL_THRESHOLD = 3.0   
        self.is_pressing_pos = False     

        # 3. Motion & Thresholds
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=20, detectShadows=False
        )
        self.MOTION_THRESHOLD = 0.05 
        self.ai_cooldown = 0         
        self.is_sleeping = False     
        self.EDGE_THRESHOLD = 0.05
        self.COLOR_THRESHOLD = 0.3
        
        self.DRAWER_OPEN_MAX_TIME = 20.0 
        self.drawer_open_start_time = 0  
        self.last_transaction_end_time = 0 
        self.POST_TRANSACTION_COOLDOWN = 4.0

        # --- TỐI ƯU: ĐỊNH NGHĨA HẰNG SỐ MÀU SẮC 1 LẦN ---
        self.lower_red1 = np.array([0, 50, 50])
        self.upper_red1 = np.array([8, 255, 255])
        self.lower_red2 = np.array([160, 60, 50])
        self.upper_red2 = np.array([180, 255, 255])
        
        # MediaPipe Timestamp
        self.mp_internal_timestamp_ms = 0

    def reset(self):
        """Reset trạng thái logic, KHÔNG reload model để tránh lag"""
        self.state = "IDLE"
        self.frame_count = 0
        self.drawer_buffer.clear()
        self.pos_enter_time = None
        self.is_pressing_pos = False
        self.ai_cooldown = 0
        self.last_transaction_end_time = 0
        self.drawer_open_start_time = 0
        self.close_confirm_counter = 0
        
    def is_inside_roi(self, x, y, roi):
        x1, y1, x2, y2 = roi
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def calculate_edge_density(self, roi_img):
        # roi_img đã được cắt sẵn
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        density = np.count_nonzero(edges) / edges.size
        return density
    
    def calculate_color_HSV(self, roi_img):
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask_pink = mask1 | mask2 
        pink_ratio = cv2.countNonZero(mask_pink) / mask_pink.size
        return pink_ratio

    def _check_motion(self, frame_gray):
        mask = self.bg_subtractor.apply(frame_gray)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        for roi in [self.pos_roi, self.drawer_roi]:
            x1, y1, x2, y2 = roi
            h, w = mask.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            roi_mask = mask[y1:y2, x1:x2]
            if roi_mask.size == 0: continue
            
            if (cv2.countNonZero(roi_mask) / roi_mask.size) > self.MOTION_THRESHOLD:
                return True
        return False

    def classify_drawer(self, frame):
        """
        Logic phân loại két tiền:
        TFLite -> (Nếu mơ hồ) -> Check Logic ảnh (Edge + Color)
        """
        x1, y1, x2, y2 = self.drawer_roi
        h, w, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        roi_img = frame[y1:y2, x1:x2]
        if roi_img.size == 0: return "CLOSED"

        # 1. Inference TFLite
        target_h, target_w = self.input_shape[1], self.input_shape[2]
        img_resized = cv2.resize(roi_img, (target_w, target_h))
        input_data = (np.float32(img_resized) / 127.5) - 1.0
        input_data = np.expand_dims(input_data, axis=0)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # open_score: Xác suất mở
        open_score = output_data[0][0]
        
        is_open = False

        # --- OPTIMIZATION: SHORT-CIRCUIT EVALUATION ---
        # Chỉ chạy thuật toán xử lý ảnh nặng nếu TFLite không chắc chắn
        # Ngưỡng dưới 0.1: Chắc chắn Đóng. Ngưỡng trên 0.6: Chắc chắn Mở.
        # Khoảng 0.1 - 0.6: Cần kiểm tra thêm bằng Edge/Color
        
        if open_score > 0.6:
            is_open = True
        elif open_score < 0.1:
            is_open = False
        else:
            # Vùng không chắc chắn -> Dùng Logic Ảnh cổ điển để verify
            edge_density = self.calculate_edge_density(roi_img)
            pink_ratio = self.calculate_color_HSV(roi_img)
            
            if edge_density > self.EDGE_THRESHOLD and pink_ratio > self.COLOR_THRESHOLD:
                is_open = True
            else:
                is_open = False

        # Logic Buffer để làm mượt kết quả (chống nhấp nháy)
        self.drawer_buffer.append(is_open)
        if sum(self.drawer_buffer) >= (self.drawer_buffer.maxlen * 0.4): # Giảm ngưỡng 1 chút
            return "OPEN"
        else:
            return "CLOSED"

    def update_pos_dwell_logic(self, hand_in_pos, current_time):
        valid_click = False
        if hand_in_pos:
            if self.pos_enter_time is None:
                self.pos_enter_time = current_time 
            elapsed = current_time - self.pos_enter_time
            if elapsed >= self.POS_DWELL_THRESHOLD:
                self.is_pressing_pos = True
                valid_click = True
        else:
            self.pos_enter_time = None
            self.is_pressing_pos = False
        return valid_click

    def update_fsm(self, drawer_status, hand_in_pos, hand_in_drawer):
        event = None
        current_time = time.time()
        
        # Logic Cooldown sau khi giao dịch
        in_cooldown = (current_time - self.last_transaction_end_time) < self.POST_TRANSACTION_COOLDOWN
        
        if in_cooldown:
            is_valid_pos_action = False 
            self.pos_enter_time = None 
            self.is_pressing_pos = False
        else:
            is_valid_pos_action = self.update_pos_dwell_logic(hand_in_pos, current_time)

        # --- STATE MACHINE ---
        if self.state == "IDLE":
            if is_valid_pos_action:
                time_since_last_txn = current_time - self.last_transaction_end_time
                if time_since_last_txn < 10.0 and self.last_transaction_end_time > 0:
                    event = "⚠️ STEP 1: Staff Inputting Order (Fast Repetition)"
                else:
                    event = "1️⃣ STEP 1: Staff Inputting Order (Verified)"
                     
                self.state = "POS_INTERACTED"
                self.last_pos_time = current_time
                
            elif drawer_status == "OPEN":
                self.state = "SUSPICIOUS"
                event = "🚨 ALARM: Drawer Opened without POS (Theft Detected!)"

        elif self.state == "POS_INTERACTED":
            if is_valid_pos_action:
                self.last_pos_time = current_time   
                
            if drawer_status == "OPEN":
                if current_time - self.last_pos_time <= self.pos_timeout:
                    self.state = "DRAWER_OPENED"
                    event = "2️⃣ STEP 2: Drawer Opened (Valid)"
                    self.drawer_open_start_time = current_time
                else:
                    self.state = "SUSPICIOUS"
                    event = "🚨 ALARM: Drawer Opened too late (Timeout)"
            elif (current_time - self.last_pos_time) > self.pos_timeout:
                self.state = "IDLE"

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
                if (current_time - self.drawer_open_start_time) > self.DRAWER_OPEN_MAX_TIME:
                    self.state = "SUSPICIOUS"
                    event = "🚨 ALARM: Drawer Left Open TOO LONG (> 60s)!"
                elif hand_in_drawer:
                    self.state = "MONEY_ACCESSED"
                    event = "3️⃣ STEP 3: Money Access / Change Given"

        elif self.state == "MONEY_ACCESSED":
            if drawer_status == "CLOSED":
                self.close_confirm_counter += 1
                if self.close_confirm_counter > self.CLOSE_THRESHOLD:
                    self.state = "IDLE"
                    event = "✅ STEP 4: Cycle Complete - Drawer Closed"
                    self.last_transaction_end_time = current_time
                    self.close_confirm_counter = 0
            else:
                self.close_confirm_counter = 0
                if (current_time - self.drawer_open_start_time) > self.DRAWER_OPEN_MAX_TIME:
                    self.state = "SUSPICIOUS"
                    event = "🚨 ALARM: Drawer Left Open TOO LONG (> 60s)!"

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
        
        # --- BƯỚC 1: MOTION GATING (Tầng lọc thô) ---
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Chỉ check motion mỗi 2 frame để giảm tải CPU
        if self.frame_count % 2 == 0:
            has_motion = self._check_motion(gray)
            if has_motion:
                self.ai_cooldown = 120 # Giữ AI thức trong 2s (60fps*2) hoặc 4s (30fps)
            elif self.ai_cooldown > 0:
                self.ai_cooldown -= 1
        else:
            # Frame lẻ thì giữ nguyên trạng thái cũ, chỉ giảm cooldown
            if self.ai_cooldown > 0: self.ai_cooldown -= 1
            
        # QUYẾT ĐỊNH NGỦ ĐÔNG
        is_drawer_safe_to_sleep = (self.last_drawer_status == "CLOSED")
        if self.ai_cooldown == 0 and self.state == "IDLE" and is_drawer_safe_to_sleep:
            self.is_sleeping = True
            return None, None, self.last_drawer_status

        self.is_sleeping = False
        
        # --- BƯỚC 2: AI LOGIC ---
        
        # 2.1 Check Drawer
        # Nếu đang có giao dịch (POS_INTERACTED) thì check liên tục (độ ưu tiên cao)
        # Nếu đang rảnh, check thưa hơn (mỗi 3 frame) để tiết kiệm
        check_drawer_now = True
        if self.state == "IDLE":
             if self.frame_count % 3 != 0: check_drawer_now = False
        
        if check_drawer_now:
            drawer_status = self.classify_drawer(frame)
            self.last_drawer_status = drawer_status
        else:
            drawer_status = self.last_drawer_status

        # 2.2 Check Hand (MediaPipe)
        # Convert MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        current_sys_time_ms = int(time.time() * 1000)
        # Đảm bảo timestamp luôn lớn hơn lần trước ít nhất 1ms (phòng trường hợp máy chạy quá nhanh)
        if current_sys_time_ms <= self.mp_internal_timestamp_ms:
            current_sys_time_ms = self.mp_internal_timestamp_ms + 1
            
        self.mp_internal_timestamp_ms = current_sys_time_ms
        
        detection_result = self.hand_detector.detect_for_video(mp_image, self.mp_internal_timestamp_ms)
        
        hand_in_pos = False
        hand_in_drawer = False
        h, w, _ = frame.shape

        if detection_result.hand_landmarks:
            for landmarks in detection_result.hand_landmarks:
                # Chỉ lấy đầu ngón trỏ (8) và cổ tay (0) để check nhanh
                # (Bỏ bớt các điểm khác để loop nhanh hơn)
                points_to_check = [landmarks[8], landmarks[0]]
                
                # Check POS
                if not hand_in_pos: # Nếu đã True rồi thì thôi không check nữa
                    if any(self.is_inside_roi(pt.x * w, pt.y * h, self.pos_roi) for pt in points_to_check):
                        hand_in_pos = True
                
                # Check Drawer
                if not hand_in_drawer:
                    if any(self.is_inside_roi(pt.x * w, pt.y * h, self.drawer_roi) for pt in points_to_check):
                        hand_in_drawer = True
        
        # 3. Cập nhật FSM Logic
        event = self.update_fsm(drawer_status, hand_in_pos, hand_in_drawer)
        
        return detection_result, event, drawer_status