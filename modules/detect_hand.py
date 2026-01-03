import cv2
import os
import warnings
import numpy as np
import time
import tensorflow.lite as tflite
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Tắt cảnh báo TensorFlow oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Tắt các cảnh báo Deprecation
warnings.filterwarnings('ignore')

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
        self.pos_timeout = 30.0 # Thời gian chờ từ lúc bấm POS đến lúc mở két
        self.drawer_buffer = deque(maxlen=5) # Bộ đệm chống nhiễu cho két

        self.frame_count = 0 
        self.last_drawer_status = "CLOSED"

        # Biến đếm số frame xác nhận đóng két
        self.close_confirm_counter = 0 
        self.CLOSE_THRESHOLD = 30

    def is_inside_roi(self, x, y, roi):
        x1, y1, x2, y2 = roi
        return x1 <= x <= x2 and y1 <= y <= y2

    def classify_drawer(self, frame):
        x1, y1, x2, y2 = self.drawer_roi
        h, w, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0: return "CLOSED"

        target_h, target_w = self.input_shape[1], self.input_shape[2]
        img = cv2.resize(roi, (target_w, target_h))
        input_data = (np.float32(img) / 127.5) - 1.0
        input_data = np.expand_dims(input_data, axis=0)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # LƯU Ý: Kiểm tra file labels.txt. 
        # Index 0 = OPEN, Index 1 = CLOSED
        open_score = output_data[0][0]
        closed_score = output_data[0][1]
        
        is_open = open_score > closed_score
        self.drawer_buffer.append(is_open)
        
        if sum(self.drawer_buffer) >= (self.drawer_buffer.maxlen * 0.8):
            return "OPEN"
        else:
            return "CLOSED"

    def update_fsm(self, drawer_status, hand_in_pos, hand_in_drawer):
        """
        LOGIC TUẦN TỰ (STRICT SEQUENTIAL FLOW):
        IDLE -> POS_INTERACTED -> DRAWER_OPENED -> MONEY_ACCESSED -> IDLE
        """
        event = None
        current_time = time.time()
        
        # --- LOGIC CHUYỂN TRẠNG THÁI ---

        # 1. TRẠNG THÁI: IDLE (Chờ khách)
        if self.state == "IDLE":
            if hand_in_pos:
                self.state = "POS_INTERACTED"
                self.last_pos_time = current_time
                event = "1️⃣ STEP 1: Staff Inputting Order"
            elif drawer_status == "OPEN":
                # Két mở bất ngờ mà không qua bước 1
                self.state = "SUSPICIOUS"
                event = "🚨 ALARM: Drawer Opened without POS!"

        # 2. TRẠNG THÁI: POS_INTERACTED (Đã bấm máy, chờ mở két)
        elif self.state == "POS_INTERACTED":
            if hand_in_pos:
                self.last_pos_time = current_time # Reset timeout nếu vẫn đang bấm
            
            # Kiểm tra xem két có mở không
            if drawer_status == "OPEN":
                # Kiểm tra thời gian từ lần cuối bấm POS
                if current_time - self.last_pos_time <= self.pos_timeout:
                    self.state = "DRAWER_OPENED"
                    event = "2️⃣ STEP 2: Drawer Opened (Valid)"
                else:
                    self.state = "SUSPICIOUS"
                    event = "🚨 ALARM: Drawer Opened too late (Timeout)"
            
            # Reset nếu chờ quá lâu mà không mở két (Khách hủy đơn)
            elif (current_time - self.last_pos_time) > self.pos_timeout:
                self.state = "IDLE"
                # event = "Info: Transaction Reset"

        # 3. TRẠNG THÁI: DRAWER_OPENED (Két đã mở, chờ lấy tiền)
        elif self.state == "DRAWER_OPENED":
            if drawer_status == "CLOSED":
                self.close_confirm_counter += 1
                if self.close_confirm_counter > self.CLOSE_THRESHOLD:
                    # Két đóng mà chưa thấy tay thò vào -> Có thể chỉ mở ra nhìn?
                    # Vẫn tính là xong chu trình nhưng có thể warning nhẹ
                    self.state = "IDLE"
                    event = "✅ Transaction Ended (No money access detected)"
                    self.close_confirm_counter = 0
            else:
                # Nếu bỗng dưng thấy OPEN lại (do lúc nãy chỉ bị che) -> Reset đếm về 0
                self.close_confirm_counter = 0
                if hand_in_drawer:
                    # Phát hiện tay trong vùng két -> Đúng quy trình lấy tiền
                    self.state = "MONEY_ACCESSED"
                    event = "3️⃣ STEP 3: Money Access / Change Given"

        # 4. TRẠNG THÁI: MONEY_ACCESSED (Đang lấy tiền)
        elif self.state == "MONEY_ACCESSED":
            if drawer_status == "CLOSED":
                self.close_confirm_counter += 1
                if self.close_confirm_counter > self.CLOSE_THRESHOLD:
                    # Đóng két -> Hoàn thành chu trình
                    self.state = "IDLE"
                    event = "✅ STEP 4: Cycle Complete - Drawer Closed"
                    self.close_confirm_counter = 0
            
            else:
                # Két vẫn mở hoặc AI detect lại được OPEN -> Reset đếm
                self.close_confirm_counter = 0

        # 5. TRẠNG THÁI: SUSPICIOUS (Cảnh báo)
        elif self.state == "SUSPICIOUS":
            # Thoát cảnh báo nếu làm lại từ đầu đúng quy trình
            if drawer_status == "CLOSED" and hand_in_pos:
                self.state = "POS_INTERACTED"
                self.last_pos_time = current_time
                event = "🔄 Info: System Reset - New Transaction"
            elif drawer_status == "CLOSED":
                # Tự động reset khi đóng két
                self.state = "IDLE"

        return event

    def process_frame(self, frame, timestamp_ms):
        self.frame_count += 1
        
        # 1. Kiểm tra: Nếu vừa bấm POS trong vòng 5 giây, thì check két LIÊN TỤC (skip=1)
        # Nếu đang rảnh (IDLE), thì check thưa ra (skip=5) để đỡ nóng máy
        is_urgent = (time.time() - self.last_pos_time < 5.0) and (self.state == "POS_INTERACTED")

        if is_urgent or (self.frame_count % 5 == 0):
            drawer_status = self.classify_drawer(frame)
            self.last_drawer_status = drawer_status
        else:
            drawer_status = self.last_drawer_status

        # 2. AI Nhận thức (Perception)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = self.hand_detector.detect_for_video(mp_image, timestamp_ms)
        
        hand_in_pos = False
        hand_in_drawer = False
        
        if detection_result.hand_landmarks:
            for landmarks in detection_result.hand_landmarks:
                wrist = landmarks[0]
                index_finger = landmarks[8]
                h, w, _ = frame.shape
                cx, cy = int(((wrist.x + index_finger.x) / 2) * w), int(((wrist.y + index_finger.y) / 2) * h)
                
                # Check vùng POS
                if self.is_inside_roi(cx, cy, self.pos_roi):
                    hand_in_pos = True
                
                # Check vùng Drawer (Chỉ có ý nghĩa khi logic cần, nhưng cứ detect trước)
                if self.is_inside_roi(cx, cy, self.drawer_roi):
                    hand_in_drawer = True
        
        # 3. Máy Trạng Thái (Logic)
        event = self.update_fsm(drawer_status, hand_in_pos, hand_in_drawer)
        
        return detection_result, event, drawer_status