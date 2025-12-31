import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

class FraudDetector:
    def __init__(self, model_path, drawer_roi, pos_roi):
        # drawer_roi format: [x1, y1, x2, y2]
        self.drawer_roi = drawer_roi
        self.pos_roi = pos_roi
        
        # Các trạng thái: IDLE, POS_TOUCHED, DRAWER_ACCESSED, SUSPICIOUS
        self.state = "IDLE" 
        self.last_event = None
        self.last_pos_time = 0
        self.pos_timeout = 10.0 # Thời gian cho phép từ lúc bấm POS đến lúc mở két (giây)
        
        # Cấu hình MediaPipe Task
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO, # Tối ưu cho luồng video, chọn LIVE_STREAM nếu cần xử lý thời gian thực
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def is_inside_roi(self, x, y, roi):
        x1, y1, x2, y2 = roi
        return x1 <= x <= x2 and y1 <= y <= y2

    def update_fsm(self, hand_in_drawer, hand_in_pos):
        """
        Finite State Machine
        Quy trình chuẩn: 
        1. IDLE: Chờ đợi.
        2. POS_TOUCHED: Nhân viên thao tác POS (Order/Nhập tiền).
        3. DRAWER_ACCESSED: Mở két sau khi đã thao tác POS (Hợp lệ).
        4. IDLE: Đóng két (Kết thúc chu trình).
        
        Nghi vấn:
        - Mở két khi chưa chạm POS (Hoặc quá thời gian chờ).
        """
        event = None
        current_time = time.time()
        
        # 1. Trạng thái IDLE (Chờ khách/đơn hàng mới)
        if self.state == "IDLE":
            if hand_in_pos:
                self.state = "POS_TOUCHED"
                self.last_pos_time = current_time
                event = "STEP 1: Staff Inputting Order on POS"
            elif hand_in_drawer:
                # CẢNH BÁO: Vào két mà chưa qua POS
                self.state = "SUSPICIOUS"
                event = "ALARM: Suspicious! Drawer opened without POS entry."

        # 2. Trạng thái ĐÃ CHẠM POS (Đang chờ mở két)
        elif self.state == "POS_TOUCHED":
            # Nếu tiếp tục chạm POS, cập nhật lại thời gian (đang nhập liệu liên tục)
            if hand_in_pos:
                self.last_pos_time = current_time
            
            # Nếu chạm vào két
            elif hand_in_drawer:
                # Kiểm tra xem có quá lâu từ lúc bỏ tay khỏi POS không
                if current_time - self.last_pos_time <= self.pos_timeout:
                    self.state = "DRAWER_ACCESSED"
                    event = "STEP 2: Valid Transaction - Drawer Opened"
                else:
                    self.state = "SUSPICIOUS"
                    event = "ALARM: Suspicious! POS sequence timed out before Drawer access."
            
            # Nếu không làm gì quá lâu thì reset về IDLE (khách bỏ đi hoặc đã xong order nhưng không thanh toán tiền mặt)
            elif (current_time - self.last_pos_time > self.pos_timeout):
                self.state = "IDLE"
                event = "INFO: Transaction reset (Timeout)"

        # 3. Trạng thái ĐANG MỞ KÉT (Lấy tiền/trả tiền thừa)
        elif self.state == "DRAWER_ACCESSED":
            if not hand_in_drawer:
                # Tay đã rút ra -> Giả định đóng két -> Kết thúc chu trình
                self.state = "IDLE"
                event = "STEP 3: Transaction Complete - Drawer Closed"

        # 4. Trạng thái NGHI VẤN (Cần reset thủ công hoặc tự động sau khi rút tay)
        elif self.state == "SUSPICIOUS":
            if not hand_in_drawer and not hand_in_pos:
                self.state = "IDLE" # Reset để bắt đầu chu trình mới
                event = "INFO: Alarm cleared - Monitoring continues"

        if event:
            self.last_event = {"event": event, "timestamp": current_time}
        return event

    def process_frame(self, frame, frame_timestamp_ms):
        # Chuyển đổi định dạng cho MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # Nhận diện landmark
        detection_result = self.detector.detect_for_video(mp_image, frame_timestamp_ms)
        
        hand_in_drawer = False
        hand_in_pos = False
        
        if detection_result.hand_landmarks:
            for landmarks in detection_result.hand_landmarks:
                # WRIST là landmark đầu tiên (index 0)
                wrist = landmarks[0]
                h, w, _ = frame.shape
                cx, cy = int(wrist.x * w), int(wrist.y * h)
                
                if self.is_inside_roi(cx, cy, self.drawer_roi):
                    hand_in_drawer = True
                elif self.is_inside_roi(cx, cy, self.pos_roi):
                    hand_in_pos = True

        event = self.update_fsm(hand_in_drawer, hand_in_pos)
        return detection_result, event