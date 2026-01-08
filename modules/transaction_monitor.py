import time
import threading
from collections import deque

from .utils import PerformanceLogger
class TransactionMonitor:
    def __init__(self, match_window=15):
        """
        Quản lý logic đối chiếu giữa hành động Vật lý (Camera) và Kỹ thuật số (POS).
        
        Args:
            match_window (int): Thời gian tối đa (giây) cho phép sự chênh lệch giữa
                                hành động mở két và thao tác POS.
        """
        self.match_window = match_window
        
        # --- BỘ NHỚ ĐỆM (BUFFERS) ---
        # 1. Danh sách các lần mở két chưa được giải trình (Đợi POS Log)
        # Cấu trúc: [{'time': timestamp, 'status': 'PENDING'}]
        self.pending_opens = [] 
        
        # 2. Danh sách các lệnh POS đã bấm nhưng chưa thấy két mở (Đợi Camera)
        # Cấu trúc: [{'time': timestamp, 'action': 'PAY', 'amount': 50000}]
        self.recent_pos_logs = []
        
        # --- THREAD SAFETY ---
        # Khóa để đảm bảo an toàn khi 2 luồng (Video & UI) cùng truy cập
        self.lock = threading.Lock()
        
        self.logger = PerformanceLogger()

    def add_physical_event(self, event_type):
        """
        [ĐƯỢC GỌI TỪ THREAD XỬ LÝ VIDEO]
        Nhận sự kiện từ Camera (FraudDetector) khi phát hiện két mở.
        """
        if event_type != "DRAWER_OPENED":
            return

        current_time = time.time()
        
        with self.lock: # <--- Bắt đầu khóa an toàn
            print(f"👁️ [VISION] Két mở lúc {time.strftime('%H:%M:%S')} -> Kiểm tra đối chiếu...")
            
            # --- LOGIC MỚI: CHECK NGƯỢC (POS TRƯỚC - KÉT SAU) ---
            # Kiểm tra xem có lệnh PAY nào đang chờ sẵn trong recent_pos_logs không?
            matched_pos_idx = -1
            
            # Duyệt ngược từ mới nhất về cũ nhất
            for i in range(len(self.recent_pos_logs) - 1, -1, -1):
                pos_log = self.recent_pos_logs[i]
                time_diff = current_time - pos_log['time']
                
                # Nếu lệnh POS nằm trong khoảng thời gian cho phép (ví dụ: bấm trước đó 2 giây)
                if 0 <= time_diff <= self.match_window:
                    matched_pos_idx = i
                    break
            
            if matched_pos_idx != -1:
                # ==> TÌM THẤY KHỚP! (Nhân viên bấm Bill rồi két mới bung ra)
                print(f"✅ MATCHED: Khớp với lệnh POS trước đó (Độ trễ: {current_time - self.recent_pos_logs[matched_pos_idx]['time']:.2f}s)")
                
                self.logger.log(0, f"✅ MATCHED: Khớp với lệnh POS trước đó (Độ trễ: {current_time - self.recent_pos_logs[matched_pos_idx]['time']:.2f}s)")
                # Xóa log POS đã dùng để tránh dùng lại cho lần sau
                self.recent_pos_logs.pop(matched_pos_idx)
                
                # Không cần thêm vào pending_opens nữa vì đã hợp lệ ngay lập tức
                return 

            # --- LOGIC CŨ: KÉT TRƯỚC - POS SAU ---
            # Nếu không tìm thấy POS log nào chờ sẵn, thêm vào danh sách Pending
            self.pending_opens.append({'time': current_time, 'status': 'PENDING'})
            self.logger.log(0, "⏳ Waiting for POS: Két đã mở, đang chờ tín hiệu POS...")
            print(f"⏳ Đang chờ tín hiệu từ POS...")

    def add_pos_log(self, action, amount=0):
        """
        [ĐƯỢC GỌI TỪ THREAD UI STREAMLIT]
        Nhận sự kiện từ giả lập POS (PAY hoặc VOID).
        Trả về chuỗi cảnh báo nếu phát hiện gian lận ngay lập tức.
        """
        current_time = time.time()
        
        self.logger.log(0, f"📠 POS INPUT: {action} - Số tiền: {amount}")
        
        with self.lock: # <--- Bắt đầu khóa an toàn
            print(f"📠 [POS] Nhận tín hiệu: {action} ({amount}đ)")
            
            # 1. TÌM KÉT ĐANG CHỜ (Két mở trước -> Giờ mới bấm POS)
            matched_idx = -1
            for i in range(len(self.pending_opens) - 1, -1, -1):
                if self.pending_opens[i]['status'] == 'PENDING':
                    time_diff = current_time - self.pending_opens[i]['time']
                    if 0 <= time_diff <= self.match_window:
                        matched_idx = i
                        break
            
            # --- XỬ LÝ NGHIỆP VỤ ---
            
            # CASE A: HỦY BILL (VOID/CANCEL)
            if action in ["VOID", "CANCEL", "DELETE"]:
                if matched_idx != -1:
                    # Tìm thấy két mở trước đó -> GIAN LẬN: Lấy tiền rồi hủy bill
                    self.pending_opens[matched_idx]['status'] = 'FRAUD_VOID'
                    # Xóa khỏi danh sách chờ để tránh check timeout lại
                    wait_time = int(current_time - self.pending_opens[matched_idx]['time'])
                    self.pending_opens.pop(matched_idx) 
                    self.logger.log(0, f"🚨 GIAN LẬN: Hủy bill sau khi đã mở két! (Cách nhau {wait_time}s)")
                    return f"🚨 GIAN LẬN: Hủy bill sau khi đã mở két! (Cách nhau {wait_time}s)"
                else:
                    self.logger.log(0, "ℹ️ POS: Hủy đơn bình thường (Két chưa mở).")
                    return "ℹ️ POS: Hủy đơn bình thường (Két chưa mở)."

            # CASE B: THANH TOÁN (PAYMENT)
            elif action == "PAY":
                if matched_idx != -1:
                    # Tìm thấy két mở -> HỢP LỆ
                    self.pending_opens[matched_idx]['status'] = 'MATCHED_OK'
                    self.pending_opens.pop(matched_idx) # Xóa sự kiện đã xử lý xong
                    self.logger.log(0, "✅ Giao dịch hợp lệ (Verified Transaction)")
                    return "✅ Giao dịch hợp lệ (Verified Transaction)"
                else:
                    # Không thấy két mở trước đó. Có thể là:
                    # 1. Két sắp mở (đang trễ tín hiệu video) -> Lưu vào buffer đợi
                    # 2. Thanh toán thẻ (không mở két)
                    
                    # Lưu vào buffer chờ két mở (Logic đồng bộ 2 chiều)
                    self.recent_pos_logs.append({'time': current_time, 'action': action})
                    
                    # Dọn dẹp buffer nếu quá đầy (chỉ giữ log trong 2 lần match_window)
                    self._cleanup_pos_logs(current_time)
                    
                    return "⏳ Đã ghi nhận thanh toán, chờ mở két..."
            
            return None

    def check_timeouts(self):
        """
        [ĐƯỢC GỌI TỪ THREAD XỬ LÝ VIDEO - MỖI FRAME]
        Kiểm tra các sự kiện mở két bị 'treo' quá lâu mà không có POS log.
        """
        current_time = time.time()
        alert_msg = None
        
        with self.lock: # <--- Bắt đầu khóa an toàn
            # Duyệt bản sao của list [:] để an toàn khi remove phần tử trong vòng lặp
            for event in self.pending_opens[:]:
                if event['status'] == 'PENDING':
                    elapsed = current_time - event['time']
                    
                    # Nếu quá thời gian cho phép mà chưa thấy POS Payment
                    if elapsed > self.match_window:
                        event['status'] = 'THEFT_TIMEOUT'
                        alert_msg = f"🚨 BÁO ĐỘNG: Mở két {int(elapsed)}s mà KHÔNG nhập POS! (Nghi vấn trộm tiền)"
                        
                        self.logger.log(0, alert_msg)
                        
                        # Xóa sự kiện này khỏi danh sách pending
                        self.pending_opens.remove(event)
                        
                        # Chỉ báo động 1 lần cho sự kiện này rồi thoát
                        # (Hàm này gọi mỗi frame nên return ngay để tránh spam log)
                        return alert_msg
        
        return None

    def _cleanup_pos_logs(self, current_time):
        """Hàm nội bộ: Dọn dẹp các log POS cũ quá hạn để tránh đầy bộ nhớ"""
        # Giữ lại các log còn mới (trong vòng 2 lần match_window)
        # Ví dụ: match_window = 10s, thì giữ log trong 20s.
        threshold = self.match_window * 2
        self.recent_pos_logs = [
            log for log in self.recent_pos_logs 
            if (current_time - log['time']) < threshold
        ]