import time
from collections import deque

class TransactionMonitor:
    def __init__(self, match_window=10):
        """
        match_window: (Giây) Thời gian tối đa cho phép giữa hành động mở két và thao tác POS.
        """
        self.match_window = match_window
        
        # Danh sách các lần mở két chưa được giải trình (Unmatched Physical Events)
        # Cấu trúc: [{'time': 170000.0, 'status': 'PENDING'}]
        self.pending_opens = [] 
        
        # Lưu log cảnh báo để UI hiển thị
        self.alert_log = None 

    def add_physical_event(self, event_type):
        """Nhận sự kiện từ Camera (FraudDetector)"""
        if event_type == "DRAWER_OPENED":
            current_time = time.time()
            # Thêm vào danh sách chờ đối chiếu
            self.pending_opens.append({'time': current_time, 'status': 'PENDING'})
            print(f"👁️ [VISION] Két mở lúc {time.strftime('%H:%M:%S', time.localtime(current_time))} -> Đang chờ POS giải trình...")

    def add_pos_log(self, action, amount=0):
        """
        Nhận sự kiện từ POS (PAY hoặc VOID).
        Trả về chuỗi cảnh báo nếu phát hiện gian lận.
        """
        current_time = time.time()
        print(f"📠 [POS] Nhận tín hiệu: {action} - {amount}đ")
        
        # Tìm xem có lần mở két nào gần đây (trong match_window) chưa được xử lý không?
        # Lấy sự kiện mở két gần nhất (Last In First Out logic thường dùng cho giao dịch liền kề)
        matched_idx = -1
        for i in range(len(self.pending_opens) - 1, -1, -1):
            if self.pending_opens[i]['status'] == 'PENDING':
                time_diff = current_time - self.pending_opens[i]['time']
                if 0 <= time_diff <= self.match_window:
                    matched_idx = i
                    break
        
        # --- LOGIC XỬ LÝ ---
        
        # CASE 1: HỦY BILL (VOID)
        if action in ["VOID", "CANCEL", "DELETE"]:
            if matched_idx != -1:
                # Tìm thấy két mở trước đó -> Gian lận: Lấy tiền rồi hủy bill
                self.pending_opens[matched_idx]['status'] = 'FRAUD_VOID'
                self.alert_log = f"🚨 GIAN LẬN: Hủy bill sau khi đã mở két! (Cách nhau {int(current_time - self.pending_opens[matched_idx]['time'])}s)"
                return self.alert_log
            else:
                return "ℹ️ POS: Hủy đơn bình thường (Két chưa mở)."

        # CASE 2: THANH TOÁN (PAYMENT)
        elif action == "PAY":
            if matched_idx != -1:
                # Tìm thấy két mở -> Hợp lệ: Mở két để thối tiền/nhận tiền
                self.pending_opens[matched_idx]['status'] = 'MATCHED_OK'
                # Xóa khỏi danh sách chờ (hoặc đánh dấu đã xử lý để không check lại)
                # Ở đây ta xóa luôn cho nhẹ bộ nhớ
                self.pending_opens.pop(matched_idx)
                return "✅ Giao dịch hợp lệ (Verified Transaction)"
            else:
                # Có thể là thanh toán thẻ hoặc chuyển khoản (không mở két)
                return "ℹ️ POS: Thanh toán không dùng tiền mặt."
        
        return None

    def check_timeouts(self):
        """
        Hàm này cần được gọi liên tục trong vòng lặp chính (main loop).
        Nó kiểm tra xem có lần mở két nào bị 'treo' quá lâu mà không có POS log không.
        """
        current_time = time.time()
        
        # Duyệt qua danh sách các lần mở két
        # Lưu ý: Duyệt bản copy để an toàn khi remove phần tử
        for event in self.pending_opens[:]:
            if event['status'] == 'PENDING':
                elapsed = current_time - event['time']
                
                # Nếu quá thời gian cho phép (ví dụ 60s) mà chưa thấy POS Payment/Void
                if elapsed > self.match_window:
                    event['status'] = 'THEFT_TIMEOUT'
                    self.alert_log = f"🚨 BÁO ĐỘNG: Mở két {int(elapsed)}s mà KHÔNG nhập POS! (Nghi vấn trộm tiền)"
                    
                    # Xóa sự kiện này khỏi danh sách pending để tránh báo lặp lại liên tục
                    self.pending_opens.remove(event)
                    return self.alert_log
        
        return None