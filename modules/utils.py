import json
import time
import threading
import os
from datetime import datetime

class PerformanceLogger:
    _file_lock = threading.Lock()
    def __init__(self, filename="event_logs.json"):
        self.filename = filename
        self.logs = []

    def log(self, process_time_ms, event_content):
        """
        Chỉ lưu nếu có event_content.
        """
        if event_content:
            entry = {
                "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3], # Giờ:Phút:Giây.ms
                "process_time_ms": round(process_time_ms, 2),
                "event": event_content
            }
            self.logs.append(entry)
            print(f"💾 Saved: {event_content} ({process_time_ms:.1f}ms)")
            
            with PerformanceLogger._file_lock:
                try:
                    data = []
                    # 1. Đọc dữ liệu cũ
                    if os.path.exists(self.filename):
                        # Dùng 'r' để đọc, nếu file lỗi/trống thì coi như mảng rỗng
                        try:
                            with open(self.filename, 'r', encoding='utf-8') as f:
                                content = f.read()
                                if content:
                                    data = json.loads(content)
                        except json.JSONDecodeError:
                            data = [] 
                    
                    # 2. Nối thêm dữ liệu mới
                    data.append(entry)

                    # 3. Ghi đè lại file an toàn
                    with open(self.filename, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=4, ensure_ascii=False)
                        
                except Exception as e:
                    print(f"❌ Lỗi ghi file (Race Condition): {e}")
    
    def save_to_file(self):
        pass