import streamlit as st
import cv2
import numpy as np
import time
import os
import threading
from collections import deque
from datetime import datetime

from modules.detect_hand import FraudDetector

# --- EVIDENCES RECORDING ---
class EvidenceRecorder:
    def __init__(self, output_folder="evidence_clips", fps=30, buffer_seconds=30):
        self.output_folder = output_folder
        self.fps = fps
        # Ring Buffer: Giữ video quá khứ
        self.ring_buffer = deque(maxlen=int(fps * buffer_seconds))
        self.is_recording = False
        self.frames_to_record = 0
        self.temp_evidence = []
        self.event_type = "UNKNOWN" # Lưu loại sự kiện
        
        if not os.path.exists(output_folder): os.makedirs(output_folder)

    def add_frame(self, frame):
        self.ring_buffer.append(frame) # Luôn lưu vào bộ nhớ tạm
        if self.is_recording:
            self.temp_evidence.append(frame)
            self.frames_to_record -= 1
            if self.frames_to_record <= 0: self.stop_and_save()

    def trigger_save(self, event_type="ALARM", duration_future=30):
        """
        event_type: "ALARM" hoặc "WARNING"
        duration_future: Số giây muốn ghi thêm vào tương lai (mặc định 30s)
        """
        # Nếu đang ghi WARNING mà chuyển sang ALARM -> Cập nhật nhãn thành ALARM (ưu tiên cao hơn)
        if self.is_recording:
            if event_type == "ALARM" and self.event_type == "WARNING":
                self.event_type = "ALARM"
                # Gia hạn thêm thời gian ghi nếu cần
                self.frames_to_record = max(self.frames_to_record, int(self.fps * duration_future))
            return False # Đang ghi rồi thì không kích hoạt mới

        # Nếu chưa ghi thì bắt đầu ghi
        self.is_recording = True
        self.event_type = event_type
        self.frames_to_record = int(self.fps * duration_future)
        self.temp_evidence = list(self.ring_buffer) # Lấy 30s quá khứ đắp vào
        return True

    def _save(self, frames, event_label):
        if not frames: return
        # Tên file sẽ có dạng: evidence_ALARM_2024... hoặc evidence_WARNING_2024...
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.output_folder}/evidence_{event_label}_{timestamp}.mp4"

        h, w, _ = frames[0].shape
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (w, h))
        for f in frames: out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR)) # Convert lại BGR để lưu
        out.release()
        print(f"✅ Đã lưu: {filename}")
    
    def stop_and_save(self):
        self.is_recording = False
        # Truyền event_type hiện tại vào thread save
        threading.Thread(target=self._save, args=(self.temp_evidence.copy(), self.event_type)).start()
        self.temp_evidence = []


# --- SETUP STREAMLIT ---
st.set_page_config(page_title="Smart Retail Monitor", layout="wide")
st.title("🛡️ AI Fraud Detection: Sequential Logic (FSM + Classification)")

# --- CẤU HÌNH SIDEBAR ---
st.sidebar.header("1. Cấu hình Model")
model_hand_path = st.sidebar.text_input("Đường dẫn Model Tay (.task)", "./models/hand_landmarker.task")
model_drawer_path = st.sidebar.text_input("Đường dẫn Model Két (.tflite)", "./models/pos_classification.tflite")

st.sidebar.header("2. Cấu hình Vùng (ROI)")
st.sidebar.info("💡 Kéo thanh trượt sao cho khung khớp vị trí thực tế.")

# 1. Cấu hình POS (Green)
st.sidebar.subheader("Vùng POS (Cảm ứng)")
pos_x1 = st.sidebar.slider("POS X1", 0, 1280, 427, key="p_x1")
pos_y1 = st.sidebar.slider("POS Y1", 0, 720, 185, key="p_y1")
pos_x2 = st.sidebar.slider("POS X2", 0, 1280, 680, key="p_x2")
pos_y2 = st.sidebar.slider("POS Y2", 0, 720, 406, key="p_y2")
pos_roi = [pos_x1, pos_y1, pos_x2, pos_y2]

# 2. Cấu hình Két Tiền (Red/Dynamic)
st.sidebar.subheader("Vùng Két Tiền (Ngăn kéo)")
drawer_x1 = st.sidebar.slider("Drawer X1", 0, 1280, 650, key="d_x1")
drawer_y1 = st.sidebar.slider("Drawer Y1", 0, 720, 98, key="d_y1")
drawer_x2 = st.sidebar.slider("Drawer X2", 0, 1280, 830, key="d_x2")
drawer_y2 = st.sidebar.slider("Drawer Y2", 0, 720, 260, key="d_y2")
drawer_roi = [drawer_x1, drawer_y1, drawer_x2, drawer_y2]

# --- INIT SYSTEM ---
# Kiểm tra file tồn tại chưa để tránh crash
if not os.path.exists(model_hand_path) or not os.path.exists(model_drawer_path):
    st.error("⚠️ Không tìm thấy file model! Hãy kiểm tra lại đường dẫn trong Sidebar.")
    st.stop()

# Khởi tạo Detector 
try:
    detector = FraudDetector(model_drawer_path, model_hand_path, drawer_roi, pos_roi)
except Exception as e:
    st.error(f"Lỗi khởi tạo: {e}")
    st.stop()

# --- MAIN APP LOOP ---
video_source = st.file_uploader("Tải video giám sát (Test)", type=['mp4', 'mov', 'avi'])
default_video_path = "./samples/temp_sample.mp4"

# Ưu tiên dùng video upload, nếu không có thì dùng video mặc định
final_video_path = None
if video_source:
    with open("temp_upload.mp4", "wb") as f:
        f.write(video_source.read())
    final_video_path = "temp_upload.mp4"
elif os.path.exists(default_video_path):
    final_video_path = default_video_path

if final_video_path:
    cap = cv2.VideoCapture(final_video_path, cv2.CAP_FFMPEG)

    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    recorder = EvidenceRecorder(fps=fps, buffer_seconds=30)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st_frame = st.empty()
    with col2:
        st.subheader("📡 Nhật ký Hệ thống")
        st_log = st.empty()
        st_state_info = st.empty()
    
    logs = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_timestamp_ms = int(1000 * frame_count / fps)
        frame_count += 1
        
        # --- GỌI XỬ LÝ LOGIC ---
        # Hàm trả về detection_result (tay), event (sự kiện logic), drawer_status (trạng thái két)
        detection_result, event, drawer_status = detector.process_frame(frame_rgb, frame_timestamp_ms)

        recorder.add_frame(frame_rgb) # Nạp frame vào bộ nhớ

        if event:
            # 1. Trường hợp BÁO ĐỘNG ĐỎ (Trộm / Ghost Refund)
            if "ALARM" in event:
                # Ghi ngay, gán nhãn ALARM, ghi thêm 30s tương lai
                if recorder.trigger_save(event_type="ALARM", duration_future=30):
                    st.toast("🚨 PHÁT HIỆN VI PHẠM! Đang lưu bằng chứng...", icon="🔥")
            
            # 2. Trường hợp CẢNH BÁO VÀNG (Mở két trước - Chờ Refund)
            elif "WARNING" in event:
                # Ghi ngay, gán nhãn WARNING, ghi thêm 30s tương lai (để chờ xem có nhập POS không)
                if recorder.trigger_save(event_type="WARNING", duration_future=30):
                    st.toast("⚠️ Cảnh báo quy trình! Đang lưu clip đối soát.", icon="📹")

        # --- HIỂN THỊ TRẠNG THÁI GHI ---
        if recorder.is_recording:
            # Đổi màu icon REC dựa theo loại sự kiện
            rec_color = (255, 0, 0) if recorder.event_type == "ALARM" else (0, 165, 255) # Đỏ hoặc Cam
            cv2.circle(frame_rgb, (30, 30), 10, rec_color, -1)
            cv2.putText(frame_rgb, f"REC [{recorder.event_type}]", (50, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, rec_color, 2)
        
        # --- VẼ GIAO DIỆN (VISUALIZATION) ---
        
        # 1. Vẽ Vùng POS (Luôn cố định màu Xanh Lá)
        cv2.rectangle(frame_rgb, (pos_roi[0], pos_roi[1]), (pos_roi[2], pos_roi[3]), (0, 255, 0), 2)
        cv2.putText(frame_rgb, "POS AREA", (pos_roi[0], pos_roi[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 2. Vẽ Vùng Két (Thay đổi màu theo trạng thái Detected)
        if drawer_status == "OPEN":
            box_color = (255, 0, 0) # Đỏ đậm báo động
            box_thick = 3
            status_lbl = "DRAWER OPEN [DETECTED]"
        else:
            box_color = (128, 128, 128) # Màu xám nhạt (Két đóng)
            box_thick = 1
            status_lbl = "Drawer Closed"
            
        cv2.rectangle(frame_rgb, (drawer_roi[0], drawer_roi[1]), (drawer_roi[2], drawer_roi[3]), box_color, box_thick)
        cv2.putText(frame_rgb, status_lbl, (drawer_roi[0], drawer_roi[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        # 3. Vẽ Tay (Đơn giản hóa: Chỉ vẽ các điểm đầu ngón tay nếu có)
        # if detection_result.hand_landmarks:
        #     for landmarks in detection_result.hand_landmarks:
        #         wrist = landmarks[0]
        #         index_finger = landmarks[8]
        #         h, w, _ = frame.shape
        #         cx, cy = int(((wrist.x + index_finger.x) / 2) * w), int(((wrist.y + index_finger.y) / 2) * h)
        #         cv2.circle(frame_rgb, (cx, cy), 5, (255, 255, 0), -1) # Màu vàng

        # 4. Hiển thị Thông tin Trạng thái (Góc trên trái)
        # State hiện tại
        state_color = (255, 0, 0) if detector.state == "SUSPICIOUS" else (0, 255, 0)
        cv2.putText(frame_rgb, f"STATE: {detector.state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, state_color, 2)
        
        # Đếm ngược giấy phép POS (Chỉ hiện khi vừa bấm POS)
        time_diff = time.time() - detector.last_pos_time
        if time_diff < detector.pos_timeout:
            perm_text = f"POS Permission: VALID ({int(detector.pos_timeout - time_diff)}s)"
            cv2.putText(frame_rgb, perm_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(frame_rgb, "POS Permission: EXPIRED", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

        # --- UPDATE LOGS ---
        if event:
            timestamp = time.strftime('%H:%M:%S')
            
            # Icon phân loại log
            if "ALARM" in event: icon = "🚨"
            elif "STEP" in event: icon = "👣"
            elif "Complete" in event: icon = "✅"
            else: icon = "ℹ️"
            
            log_entry = f"{icon} [{timestamp}] {event}"
            logs.append(log_entry)
            
            # Cập nhật khung log bên phải
            log_text = "  \n".join(logs[::-1]) # Mới nhất lên đầu
            st_log.markdown(log_text)
            
            # Cập nhật bảng thông tin nhanh
            st_state_info.info(f"Last Event: {event}")

        st_frame.image(frame_rgb, channels="RGB")

    cap.release()
else:
    st.info("Vui lòng tải video lên hoặc đảm bảo file mẫu tồn tại.")