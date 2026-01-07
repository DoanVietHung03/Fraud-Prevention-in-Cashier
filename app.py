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
        self.event_type = "UNKNOWN"     # Lưu loại sự kiện (WARNING, ALARM, UNKNOWN)
        
        if not os.path.exists(output_folder): os.makedirs(output_folder)

    def add_frame(self, frame):
        self.ring_buffer.append(frame)      # Luôn lưu vào bộ nhớ tạm
        if self.is_recording:
            self.temp_evidence.append(frame)
            self.frames_to_record -= 1
            if self.frames_to_record <= 0: self.stop_and_save()

    def trigger_save(self, event_type="ALARM", duration_future=30):
        """
        event_type: "ALARM" hoặc "WARNING"
        duration_future: Số giây muốn ghi thêm vào tương lai (mặc định 30s)
        """
        if self.is_recording:
            # Nếu đang ghi WARNING mà có ALARM -> Nâng cấp lên ALARM
            if event_type == "ALARM" and self.event_type == "WARNING":
                self.event_type = "ALARM"
                
                # Gia hạn thêm thời gian ghi nếu cần
                self.frames_to_record = max(self.frames_to_record, int(self.fps * duration_future))
            return False    # Đang ghi rồi thì không kích hoạt mới

        # Nếu chưa ghi thì bắt đầu ghi
        self.is_recording = True
        self.event_type = event_type
        self.frames_to_record = int(self.fps * duration_future)
        self.temp_evidence = list(self.ring_buffer) 
        return True

    def _save(self, frames, event_label):
        if not frames: return
        
        # Tên file sẽ có dạng: evidence_ALARM_2024... hoặc evidence_WARNING_2024...
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.output_folder}/evidence_{event_label}_{timestamp}.mp4"

        h, w, _ = frames[0].shape
        try:
            out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (w, h))
            for f in frames: out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR)) 
            out.release()
            print(f"✅ Saved: {filename}")
        except Exception as e:
            print(f" ❌ Error saving video: {e}")
    
    def stop_and_save(self):
        self.is_recording = False
        
        # Truyền event_type hiện tại vào thread save
        threading.Thread(target=self._save, args=(self.temp_evidence.copy(), self.event_type)).start()
        self.temp_evidence = []


# --- SETUP STREAMLIT ---
st.set_page_config(page_title="Smart Retail Monitor", layout="wide")
st.title("🛡️ AI Fraud Detection: Hybrid (Motion Gate + FSM)")

# --- CẤU HÌNH SIDEBAR ---
st.sidebar.header("1. Cấu hình Model")
model_hand_path = st.sidebar.text_input("Đường dẫn Model Tay (.task)", "./models/hand_landmarker.task")
model_drawer_path = st.sidebar.text_input("Đường dẫn Model Két (.tflite)", "./models/demo_sample/model_unquant.tflite")

st.sidebar.header("2. Cấu hình Vùng (ROI)")
st.sidebar.info("💡 Kéo thanh trượt sao cho khung khớp vị trí thực tế.")

# 1. Cấu hình POS (Green)
pos_x1 = st.sidebar.slider("POS X1", 0, 1280, 206, key="p_x1")
pos_y1 = st.sidebar.slider("POS Y1", 0, 720, 180, key="p_y1")
pos_x2 = st.sidebar.slider("POS X2", 0, 1280, 370, key="p_x2")
pos_y2 = st.sidebar.slider("POS Y2", 0, 720, 292, key="p_y2")
pos_roi = [pos_x1, pos_y1, pos_x2, pos_y2]

# 2. Cấu hình Két Tiền (Red/Dynamic)
drawer_x1 = st.sidebar.slider("DRAWER X1", 0, 1280, 164, key="d_x1")
drawer_y1 = st.sidebar.slider("DRAWER Y1", 0, 720, 150, key="d_y1")
drawer_x2 = st.sidebar.slider("DRAWER X2", 0, 1280, 306, key="d_x2")
drawer_y2 = st.sidebar.slider("DRAWER Y2", 0, 720, 200, key="d_y2")
drawer_roi = [drawer_x1, drawer_y1, drawer_x2, drawer_y2]

# --- INIT SYSTEM ---
if 'detector' not in st.session_state:
    try:
        # Kiểm tra file tồn tại chưa để tránh crash
        if os.path.exists(model_hand_path) and os.path.exists(model_drawer_path):
            st.session_state.detector = FraudDetector(model_drawer_path, model_hand_path, drawer_roi, pos_roi)
        else:
            st.error("⚠️ Model file not found! Please check paths.")
            st.stop()
    except Exception as e:
        st.error(f"Lỗi khởi tạo: {e}")
        st.stop()

detector = st.session_state.detector

# Cập nhật ROI realtime khi kéo slider
detector.pos_roi = pos_roi
detector.drawer_roi = drawer_roi

# --- MAIN APP LOOP ---
video_source = st.file_uploader("Tải video giám sát (Test)", type=['mp4', 'mov', 'avi'])
default_video_path = "./samples_demo/good_procedure/test.mp4"

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

    detector.reset()
    
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    recorder = EvidenceRecorder(fps=fps, buffer_seconds=30)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st_frame = st.empty()
    with col2:
        st.subheader("📡 Nhật ký Hệ thống")
        st_mode = st.empty() # Hiển thị chế độ Ngủ/Thức
        st_log = st.empty()
        st_state_info = st.empty()
    
    logs = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (640, 640))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_timestamp_ms = int(1000 * frame_count / fps)
        frame_count += 1
        
        # --- GỌI XỬ LÝ LOGIC ---
        # detection_result sẽ là None nếu AI đang ngủ
        detection_result, event, drawer_status = detector.process_frame(frame_rgb, frame_timestamp_ms)

        recorder.add_frame(frame_rgb)

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

        # --- HIỂN THỊ TRẠNG THÁI REC ---
        if recorder.is_recording:
            # Đổi màu icon REC dựa theo loại sự kiện
            rec_color = (255, 0, 0) if recorder.event_type == "ALARM" else (0, 165, 255)
            cv2.circle(frame_rgb, (30, 30), 10, rec_color, -1)
            cv2.putText(frame_rgb, f"REC [{recorder.event_type}]", (50, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, rec_color, 1)
        
        # --- HIỂN THỊ TRẠNG THÁI NGỦ/THỨC ---
        if detector.is_sleeping:
            st_mode.success("🌙 MODE: SLEEP (Motion Gate Active)")
            # Vẽ overlay mờ để báo hiệu hệ thống đang tiết kiệm điện
            overlay = frame_rgb.copy()
            cv2.rectangle(overlay, (0,0), (400, 60), (0,0,0), -1)
            cv2.addWeighted(overlay, 0.5, frame_rgb, 0.5, 0, frame_rgb)
            cv2.putText(frame_rgb, "💤 AI SLEEPING (NO MOTION)", (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            st_mode.warning("⚡ MODE: ACTIVE (AI Processing)")
            # Chỉ vẽ tay khi có kết quả detect (không phải None)
            if detection_result and detection_result.hand_landmarks:
                for landmarks in detection_result.hand_landmarks:
                    wrist = landmarks[0]
                    index_finger = landmarks[8]
                    h, w, _ = frame.shape
                    cx, cy = int(((wrist.x + index_finger.x) / 2) * w), int(((wrist.y + index_finger.y) / 2) * h)
                    cv2.circle(frame_rgb, (cx, cy), 5, (255, 255, 0), -1)

        # --- VẼ GIAO DIỆN ROI ---
        cv2.rectangle(frame_rgb, (pos_roi[0], pos_roi[1]), (pos_roi[2], pos_roi[3]), (0, 255, 0), 2)
        cv2.putText(frame_rgb, "POS AREA", (pos_roi[0], pos_roi[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        if drawer_status == "OPEN":
            box_color = (255, 0, 0)
            box_thick = 3
            status_lbl = "DRAWER OPEN"
        else:
            box_color = (128, 128, 128)
            box_thick = 1
            status_lbl = "Drawer Closed"
            
        cv2.rectangle(frame_rgb, (drawer_roi[0], drawer_roi[1]), (drawer_roi[2], drawer_roi[3]), box_color, box_thick)
        cv2.putText(frame_rgb, status_lbl, (drawer_roi[0], drawer_roi[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 1)

        # Hiển thị State
        state_color = (255, 0, 0) if detector.state == "SUSPICIOUS" else (0, 255, 0)
        cv2.putText(frame_rgb, f"STATE: {detector.state}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 1)
        
        # --- UPDATE LOGS ---
        if event:
            timestamp = time.strftime('%H:%M:%S')
            if "ALARM" in event: icon = "🚨"
            elif "STEP" in event: icon = "👣"
            elif "Complete" in event: icon = "✅"
            else: icon = "ℹ️"
            
            log_entry = f"{icon} [{timestamp}] {event}"
            logs.append(log_entry)
            st_log.markdown("  \n".join(logs[::-1]))
            st_state_info.info(f"Last Event: {event}")

        st_frame.image(frame_rgb, channels="RGB")
        time.sleep(0.01)  # Giảm tải CPU

    cap.release()
else:
    st.info("Vui lòng tải video lên hoặc đảm bảo file mẫu tồn tại.")