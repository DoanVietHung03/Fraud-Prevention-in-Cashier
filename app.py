import streamlit as st
import cv2
import numpy as np
import time
import os
import threading
import queue
from collections import deque
from datetime import datetime
from streamlit_image_coordinates import streamlit_image_coordinates

from modules.detect_hand import FraudDetector

# --- 1. SETUP & UTILS ---
class EvidenceRecorder:
    def __init__(self, output_folder="evidence_clips", fps=30, buffer_seconds=30):
        self.output_folder = output_folder
        self.fps = fps
        self.ring_buffer = deque(maxlen=int(fps * buffer_seconds))
        self.is_recording = False
        self.frames_to_record = 0
        self.temp_evidence = []
        self.event_type = "UNKNOWN"
        if not os.path.exists(output_folder): os.makedirs(output_folder)

    def add_frame(self, frame):
        self.ring_buffer.append(frame)
        if self.is_recording:
            self.temp_evidence.append(frame)
            self.frames_to_record -= 1
            if self.frames_to_record <= 0: self.stop_and_save()

    def trigger_save(self, event_type="ALARM", duration_future=30):
        if self.is_recording:
            if event_type == "ALARM" and self.event_type == "WARNING":
                self.event_type = "ALARM"
                self.frames_to_record = max(self.frames_to_record, int(self.fps * duration_future))
            return False
        self.is_recording = True
        self.event_type = event_type
        self.frames_to_record = int(self.fps * duration_future)
        self.temp_evidence = list(self.ring_buffer) 
        return True

    def _save(self, frames, event_label):
        if not frames: return
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
        threading.Thread(target=self._save, args=(self.temp_evidence.copy(), self.event_type)).start()
        self.temp_evidence = []

class VideoProcessorThread(threading.Thread):
    def __init__(self, video_path, detector, output_queue):
        threading.Thread.__init__(self)
        self.video_path = video_path
        self.detector = detector
        self.output_queue = output_queue
        self.stopped = False
        self.fps = 30
        self.recorder = None

    def run(self):
        cap = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG)
        self.fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
        self.recorder = EvidenceRecorder(fps=self.fps, buffer_seconds=30)
        
        self.detector.reset()
        frame_count = 0

        while not self.stopped and cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_timestamp_ms = int(1000 * frame_count / self.fps)
            frame_count += 1

            detection_result, event, drawer_status = self.detector.process_frame(frame_rgb, frame_timestamp_ms)
            
            self.recorder.add_frame(frame_rgb)
            toast_msg = None
            if event:
                if "ALARM" in event:
                    if self.recorder.trigger_save(event_type="ALARM", duration_future=30):
                        toast_msg = ("🚨 PHÁT HIỆN VI PHẠM!", "🔥")
                elif "WARNING" in event:
                    if self.recorder.trigger_save(event_type="WARNING", duration_future=30):
                         toast_msg = ("⚠️ Cảnh báo quy trình!", "📹")

            # Visualize
            cv2.rectangle(frame_rgb, (self.detector.pos_roi[0], self.detector.pos_roi[1]), 
                          (self.detector.pos_roi[2], self.detector.pos_roi[3]), (0, 255, 0), 2)
            
            if drawer_status == "OPEN":
                box_color = (255, 0, 0)
                status_lbl = "DRAWER OPEN"
            else:
                box_color = (128, 128, 128)
                status_lbl = "Drawer CLOSED"
            cv2.rectangle(frame_rgb, (self.detector.drawer_roi[0], self.detector.drawer_roi[1]), 
                          (self.detector.drawer_roi[2], self.detector.drawer_roi[3]), box_color, 2)
            
            state_color = (255, 0, 0) if self.detector.state == "SUSPICIOUS" else (0, 255, 0)
            cv2.putText(frame_rgb, f"STATE: {self.detector.state}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)

            if self.output_queue.full():
                try: self.output_queue.get_nowait()
                except queue.Empty: pass
            
            packet = {
                "frame": frame_rgb, "event": event, "drawer_status": drawer_status,
                "is_sleeping": self.detector.is_sleeping, "toast": toast_msg
            }
            self.output_queue.put(packet)
            
        cap.release()
        self.stopped = True

    def stop(self):
        self.stopped = True

# --- 2. STREAMLIT APP ---
st.set_page_config(page_title="Smart Retail Monitor", layout="wide")
st.title("🛡️ AI Fraud Detection: Click-to-Setup")

# --- KHỞI TẠO STATE ---
defaults = {
    "p_x1": 206, "p_y1": 424, "p_x2": 370, "p_y2": 624,
    "d_x1": 164, "d_y1": 352, "d_x2": 391, "d_y2": 444,
    "last_processed_click": None 
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# --- SIDEBAR: CẤU HÌNH ---
video_source = st.file_uploader("Tải video giám sát", type=['mp4', 'mov', 'avi'])
default_video_path = "./samples_demo/drawer_obscured/test.mp4"
final_video_path = None

if video_source:
    with open("temp_upload.mp4", "wb") as f: f.write(video_source.read())
    final_video_path = "temp_upload.mp4"
elif os.path.exists(default_video_path):
    final_video_path = default_video_path

st.sidebar.header("1. Cấu hình AI")
model_hand_path = st.sidebar.text_input("Model Tay", "./models/hand_landmarker.task")
model_drawer_path = st.sidebar.text_input("Model Két", "./models/demo_sample/model_unquant.tflite")

st.sidebar.divider()

# --- SIDEBAR: CHỌN ĐIỂM ĐỂ CLICK ---
setup_mode = st.sidebar.checkbox("🎯 Chế độ lấy tọa độ (Setup Mode)", value=False)
target_point = None
if setup_mode:
    st.sidebar.info("Chọn điểm cần chỉnh bên dưới, sau đó click vào ảnh.")
    target_point = st.sidebar.radio(
        "Đang chỉnh tọa độ cho:",
        ["POS: Top-Left (Góc Trái-Trên)", 
         "POS: Bottom-Right (Góc Phải-Dưới)", 
         "DRAWER: Top-Left (Góc Trái-Trên)", 
         "DRAWER: Bottom-Right (Góc Phải-Dưới)"]
    )

st.sidebar.divider()

# --- SIDEBAR: HIỂN THỊ KẾT QUẢ ---
st.sidebar.subheader("📍 Tọa độ hiện tại")
st.sidebar.markdown(f"""
**🟩 Vùng POS:**
- `x1, y1`: **{st.session_state.p_x1}, {st.session_state.p_y1}**
- `x2, y2`: **{st.session_state.p_x2}, {st.session_state.p_y2}**

**🟥 Vùng Két (Drawer):**
- `x1, y1`: **{st.session_state.d_x1}, {st.session_state.d_y1}**
- `x2, y2`: **{st.session_state.d_x2}, {st.session_state.d_y2}**
""")

# --- MAIN LOGIC ---

# 1. SETUP MODE: HIỂN THỊ ẢNH ĐỂ CLICK
if final_video_path and setup_mode:
    st.info(f"👉 Hãy click vào ảnh để đặt tọa độ cho: **{target_point}**")
    
    cap = cv2.VideoCapture(final_video_path)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Vẽ các hộp chữ nhật hiện tại lên ảnh để dễ căn
        p1 = (st.session_state["p_x1"], st.session_state["p_y1"])
        p2 = (st.session_state["p_x2"], st.session_state["p_y2"])
        d1 = (st.session_state["d_x1"], st.session_state["d_y1"])
        d2 = (st.session_state["d_x2"], st.session_state["d_y2"])

        cv2.rectangle(frame_rgb, p1, p2, (0, 255, 0), 2)
        cv2.rectangle(frame_rgb, d1, d2, (255, 0, 0), 2)
        
        # COMPONENT NHẬN CLICK
        value = streamlit_image_coordinates(frame_rgb, key="click_capture")
        
        # XỬ LÝ CLICK
        if value is not None and value != st.session_state["last_processed_click"]:
            st.session_state["last_processed_click"] = value
            
            x_click = value["x"]
            y_click = value["y"]
            
            # Cập nhật state tùy theo Radio đang chọn
            if "POS: Top-Left" in target_point:
                st.session_state["p_x1"] = x_click
                st.session_state["p_y1"] = y_click
            elif "POS: Bottom-Right" in target_point:
                st.session_state["p_x2"] = x_click
                st.session_state["p_y2"] = y_click
            elif "DRAWER: Top-Left" in target_point:
                st.session_state["d_x1"] = x_click
                st.session_state["d_y1"] = y_click
            elif "DRAWER: Bottom-Right" in target_point:
                st.session_state["d_x2"] = x_click
                st.session_state["d_y2"] = y_click
            
            st.rerun() # Refresh để cập nhật số hiển thị bên Sidebar

# 2. RUN MODE: CHẠY AI
elif final_video_path and not setup_mode:
    # Lấy tọa độ từ Session State
    pos_roi = [st.session_state.p_x1, st.session_state.p_y1, st.session_state.p_x2, st.session_state.p_y2]
    drawer_roi = [st.session_state.d_x1, st.session_state.d_y1, st.session_state.d_x2, st.session_state.d_y2]
    
    # Init Detector
    if 'detector' not in st.session_state:
        try:
            if os.path.exists(model_hand_path) and os.path.exists(model_drawer_path):
                st.session_state.detector = FraudDetector(model_drawer_path, model_hand_path, drawer_roi, pos_roi)
            else:
                st.error("⚠️ Model file not found!")
                st.stop()
        except Exception as e:
            st.error(f"Lỗi khởi tạo: {e}")
            st.stop()

    # Cập nhật ROI mới nhất
    st.session_state.detector.pos_roi = pos_roi
    st.session_state.detector.drawer_roi = drawer_roi
    
    # Giao diện chạy
    col1, col2 = st.columns([3, 1])
    with col1:
        st_frame = st.empty()
    with col2:
        st.subheader("📡 Nhật ký")
        st_mode = st.empty()
        st_log = st.empty()
        st_state_info = st.empty()
    
    logs = []
    start_btn = st.button("▶️ Bắt đầu / Reset")
    stop_btn = st.button("⏹️ Dừng")

    if 'frame_queue' not in st.session_state:
        st.session_state.frame_queue = queue.Queue(maxsize=1)
    
    if start_btn:
        if 'thread' in st.session_state and st.session_state.thread.is_alive():
            st.session_state.thread.stop()
            st.session_state.thread.join()
        
        st.session_state.thread = VideoProcessorThread(final_video_path, st.session_state.detector, st.session_state.frame_queue)
        st.session_state.thread.start()
    
    if stop_btn:
        if 'thread' in st.session_state:
            st.session_state.thread.stop()

    if 'thread' in st.session_state and st.session_state.thread.is_alive():
        while True:
            try:
                data = st.session_state.frame_queue.get(timeout=1.0)
            except queue.Empty:
                if not st.session_state.thread.is_alive(): break
                continue
            
            st_frame.image(data['frame'], channels="RGB", width='stretch')
            
            if data['is_sleeping']: st_mode.success("🌙 MODE: SLEEP")
            else: st_mode.warning("⚡ MODE: ACTIVE")

            event = data['event']
            if event:
                timestamp = time.strftime('%H:%M:%S')
                icon = "🚨" if "ALARM" in event else "ℹ️"
                logs.append(f"{icon} [{timestamp}] {event}")
                st_log.markdown("  \n".join(logs[::-1]))
            
            if data['toast']:
                msg, icon_toast = data['toast']
                st.toast(msg, icon=icon_toast)
            
            if not st.session_state.thread.is_alive() and st.session_state.frame_queue.empty():
                st.info("Video kết thúc.")
                break