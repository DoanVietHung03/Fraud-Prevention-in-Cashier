import streamlit as st
import cv2
import numpy as np
import time

from modules.detect_hand import FraudDetector

st.set_page_config(page_title="Monitor AI - Cafe Cam", layout="wide")
st.title("🛡️ Coffee Shop Monitor (Vectron POS Setup)")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("Cấu hình Vùng Nhận Diện")

# Hướng dẫn
st.sidebar.info("💡 Điều chỉnh khung trùng khớp với thiết bị trong hình.")

# 1. Cấu hình POS (Màn hình bên Trái) - Màu Xanh Lá
st.sidebar.subheader("1. Màn hình POS (Green)")
# Ước lượng vị trí POS dựa trên ảnh của bạn (Góc trái dưới/giữa)
pos_x1 = st.sidebar.slider("POS X1", 0, 1280, 150, key="p_x1")
pos_y1 = st.sidebar.slider("POS Y1", 0, 720, 250, key="p_y1")
pos_x2 = st.sidebar.slider("POS X2", 0, 1280, 550, key="p_x2")
pos_y2 = st.sidebar.slider("POS Y2", 0, 720, 550, key="p_y2")
pos_roi = [pos_x1, pos_y1, pos_x2, pos_y2]

# 2. Cấu hình Két Tiền (Ngăn kéo bên Phải) - Màu Đỏ
st.sidebar.subheader("2. Két Tiền Mở (Red)")
# Ước lượng vị trí Két khi mở ra (Góc phải trên)
drawer_x1 = st.sidebar.slider("Drawer X1", 0, 1280, 600, key="d_x1")
drawer_y1 = st.sidebar.slider("Drawer Y1", 0, 720, 50, key="d_y1")
drawer_x2 = st.sidebar.slider("Drawer X2", 0, 1280, 950, key="d_x2")
drawer_y2 = st.sidebar.slider("Drawer Y2", 0, 720, 350, key="d_y2")
drawer_roi = [drawer_x1, drawer_y1, drawer_x2, drawer_y2]

# --- INIT DETECTOR ---
model_path = "./models/hand_landmarker.task"
# Khởi tạo detector với tham số từ sidebar
detector = FraudDetector(model_path, drawer_roi, pos_roi)

# --- MAIN APP ---
video_source = st.file_uploader("Tải video giám sát (Góc quay từ trên xuống)", type=['mp4', 'mov', 'avi'])
video_path = "./samples/temp_sample.mp4"

if video_source:
    with open(video_path, "wb") as f:
        f.write(video_source.read())
    
    cap = cv2.VideoCapture(video_path)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st_frame = st.empty()
    with col2:
        st.subheader("Nhật ký Hoạt động")
        st_log = st.empty()
    
    logs = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Resize frame nếu video quá to để xử lý nhanh hơn (tùy chọn)
        # frame = cv2.resize(frame, (1280, 720))
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_timestamp_ms = int(1000 * frame_count / fps)
        frame_count += 1
        
        # Xử lý Logic
        result, event = detector.process_frame(frame_rgb, frame_timestamp_ms)
        
        # --- VẼ GIAO DIỆN ---
        # 1. Vẽ vùng POS - Green
        cv2.rectangle(frame_rgb, (pos_roi[0], pos_roi[1]), (pos_roi[2], pos_roi[3]), (0, 255, 0), 2)
        cv2.putText(frame_rgb, "POS INPUT", (pos_roi[0], pos_roi[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 2. Vẽ vùng Két (Drawer) - Red
        cv2.rectangle(frame_rgb, (drawer_roi[0], drawer_roi[1]), (drawer_roi[2], drawer_roi[3]), (255, 0, 0), 2)
        cv2.putText(frame_rgb, "CASH DRAWER", (drawer_roi[0], drawer_roi[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # 3. Hiển thị trạng thái
        status_text = f"STATE: {detector.state}"
        # Màu chữ: Đỏ nếu nghi vấn, Xanh nếu bình thường
        text_color = (255, 0, 0) if detector.state == "SUSPICIOUS" else (0, 255, 0)
        cv2.putText(frame_rgb, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

        # Xử lý Log
        if event:
            timestamp = time.strftime('%H:%M:%S')
            prefix = "🚨 " if "ALARM" in event else "✅ "
            log_entry = f"[{timestamp}] {prefix}{event}"
            logs.append(log_entry)
            
            # Hiển thị log dạng cuộn
            log_text = "\n\n".join(logs[::-1]) 
            st_log.markdown(f"**Log:**\n```\n{log_text}\n```")

        st_frame.image(frame_rgb, channels="RGB")

    cap.release()