import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import time
import pandas as pd
import collections


ROLLING_WINDOW_SECONDS = 60  # how many seconds of data to show live
FRAME_INTERVAL_UI_UPDATE = 10  # update UI every 10 frames
FRAME_SLEEP_TIME = 0.05  # 20 FPS roughly


mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
important_body_indices = [
    mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    mp_pose.PoseLandmark.LEFT_ELBOW.value,
    mp_pose.PoseLandmark.RIGHT_ELBOW.value,
    mp_pose.PoseLandmark.LEFT_HIP.value,
    mp_pose.PoseLandmark.RIGHT_HIP.value,
]



def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle



class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.left_angle = 0
        self.right_angle = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * img.shape[1],
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * img.shape[0]]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * img.shape[1],
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * img.shape[0]]
            self.left_angle = 180 - calculate_angle(left_elbow, left_shoulder,
                                                    [left_shoulder[0], left_shoulder[1] - 100])

            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * img.shape[1],
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * img.shape[0]]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * img.shape[1],
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * img.shape[0]]
            self.right_angle = 180 - calculate_angle(right_elbow, right_shoulder,
                                                     [right_shoulder[0], right_shoulder[1] - 100])

        return img


# -------------- Streamlit App --------------
def main():
    st.set_page_config(page_title="NeuroTrack Pro - Safe Deployment", layout="wide")
    st.title("🧠 NeuroTrack Pro - Safe Deployment Version")

    # Initialize session state
    if "start_time" not in st.session_state:
        st.session_state.start_time = time.time()
    if "rolling_data" not in st.session_state:
        st.session_state.rolling_data = collections.deque()
    if "full_summary" not in st.session_state:
        st.session_state.full_summary = []

    # UI Layout
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        ctx = webrtc_streamer(
            key="stream",
            video_transformer_factory=VideoTransformer,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
        )

    meter_left = col2.empty()
    meter_right = col3.empty()
    chart_placeholder = st.empty()
    timer_placeholder = st.empty()

    frame_counter = 0

    while ctx.state.playing:
        if ctx.video_transformer:
            now = time.time()
            elapsed = now - st.session_state.start_time

            left_angle = ctx.video_transformer.left_angle
            right_angle = ctx.video_transformer.right_angle

            # Add to full summary (light-weight)
            st.session_state.full_summary.append((elapsed, left_angle, right_angle))

            # Add to rolling window (time limited)
            st.session_state.rolling_data.append((elapsed, left_angle, right_angle))

            # Purge old data from rolling window
            while st.session_state.rolling_data and (elapsed - st.session_state.rolling_data[0][0]) > ROLLING_WINDOW_SECONDS:
                st.session_state.rolling_data.popleft()

            # UI update only every few frames
            if frame_counter % FRAME_INTERVAL_UI_UPDATE == 0:
                meter_left.metric("Left Arm", f"{int(left_angle)}°")
                meter_right.metric("Right Arm", f"{int(right_angle)}°")

                if len(st.session_state.rolling_data) >= 5:
                    df = pd.DataFrame(st.session_state.rolling_data, columns=["Time", "Left", "Right"])
                    df.set_index("Time", inplace=True)
                    chart_placeholder.line_chart(df)

                mins, secs = divmod(int(elapsed), 60)
                timer_placeholder.write(f"⏱ Session Time: {mins:02d}:{secs:02d}")

            frame_counter += 1
            time.sleep(FRAME_SLEEP_TIME)

    # After session ends
    if not ctx.state.playing and st.session_state.full_summary:
        st.success("✅ Session Completed!")
        df = pd.DataFrame(st.session_state.full_summary, columns=["Time", "Left Angle", "Right Angle"])
        st.line_chart(df.set_index("Time")[["Left Angle", "Right Angle"]])

        csv = df.to_csv(index=False)
        st.download_button("📥 Download Full Session Data", data=csv, file_name="session_data.csv", mime="text/csv")


if __name__ == "__main__":
    main()
