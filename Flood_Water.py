"""
streamlit_app.py
================
Web app for flood water-level detection.

Run locally :  streamlit run streamlit_app.py
Deploy      :  push this folder to GitHub -> https://share.streamlit.io
               (main file = streamlit_app.py)

Tabs: Image upload | Video upload | Camera snapshot | Live camera (WebRTC)
Uses water_detector.py (must be in the same folder).
"""

import json
import os
import tempfile
import threading

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from water_detector import (LevelTracker, analyze_image, draw_panel,
                            format_report, result_to_dict)

st.set_page_config(page_title="Flood Water Level Detector", page_icon="🌊", layout="wide")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def decode_upload(file):
    data = np.frombuffer(file.getvalue(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def jpg_bytes(bgr):
    return cv2.imencode(".jpg", bgr)[1].tobytes()


SEVERITY_BOX = {0: st.success, 1: st.warning, 2: st.warning, 3: st.error}


def show_result(res, trend=None, key="r", name="result"):
    """Detailed display of one analysis result."""
    SEVERITY_BOX[res["severity"]](f"{res['emoji']} **{res['label']}** - {res['description']}")

    c = st.columns(5)
    c[0].metric("Water level", f"{res['level_percent']}%", help="% of frame height, from the bottom")
    c[1].metric("Approx. depth", f"~{res['estimated_depth_cm']} cm",
                help=f"Assumes the visible scene is {res['scene_height_cm_assumed']:.0f} cm tall")
    c[2].metric("Water area", f"{res['water_area_percent']}%")
    c[3].metric("Confidence", f"{res['confidence_percent']}%")
    c[4].metric("Water regions", res["num_water_regions"])

    if trend:
        t = st.columns(3)
        t[0].metric("Trend", trend["trend"], f"{trend['rate_percent_per_min']:+.1f} %/min")
        t[1].metric("Smoothed level", f"{trend['smoothed_level_percent']}%")
        t[2].metric("Peak level", f"{trend['peak_level_percent']}%")

    left, right = st.columns(2)
    left.image(to_rgb(res["resized_image"]), caption="Original", width="stretch")
    right.image(to_rgb(draw_panel(res, trend)), caption="Detected water (blue) + waterline (red)",
                width="stretch")

    with st.expander("Full details (report)"):
        st.code(format_report(res, trend), language="text")
        st.info(f"**Advice:** {res['advice']}")
    with st.expander("Water mask"):
        st.image(res["mask"], caption="White = detected water", width="stretch")

    d1, d2 = st.columns(2)
    d1.download_button("Download result image", jpg_bytes(draw_panel(res, trend)),
                       file_name=f"{name}_result.jpg", mime="image/jpeg", key=f"{key}_img")
    d2.download_button("Download details (JSON)",
                       json.dumps(result_to_dict(res, trend), indent=2, ensure_ascii=False),
                       file_name=f"{name}_details.json", mime="application/json", key=f"{key}_json")


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
st.title("🌊 Flood Water Level Detector")
st.caption("Visual estimate from image / video / camera. Not a calibrated or safety-certified measurement.")

with st.sidebar:
    st.header("Settings")
    scene_h = st.number_input(
        "Visible scene height (cm)", min_value=30.0, max_value=1000.0, value=150.0, step=10.0,
        help="How tall (in real life) the area visible in the frame is, at the water's position. "
             "A known reference (pole, wall, door) makes the cm estimate more meaningful.")
    st.markdown("---")
    st.markdown(
        "**Severity bands**\n\n"
        "- 🟢 under 8%: minimal\n- 🟡 8-25%: low (ankle)\n"
        "- 🟠 25-55%: moderate (knee-waist)\n- 🔴 55%+: severe\n\n"
        "The percentage is the share of the frame height, from the bottom, covered by water."
    )

tab_img, tab_vid, tab_cam, tab_live = st.tabs(
    ["🖼️ Image", "🎞️ Video", "📸 Camera snapshot", "🔴 Live camera"])

# ----------------------------------------------------------------------
# Image tab
# ----------------------------------------------------------------------
with tab_img:
    files = st.file_uploader("Upload one or more flood photos", type=["jpg", "jpeg", "png", "bmp", "webp"],
                             accept_multiple_files=True, key="img_up")
    for i, f in enumerate(files or []):
        img = decode_upload(f)
        if img is None:
            st.error(f"Could not read {f.name}")
            continue
        st.subheader(f.name)
        with st.spinner("Analyzing..."):
            res = analyze_image(img, scene_height_cm=scene_h)
        show_result(res, key=f"img{i}", name=os.path.splitext(f.name)[0])
        st.markdown("---")

# ----------------------------------------------------------------------
# Video tab
# ----------------------------------------------------------------------
with tab_vid:
    vf = st.file_uploader("Upload a flood video", type=["mp4", "avi", "mov", "mkv", "webm"], key="vid_up")
    c1, c2 = st.columns(2)
    every = c1.slider("Analyze every Nth frame", 1, 60, 10)
    max_frames = c2.slider("Max frames to analyze", 20, 600, 200,
                           help="Keeps processing time reasonable on a hosted server.")

    if vf is not None and st.button("Analyze video", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(vf.name)[1]) as tmp:
            tmp.write(vf.getvalue())
            path = tmp.name

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            st.error("Could not open this video.")
        else:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            tracker = LevelTracker(window_seconds=120)
            rows, peak, last = [], None, None
            bar = st.progress(0.0, text="Analyzing video...")
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                idx += 1
                if idx % every:
                    continue
                res = analyze_image(frame, scene_height_cm=scene_h, max_width=640)
                t = idx / fps
                tr = tracker.update(res["level_percent"], t=t)
                rows.append({"time_s": round(t, 2), "level_percent": res["level_percent"],
                             "smoothed_percent": tr["smoothed_level_percent"],
                             "depth_cm_approx": res["estimated_depth_cm"],
                             "water_area_percent": res["water_area_percent"],
                             "confidence_percent": res["confidence_percent"],
                             "severity": res["label"], "trend": tr["trend"],
                             "rate_percent_per_min": tr["rate_percent_per_min"]})
                last = (res, tr)
                if peak is None or res["level_percent"] > peak[0]["level_percent"]:
                    peak = (res, tr, t)
                if total:
                    bar.progress(min(idx / total, 1.0), text=f"Analyzing... frame {idx}/{total}")
                if len(rows) >= max_frames:
                    break
            cap.release()
            os.unlink(path)
            bar.empty()

            if not rows:
                st.warning("No frames analyzed.")
            else:
                df = pd.DataFrame(rows)
                st.subheader("Water level over time")
                st.line_chart(df.set_index("time_s")[["level_percent", "smoothed_percent"]])

                s = st.columns(4)
                s[0].metric("Frames analyzed", len(df))
                s[1].metric("Peak level", f"{df['level_percent'].max()}%")
                s[2].metric("Average level", f"{df['level_percent'].mean():.1f}%")
                s[3].metric("Final trend", last[1]["trend"], f"{last[1]['rate_percent_per_min']:+.1f} %/min")

                st.subheader(f"Peak moment (at {peak[2]:.1f}s)")
                show_result(peak[0], peak[1], key="vid_peak", name="video_peak")

                st.subheader("Last analyzed frame")
                show_result(last[0], last[1], key="vid_last", name="video_last")

                st.dataframe(df, width="stretch")
                st.download_button("Download log (CSV)", df.to_csv(index=False),
                                   file_name="video_log.csv", mime="text/csv")

# ----------------------------------------------------------------------
# Camera snapshot tab (works on any deployed app, uses the browser camera)
# ----------------------------------------------------------------------
with tab_cam:
    st.write("Take a photo with your device camera and analyze it.")
    shot = st.camera_input("Camera", key="cam_snap")
    if shot is not None:
        img = decode_upload(shot)
        res = analyze_image(img, scene_height_cm=scene_h)
        show_result(res, key="cam", name="camera_snapshot")

# ----------------------------------------------------------------------
# Live camera tab (WebRTC: browser camera -> server -> annotated video)
# ----------------------------------------------------------------------
_lock = threading.Lock()
_live_tracker = LevelTracker(alpha=0.3, window_seconds=120)
_live_state = {"res": None, "trend": None}

with tab_live:
    try:
        import av
        from streamlit_webrtc import RTCConfiguration, webrtc_streamer
        WEBRTC_OK = True
    except Exception:
        WEBRTC_OK = False

    if not WEBRTC_OK:
        st.warning("Live mode needs `streamlit-webrtc` and `av`. Add them to requirements.txt "
                   "(already included in the provided file).")
    else:
        st.write("Live annotated video from your browser camera. Allow camera access when asked. "
                 "Water level, depth and trend are drawn on the video.")

        def video_frame_callback(frame):
            img = frame.to_ndarray(format="bgr24")
            res = analyze_image(img, scene_height_cm=scene_h, max_width=480)
            with _lock:
                trend = _live_tracker.update(res["level_percent"])
                _live_state["res"], _live_state["trend"] = res, trend
            return av.VideoFrame.from_ndarray(draw_panel(res, trend), format="bgr24")

        ctx = webrtc_streamer(
            key="flood-live",
            video_frame_callback=video_frame_callback,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        st.caption("Note: this uses your phone/laptop camera through the browser. "
                   "A CCTV/RTSP camera needs the local `flood_app.py camera` tool instead.")
        if ctx.state.playing and st.button("Show latest detailed report"):
            with _lock:
                res, trend = _live_state["res"], _live_state["trend"]
            if res is not None:
                st.code(format_report(res, trend), language="text")
