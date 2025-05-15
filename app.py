# import streamlit as st
# import tempfile
# import os
# from process_video_logic.process_video import process_video

# st.set_page_config(page_title="Stop Sign Violation Detection", layout="wide")

# st.title("🚦 Stop Sign Violation Detection with OCR + Direction + Speed Analysis")

# # Upload section
# st.sidebar.header("📤 Upload Files")
# uploaded_video = st.sidebar.file_uploader("Upload Video (.mp4)", type=["mp4"])
# uploaded_json = st.sidebar.file_uploader("Upload Speed JSON", type=["json"])

# if uploaded_video and uploaded_json:
#     # Save uploaded files to temp locations
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
#         tmp_vid.write(uploaded_video.read())
#         video_path = tmp_vid.name

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_json:
#         tmp_json.write(uploaded_json.read())
#         json_path = tmp_json.name

#     st.sidebar.success("Files uploaded successfully!")

#     # Run processing
#     st.info("Processing video... please wait ⏳")
#     status, processed_video_path, speed_graph_path = process_video(video_path, json_path)
    
#     # Display result
#     st.success(status)
#     st.video(processed_video_path)

#     st.markdown("### 📈 Speed Variation Over Time")
#     st.image(speed_graph_path, use_column_width=True)

# else:
#     st.warning("Please upload both a video file and a speed JSON file.")

import streamlit as st
import tempfile
import os
from process_video_logic.process_video import process_video

st.set_page_config(page_title="Stop Sign Violation Detection", layout="wide")
st.title("🚦 Stop Sign Violation Detection with OCR + Direction + Speed Analysis")

# Upload section
st.sidebar.header("📤 Upload Files")
uploaded_video = st.sidebar.file_uploader("Upload Video (.mp4)", type=["mp4"])
uploaded_json = st.sidebar.file_uploader("Upload Speed JSON", type=["json"])

if uploaded_video and uploaded_json:
    # Save uploaded files to temp locations
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
        tmp_vid.write(uploaded_video.read())
        video_path = tmp_vid.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_json:
        tmp_json.write(uploaded_json.read())
        json_path = tmp_json.name

    st.sidebar.success("✅ Files uploaded successfully!")

    # Run processing
    st.info("🔍 Processing video... please wait ⏳")
    status, processed_video_path, speed_graph_path,txt_file_path = process_video(video_path, json_path)

    # Show result
    st.markdown("## 🧾 Violation Result")

    if "Violation" in status and "Rejected" not in status:
        st.markdown(f"<span style='color:red; font-size:24px;'>🛑 {status}</span>", unsafe_allow_html=True)
    elif "Rejected" in status:
        st.markdown(f"<span style='color:orange; font-size:24px;'>⚠️ {status}</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"<span style='color:green; font-size:24px;'>✅ {status}</span>", unsafe_allow_html=True)

    st.markdown("### 🎥 Processed Video")
    st.video(processed_video_path)

    st.markdown("### 📈 Speed Variation Over Time")
    st.image(speed_graph_path, use_column_width=True)

else:
    st.warning("⚠️ Please upload both a video file and a speed JSON file.")

# import streamlit as st
# import tempfile
# import os
# from process_video_logic.process_video import process_video

# # Set up the Streamlit app
# st.set_page_config(page_title="Stop Sign Violation Detection", layout="wide")
# st.title("🚦 Stop Sign Violation Detection with OCR + Direction + Speed Analysis")

# # Sidebar for file uploads
# st.sidebar.header("📤 Upload Files")
# uploaded_video = st.sidebar.file_uploader("Upload Video (.mp4)", type=["mp4"])
# uploaded_json = st.sidebar.file_uploader("Upload Speed JSON", type=["json"])

# if uploaded_video and uploaded_json:
#     # Save uploaded files to temporary locations
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
#         tmp_vid.write(uploaded_video.read())
#         video_path = tmp_vid.name

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_json:
#         tmp_json.write(uploaded_json.read())
#         json_path = tmp_json.name

#     st.sidebar.success("✅ Files uploaded successfully!")

#     # Run the video processing
#     st.info("🔍 Processing video... please wait ⏳")
#     try:
#         status, processed_video_path, speed_graph_path = process_video(video_path, json_path)

#         # Display results
#         st.markdown("## 🧾 Violation Result")
#         if "Violation" in status and "Rejected" not in status:
#             st.markdown(f"<span style='color:red; font-size:24px;'>🛑 {status}</span>", unsafe_allow_html=True)
#         elif "Rejected" in status:
#             st.markdown(f"<span style='color:orange; font-size:24px;'>⚠️ {status}</span>", unsafe_allow_html=True)
#         else:
#             st.markdown(f"<span style='color:green; font-size:24px;'>✅ {status}</span>", unsafe_allow_html=True)

#         # Display processed video
#         st.markdown("### 🎥 Processed Video")
#         st.video(processed_video_path)

#         # Display speed graph
#         st.markdown("### 📈 Speed Variation Over Time")
#         st.image(speed_graph_path, use_column_width=True)

#     except Exception as e:
#         st.error(f"❌ An error occurred during processing: {e}")

# else:
#     st.warning("⚠️ Please upload both a video file and a speed JSON file.")

