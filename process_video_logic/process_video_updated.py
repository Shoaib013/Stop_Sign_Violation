import cv2
import torch
import pytesseract
from ultralytics import YOLO
import numpy as np
from collections import deque
import os
import json
import matplotlib.pyplot as plt

# OCR function
def apply_ocr(image, box):
    x1, y1, x2, y2 = map(int, box)
    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(roi, config='--psm 6').strip().lower()
    return text

# Get lane line data for a specific frame
def get_lane_lines_for_frame(json_path, frame_index):
    with open(json_path, 'r') as f:
        data = json.load(f)
    try:
        samples = data[frame_index]['perception_engine']['urm_data']['lane_lines']
    except (IndexError, KeyError):
        raise ValueError(f"Invalid JSON structure or missing data for frame {frame_index}.")
    
    m, c, avg_keypoints_conf = [], [], []
    for lane_line in samples:
        m.append(lane_line['m'])
        c.append(lane_line['c'])
        avg_keypoints_conf.append(lane_line['avg_keypoints_conf'])
    
    return m, c, avg_keypoints_conf

# Estimate driving direction
def estimate_direction_from_lane_lines(m_values, conf_values, conf_threshold=0.3):
    filtered_m = [m for m, conf in zip(m_values, conf_values) if conf >= conf_threshold]
    if not filtered_m:
        return "Straight"
    
    left_slopes = [m for m in filtered_m if m < -0.05]
    right_slopes = [m for m in filtered_m if m > 0.05]
    
    if right_slopes and not left_slopes:
        return "Right"
    elif left_slopes and not right_slopes:
        return "Left"
    elif left_slopes and right_slopes:
        avg_left = sum(left_slopes) / len(left_slopes)
        avg_right = sum(right_slopes) / len(right_slopes)
        diff = avg_right - avg_left
        if diff > 0.2:
            return "Right"
        elif diff < -0.2:
            return "Left"
    
    return "Straight"

# Extract speed from JSON
def extract_speed_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    try:
        samples = data[0]['data']['telematic']['vehicle']['samples']
    except (IndexError, KeyError):
        raise ValueError("Invalid JSON structure. Ensure 'samples' exists.")
    
    return [sample["speed_kph"] for sample in samples]

# Plot speed bar graph
def plot_speed_graph(speeds, output_path):
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(speeds)), speeds, color='skyblue')
    plt.xlabel('Frame Index')
    plt.ylabel('Speed (kph)')
    plt.title('Speed Variation Over Time')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# MAIN VIDEO PROCESSING FUNCTION
def process_video(video_path, json_path, model_path='weights/best.pt', output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    model = YOLO(model_path)
    speeds = extract_speed_from_json(json_path)

    # Output file paths
    video_name = os.path.basename(video_path).replace('.mp4', '_processed.mp4')
    output_video_path = os.path.join(output_dir, video_name)
    speed_graph_path = os.path.join(output_dir, "speed_graph.png")

    plot_speed_graph(speeds, speed_graph_path)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(3)), int(cap.get(4))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    vehicle_trail = deque(maxlen=int(fps * 3))
    ocr_type = None
    stop_frame_index = -1
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        detections = results.boxes.data.cpu().numpy() if results.boxes is not None else []

        speed = speeds[frame_index] if frame_index < len(speeds) else 0
        cv2.putText(frame, f"Speed: {speed:.1f} kph", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det[:6]
            cls_id = int(cls_id)
            label = model.names[cls_id]
            color = (0, 255, 0) if 'stop' in label else (255, 0, 0)

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if label == 'stop_sign':
                stop_frame_index = frame_index
                ocr_box = (x1, y2, x2, min(y2 + (y2 - y1), frame.shape[0]))
                text = apply_ocr(frame, ocr_box)
                if "except" in text:
                    ocr_type = "except_right_turn"
                elif "right turn only" in text:
                    ocr_type = "right_turn_only"
                elif "arrow" in text:
                    ocr_type = "arrow_right_turn_only"
                cv2.putText(frame, f"OCR: {text}", (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            if label in ["vehicle"]:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                vehicle_trail.append((cx, cy))
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        if ocr_type:
            cv2.putText(frame, f"OCR Type: {ocr_type}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Estimate direction
        try:
            m_values, _, conf_values = get_lane_lines_for_frame(json_path, frame_index)
        except ValueError as e:
            print(f"Frame {frame_index}: {e}")
            m_values, conf_values = [], []
        
        direction = estimate_direction_from_lane_lines(m_values, conf_values)
        cv2.putText(frame, f"Dir: {direction}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        if frame_index == stop_frame_index + int(fps * 3) + 1:
                if violation:
                    if ocr_type in ["right_turn_only", "arrow_right_turn_only"] and direction == "Right":
                        violation_reason = "Violation Rejected: Right Turn Only"
                    elif ocr_type == "except_right_turn" and direction == "Right":
                        violation_reason = "Violation Rejected: Except Right Turn"
                    else:
                        violation_reason = "Violation: Did not slow before stop sign"
                else:
                    violation_reason = "No Violation"

                color = (0, 0, 255) if "Violation" in violation_reason and "Rejected" not in violation_reason else (0, 255, 0)
                y_offset = 110
                for line in violation_reason.split(": "):
                    cv2.putText(frame, line.strip(), (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    y_offset += 40  # spacing between lines
        out.write(frame)
        frame_index += 1

    cap.release()
    out.release()
    return violation_reason,output_video_path, speed_graph_path
