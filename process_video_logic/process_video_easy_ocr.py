# import cv2
# import torch
# import easyocr
# from ultralytics import YOLO
# import numpy as np
# from collections import deque
# import os
# import json
# import matplotlib.pyplot as plt


# def extract_speed_from_json(json_path):
#     with open(json_path, 'r') as f:
#         data = json.load(f)
#     try:
#         samples = data[0]['data']['telematic']['vehicle']['samples']
#     except (IndexError, KeyError):
#         raise ValueError("Invalid JSON structure. Ensure 'samples' exists in the JSON file.")
#     speed_array = []
#     for sample in samples:  # Iterate over the list
#         speed_array.append(sample["speed_kph"])
#     return speed_array

# def plot_speed_graph(speeds, output_path):
#     plt.figure(figsize=(10, 4))
#     plt.bar(range(len(speeds)), speeds, color='skyblue')
#     plt.xlabel('Frame Index')
#     plt.ylabel('Speed (kph)')
#     plt.title('Speed Variation Over Time')
#     plt.tight_layout()
#     plt.savefig(output_path)
#     plt.close()

# ## Initialize the EasyOCR
# def extract_text_from_roi(image,bbox,reader):
#     x1,y1,x2,y2 = bbox
#     roi = image[y1:y2, x1:x2]
#     results = reader.readtext(roi)
#     return [res[1].upper() for res in results]

# def detect_modifier(text_list):
#     for text in text_list:
#         if "Except Right Turn" in text:
#             return "except_right_turn"
#         elif "Right Turn Only" in text:
#             return "right_turn_only"
#     return None
# def process_video(video_path, json_path, model_path='/home/shoaibkhan/Desktop/YOLO_SSV_Model/Complete_Project/weights/best.pt'):
#     model = YOLO(model_path)
#     reader = easyocr.Reader(['en'], gpu=True)
#     speeds = extract_speed_from_json(json_path)

#     # Plot the speed bar graph
#     speed_graph_path = os.path.join("/home/shoaibkhan/Desktop/YOLO_SSV_Model/Complete_Project/results", "speed_graph.png")
#     plot_speed_graph(speeds, speed_graph_path)

#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width, height = int(cap.get(3)), int(cap.get(4))

#     filename = os.path.basename(video_path).replace('.mp4', '_processed.mp4')
#     output_path = os.path.join("results", filename)
#     os.makedirs("results", exist_ok=True)

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     vehicle_trail = deque(maxlen=int(fps * 3))
#     ocr_type = None
#     violation = False
#     stop_frame_index = -1
#     direction = None
#     frame_index = 0
#     violation_reason = "No Violation"
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         results = model(frame)[0]
#         detections = results.boxes.data.cpu().numpy() if results.boxes is not None else []

#         speed = speeds[frame_index] if frame_index < len(speeds) else 0
#         cv2.putText(frame, f"Speed: {speed:.1f} kph", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

#         for det in detections:
#             x1, y1, x2, y2, conf, cls_id = det[:6]
#             cls_id = int(cls_id)
#             label = model.names[cls_id]
#             color = (0, 255, 0) if 'stop' in label else (255, 0, 0)

#             x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
#             cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#             if label == 'stop_sign':
#                 stop_frame_index = frame_index
#                 roi_text = extract_text_from_roi(frame, (x1, y1, x2, y2), reader)
#                 modifier = detect_modifier(roi_text)
#                 ocr_type = modifier 
#                 for i,t in enumerate(roi_text):
#                     cv2.putText(frame, t, (x1, y1 + 20 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
#             if label == "vehicle":
#                 cx, cy = int((x1 + x2) // 2), int((y1 + y2) // 2)
#                 vehicle_trail.append((cx, cy))
#                 cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
#         if ocr_type:
#             cv2.putText(frame, f"OCR Type: {ocr_type}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

#         # Check for violation
#         if stop_frame_index != -1 and frame_index == stop_frame_index + int(fps * 3):
#             speeds_before = speeds[max(0, stop_frame_index - int(fps) * 3):stop_frame_index]
#             min_speed = min(speeds_before) if speeds_before else 999

#             if ocr_type is None:
#                 violation = min_speed > 8
#             elif ocr_type in ["right_turn_only", "arrow_right_turn_only"]:
#                 violation = min_speed > 8 and direction == "Right"
#             elif ocr_type == "except_right_turn":
#                 violation = min_speed > 8 and direction != "Right"

#         if frame_index == stop_frame_index + int(fps * 3) + 1:
#             if violation:
#                 if ocr_type in ["right_turn_only", "arrow_right_turn_only"] and direction == "Right":
#                    violation_reason = "Violation Rejected: Right Turn Only"
#                 elif ocr_type == "except_right_turn" and direction == "Right":
#                    violation_reason = "Violation Rejected: Except Right Turn"
#                 else:
#                    violation_reason = "Violation: Did not slow before stop sign"
#             else:
#                 violation_reason = "No Violation"

#             color = (0, 0, 255) if "Violation" in violation_reason and "Rejected" not in violation_reason else (0, 255, 0)
#             y_offset = 110
#             for line in violation_reason.split(": "):
#                 cv2.putText(frame, line.strip(), (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
#                 y_offset += 40  # spacing between lines

#         out.write(frame)
#         frame_index += 1

#     cap.release()
#     out.release()

#     return violation_reason, output_path, speed_graph_path


# import cv2
# import torch
# import easyocr
# from ultralytics import YOLO
# import numpy as np
# from collections import deque
# import os
# import json
# import matplotlib.pyplot as plt

# def extract_speed_from_json(json_path):
#     with open(json_path, 'r') as f:
#         data = json.load(f)
#     try:
#         samples = data[0]['data']['telematic']['vehicle']['samples']
#     except (IndexError, KeyError):
#         raise ValueError("Invalid JSON structure. Ensure 'samples' exists in the JSON file.")
#     speed_array = []
#     for sample in samples:
#         speed_array.append(sample["speed_kph"])
#     return speed_array

# def plot_speed_graph(speeds, output_path):
#     plt.figure(figsize=(10, 4))
#     plt.bar(range(len(speeds)), speeds, color='skyblue')
#     plt.xlabel('Frame Index')
#     plt.ylabel('Speed (kph)')
#     plt.title('Speed Variation Over Time')
#     plt.tight_layout()
#     plt.savefig(output_path)
#     plt.close()

# ## Updated OCR Function: Visualize & Save ROI
# def extract_text_from_roi(image, bbox, reader, frame_index=None):
#     x1, y1, x2, y2 = bbox
#     h, w = image.shape[:2]
#     pad = 10
#     x1 = max(x1 - pad, 0)
#     y1 = max(y1 - pad, 0)
#     x2 = min(x2 + pad, w)
#     y2 = min(y2 + pad, h)

#     roi = image[y1:y2, x1:x2]
#     roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

#     # Save ROI image for debugging
#     if frame_index is not None:
#         os.makedirs("ocr_debug_rois", exist_ok=True)
#         cv2.imwrite(f"ocr_debug_rois/frame_{frame_index}_roi.png", roi)

#     results = reader.readtext(roi_gray)
#     text_list = [res[1].upper() for res in results if res[2] > 0.5]

#     # Optional: fallback to Tesseract
#     if not text_list:
#         try:
#             import pytesseract
#             text_list = [pytesseract.image_to_string(roi_gray).strip().upper()]
#         except Exception as e:
#             print("Fallback OCR failed:", e)

#     return text_list

# def detect_modifier(text_list):
#     for text in text_list:
#         if "EXCEPT RIGHT TURN" in text:
#             return "except_right_turn"
#         elif "RIGHT TURN ONLY" in text:
#             return "right_turn_only"
#     return None

# def process_video(video_path, json_path, model_path='/home/shoaibkhan/Desktop/YOLO_SSV_Model/Complete_Project/weights/best.pt'):
#     model = YOLO(model_path)
#     reader = easyocr.Reader(['en'], gpu=True)
#     speeds = extract_speed_from_json(json_path)

#     # Plot the speed bar graph
#     speed_graph_path = os.path.join("results", "speed_graph.png")
#     plot_speed_graph(speeds, speed_graph_path)

#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width, height = int(cap.get(3)), int(cap.get(4))

#     filename = os.path.basename(video_path).replace('.mp4', '_processed.mp4')
#     output_path = os.path.join("results", filename)
#     os.makedirs("results", exist_ok=True)

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     vehicle_trail = deque(maxlen=int(fps * 3))
#     ocr_type = None
#     violation = False
#     stop_frame_index = -1
#     direction = None
#     frame_index = 0
#     violation_reason = "No Violation"

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         results = model(frame)[0]
#         detections = results.boxes.data.cpu().numpy() if results.boxes is not None else []

#         speed = speeds[frame_index] if frame_index < len(speeds) else 0
#         cv2.putText(frame, f"Speed: {speed:.1f} kph", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

#         for det in detections:
#             x1, y1, x2, y2, conf, cls_id = det[:6]
#             cls_id = int(cls_id)
#             label = model.names[cls_id]
#             color = (0, 255, 0) if 'stop' in label else (255, 0, 0)

#             x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#             if label == 'stop_sign':
#                 stop_frame_index = frame_index
#                 roi_text = extract_text_from_roi(frame, (x1, y1, x2, y2), reader, frame_index)
#                 modifier = detect_modifier(roi_text)
#                 ocr_type = modifier 
#                 for i, t in enumerate(roi_text):
#                     cv2.putText(frame, t, (x1, y1 + 20 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

#             if label == "vehicle":
#                 cx, cy = int((x1 + x2) // 2), int((y1 + y2) // 2)
#                 vehicle_trail.append((cx, cy))
#                 cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

#         if ocr_type:
#             cv2.putText(frame, f"OCR Type: {ocr_type}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

#         # Check for violation
#         if stop_frame_index != -1 and frame_index == stop_frame_index + int(fps * 3):
#             speeds_before = speeds[max(0, stop_frame_index - int(fps) * 3):stop_frame_index]
#             min_speed = min(speeds_before) if speeds_before else 999

#             if ocr_type is None:
#                 violation = min_speed > 8
#             elif ocr_type in ["right_turn_only", "arrow_right_turn_only"]:
#                 violation = min_speed > 8 and direction == "Right"
#             elif ocr_type == "except_right_turn":
#                 violation = min_speed > 8 and direction != "Right"

#         if frame_index == stop_frame_index + int(fps * 3) + 1:
#             if violation:
#                 if ocr_type in ["right_turn_only", "arrow_right_turn_only"] and direction == "Right":
#                     violation_reason = "Violation Rejected: Right Turn Only"
#                 elif ocr_type == "except_right_turn" and direction == "Right":
#                     violation_reason = "Violation Rejected: Except Right Turn"
#                 else:
#                     violation_reason = "Violation: Did not slow before stop sign"
#             else:
#                 violation_reason = "No Violation"

#             color = (0, 0, 255) if "Violation" in violation_reason and "Rejected" not in violation_reason else (0, 255, 0)
#             y_offset = 110
#             for line in violation_reason.split(": "):
#                 cv2.putText(frame, line.strip(), (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
#                 y_offset += 40

#         out.write(frame)
#         frame_index += 1

#     cap.release()
#     out.release()

#     return violation_reason, output_path, speed_graph_path

import cv2
import torch
import pytesseract
from ultralytics import YOLO
import numpy as np
from collections import deque
import os
import json
import matplotlib.pyplot as plt


def extract_speed_from_json(json_path):
     with open(json_path, 'r') as f:
         data = json.load(f)
     try:
         samples = data[0]['data']['telematic']['vehicle']['samples']
     except (IndexError, KeyError):
         raise ValueError("Invalid JSON structure. Ensure 'samples' exists in the JSON file.")
     speed_array = []
     for sample in samples:
         speed_array.append(sample["speed_kph"])
     return speed_array
# Applying OCR
def apply_ocr(image,box):
    x1,y1,x2,y2 = map(int,box)
    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray,150,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(roi, config='--psm 6').strip().lower()
    return text

def get_lane_lines_for_frame(json_path, frame_index):
    with open(json_path, 'r') as f:
        data = json.load(f)
    try:
        samples = data[frame_index]['perception_engine']['urm_data']['lane_lines']
    except (IndexError, KeyError):
        raise ValueError(f"Invalid JSON structure or missing data for frame {frame_index}.")
    
    m = []
    c = []
    avg_keypoints_conf = []
    for lane_line in samples:
        m.append(lane_line['m'])
        c.append(lane_line['c'])
        avg_keypoints_conf.append(lane_line['avg_keypoints_conf'])
    
    return m, c, avg_keypoints_conf

def estimate_direction_from_lane_lines(m_values, conf_values, conf_threshold=0.3):
    filtered_m = [m for m, conf in zip(m_values, conf_values) if conf >= conf_threshold]
    if len(filtered_m) == 0:
        return "Straight"
    left_slopes = [m for m in filtered_m if m < -0.05]
    right_slopes = [m for m in filtered_m if m > 0.05]
    if len(right_slopes) > 0 and len(left_slopes) == 0:
        return "Right"
    elif len(left_slopes) > 0 and len(right_slopes) == 0:
        return "Left"
    elif len(left_slopes) > 0 and len(right_slopes) > 0:
        avg_left_slope = sum(left_slopes) / len(left_slopes)
        avg_right_slope = sum(right_slopes) / len(right_slopes)
        diff = avg_right_slope - avg_left_slope
        if diff > 0.2:
            return "Right"
        elif diff < -0.2:
            return "Left"
    return "Straight"

def plot_speed_graph(speeds, output_path):
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(speeds)), speeds, color='skyblue')
    plt.xlabel('Frame Index')
    plt.ylabel('Speed (kph)')
    plt.title('Speed Variation Over Time')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def process_video(video_path, json_path, model_path='/home/shoaibkhan/Desktop/YOLO_SSV_Model/Complete_Project/weights/best.pt'):
    model = YOLO(model_path)
    speeds = extract_speed_from_json(json_path)

    # Plot the speed bar graph
    speed_graph_path = os.path.join("/home/shoaibkhan/Desktop/YOLO_SSV_Model/Complete_Project/results", "speed_graph.png")
    plot_speed_graph(speeds, speed_graph_path)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(3)), int(cap.get(4))

    filename = os.path.basename(video_path).replace('.mp4', '_processed.mp4')
    output_path = os.path.join("results", filename)
    os.makedirs("results", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    vehicle_trail = deque(maxlen=int(fps * 3))
    ocr_type = None
    violation = False
    stop_frame_index = -1
    direction = None
    frame_index = 0
    violation_reason = "No Violation"

    # Create and open the text file to save the output
    txt_file_path = os.path.join("results", "detected_classes.txt")
    with open(txt_file_path, 'w') as txt_file:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)[0]
            detections = results.boxes.data.cpu().numpy() if results.boxes is not None else []

            speed = speeds[frame_index] if frame_index < len(speeds) else 0
            cv2.putText(frame, f"Speed: {speed:.1f} kph", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

            detected_classes = []  # Store detected classes for this frame

            for det in detections:
                x1, y1, x2, y2, conf, cls_id = det[:6]
                cls_id = int(cls_id)
                label = model.names[cls_id]
                color = (0, 255, 0) if 'stop' in label else (255, 0, 0)

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Append detected class to the list
                detected_classes.append(label)

            # Write the detected classes for this frame to the text file
            if detected_classes:
                txt_file.write(f"Frame {frame_index}: Detected Classes: {', '.join(detected_classes)}\n")

            out.write(frame)
            frame_index += 1

        cap.release()
        out.release()

    return violation_reason, output_path, speed_graph_path, txt_file_path  # Return the path to the text file
