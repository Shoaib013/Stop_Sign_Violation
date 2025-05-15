import cv2
import torch
import pytesseract
from ultralytics import YOLO
import numpy as np
from collections import deque
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
# Applying OCR
def apply_ocr(image,box):
    x1,y1,x2,y2 = map(int,box)
    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) # added new
    gray = cv2.threshold(gray,150,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] # added new
    text = pytesseract.image_to_string(roi, config='--psm 6').strip().lower()
    # text = text.replace("\m", " ").replace("\x0c", "")
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
def estimate_direction_from_lane_lines(m_values,conf_values,conf_threshold=0.3):
    filtered_m = [m for m,conf in zip(m_values,conf_values) if conf >= conf_threshold]
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

        print(f"Left Slope:{left_slopes}")
        print(f"Right Slope:{right_slopes}")
        print(f"Avg Left Slope:{avg_left_slope:.2f},Avg Right Slope:{avg_right_slope:.2f},Diff:{diff:.2f}")

        if diff > 0.2:
            return "Right"
        elif diff < -0.2:
            return "Left"
    # if len(left_slopes) > 0 and len(right_slopes) > 0:
    #     avg_left_slope = sum(left_slopes) / len(left_slopes)
    #     avg_right_slope = sum(right_slopes) / len(right_slopes)
    #     if avg_left_slope < -0.3 and avg_right_slope > 0.3:
    #         return "Right"
    #     elif avg_left_slope < -0.3:
    #         return "Left"
    #     elif avg_right_slope > 0.3:
    #         return "Right"
    return "Straight"
# def extract_speed_from_json(json_path):
#     with open(json_path, 'r') as f:
#         data = json.load(f)
#     try:
#         samples = data[0]['samples']
#     except (IndexError, KeyError):
#         raise ValueError("Invalid JSON structure. Ensure 'samples' exists in the JSON file.")
#     speed_array = []
#     for sample in samples:  # Iterate over the list
#         speed_array.append(sample["eld"]["vehicle"]["road_speed_smooth_kph"])
#     return speed_array

# VG5

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

# Hubble is below one

def extract_speed_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    try:
        speed_array = []
        for i in range(len(data[0]['dpe'])):
            if data[0]['dpe'][i]['type'] == 13:
                # print("Found DPE with type 13")
                try:
                    start_time = datetime.strptime(data[0]['dpe'][i]['start_time'], "%Y-%m-%dT%H:%M:%S.%fZ")
                except ValueError:
                    start_time = datetime.strptime(data[0]['dpe'][i]['start_time'], "%Y-%m-%dT%H:%M:%SZ")
                try:
                    end_time = datetime.strptime(data[0]['dpe'][i]['end_time'], "%Y-%m-%dT%H:%M:%S.%fZ")
                except ValueError:
                    end_time = datetime.strptime(data[0]['dpe'][i]['end_time'], "%Y-%m-%dT%H:%M:%SZ")
                # print("Start time:", start_time)
                # print("End time:", end_time)
        samples = data[0]['samples']
        
        for sample in samples:
            try:
               frame_time = datetime.strptime(sample['perception_engine']['time'], "%Y-%m-%dT%H:%M:%S.%fZ")
            except ValueError:
                try:
                    frame_time = datetime.strptime(sample['perception_engine']['time'], "%Y-%m-%dT%H:%M:%SZ")
                except KeyError:
                        print(f"Missing 'time' key in sample: {sample}")
            if start_time <= frame_time <= end_time:
                speed_array.append(sample['ae_frame_data']['processed_speed_kph'])
        
    except (IndexError, KeyError):
        raise ValueError("Invalid JSON structure. Ensure 'samples' exists in the JSON file.")
    return speed_array




# def estimate_direction(trail):
#     if len(trail) < 2:
#         return None
#     dx = trail[-1][0] - trail[0][0]
#     if abs(dx) < 5:
#         return "Straight"
#     return "Right" if dx > 0 else "Left"

def plot_speed_graph(speeds, output_path):
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(speeds)), speeds, color='skyblue')
    plt.xlabel('Frame Index')
    plt.ylabel('Speed (kph)')
    plt.title('Speed Variation Over Time')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# def process_video(video_path, json_path, model_path='/home/shoaibkhan/Desktop/YOLO_SSV_Model/Complete_Project/weights/best.pt'):
#     model = YOLO(model_path)
#     speeds = extract_speed_from_json(json_path)
#     video_name = os.path.splitext(os.path.basename(video_path))[0]

#     # Define unique output file paths based on the video name
#     results_dir = "/home/shoaibkhan/Desktop/YOLO_SSV_Model/Complete_Project/results"
#     os.makedirs(results_dir, exist_ok=True)
#     processed_video_path = os.path.join(results_dir, f"{video_name}_processed.mp4")
#     speed_graph_path = os.path.join(results_dir, f"{video_name}_speed_graph.png")
#     txt_file_path = os.path.join(results_dir, f"{video_name}_detected_classes.txt")

#     # Plot the speed bar graph
#     plot_speed_graph(speeds, speed_graph_path)

#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width, height = int(cap.get(3)), int(cap.get(4))

#     # filename = os.path.basename(video_path).replace('.mp4', '_processed.mp4')
#     # output_path = os.path.join("results", filename)
#     # os.makedirs("results", exist_ok=True)

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))

#     vehicle_trail = deque(maxlen=int(fps * 3))
#     ocr_type = None
#     violation = False
#     stop_frame_index = -1
#     direction = None
#     frame_index = 0
#     violation_reason = "No Violation"
#     txt_file_path = os.path.join("results", "detected_classes.txt")
#     with open(txt_file_path, mode="w") as txt_file:
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             results = model(frame)[0]
#             detections = results.boxes.data.cpu().numpy() if results.boxes is not None else []

#             speed = speeds[frame_index] if frame_index < len(speeds) else 0
#             cv2.putText(frame, f"Speed: {speed:.1f} kph", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
#             detected_classes = []
#             for det in detections:
#                 x1, y1, x2, y2, conf, cls_id = det[:6]
#                 cls_id = int(cls_id)
#                 label = model.names[cls_id]
#                 color = (0, 255, 0) if 'stop' in label else (255, 0, 0)

#                 cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
#                 cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#                 detected_classes.append(label)
#                 if label == 'stop_sign':
#                     stop_frame_index = frame_index
#                     x1, y1, x2, y2 = map(int, [x1, y1, x2, y2]) # New added from here
#                     ocr_box = (x1, y2, x2, min(y2 + (y2 - y1), frame.shape[0]))  # box below the stop sign
#                     text = apply_ocr(frame, ocr_box)

#                     if "except" in text.lower():
#                         ocr_type = "except_right_turn"
#                     elif "right turn only" in text.lower():
#                         ocr_type = "right_turn_only"
#                     elif "arrow" in text.lower():
#                         ocr_type = "arrow_right_turn_only"
        
#                     cv2.putText(frame, f"OCR: {text}", (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3) # New added till here


#                 if label in ["right_turn_only", "except_right_turn", "arrow_right_turn_only"]:
#                     text = apply_ocr(frame, (x1, y1, x2, y2))
#                     ocr_type = label
#                     cv2.putText(frame, f"OCR: {text}", (int(x1), int(y2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#                 if label == "vehicle":
#                     cx, cy = int((x1 + x2) // 2), int((y1 + y2) // 2)
#                     vehicle_trail.append((cx, cy))
#                     cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
#             if ocr_type:
#                 cv2.putText(frame, f"OCR Type: {ocr_type}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

#             # Estimate direction
#             # direction = estimate_direction(vehicle_trail)
#             # if direction:
#             #     print(f"Frame {frame_index}: Direction: {direction}")
#             #     cv2.putText(frame, f"Dir: {direction}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#             # else: 
#             #     print(f"Frame {frame_index}: Direction: Not enough data")
#             try:
#                 m_values, c_values, conf_values = get_lane_lines_for_frame(json_path, frame_index)
#             except ValueError as e:
#                 print(f"Frame {frame_index}: {e}")
#                 m_values, c_values, conf_values = [], [], []
#             direction = estimate_direction_from_lane_lines(m_values, conf_values)
#             print(f"Frame {frame_index}: Direction: {direction}")
#             cv2.putText(frame, f"Dir: {direction}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#             # Check for violation
#             if stop_frame_index != -1 and frame_index == stop_frame_index + int(fps * 3):
#                 speeds_before = speeds[max(0, stop_frame_index - int(fps) * 3):stop_frame_index]
#                 min_speed = min(speeds_before) if speeds_before else 999

#                 if ocr_type is None:
#                     violation = min_speed > 8
#                 elif ocr_type in ["right_turn_only", "arrow_right_turn_only"]:
#                     violation = min_speed > 8 and direction == "Right"
#                 elif ocr_type == "except_right_turn":
#                     violation = min_speed > 8 and direction != "Right"

#             # if frame_index == stop_frame_index + int(fps * 3) + 1:
#             #     status_text = "Violation Detected!" if violation else "No Violation"
#             #     color = (0, 0, 255) if violation else (0, 255, 0)
#             #     cv2.putText(frame, status_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

#             if frame_index == stop_frame_index + int(fps * 3) + 1:
#                 if violation:
#                     if ocr_type in ["right_turn_only", "arrow_right_turn_only"] and direction == "Right":
#                         violation_reason = "Violation Rejected: Right Turn Only"
#                     elif ocr_type == "except_right_turn" and direction == "Right":
#                         violation_reason = "Violation Rejected: Except Right Turn"
#                     else:
#                         violation_reason = "Violation: Did not slow before stop sign"
#                 else:
#                     violation_reason = "No Violation"

#                 color = (0, 0, 255) if "Violation" in violation_reason and "Rejected" not in violation_reason else (0, 255, 0)
#                 y_offset = 110
#                 for line in violation_reason.split(": "):
#                     cv2.putText(frame, line.strip(), (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
#                     y_offset += 40  # spacing between lines
#             if detected_classes:
#                 txt_file.write(f"Frame {frame_index}: Detected Classes: {', '.join(detected_classes)}\n, OCR Type: {ocr_type}\n")
#             out.write(frame)
#             frame_index += 1

#         cap.release()
#         out.release()

#     return violation_reason, processed_video_path, speed_graph_path,txt_file_path




# # New Updated Code
# def process_video(video_path, json_path, model_path='/home/shoaibkhan/Desktop/YOLO_SSV_Model/Complete_Project/weights/best.pt'):
#     model = YOLO(model_path)
#     speeds = extract_speed_from_json(json_path)
#     video_name = os.path.splitext(os.path.basename(video_path))[0]

#     # Define unique output file paths based on the video name
#     results_dir = "/home/shoaibkhan/Desktop/YOLO_SSV_Model/Complete_Project/results"
#     os.makedirs(results_dir, exist_ok=True)
#     processed_video_path = os.path.join(results_dir, f"{video_name}_processed.mp4")
#     speed_graph_path = os.path.join(results_dir, f"{video_name}_speed_graph.png")
#     txt_file_path = os.path.join(results_dir, f"{video_name}_detected_classes.txt")

#     # Plot the speed bar graph
#     plot_speed_graph(speeds, speed_graph_path)

#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))

#     vehicle_trail = deque(maxlen=int(fps * 3))
#     ocr_type = None
#     violation = False
#     stop_frame_index = -1
#     direction = None
#     frame_index = 0
#     violation_reason = "No Violation"

#     with open(txt_file_path, mode="w") as txt_file:
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 print(f"Error: Unable to read frame {frame_index}")
#                 break

#             # Debug: Check if the frame is being processed
#             print(f"Processing frame {frame_index}")

#             results = model(frame)[0]
#             detections = results.boxes.data.cpu().numpy() if results.boxes is not None else []

#             speed = speeds[frame_index] if frame_index < len(speeds) else 0
#             cv2.putText(frame, f"Speed: {speed:.1f} kph", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
#             detected_classes = []

#             for det in detections:
#                 x1, y1, x2, y2, conf, cls_id = det[:6]
#                 cls_id = int(cls_id)
#                 label = model.names[cls_id]
#                 color = (0, 255, 0) if 'stop' in label else (255, 0, 0)

#                 cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
#                 cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#                 detected_classes.append(label)

#             if detected_classes:
#                 txt_file.write(f"Frame {frame_index}: Detected Classes: {', '.join(detected_classes)}\n")

#             # Write the frame to the output video
#             out.write(frame)
#             frame_index += 1

#     cap.release()
#     out.release()

#     return violation_reason, processed_video_path, speed_graph_path, txt_file_path



def process_video(video_path, json_path, model_path='/home/shoaibkhan/Desktop/YOLO_SSV_Model/Complete_Project/weights/best.pt'):
    model = YOLO(model_path)
    speeds = extract_speed_from_json(json_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Define unique output file paths based on the video name
    results_dir = "/home/shoaibkhan/Desktop/YOLO_SSV_Model/Complete_Project/results"
    os.makedirs(results_dir, exist_ok=True)
    processed_video_path = os.path.join(results_dir, f"{video_name}_processed.mp4")
    speed_graph_path = os.path.join(results_dir, f"{video_name}_speed_graph.png")
    txt_file_path = os.path.join(results_dir, f"{video_name}_detected_classes.txt")

    # Plot the speed bar graph
    plot_speed_graph(speeds, speed_graph_path)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))

    vehicle_trail = deque(maxlen=int(fps * 3))
    ocr_type = None
    violation = False
    stop_frame_index = -1
    direction = None
    frame_index = 0
    violation_reason = "No Violation"

    with open(txt_file_path, mode="w") as txt_file:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Unable to read frame {frame_index}")
                break

            # Debug: Check if the frame is being processed
            # print(f"Processing frame {frame_index}")

            results = model(frame)[0]
            detections = results.boxes.data.cpu().numpy() if results.boxes is not None else []

            speed = speeds[frame_index] if frame_index < len(speeds) else 0
            cv2.putText(frame, f"Speed: {speed:.1f} kph", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
            detected_classes = []

            for det in detections:
                x1, y1, x2, y2, conf, cls_id = det[:6]
                cls_id = int(cls_id)
                label = model.names[cls_id]
                color = (0, 255, 0) if 'stop' in label else (255, 0, 0)

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                detected_classes.append(label)

                if label == 'stop_sign':
                    stop_frame_index = frame_index
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    ocr_box = (x1, y2, x2, min(y2 + (y2 - y1), frame.shape[0]))
                    text = apply_ocr(frame, ocr_box)

                    if "except" in text.lower():
                        ocr_type = "except_right_turn"
                    elif "right turn only" in text.lower():
                        ocr_type = "right_turn_only"
                    elif "arrow" in text.lower():
                        ocr_type = "arrow_right_turn_only"

                    # Display OCR text on the video
                    cv2.putText(frame, f"OCR: {text}", (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

            if detected_classes:
                txt_file.write(f"Frame {frame_index}: Detected Classes: {', '.join(detected_classes)}\n")

            # Check for violation
            if stop_frame_index != -1 and frame_index == stop_frame_index + int(fps * 3):
                speeds_before = speeds[max(0, stop_frame_index - int(fps) * 3):stop_frame_index]
                min_speed = min(speeds_before) if speeds_before else 999

                if ocr_type is None:
                    violation = min_speed > 8
                elif ocr_type in ["right_turn_only", "arrow_right_turn_only"]:
                    violation = min_speed > 8 and direction == "Right"
                elif ocr_type == "except_right_turn":
                    violation = min_speed > 8 and direction != "Right"

            # Display violation reason
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
                if ocr_type:
                    cv2.putText(frame, f"OCR Type: {ocr_type}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                for _ in range(int(fps * 3)):
                    out.write(frame)
            # Write the frame to the output video
            out.write(frame)
            frame_index += 1

    cap.release()
    out.release()

    return violation_reason, processed_video_path, speed_graph_path, txt_file_path