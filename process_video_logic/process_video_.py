import cv2
import torch
import pytesseract
from ultralytics import YOLO
import numpy as np
from collections import deque
import os
import json
import matplotlib.pyplot as plt

# Applying OCR
def apply_ocr(image,box):
    x1,y1,x2,y2 = map(int,box)
    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) # added new
    gray = cv2.threshold(gray,150,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] # added new
    text = pytesseract.image_to_string(gray, config='--psm 6').strip().lower()
    # text = text.replace("\m", " ").replace("\x0c", "")
    return text

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
def get_speed_for_frame(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    samples = data[0]['data']['telematic']['vehicle']['samples']
    return [sample['speed_kph'] for sample in samples]

def extract_heading_from_json(json_path):
    with open(json_path,'r') as f:
        data = json.load(f)
    samples = data[0]['data']['telematic']['gnss']['samples']
    return [sample['heading_deg'] for sample in samples]

def estimate_direction_from_heading(heading_values,current_frame_index,threshold_deg=5):
    if (
        heading_values is None or
        not isinstance(heading_values, (list,np.ndarray)) or
        current_frame_index is None or
        current_frame_index < 5 or
        current_frame_index >= len(heading_values)
    ):
        return 'Unknown'
    
    if current_frame_index < 5 or current_frame_index >= len(heading_values):
        return 'Straight'
    past_heading = heading_values[current_frame_index - 5]
    current_heading = heading_values[current_frame_index]
    delta_heading = (current_heading - past_heading + 180) % 360 - 180
    if delta_heading > threshold_deg:
        return 'Right'
    elif delta_heading < -threshold_deg:
        return 'Left'
    else:
        return 'Straight'

def overlay_text_with_background(frame,text,position,font_scale=0.7,thickness= 2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    text_w, text_h = text_size

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y - text_h - 10), (x + text_w + 10, y + 5), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
    cv2.putText(frame, text, (x + 5, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame
def draw_direction_arrow(frame, direction, position=(100, 150), length=60, color=(0, 255, 0), thickness=4):
    x, y = position
    if direction == "Left":
        end_point = (x - length, y)
    elif direction == "Right":
        end_point = (x + length, y)
    else:  # Straight
        end_point = (x, y - length)
    
    cv2.arrowedLine(frame, (x, y), end_point, color, thickness, tipLength=0.4)
    return frame
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

def process_video(video_path, json_path, model_path='/home/shoaibkhan/Desktop/YOLO_SSV_Model/Complete_Project/weights/best.pt'):
    model = YOLO(model_path)
    speeds = get_speed_for_frame(json_path)
    heading_values = extract_heading_from_json(json_path)
    # Plot the speed bar graph
    speed_graph_path = os.path.join("/home/shoaibkhan/Desktop/YOLO_SSV_Model/Complete_Project/results", "speed_graph.png")
    plot_speed_graph(speeds, speed_graph_path)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(3)), int(cap.get(4))
    print("FPS of Video", fps)
    filename = os.path.basename(video_path).replace('.mp4', '_processed.mp4')
    output_path = os.path.join("results", filename)
    print("Output path", output_path)
    os.makedirs("results", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    vehicle_trail = deque(maxlen=int(fps * 3))
    ocr_type = None
    violation = False
    stop_frame_index = -1
    direction = None
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

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if label == 'stop_sign':
                stop_frame_index = frame_index
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2]) # New added from here
                ocr_box = (x1, y2, x2, min(y2 + (y2 - y1), frame.shape[0]))  # box below the stop sign
                text = apply_ocr(frame, ocr_box)

                if "except" in text.lower():
                    ocr_type = "except_right_turn"
                elif "right turn only" in text.lower():
                    ocr_type = "right_turn_only"
                elif "arrow" in text.lower():
                    ocr_type = "arrow_right_turn_only"
    
                cv2.putText(frame, f"OCR: {text}", (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2) # New added till here


            if label in ["right_turn_only", "except_right_turn", "arrow_right_turn_only"]:
                text = apply_ocr(frame, (x1, y1, x2, y2))
                ocr_type = label
                cv2.putText(frame, f"OCR: {text}", (int(x1), int(y2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # if label == "vehicle":
            #     cx, cy = int((x1 + x2) // 2), int((y1 + y2) // 2)
            #     vehicle_trail.append((cx, cy))
            #     cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        if ocr_type:
            cv2.putText(frame, f"OCR Type: {ocr_type}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        direction = estimate_direction_from_heading(heading_values, frame_index)
        print(f"Frame {frame_index}: Direction: {direction}")
        cv2.putText(frame, f"Dir: {direction}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        # Check for violation
        if stop_frame_index != -1 and frame_index == stop_frame_index + int(fps * 3):
            print(f"Stop sign detected at frame {stop_frame_index}.")
            print(f"Current frame index: {frame_index}")
            print(f"Target frame index: {stop_frame_index + int(fps * 3)}")
            speeds_before = speeds[max(0, stop_frame_index - int(fps) * 3):stop_frame_index]
            print(f"Speeds before stop sign: {speeds_before}")
            if not speeds_before:
                print("No speed data available before stop sign.")
            min_speed = min(speeds_before) if speeds_before else 0
            print(f"Minimum speed before stop sign: {min_speed}")
            if ocr_type is None:
                violation = min_speed > 8
            elif ocr_type in ["right_turn_only", "arrow_right_turn_only"]:
                violation = min_speed > 8 and direction == "Right"
            elif ocr_type == "except_right_turn":
                violation = min_speed > 8 and direction != "Right"
            print(f"Violation Detected :{violation}")

        if frame_index == stop_frame_index + int(fps * 2) + 1:
            print(f"Checking for violation at frame {frame_index}, target frame: {stop_frame_index + int(fps * 2) + 1}")
            print(f'violation: {violation}')
            print(f'ocr_type: {ocr_type}')
            print(f'direction: {direction}')
            if violation:
                if ocr_type in ["right_turn_only", "arrow_right_turn_only"] and direction == "Right":
                   violation_reason = "Violation Rejected: Right Turn Only"
                elif ocr_type == "except_right_turn" and direction == "Right":
                   violation_reason = "Violation Rejected: Except Right Turn"
                else:
                   violation_reason = "Violation: Did not slow before stop sign"
            else:
                violation_reason = "No Violation"
            print(f"Final Violation Reason: {violation_reason}")

            color = (0, 0, 255) if "Violation" in violation_reason and "Rejected" not in violation_reason else (0, 255, 0)
            y_offset = 110
            for line in violation_reason.split(": "):
                cv2.putText(frame, line.strip(), (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                y_offset += 40  # spacing between lines
        frame = overlay_text_with_background(frame, f"Speed: {speed:.2f} kph", (10, 30))
        frame = overlay_text_with_background(frame, f"Direction: {direction}", (10, 70))
        # Overlay heading value for the current frame
        if isinstance(heading_values, list) and frame_index < len(heading_values):
            current_heading = heading_values[frame_index]
            frame = overlay_text_with_background(frame, f"Heading: {current_heading:.2f}°", (10, 110))
        else:
            frame = overlay_text_with_background(frame, "Heading: Unknown", (10, 110))
        
        # Draw arrow
        frame = draw_direction_arrow(frame, direction, position=(width - 120, height - 80))
        out.write(frame)
        frame_index += 1

    cap.release()
    out.release()

    return violation_reason, output_path, speed_graph_path
