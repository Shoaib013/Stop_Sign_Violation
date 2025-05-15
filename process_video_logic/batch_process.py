# import os 
# import csv
# from tqdm import tqdm
# from process_video import process_video

# video_dir = "/home/shoaibkhan/Desktop/SSV_Model_Testing_Videos_JSONs/2_fps"
# json_dir = "/home/shoaibkhan/Desktop/SSV_Model_Testing_Videos_JSONs/jsons/"
# output_csv = "results_summary.csv"

# if not os.path.exists(video_dir):
#     raise FileNotFoundError(f"Output CSV file '{video_dir}' does not exist. Please check the path.")
# if not os.path.exists(json_dir):
#     raise FileNotFoundError(f"Output CSV file '{json_dir}' does not exist. Please check the path.")

# video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
# json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

# if len(video_files) != len(json_files):
#     raise ValueError("The number of video files and JSON files do not match. Please check the directories.")

# with open(output_csv, mode="w", newline="") as csv_file:
#     csv_writer = csv.writer(csv_file)
#     # Write the header row
#     csv_writer.writerow(["Video Name", "Violation Reason", "Processed Video Path", "Speed Graph Path", "Detected Classes File"])

#     # Process each video and JSON file
#     for video_file, json_file in tqdm(zip(video_files, json_files), total=len(video_files), desc="Processing Videos"):
#         video_path = os.path.join(video_dir, video_file)
#         json_path = os.path.join(json_dir, json_file)

#         try:
#             # Process the video and JSON file
#             violation_reason, processed_video_path, speed_graph_path, txt_file_path = process_video(video_path, json_path)

#             # Write the results to the CSV file
#             csv_writer.writerow([video_file, violation_reason, processed_video_path, speed_graph_path, txt_file_path])

#         except Exception as e:
#             print(f"Error processing {video_file} with {json_file}: {e}")
#             csv_writer.writerow([video_file, f"Error: {e}", "", "", ""])

# print(f"Processing complete. Results saved to {output_csv}.")





import os
import csv
from tqdm import tqdm
from process_video import process_video

# Define directories
video_dir = "/home/shoaibkhan/Desktop/SSV_Model_Testing_Videos_JSONs/30_fps/"
json_dir = "/home/shoaibkhan/Desktop/SSV_Model_Testing_Videos_JSONs/30_fpsjson/"
results_dir = "/home/shoaibkhan/Desktop/YOLO_SSV_Model/Complete_Project/results"
output_csv = os.path.join(results_dir, "results_summary.csv")

# Ensure directories exist
if not os.path.exists(video_dir):
    raise FileNotFoundError(f"Video directory '{video_dir}' does not exist. Please check the path.")
if not os.path.exists(json_dir):
    raise FileNotFoundError(f"JSON directory '{json_dir}' does not exist. Please check the path.")

# Ensure results directory exists
os.makedirs(results_dir, exist_ok=True)

# Get video and JSON files
video_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')])
json_files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])

# Check for empty directories
if not video_files:
    raise ValueError(f"No video files found in directory '{video_dir}'.")
if not json_files:
    raise ValueError(f"No JSON files found in directory '{json_dir}'.")

# Ensure the number of video and JSON files match
if len(video_files) != len(json_files):
    raise ValueError("The number of video files and JSON files do not match. Please check the directories.")

# Open the CSV file for writing
with open(output_csv, mode="w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    # Write the header row
    csv_writer.writerow(["Video Name", "Violation Reason", "Processed Video Path", "Speed Graph Path", "Detected Classes File"])

    # Process each video and JSON file
    for video_file, json_file in tqdm(zip(video_files, json_files), total=len(video_files), desc="Processing Videos"):
        video_path = os.path.join(video_dir, video_file)
        json_path = os.path.join(json_dir, json_file)

        try:
            # Process the video and JSON file
            violation_reason, processed_video_path, speed_graph_path, txt_file_path = process_video(video_path, json_path)

            # Write the results to the CSV file
            csv_writer.writerow([video_file, violation_reason, processed_video_path, speed_graph_path, txt_file_path])

        except Exception as e:
            print(f"Error processing {video_file} with {json_file}: {e}")
            csv_writer.writerow([video_file, f"Error: {e}", "", "", ""])

print(f"Processing complete. Results saved to {output_csv}.")