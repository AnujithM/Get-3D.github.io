import cv2
import numpy as np
import torch
from ultralytics import YOLO
import torchvision.transforms as transforms
from tqdm import tqdm

# Define paths
input_video_path = 'D:\\Get-3D\\wolf.mp4'
output_video_path = 'D:\\Get-3D\\output12.mp4'
model_path = 'D:\\Get-3D\\yolov8x-seg.pt'  # Path to YOLOv8s instance segmentation model

# Check if GPU is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load YOLOv8 model for instance segmentation
model = YOLO(model_path).to(device)  # Load the model from the specified path and move to GPU if available

# Load MiDaS depth estimation model
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS").to(device)
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").default_transform

# Function to estimate depth using MiDaS
def estimate_depth(frame):
    input_tensor = midas_transforms(frame).to(device)
    with torch.no_grad():
        prediction = midas(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map = prediction.cpu().numpy()
    return depth_map

# Function to perform segmentation using YOLOv8
def segment_frame(frame):
    results = model(frame)
    masks = results[0].masks  # Get masks from the results
    
    person_masks = []
    if masks is not None:
        person_masks = [masks[i].data.cpu().numpy().astype(np.uint8) * 255 for i in range(len(masks))]

    if person_masks:
        person_masks = [np.squeeze(mask) for mask in person_masks]  # Remove any singleton dimensions
    return person_masks

# Function to apply depth effect
def apply_depth_effect(frame, masks, depth_map, depth_threshold=4500.0, min_mask_area=15000, zoom_factor=1.1, bar_ratio=0.2, vertical_bar_width=40, vertical_bar_position_ratio=0.7):
    depth_map_resized = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))
    
    combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for mask in masks:
        if mask.shape != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

        median_depth = np.median(depth_map_resized[mask > 0])
        mask_area = np.sum(mask > 0)

        if median_depth <= depth_threshold and mask_area >= min_mask_area:
            combined_mask = cv2.bitwise_or(combined_mask, mask)

    if combined_mask.ndim == 3 and combined_mask.shape[2] == 3:
        combined_mask = cv2.cvtColor(combined_mask, cv2.COLOR_BGR2GRAY)
    if combined_mask.dtype != np.uint8:
        combined_mask = combined_mask.astype(np.uint8)

    object_part = cv2.bitwise_and(frame, frame, mask=combined_mask)
    background_part = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(combined_mask))

    bar_height = int(frame.shape[0] * bar_ratio)
    background_part[:bar_height, :] = 0
    background_part[-bar_height:, :] = 0

    bar_position = int(frame.shape[1] * vertical_bar_position_ratio)
    background_part[:, bar_position - vertical_bar_width // 2: bar_position + vertical_bar_width // 2] = 0
    background_part[:, -bar_position - vertical_bar_width // 2: -bar_position + vertical_bar_width // 2] = 0

    center = (frame.shape[1] // 2, frame.shape[0] // 2)
    matrix = cv2.getRotationMatrix2D(center, 0, zoom_factor)
    object_part = cv2.warpAffine(object_part, matrix, (frame.shape[1], frame.shape[0]))
    enlarged_mask = cv2.warpAffine(combined_mask, matrix, (frame.shape[1], frame.shape[0]))

    enlarged_mask_3channel = cv2.cvtColor(enlarged_mask, cv2.COLOR_GRAY2BGR)
    combined = cv2.addWeighted(background_part, 1, enlarged_mask_3channel, -1, 0)
    combined = cv2.add(combined, object_part)
    
    return combined

# Read input video
cap = cv2.VideoCapture(input_video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

output_width, output_height = 1600, 900
out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))

frame_count = 0

# Initialize tqdm progress bar
with tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (output_width, output_height))
        
        # Estimate depth using MiDaS
        depth_map = estimate_depth(frame)
        depth_map_resized = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))
        median_depth = np.median(depth_map_resized)
        
        # Check if depth threshold criteria are met
        if median_depth <= 4500.0:
            # Perform segmentation using YOLOv8
            masks = segment_frame(frame)
            
            if masks:
                frame_3d = apply_depth_effect(frame, masks, depth_map, depth_threshold=4500.0, min_mask_area=15000)
            else:
                frame_3d = frame
        else:
            frame_3d = frame
        
        out.write(frame_3d)
        cv2.imshow('Processing Video', frame_3d)  # Show the processing video live

        frame_count += 1
        pbar.update(1)  # Update the progress bar

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()

print("\nVideo processing complete. Now playing the processed video...")

# Play the processed video in a loop
cap_processed = cv2.VideoCapture(output_video_path)
frame_delay = int(1000 / fps)  # Calculate the delay between frames in milliseconds

while True:
    ret, frame = cap_processed.read()
    if not ret:
        cap_processed.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    
    cv2.imshow('Processed Video', frame)
    
    if cv2.waitKey(frame_delay) & 0xFF == ord('q'):  # Use frame_delay for correct playback speed
        break

cap_processed.release()
cv2.destroyAllWindows()

print("3D video generated and saved successfully.")
