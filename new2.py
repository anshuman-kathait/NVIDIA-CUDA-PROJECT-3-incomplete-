import cv2
import numpy as np
import cudf
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Step 1: Verify video path
video_path = '/mnt/c/Users/katha/OneDrive/Desktop/NVIDIA FINAL PROJECT/APP2.final/traffic_video.mp4'
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video file not found: {video_path}")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("Error: Cannot open video. Check file path and format.")

# Read first frame
ret, prev_frame = cap.read()
if not ret:
    raise RuntimeError("Error: Cannot read the first frame of the video.")

prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
motion_data = []

# Step 2: Initialize GPU Optical Flow
try:
    gpu_flow = cv2.cuda_OpticalFlowDual_TVL1.create()
except AttributeError:
    raise RuntimeError("OpenCV was not compiled with CUDA support. Reinstall or compile OpenCV with CUDA.")

# Step 3: Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = 'processed_video.mp4'
out = cv2.VideoWriter(output_video, fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (prev_frame.shape[1], prev_frame.shape[0]))

frame_count = 0
while cap.isOpened():
    ret, curr_frame = cap.read()
    if not ret:
        break

    curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Upload frames to GPU
    gpu_prev_frame = cv2.cuda_GpuMat()
    gpu_curr_frame = cv2.cuda_GpuMat()
    gpu_prev_frame.upload(prev_frame_gray)
    gpu_curr_frame.upload(curr_frame_gray)

    # Compute Optical Flow on GPU
    flow_gpu = gpu_flow.calc(gpu_prev_frame, gpu_curr_frame, None)
    flow = flow_gpu.download()

    # Compute motion metrics
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    avg_magnitude = np.mean(magnitude)
    avg_angle = np.mean(angle)
    motion_data.append([frame_count, avg_magnitude, avg_angle, datetime.now()])

    # Create visualization
    hsv = np.zeros_like(prev_frame)
    hsv[..., 1] = 255
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    motion_display = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    overlay_frame = cv2.addWeighted(curr_frame, 0.7, motion_display, 0.3, 0)

    # Write frame to video
    out.write(overlay_frame)

    prev_frame_gray = curr_frame_gray
    frame_count += 1

cap.release()
out.release()

# Step 4: Process motion data with RAPIDS
gdf = cudf.DataFrame(motion_data, columns=["Frame", "Magnitude", "Angle", "Timestamp"])

# Summarize and visualize
summary = gdf.groupby("Frame")[["Magnitude", "Angle"]].mean().to_pandas()
plt.figure(figsize=(10, 5))
plt.plot(summary["Magnitude"], label="Average Speed")
plt.xlabel("Frame")
plt.ylabel("Motion Intensity")
plt.title("Traffic Flow Analysis")
plt.legend()
plt.savefig("traffic_flow_summary.png")
plt.show()

# Save results
summary.to_csv("traffic_flow_metrics.csv", index=False)
print("Analysis complete.")
print(f"- Processed video: {output_video}")
print("- Motion metrics: traffic_flow_metrics.csv")
print("- Motion intensity plot: traffic_flow_summary.png")
