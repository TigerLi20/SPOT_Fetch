import os
import glob
import cv2
import pandas as pd
from pose_detection.old_calibration_apriltag import *
from gesture_util import smooth_depth_image
import mediapipe as mp

# Camera intrinsic parameters
fx = 3629.5913404959015
fy = 3629.5913404959015
px = 2104
py = 1560
K = np.array([[fx, 0, px],[0, fy, py],[0, 0, 1]])
tag_size_m = 0.165
pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence = 0.5,  model_complexity = 2)

# Folder paths
folder_path = '/Users/ivy/Desktop/spot_gesture_eval/1003/consistency_test/2m'
output_csv_path = '/Users/ivy/Desktop/spot_gesture_eval/1003/consistency_test/2m_compare.csv'

# Get a list of all color images in the folder (assume .jpg)
color_images = sorted(glob.glob(os.path.join(folder_path, '*color_image_*.jpg')))
depth_images = sorted(glob.glob(os.path.join(folder_path, '*depth_in_hand_color_frame_*.png')))

if len(color_images) == 0 or len(depth_images) == 0:
    print("No images found in the specified folder.")
    exit()

# Ensure the color and depth image lists have matching lengths
if len(color_images) != len(depth_images):
    print("Mismatch between number of color and depth images.")
    exit()

# Initialize DataFrame to store results
data = pd.DataFrame(columns=['frame', 'ground_plane', 'tag_centroids', 'landmarks_2d', 'landmarks_3d'])

# Initialize frame counter
frame_num = 0
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Loop through the images
for color_image_path, depth_image_path in zip(color_images, depth_images):
    
    print(f"Processing {color_image_path} and {depth_image_path}")

    # Load color and depth images
    image = cv2.imread(color_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

    # Process the images using the calibrate function
    K = np.array([[fx, 0, px],[0, fy, py],[0, 0, 1]])
    results, new_ground, tag_3d_points, tag_centroids = calibrate(image, pose, depth_image, K, tag_size_m)

    # Ensure results and ground plane exist before proceeding
    if results is not None and new_ground is not None:
        # Extract landmark data (pose_landmarks and world_pose_landmarks)
        landmarks_2d = results.pose_landmarks.landmark if results.pose_landmarks else []
        landmarks_3d = results.pose_world_landmarks.landmark if results.pose_world_landmarks else []
        if tag_3d_points is not None: 
        # Prepare the row data for CSV
            new_row = {
                'frame': frame_num,
                'ground_plane': new_ground if new_ground is not None else '',
                'tag_centroids': [tag_centroids] if tag_centroids is not None else '',
                'landmarks_2d': landmarks_2d, 
                'landmarks_3d': landmarks_3d, 
            }
            # Append the row to the DataFrame
            data = pd.concat([data, pd.DataFrame(new_row, index=[0])], ignore_index=True)

            # Plot the processed data (AprilTag, human skeleton, ground plane)
        
            ax = plot_tag_and_landmarks(tag_3d_points, tag_centroids, results.pose_world_landmarks.landmark, new_ground, ax)

    else:
        print(f"Skipping frame {frame_num} due to missing calibration results.")

    # Increment frame number
    frame_num += 1
elev = - 53
azim = -90
ax.view_init(elev, azim)
plt.show()
# Save the results to a CSV file
data.to_csv(output_csv_path, index=False)
print(f"Results saved to {output_csv_path}")