import os
import shutil
import time
import sys
import json
# sys.path.append('/home/xhe71/Documents/GitHub/LEGS-POMDP/')
# sys.path.append('/home/xhe71/Documents/GitHub/LEGS-POMDP/object_detection/')
# sys.path.append('/home/xhe71/Documents/GitHub/LEGS-POMDP/pose_detection/')
# sys.path.append('/home/xhe71/Documents/GitHub/LEGS-POMDP/object_detection/SoM/task_adapter/')

# Example directory containing the input files
user_input = input("input image folder path: ")
print(f"You entered: {user_input}")
if len(user_input)==0:
    example_path = "/home/xhe71/Desktop/LEGS_eval/system_belief_update/gouger/blue_white_binary_cube_0207/gesture_white/"
else:
    example_path = user_input
# example_path = "/home/xhe71/Desktop/object_seg_example/"
# example_path = "/home/xhe71/Desktop/LEGS_eval/system_belief_update/gouger/blue_white_binary_cube_0207/gesture_white/"
# example_path = "/home/xhe71/Desktop/LEGS_eval/system_belief_update/gouger/blue_white_binary_cube_0207/gesture_white/"
# Output paths
save_image_path = "/home/xhe71/Documents/GitHub/LEGS-POMDP/spot_util/hand_color_image.png"
save_depth_path = "/home/xhe71/Documents/GitHub/LEGS-POMDP/spot_util/hand_depth_image.png"
save_transformation_path = "/home/xhe71/Documents/GitHub/LEGS-POMDP/spot_util/hand_intrinsics_and_transform.json"
detection_path = "/home/xhe71/Documents/GitHub/LEGS-POMDP/.tmp/detection_tmp.json"
SoM_annotated_image_path = "./.tmp/annotated_image.png"
# Get all files in the folder
files = os.listdir(example_path)

# Extract the unique frame numbers from file names
frames = set()
for file in files:
    if file.startswith("hand_color_image_") and file.endswith(".png"):
        frame_number = int(file.split("_")[-1].split(".")[0])
        frames.add(frame_number)

# Sort frame numbers to process sequentially
frames = sorted(frames)

# Loop over each frame
while True:

    for frame in frames:

        # Generate paths for the current frame
        image_path = os.path.join(example_path, f"hand_color_image_{frame}.png")
        depth_path = os.path.join(example_path, f"hand_depth_image_{frame}.png")
        transformation_matrix_path = os.path.join(example_path, f"hand_intrinsics_and_transform_{frame}.json")
        
        detection_matrix_path = os.path.join(example_path, f"detection_confidence_{frame}.json")
        
        # Verify that all required files exist
        if os.path.exists(image_path) and os.path.exists(depth_path) and os.path.exists(transformation_matrix_path):
            # from SoM_segmentation import run_segmentation
            # from SoM_GPT4 import combined_probability, run_SoM
            # if not os.path.exists(detection_matrix_path):
            #     marks = run_segmentation(image_path, depth_path, transformation_matrix_path)
            #     resp, marks = run_SoM(SoM_annotated_image_path, marks)
            #     with open(detection_matrix_path, 'w') as f:
            #         json.dump(marks, f, indent=4)
            # Copy files to the destination
            shutil.copy(image_path, save_image_path)
            shutil.copy(depth_path, save_depth_path)
            shutil.copy(transformation_matrix_path, save_transformation_path)
            # shutil.copy(detection_matrix_path, detection_path)

            print(f"Processed and saved frame {frame}")
        else:
            print(f"Missing files for frame {frame}, skipping.")

        time.sleep(1)