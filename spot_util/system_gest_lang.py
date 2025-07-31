from spot_util.pointcloud import *

from object_detection.SoM_segmentation import run_segmentation

from pose_detection.gesture_extractor import PointingGestureDetector
from pose_detection.gesture_system import run_gesture
from pose_detection.gesture_spot import map_to_3d, image_pt_to_pointcloud_pt, load_images_and_resize_depth
from joint_probability import combine_prob

# step 0: clear all files


# step 1: activate spot for image detection

# step 2: locate and approach human 

# step 3.0: constantly detect and append object
transformation_matrix_path = './spot_util/hand_intrinsics_and_transform.json'
image_path = "./spot_util/spot_util/hand_color_image.png"
depth_image_path =  "./spot_util/spot_util/hand_depth_image.png"
run_segmentation(image_path, depth_image_path, transformation_matrix_path)



# step 3.1: detect gesture input
# if human gesture is detected, enable run_spot from gesture_system
detector = PointingGestureDetector()
run_gesture(detector)

# step 3.2: detect language input
# from SoM_Segmentation

# step 4: combine the probability
combine_prob()

