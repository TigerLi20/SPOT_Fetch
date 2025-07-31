import threading
import time
import sys
sys.path.append('pose_detection/')
from pose_detection.gesture_system import ObjectProcessor
sys.path.append('object_detection/')
sys.path.append('object_detection/SoM/')
import os
import json
from object_detection.detection_system import ObjectDetectionSystem
import numpy as np

def gesture_intersection_update(object_processor):

    post_gesture_global_mark = object_processor.update_objects_in_cone()
    
    return post_gesture_global_mark

def lang_objs_update(lang_objs):

    marks, global_marks = lang_objs.process_image_update()
    lang_objs.save_if_updated()  # Save only if updates occurred
    return marks, global_marks


def compute_joint_probabilities(objects, alpha=1e-6):
    N = len(objects)  # Number of unique objects (marks)

    # Compute modality normalizing constants
    Z_detection = sum(obj["predicted_iou"] for obj in objects) + alpha * N
    Z_language = sum(obj["lang_prob"] for obj in objects) + alpha * N
    Z_gesture = sum(obj["gest_prob"] for obj in objects) + alpha * N

    # Apply Laplace smoothing to each modality
    for obj in objects:
        dete_prob = (obj["predicted_iou"] + alpha) / Z_detection
        lang_prob = (obj["lang_prob"] + alpha) / Z_language
        gest_prob = (obj["gest_prob"] + alpha) / Z_gesture
        obj["joint_prob"] = dete_prob * lang_prob * gest_prob

    # Normalize joint probabilities
    Z_joint = sum(obj["joint_prob"] for obj in objects)
    for obj in objects:
        obj["joint_prob"] /= Z_joint  
        obj["joint_prob"] = np.round(obj["joint_prob"], 6)

    return objects

def observation_update(object_processor, lang_objs):
    marks, global_marks = lang_objs_update(lang_objs)
    post_gesture_global_mark = gesture_intersection_update(object_processor)
    updated_global_marks = compute_joint_probabilities(post_gesture_global_mark)
    with open(".tmp/confidence.json", "w") as f:
        json.dump(updated_global_marks, f, indent=4)
    return updated_global_marks

def is_file_updated(file_path, last_mod_time):
    if not os.path.exists(file_path):
        return False  # File doesn't exist
    current_mod_time = os.path.getmtime(file_path)
    return current_mod_time > last_mod_time

def main():
    # Configure paths
    object_path=".tmp/detection_confidence.json"
    gesture_cone_path= ".tmp/gesture_vector.json"
    color_image_path = "spot_util/hand_color_image.png"
    depth_image_path = "spot_util/hand_depth_image.png"
    transformation_path = "spot_util/hand_intrinsics_and_transform.json"
    # Initualize models
    try:
        os.remove(object_path)
    except FileNotFoundError:
        print("deleted")
    object_processor = ObjectProcessor(
            object_list_path= object_path,
            gesture_vector_path= gesture_cone_path
        )

    lang_objs = ObjectDetectionSystem(
        color_path=color_image_path,
        depth_path=depth_image_path,
        trans_matrix_path=transformation_path,
        detection_conf_path=object_path
    )
    


    while True:
        # change to Look Action detection
        last_mod_time = os.path.getmtime(color_image_path)
        if is_file_updated(color_image_path, last_mod_time):
            LookAction = True
        else:
            LookAction = False
        if LookAction:
            updated_global_marks = observation_update(object_processor, lang_objs)

        # time.sleep(1)


if __name__ == "__main__":
    main()