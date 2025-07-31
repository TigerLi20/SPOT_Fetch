import torch
import numpy as np
from PIL import Image
from scipy.ndimage import label
import cv2
import sys
import os
import json
import pandas as pd
import random
import open3d as o3d
import urllib.request
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gc

pth = '/'.join(sys.path[0].split('/')[:-1])
sys.path.insert(0, pth)

SoM_pth =pth+'/object_detection/SoM'
sys.path.insert(0, SoM_pth)


from pose_detection.gesture_util import load_intrinsics_and_transform, image_pt_to_pointcloud_pt
# seem
import seem
from seem.modeling.BaseModel import BaseModel as BaseModel_Seem
from seem.utils.distributed import init_distributed as init_distributed_seem
from seem.modeling import build_model as build_model_seem
from object_detection.SoM.task_adapter.seem.tasks import interactive_seem_m2m_auto, inference_seem_pano, inference_seem_interactive

# semantic sam
from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from semantic_sam.utils.dist import init_distributed_mode
from semantic_sam.utils.arguments import load_opt_from_config_file
from semantic_sam.utils.constants import COCO_PANOPTIC_CLASSES
import torch
import torchvision.transforms as transforms
from torchvision import models
from detection_util import KalmanFilter, associate_marks, merge_marks, resize_depth
from object_detection.SoM.task_adapter.semantic_sam.tasks import inference_semsam_m2m_auto, prompt_switch

# sam
from segment_anything import sam_model_registry
from object_detection.SoM.task_adapter.sam.tasks.inference_sam_m2m_auto import inference_sam_m2m_auto
from object_detection.SoM.task_adapter.sam.tasks.inference_sam_m2m_interactive import inference_sam_m2m_interactive

from scipy.ndimage import label
import numpy as np
import spacy
nlp = spacy.load("en_core_web_md")  
# from sentence_transformers import SentenceTransformer, util


'''
Configurations and checkpoints
'''
semsam_cfg = "object_detection/SoM/configs/semantic_sam_only_sa-1b_swinL.yaml"
seem_cfg = "object_detection/SoM/configs/seem_focall_unicl_lang_v1.yaml"

semsam_ckpt = "object_detection/SoM/swinl_only_sam_many2many.pth"
sam_ckpt = "object_detection/SoM/sam_vit_h_4b8939.pth"
seem_ckpt = "object_detection/SoM/seem_focall_v1.pt"
# seem_ckpt = "/home/xhe71/Downloads/seem_samvitl_v1.pt"
# Load configurations
opt_semsam = load_opt_from_config_file(semsam_cfg)
opt_seem = load_opt_from_config_file(seem_cfg)
opt_seem = init_distributed_seem(opt_seem)


'''
Load models
'''
print("Loading models...")
model_semsam = BaseModel(opt_semsam, build_model(opt_semsam)).from_pretrained(semsam_ckpt).eval().cuda()
model_sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).eval().cuda()
model_seem = BaseModel_Seem(opt_seem, build_model_seem(opt_seem)).from_pretrained(seem_ckpt).eval().cuda()

# Initialize embeddings for SEEM
with torch.no_grad():
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        model_seem.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)

# Load Pretrained ResNet model
# Download the labels from PyTorch's official repository
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
imagenet_classes = urllib.request.urlopen(url).read().decode("utf-8").split("\n")

'''
Inference function
'''

# maximum distance allowed for associating a new detection with a predicted position
__MAX_DIST = 0.3

# threshold distance below which two marks are considered redundant
redundancy_threshold = 0.3

def load_image_safe(image_path):
    try:
        img = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format
        return img
    except Exception as e:
        print(f"Error loading image at {image_path}: {e}")
        return None
    
def inference(image_path, model_name='semantic-sam', mode='Automatic', slider=5, alpha=0.05, label_mode='1', anno_mode=['Mask', 'Mark']):
    # Load image
    from PIL import ImageFilter
    _image = load_image_safe(image_path)
    _image = _image.filter(ImageFilter.GaussianBlur(radius=1))
    
    if _image is None:
        print("Skipping corrupted image.")
        return None, None

    # Determine the model to use
    if model_name == 'semantic-sam':
        model = model_semsam
        level = [slider]  # Adjust granularity level if needed
        output, mask = inference_semsam_m2m_auto(model, _image, level,
                                                 text_size=640, 
                                                 label_mode=label_mode, alpha=alpha, anno_mode=anno_mode)

    elif model_name == 'sam':
        model = model_sam
        if mode == "Automatic":
            output, mask = inference_sam_m2m_auto(model, _image, text_size=480, label_mode=label_mode, alpha=alpha, anno_mode=anno_mode)
        else:
            labeled_array, num_features = label(np.asarray(_image.convert('L')))
            spatial_masks = torch.stack([torch.from_numpy(labeled_array == i + 1) for i in range(num_features)])
            output, mask = inference_sam_m2m_interactive(model, _image, spatial_masks, text_size=640, label_mode=label_mode, alpha=alpha, anno_mode=anno_mode)

    elif model_name == 'seem':
        model = model_seem
        if mode == "Automatic":
            output, mask = inference_seem_pano(model, _image, text_size=640, label_mode=label_mode, alpha=alpha, anno_mode=anno_mode)
        else:
            labeled_array, num_features = label(np.asarray(_image.convert('L')))
            spatial_masks = torch.stack([torch.from_numpy(labeled_array == i + 1) for i in range(num_features)])
            output, mask = inference_seem_interactive(model, _image, spatial_masks, text_size=640, label_mode=label_mode, alpha=alpha, anno_mode=anno_mode)

    return output, mask



def convert_numpy_to_native(data):
    """Recursively convert NumPy data types to Python native types."""
    if isinstance(data, dict):
        return {key: convert_numpy_to_native(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_to_native(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.int64, np.int32, np.int16)):
        return int(data)
    elif isinstance(data, (np.float64, np.float32)):
        return float(data)
    else:
        return data





def classify_object(rgb_image, bbox):
    """
    Classifies an object using a pretrained CNN.
    bbox: (x_min, y_min, x_max, y_max)
    """
    
    # Crop the object from the RGB image
    x_min, y_min, w, h = bbox
    x_max = x_min + w
    y_max = y_min + h
    # x_min, y_min, x_max, y_max = bbox

    cropped = rgb_image[y_min:y_max, x_min:x_max]
    image = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    output, mask = inference_seem_pano(model_seem, image, text_size=640)
    if output is None:
        return None
    category = mask[0]["category"]
    gc.collect()
    del image

    torch.cuda.empty_cache()  # Releases memory not being used
    torch.cuda.ipc_collect()  # Cleans up fragmentation
    return category


def remove_outliers(points, nb_neighbors=10, std_ratio=1.5):
    """
    Removes statistical outliers from a list of 3D points.
    - nb_neighbors: Number of neighbors to analyze for each point.
    - std_ratio: Standard deviation threshold for filtering.
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return np.asarray(cl.points)

def plot_3d_bounding_boxes_and_points(curr_marks):
    """
    Visualize 3D bounding boxes and points in a point cloud.
    Args:
        curr_marks (list): List of dictionaries containing 'bbox_xyzxyz' and 'center_3d'.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot each bounding box and its center
    for mark in curr_marks:
        if "bbox_xyzxyz" in mark and "center_3d" in mark:
            # Extract the bounding box coordinates
            bbox = mark["bbox_xyzxyz"]
            x_min, y_min, z_min, x_max, y_max, z_max = bbox

            # Define the edges of the bounding box
            edges = [
                [[x_min, x_max], [y_min, y_min], [z_min, z_min]],
                [[x_min, x_max], [y_max, y_max], [z_min, z_min]],
                [[x_min, x_max], [y_min, y_min], [z_max, z_max]],
                [[x_min, x_max], [y_max, y_max], [z_max, z_max]],
                [[x_min, x_min], [y_min, y_max], [z_min, z_min]],
                [[x_max, x_max], [y_min, y_max], [z_min, z_min]],
                [[x_min, x_min], [y_min, y_max], [z_max, z_max]],
                [[x_max, x_max], [y_min, y_max], [z_max, z_max]],
                [[x_min, x_min], [y_min, y_min], [z_min, z_max]],
                [[x_max, x_max], [y_min, y_min], [z_min, z_max]],
                [[x_min, x_min], [y_max, y_max], [z_min, z_max]],
                [[x_max, x_max], [y_max, y_max], [z_min, z_max]],
            ]

            # Plot the edges of the bounding box
            for edge in edges:
                ax.plot(edge[0], edge[1], edge[2], color="b")

            # Plot the center point
            center = mark["center_3d"]
            ax.scatter(center[0], center[1], center[2], color="r", s=50, label="Center")

    # Set axis labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.title("3D Bounding Boxes and Points")
    plt.legend(["Bounding Box", "Center"])
    plt.show()

def project_to_3d(intrinsics, transformation_matrix, segmentation_mask, depth_image):
    """
    Convert all segmented 2D pixels into 3D points using depth data.
    """
    height, width = depth_image.shape
    points_3d = []

    for y in range(height):
        for x in range(width):
            if segmentation_mask[y, x]:  # If pixel belongs to segmented object
                depth = depth_image[y, x]
                if depth > 0:  # Ignore invalid depth values
                    point_3d = image_pt_to_pointcloud_pt(intrinsics, transformation_matrix, x, y, depth)
                    points_3d.append(point_3d)

    return np.array(points_3d)

def compute_tight_bounding_box(points):
    """
    Computes a tight bounding box from filtered 3D points.
    """
    if len(points) == 0:
        return None  # Return nothing if no valid points
    elif min(np.std(points, axis = 0)) < 0.05:
        jitter = np.random.normal(scale=1e-3, size=points.shape)  # Small noise
        points += jitter
    # Convert to Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # Compute Oriented Bounding Box (tighter fit)
    obb = point_cloud.get_oriented_bounding_box()
    obb.color = (1, 0, 0)  # Red for visualization

    return obb

# Define background categories to exclude (phrase similarity will be used)
BACKGROUND_CATEGORIES = ["wall-other-merged", "floor-other-merged", "person", "ceiling", "ground", "sky"]

def is_background_category(category_name, threshold=0.8):
    """
    Check if the detected category is similar to background categories.
    Returns True if it should be skipped.
    """
    sim_list = []
    doc1 = nlp(category_name)
    for cat in BACKGROUND_CATEGORIES:
        doc2 = nlp(cat)

        # Compute similarity
        similarity_score = doc1.similarity(doc2)
        sim_list.append(similarity_score)
    
    return max(sim_list) >= threshold  # Skip if above threshold


def run_segmentation(image_path =  "./.tmp/image.png", depth_image_path = None, transformation_matrix_path = None):
    
    # Run inference
    print("Running inference...")
    output, mask = inference(image_path, model_name='semantic-sam', 
                             mode='Automatic', slider=3, alpha=0.2, label_mode='1', anno_mode=['Mask', 'Mark'])
    if output is None:
        return 
    height, width = mask[0]['segmentation'].shape
    # Create an empty RGB image to hold the combined mask
    condensed_mask = np.zeros((height, width, 3), dtype=np.uint8)
    # Save the output
    
    cv2.imwrite("./.tmp/annotated_image.png", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    image = cv2.imread(image_path)

    if depth_image_path:
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
        if depth_image is None:
            print(f"Warning: Could not load image at {depth_image}. Skipping...")
            return []
        image, depth_image = resize_depth(image_path, depth_image_path, w=width, h=height)
        cv2.imwrite("./.tmp/depth_image.png", depth_image)
    else:
        depth_image = None  
    

        
    curr_marks = []
    
    # # Initialize or load Kalman Filters

    for i in range(len(mask)):
        segmentation = mask[i]['segmentation']
        if hasattr(mask[i], "category"):
            category = mask[i]['category']
            if is_background_category(category):
                continue
        

        # red channel represent mask id
        color = [i+1, 255//len(mask)*i, random.randint(10, 245)]
        
        condensed_mask[segmentation] = color
        # xywh bounding box
        bbox = mask[i]['bbox']
        area = mask[i]['area']
        predicted_iou = mask[i]['predicted_iou']
        
        
        
        if predicted_iou > 1:
            predicted_iou = 1
        if hasattr(mask[i], "category"):
            data = {
                "mark": i+1, 
                "predicted_iou": np.round(predicted_iou, 6),
                "lang_prob": 0, 
                "gest_prob": 0, 
                "joint_prob": 0,
                "category":category,
                "bbox": bbox,
                "area": area,            
            }
        else:
                data = {
                "mark": i+1, 
                "predicted_iou": np.round(predicted_iou, 6),
                "lang_prob": 0, 
                "gest_prob": 0, 
                "joint_prob": 0,
                "category":None,
                "bbox": bbox,
                "area": area,    
            }
        
            
        if transformation_matrix_path:
        #     with open(transformation_matrix_path, 'r') as f:
        #         intrinsic_transform = json.load(f)
        #     data['intrinsics'] = intrinsic_transform['intrinsics']
        #     data['transform'] = intrinsic_transform['transform']
        
            intrinsics, transformation_matrix = load_intrinsics_and_transform(transformation_matrix_path)
        # box approach
        if depth_image is not None:

            # Process each bounding box
            x_min, y_min, width, height = bbox
            x_max = x_min + width
            y_max = y_min + height

            
            # Calculate the center of the bounding box
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2

            # Extract the region of interest (ROI) from the depth image
            roi_depth = depth_image[int(y_min):int(y_max), int(x_min):int(x_max)]
            # Calculate the average depth within the bounding box
            non_zero_depths = roi_depth[roi_depth > 0]  # Ignore zero-depth pixels
            avg_depth = np.mean(non_zero_depths) if len(non_zero_depths) > 0 else 0

            points_3d = project_to_3d(intrinsics, transformation_matrix, segmentation, depth_image)
            if len(points_3d) > 0:
                filtered_points = remove_outliers(points_3d)
            else:
                print(f"skip adding Mark {i+1}")
                continue
            # Step 3: Compute tighter bounding box
            bounding_box = compute_tight_bounding_box(filtered_points)
            

            center_3d = bounding_box.center
            extent_3d = bounding_box.extent
            data['center'] = [int(center_x), int(center_y), int(avg_depth)]
            data['center_3d'] = center_3d.tolist()
            if extent_3d[0]*extent_3d[1]*extent_3d[2] > 50:
                print(f"skip adding Mark {i+1}.... Volume Exception ....")
                continue
            if data["category"] is None:
                data["category"] = classify_object(image, data["bbox"])
            if data["category"] is not None and is_background_category(data["category"]):
                print(f"skip adding Mark {i+1}.... {data['category']} ....")
                continue
            print(f"{data['mark']}-{data['category']}:{bounding_box}")
            # 4. Compute a tight bounding box based on filtered data
            x_min, y_min, z_min = center_3d - extent_3d /2
            x_max, y_max, z_max = center_3d + extent_3d /2
            data["bbox_xyzxyz"] = np.round(np.array([x_min, y_min, z_min, x_max, y_max, z_max]), 4).tolist() # 
            
        
            
        curr_marks.append(data)
    cv2.imwrite("./.tmp/.mask.png", condensed_mask)
    gc.collect()

    torch.cuda.empty_cache()  # Releases memory not being used
    torch.cuda.ipc_collect()  # Cleans up fragmentation

    return convert_numpy_to_native(curr_marks)


def main():
    transformation_matrix_path = './spot_util/hand_intrinsics_and_transform.json'
    image_path = "/home/xhe71/Documents/GitHub/LEGS-POMDP/SoM_Eval.jpg"
    # image_path = "./spot_util/hand_color_image.png"
    depth_image_path =  "./spot_util/hand_depth_image.png"
    
    data = run_segmentation(image_path)


'''
Run locally with an input image
'''
if __name__ == "__main__":
    main()