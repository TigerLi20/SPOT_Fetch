import numpy as np
import cv2
import json
import fcntl
import os
import time
import spacy
nlp = spacy.load("en_core_web_md")  
def write_json_locked(data, json_path, max_retries=20, retry_delay=0.5):
    """Writes JSON safely with an exclusive lock (prevents concurrent reads/writes)."""
    retries = 0
    while retries < max_retries:
        try:
            with open(json_path, "w") as f:
                fcntl.flock(f, fcntl.LOCK_EX)  # Exclusive lock (blocks readers/writers)
                json.dump(data, f, indent=4)  # Write JSON data
                f.flush()  # Ensure data is written
                os.fsync(f.fileno())  # Force write to disk (prevents corruption)
                fcntl.flock(f, fcntl.LOCK_UN)  # Unlock BEFORE closing the file
                return  # Exit after successful write
        except OSError:
            print(f"Warning: Could not write to JSON. Retrying... ({retries+1}/{max_retries})")
            time.sleep(retry_delay)
            retries += 1
    
    raise RuntimeError("Error: Could not write to JSON file after multiple retries.")

def read_json_locked(json_path, max_retries=3, retry_delay=0.5):
    """Reads JSON file safely with a shared lock (prevents writes but allows multiple readers)."""
    retries = 0
    while retries < max_retries:
        try:
            with open(json_path, "r") as f:
                fcntl.flock(f, fcntl.LOCK_SH)  # Shared lock (allows multiple readers)
                data = json.load(f)  # Read JSON
                fcntl.flock(f, fcntl.LOCK_UN)  # Unlock BEFORE closing the file
                return data  # Return the parsed JSON
        except json.JSONDecodeError:
            print(f"Warning: JSON file is being updated. Retrying... ({retries+1}/{max_retries})")
            time.sleep(retry_delay)
            retries += 1
        except FileNotFoundError:
            print("Error: JSON file not found.")
            return None
    
    raise RuntimeError("Error: Could not read JSON file due to corruption or lock issues.")

# Using Kalman Filter for consistently tracking objects in different frames

class KalmanFilter:
    def __init__(self, dt, process_noise_std, measurement_noise_std):
        # State vector: [x, y, vx, vy]
        self.x = np.zeros((4, 1))

        # State transition matrix
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Measurement matrix (we measure [x, y])
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Process noise covariance
        self.Q = process_noise_std * np.eye(4)

        # Measurement noise covariance
        self.R = measurement_noise_std * np.eye(2)

        # Error covariance matrix
        self.P = np.eye(4)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        y = z - self.H @ self.x  # Measurement residual
        S = self.H @ self.P @ self.H.T + self.R  # Residual covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        self.x = self.x + K @ y  # Update state
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P  # Update error covariance

    def get_state(self):
        return self.x.flatten()
    

def compute_bbox_iou(bbox1, bbox2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        bbox1 (list or tuple): [x_min, y_min, width, height] for the first box.
        bbox2 (list or tuple): [x_min, y_min, width, height] for the second box.
    
    Returns:
        float: IoU value between 0 and 1.
    """
    # Convert bbox format [x_min, y_min, width, height] to [x_min, y_min, x_max, y_max]
    x1_min, y1_min, w1, h1 = bbox1
    x1_max, y1_max = x1_min + w1, y1_min + h1
    
    x2_min, y2_min, w2, h2 = bbox2
    x2_max, y2_max = x2_min + w2, y2_min + h2

    # Calculate the intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height

    # Calculate the area of both bounding boxes
    area1 = w1 * h1
    area2 = w2 * h2

    # Calculate the union area
    union_area = area1 + area2 - inter_area

    # Return IoU
    return inter_area / union_area if union_area > 0 else 0.0

# Function to compute 3D IoU for bounding boxes
def compute_bbox_iou_3d(bbox1, bbox2):
    inter_x_min = max(bbox1[0], bbox2[0])
    inter_y_min = max(bbox1[1], bbox2[1])
    inter_z_min = max(bbox1[2], bbox2[2])
    inter_x_max = min(bbox1[3], bbox2[3])
    inter_y_max = min(bbox1[4], bbox2[4])
    inter_z_max = min(bbox1[5], bbox2[5])

    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_depth = max(0, inter_z_max - inter_z_min)
    inter_volume = inter_width * inter_height * inter_depth

    vol1 = (bbox1[3] - bbox1[0]) * (bbox1[4] - bbox1[1]) * (bbox1[5] - bbox1[2])
    vol2 = (bbox2[3] - bbox2[0]) * (bbox2[4] - bbox2[1]) * (bbox2[5] - bbox2[2])
    union_volume = vol1 + vol2 - inter_volume

    return inter_volume / union_volume if union_volume > 0 else 0.0



def same_class(sentence1, sentence2, min_similarity=0.7):
    """
    Compute similarity score between two sentences using SentenceTransformer.

    Args:
        sentence1 (str): The first sentence.
        sentence2 (str): The second sentence.
        min_similarity(float)

    Returns:
        float: Cosine similarity score between the two sentences.
    """
    # Compute cosine similarity
    # Convert strings to spaCy Doc objects
    if sentence2 is None or sentence2 is None:
        return False
    doc1 = nlp(sentence1)
    doc2 = nlp(sentence2)

    # Compute similarity
    similarity_score = doc1.similarity(doc2)

    return similarity_score > min_similarity
    
def associate_marks(global_marks, current_marks, max_dist=0.5,min_2d_iou = 0.8, min_iou=0.4):
    global_len = len(global_marks)
    for curr_mark in current_marks:
        best_match = None
        best_score = float('inf')
        for global_mark in global_marks:
            # Compute 3D distance
            dist = np.linalg.norm(np.array(curr_mark['center_3d']) - np.array(global_mark['center_3d']))
            # Compute IoU - how much does the bounding box overlap
            iou_2d =  compute_bbox_iou(curr_mark['bbox'], global_mark['bbox'])
            iou = compute_bbox_iou_3d(curr_mark['bbox_xyzxyz'], global_mark['bbox_xyzxyz'])
            
            # Association based on distance and IoU
            if   (dist < max_dist ) or ((iou > min_iou) or (iou_2d > min_2d_iou)) and (dist < best_score):
                # print(f"Curr Mark [{curr_mark['mark']}] {curr_mark['category']}-Global Mark[ {global_mark['mark']}] {global_mark['category']}:  2D dist - {np.round(dist_2d, 3)}, dist - {np.round(dist, 3)}, 2d IOU - {round(iou_2d, 3)}, 3D IOU - {round(iou, 3)}")
                print(f"Curr Mark[{curr_mark['mark']}]-{curr_mark['category']}-lan_prob={np.round(curr_mark['lang_prob'], 2)} matches Global Mark [{global_mark['mark']}]-{global_mark['category']}")
                best_match = global_mark
                best_score = dist

        if best_match:
            # Update existing mark
            if hasattr(curr_mark, 'category'):
                best_match['category'] = curr_mark['category']
            if hasattr(curr_mark, 'area'):
                best_match['area'] = np.mean([best_match['area'], curr_mark['area']])
            if hasattr(curr_mark, 'lang_prob'):
                best_match['lang_prob'] = max(best_match['lang_prob'], curr_mark['lang_prob'])
            if hasattr(curr_mark, 'gest_prob'):
                best_match['gest_prob'] = max(best_match['gest_prob'], curr_mark['gest_prob'])
            if hasattr(curr_mark, 'combined_prob'):
                best_match['combined_prob'] = curr_mark['combined_prob']

            best_match['center'] = curr_mark['center']
            best_match['center_3d'] = curr_mark['center_3d']
            best_match['bbox'] = curr_mark['bbox']
            best_match['bbox_xyzxyz'] = curr_mark['bbox_xyzxyz']
            best_match['predicted_iou'] = np.mean([best_match['predicted_iou'], curr_mark['predicted_iou']])

            

            # print(f"skip adding curr mark [{curr_mark['mark']}]{curr_mark['category']}")
        else:
            # Add new mark
            global_len += 1
            curr_mark['mark'] = global_len
            print(f"appending new marks [{curr_mark['mark']}]{curr_mark['category']}")
            global_marks.append(curr_mark)
    return global_marks

def merge_marks(marks, max_dist=0.3):
    merged = []
    used = set()
    for i, mark1 in enumerate(marks):
        if i in used:
            continue
        group = [mark1]
        for j, mark2 in enumerate(marks):
            if j == i or j in used:
                continue
            dist = np.linalg.norm(np.array(mark1['center_3d']) - np.array(mark2['center_3d']))
            if mark1['category'] is not None:
                same_category = same_class(mark1['category'], mark2['category'])
            else:
                same_category = True
            if dist < max_dist and same_category:
                print(f"Dist{dist},merging  mark [{i}]{mark1['category']} with mark [{j}]{mark2['category']}")
                group.append(mark2)
                used.add(j)
            
                
        # Average attributes
        merged_mark = {
            'mark': i+1, 
            'category':group[0]['category'],
            'area': np.mean([m['area'] for m in group], axis=0).tolist(),
            'lang_prob': np.max([m['lang_prob'] for m in group], axis=0).tolist(), 
            
            'center': np.mean([m['center'] for m in group], axis=0).tolist(),
            'center_3d': np.mean([m['center_3d'] for m in group], axis=0).tolist(),
            'bbox': np.mean([m['bbox'] for m in group], axis=0).tolist(),
            'bbox_xyzxyz': np.mean([m['bbox_xyzxyz'] for m in group], axis=0).tolist(),
            'predicted_iou': max(m['predicted_iou'] for m in group),
            # 'intrinsics': group[0]['intrinsics'],  # Assuming same intrinsics
            # 'transform': group[0]['transform'],  # Assuming same transform
        }
        # merged_mark['lang_prob'] = max(m['lang_prob'] for m in group),
        # merged_mark['gest_prob'] = max(m['gest_prob'] for m in group),
            
        merged.append(merged_mark)
        used.add(i)
    for i in range(len(merged)):
        merged[i]['mark'] = i+1
    return merged

def process_marks(global_marks, current_marks, max_dist=0.2, min_2d_iou=0.6, min_iou=0.4, redundancy_threshold=0.3):
    """
    Process new marks: associate, merge, and apply Kalman Filters.
    
    Args:
        global_marks (list): Marks tracked globally.
        current_marks (list): Newly detected marks.
        max_dist (float): Maximum distance for associating marks.
        min_iou (float): Minimum IoU for associating marks.
        redundancy_threshold (float): Maximum distance for merging redundant marks.
    
    Returns:
        list: Updated global marks.
    """
    global global_kalman_filters

    # Associate new marks with global marks
    global_marks = associate_marks(global_marks, current_marks, max_dist,min_2d_iou,  min_iou)

    # Merge redundant marks
    global_marks = merge_marks(global_marks, redundancy_threshold)
    print(len(global_marks))

    return global_marks 

def resize_depth(color_image_path, depth_image_path, w=None, h = None):
    color_image = cv2.imread(color_image_path, cv2.IMREAD_COLOR)
    
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
    if color_image is None:
        print(f"Warning: Could not load image at {color_image_path}. Skipping...")
        return None, None
    if w==None:
        h_color, w_color, _ = color_image.shape
    else:
        h_color, w_color = h, w
    color_image_resized = cv2.resize(color_image, (w_color, h_color), interpolation=cv2.INTER_NEAREST)
    depth_image_resized = cv2.resize(depth_image, (w_color, h_color), interpolation=cv2.INTER_NEAREST)
    return color_image_resized, depth_image_resized
