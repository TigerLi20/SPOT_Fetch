import os
import numpy as np
import cv2
import json
import open3d as o3d
from scipy.spatial import cKDTree


def recreate_depth_image_from_pointcloud(color_image, point_cloud, intrinsics, color_to_world_transform):
    """
    Recreate the depth image by finding depth values in the point cloud for each color image pixel.

    Args:
        color_image (np.ndarray): Color image (HxWx3).
        point_cloud (np.ndarray): Nx3 array of 3D points in world coordinates.
        intrinsics (np.ndarray): 3x3 intrinsic matrix of the camera.
        color_to_world_transform (np.ndarray): 4x4 transformation matrix from color image to world frame.

    Returns:
        np.ndarray: Recreated depth image (HxW) in millimeters.
    """
    h, w, _ = color_image.shape
    depth_image = np.zeros((h, w), dtype=np.float32)
    depth_buffer = np.full((h, w), np.inf)  # Buffer to store the minimum depth



    # Build a KD-tree for efficient nearest neighbor search in the point cloud
    kdtree = cKDTree(point_cloud)

    # Compute the inverse of the color image's transformation matrix
    world_to_color_transform = np.linalg.inv(color_to_world_transform)
    points_homogeneous = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))
    camera_points = (world_to_color_transform @ points_homogeneous.T).T[:, :3]

    # Project the points into the image plane
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    for point in camera_points:
        x, y, z = point
        if z <= 0:  # Skip points behind the camera
            continue

        # Map 3D points to 2D pixel coordinates
        u = int((fx * x / z) + cx)
        v = int((fy * y / z) + cy)
        # Check if the pixel is within bounds
        if 0 <= u < w and 0 <= v < h:
            # Update the depth buffer with the nearest point
            if z < depth_buffer[v, u]:
                depth_image[v, u] = z
                depth_buffer[v, u] = z

    # Convert depth to millimeters
    depth_image = (depth_image * 1000).astype(np.uint16)

    return depth_image

def visualize_color_image_in_pointcloud(point_cloud, color_image, intrinsics, color_to_world_transform):
    """
    Visualize the color image within the point cloud by projecting color image pixels into the point cloud.

    Args:
        point_cloud (np.ndarray): Nx3 array of 3D points (world coordinates).
        color_image (np.ndarray): Color image (HxWx3).
        intrinsics (np.ndarray): 3x3 intrinsic matrix of the camera.
        color_to_world_transform (np.ndarray): 4x4 transformation matrix from color image to world frame.
    """
    h, w, _ = color_image.shape
    projected_points = []

    # Compute the inverse transformation matrix
    world_to_color_transform = np.linalg.inv(color_to_world_transform)

    # Intrinsic parameters
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Loop over each pixel in the color image
    for v in range(h):
        for u in range(w):
            # Project pixel (u, v) into the camera frame
            x = (u - cx) / fx
            y = (v - cy) / fy
            z = 1.0  # Assume unit depth
            camera_point = np.array([x * z, y * z, z, 1.0])

            # Transform from camera frame to world frame
            world_point = color_to_world_transform @ camera_point
            projected_points.append(world_point[:3])

    # Convert projected points to Open3D format
    projected_pcd = o3d.geometry.PointCloud()
    projected_pcd.points = o3d.utility.Vector3dVector(np.array(projected_points))

    # Assign color from the color image
    colors = np.reshape(color_image, (-1, 3)) / 255.0
    projected_pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize both point clouds
    o3d.visualization.draw_geometries([o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_cloud)), projected_pcd])
    
def load_intrinsics_and_transform(json_file_path):
    """
    Load intrinsics and transformation matrices from a JSON file.
    
    Args:
        json_file_path (str): Path to the JSON file.
    
    Returns:
        tuple: Intrinsics matrix (3x3 numpy array), transformation matrix (4x4 numpy array).
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    intrinsics = np.array(data["intrinsics"])
    rotation_matrix = np.array(data["transform"]["rotation_matrix"])
    translation_vector = np.array(data["transform"]["translation_vector"])
    
    # Combine into a single 4x4 transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation_vector
    
    return intrinsics, transformation_matrix


# Paths
gesture_folder_path = '/Users/ivy/Desktop/Desktop - Ivy’s MacBook Pro/LEGS_eval/spot_images/0115_system_test/object_segmentation/'
ply_path = "/Users/ivy/Desktop/Desktop - Ivy’s MacBook Pro/LEGS_eval/spot_images/0115_system_test/pointcloud.ply"

# List color image files to determine the number of frames
color_images = sorted([f for f in os.listdir(gesture_folder_path) if f.startswith('hand_color_image_') and f.endswith('.png')])
frames = len(color_images)
print(f"Total frames: {frames}")

# Load the full point cloud
pcd = o3d.io.read_point_cloud(ply_path)
pointcloud = np.asarray(pcd.points)

# Process each frame
for frame in range(1, frames + 1):
    print(f"Processing frame {frame}...")
    
    # Read color and depth images
    color_image_path = os.path.join(gesture_folder_path, f'hand_color_image_{frame}.png')
    depth_image_path = os.path.join(gesture_folder_path, f'hand_depth_image_{frame}.png')
    json_file_path = os.path.join(gesture_folder_path, f'hand_intrinsics_and_transform_{frame}.json')

    color_image = cv2.imread(color_image_path, cv2.IMREAD_COLOR)
    h, w, _ = color_image.shape

    
    # Load intrinsics and transformation matrix
    intrinsics, transformation_matrix = load_intrinsics_and_transform(json_file_path)

    # Visualize the color image within the point cloud
    # visualize_color_image_in_pointcloud(pointcloud, color_image, intrinsics, transformation_matrix)
    
    # Restore the depth image from the point cloud
    depth_image = recreate_depth_image_from_pointcloud(color_image, pointcloud, intrinsics, transformation_matrix)

    # Save the restored depth image
    # restored_depth_path = depth_image_path.replace(".png", "_restored.png")
    restored_depth_path = depth_image_path
    cv2.imwrite(restored_depth_path, depth_image)
    print(f"Restored depth image saved to {restored_depth_path}")