import open3d as o3d
import numpy as np
import json
import os
import sys

# Set ROOT_DIR and add to Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from pose_detection.gesture_util import read_json_locked

# Paths to JSON files
file_path = "./.tmp/confidence.json"
gesture_path = './.tmp/gesture_vector.json'
pointcloud_path = './.tmp/pointcloud.ply'

def load_data():
    """Load object detection data from JSON file."""
    object_data = read_json_locked(file_path)
    gesture_data = read_json_locked(gesture_path)
    return object_data, gesture_data

def load_pointcloud():
    """Load a point cloud from file."""
    if os.path.exists(pointcloud_path):
        return o3d.io.read_point_cloud(pointcloud_path)
    else:
        print(f"⚠️ Warning: Point cloud file '{pointcloud_path}' not found.")
        return None

def create_arrow(start, direction, length=0.5, color=[0, 1, 0]):
    """Create an arrow for gesture visualization."""
    direction = np.array(direction) / np.linalg.norm(direction) * length
    end = start + direction
    arrow = o3d.geometry.LineSet()
    arrow.points = o3d.utility.Vector3dVector([start, end])
    arrow.lines = o3d.utility.Vector2iVector([[0, 1]])
    arrow.colors = o3d.utility.Vector3dVector([color])
    return arrow

def create_cone(base, direction, angle=15, height=1, color=[0, 0.5, 0]):
    """Create a 3D confidence cone aligned with the pointing vector."""
    direction = np.array(direction)
    direction = direction / np.linalg.norm(direction)

    # Create cone geometry
    cone = o3d.geometry.TriangleMesh.create_cone(radius=np.tan(np.radians(angle)) * height, height=height)
    cone.paint_uniform_color(color)

    # Calculate rotation matrix
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(z_axis, direction)
    if np.linalg.norm(rotation_axis) > 1e-6:
        rotation_axis /= np.linalg.norm(rotation_axis)
    angle_between = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle_between)

    # Apply rotation and translate to the correct position
    cone.rotate(R, center=(0, 0, 0))
    cone.translate(base + direction * (height / 2))  # Align base of cone with origin point

    return cone
def create_sphere_marker(center, radius=0.05, color=[1, 0, 0]):
    """
    Create a small sphere at the object's center position.

    Args:
        center (list): [x, y, z] center of the object.
        radius (float): Radius of the sphere.
        color (list): RGB color for the sphere.

    Returns:
        o3d.geometry.TriangleMesh: Sphere marker.
    """
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.paint_uniform_color(color)
    sphere.translate(center)  # Move sphere to the object's center
    return sphere

def visualize_objects():
    """Visualize objects' center points, gesture direction, and confidence cone in Open3D."""
    objects, gesture = load_data()
    pointcloud = load_pointcloud()
    vis_objects = []

    # Add spheres at object centers
    for obj in objects:
        if "center_3d" in obj:
            center = np.array(obj["center_3d"])
            sphere_marker = create_sphere_marker(center, radius=0.1, color=[1, 0, 0])  # Red sphere for object centers
            vis_objects.append(sphere_marker)

    # Draw pointing vector if available
    if "origin_world" in gesture and "direction_world" in gesture:
        center = np.array(gesture["origin_world"])
        pointing_vector = np.array(gesture["direction_world"])
        vis_objects.append(create_arrow(center, pointing_vector, length=1.0, color=[0, 1, 0]))  # Green arrow
        vis_objects.append(create_cone(center, pointing_vector, angle=20, height=0.7))  # Green cone

    # Add point cloud to the scene
    if pointcloud:
        vis_objects.append(pointcloud)

    # Show visualization
    o3d.visualization.draw_geometries(vis_objects)

# Run the visualization
visualize_objects()