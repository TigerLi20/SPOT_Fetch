import threading
import queue
import time
import os
import json
import numpy as np
import open3d as o3d
import cv2
import bosdyn.client
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.frame_helpers import get_a_tform_b, ODOM_FRAME_NAME
import socket
import base64

# Shared queue for multithreaded communication
data_queue = queue.Queue(maxsize=10)

def save_pointcloud(points, colors, filename):
    """Save a point cloud to a file."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Saved point cloud to {filename}")

def load_pointcloud(filename):
    """Load a point cloud from a file."""
    if os.path.exists(filename):
        pcd = o3d.io.read_point_cloud(filename)
        print(f"Loaded point cloud from {filename}")
        return np.asarray(pcd.points), np.asarray(pcd.colors)
    else:
        print(f"File {filename} does not exist. Starting with an empty point cloud.")
        return np.empty((0, 3)), np.empty((0, 3))

def save_intrinsics_and_transform(intrinsics, transform, filename):
    """Save camera intrinsics and transformation data."""
    data = {
        "intrinsics": intrinsics.tolist(),
        "transform": {
            "rotation_matrix": transform.rotation.to_matrix().tolist(),
            "translation_vector": [transform.position.x, transform.position.y, transform.position.z]
        }
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved intrinsics and transform data to {filename}")

def generate_pointcloud_from_depth_and_greyscale(depth, greyscale, intrinsics):
    """Generate a point cloud from depth and greyscale images."""
    height, width = depth.shape
    x_indices, y_indices = np.meshgrid(np.arange(width), np.arange(height))
    z = depth / 1000.0

    x = (x_indices - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (y_indices - intrinsics[1, 2]) * z / intrinsics[1, 1]
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    colors = np.repeat(greyscale.reshape(-1, 1) / 255.0, 3, axis=1) if len(greyscale.shape) == 2 else greyscale.reshape(-1, 3) / 255.0
    colors = colors[:, [2, 1, 0]]  # BGR to RGB

    valid_mask = (points[:, 2] > 0) & (points[:, 2] < 10)
    return points[valid_mask], colors[valid_mask]

def get_img_and_ply(robot, frame_num=0, image_pairs=None):
    """
    Retrieve images from Spot, generate point clouds, and return data.
    """
    if image_pairs is None:
        image_pairs = [['hand_depth_in_hand_color_frame', 'hand_color_image']]

    image_client = robot.ensure_client(ImageClient.default_service_name)
    all_points = []
    all_colors = []

    for depth_name, greyscale_name in image_pairs:
        try:
            print(f"Processing {depth_name} and {greyscale_name} (Frame {frame_num})...")

            # Get the depth image
            depth_response = image_client.get_image_from_sources([depth_name])[0]
            depth_image = np.frombuffer(depth_response.shot.image.data, dtype=np.uint16).reshape(
                depth_response.shot.image.rows, depth_response.shot.image.cols
            )

            # Get the greyscale image
            greyscale_response = image_client.get_image_from_sources([greyscale_name])[0]
            greyscale_image = cv2.imdecode(np.frombuffer(greyscale_response.shot.image.data, dtype=np.uint8), -1)

            # Get camera intrinsics
            depth_source = next((src for src in image_client.list_image_sources() if src.name == depth_name), None)
            if not depth_source:
                print(f"Depth source {depth_name} not found.")
                continue
            
            intrinsic = depth_source.pinhole.intrinsics
            fx, fy = intrinsic.focal_length.x, intrinsic.focal_length.y
            cx, cy = intrinsic.principal_point.x, intrinsic.principal_point.y
            intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

            # Generate point cloud
            points, colors = generate_pointcloud_from_depth_and_greyscale(depth_image, greyscale_image, intrinsics)

            # Transform points to the odometry frame
            camera_frame = depth_response.shot.frame_name_image_sensor
            frame_tree = depth_response.shot.transforms_snapshot
            odom_tform_camera = get_a_tform_b(frame_tree, ODOM_FRAME_NAME, camera_frame)

            rotation_matrix = odom_tform_camera.rotation.to_matrix()
            translation_vector = np.array([
                odom_tform_camera.position.x, 
                odom_tform_camera.position.y, 
                odom_tform_camera.position.z
            ])
            points = np.dot(points, rotation_matrix.T) + translation_vector

            return greyscale_image, depth_image, intrinsics, odom_tform_camera, points, colors

        except Exception as e:
            print(f"Error processing {depth_name} and {greyscale_name}: {e}")
            return None, None, None, None, None

def receive_images(robot, stop_event, save_image=True):
    """Receive images, depth images, and transformation matrices from Spot."""
    frame = 0
    while not stop_event.is_set():
        try:
            greyscale_image, depth_image, intrinsics, transform, points, colors = get_img_and_ply(robot, frame)
            if save_image:
                cv2.imwrite('spot_util/hand_color_image.png', greyscale_image)
                cv2.imwrite('spot_util/hand_depth_image.png', depth_image.astype(np.uint16),  [cv2.IMWRITE_PNG_COMPRESSION, 0])
                with open("spot_util/hand_transformation_intrinsic.json", "w") as f:
                    json.dump(transform, f, indent = 4)
                
            if points is not None and colors is not None:
                data = {
                    "image": greyscale_image,
                    "depth_image": depth_image,
                    "transformation_matrix": transform,
                    "points": points,
                    "colors": colors,
                }
                data_queue.put(data, block=True)
                print(f"Frame {frame}: Data added to queue.")
            frame += 1
            time.sleep(0.2)
        except Exception as e:
            print(f"Error receiving data: {e}")

def send_data(robot, HOST = "127.0.0.1", PORT = 65432):
    """Send images, depth images, and transformation matrices over a socket."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print(f"Server listening on {HOST}:{PORT}")
        conn, addr = server_socket.accept()
        print(f"Connected by {addr}")

        frame = 0
        while True:
            try:
                greyscale_image, depth_image, intrinsics, transform, _, _ = get_img_and_ply(robot, frame)
                if greyscale_image is not None and depth_image is not None and transform is not None:
                    # Prepare data
                    _, greyscale_buffer = cv2.imencode('.png', greyscale_image)
                    _, depth_buffer = cv2.imencode('.png', depth_image)
                    greyscale_base64 = base64.b64encode(greyscale_buffer.tobytes()).decode('utf-8')
                    depth_base64 = base64.b64encode(depth_buffer.tobytes()).decode('utf-8')

                    data = {
                        "image": greyscale_base64,
                        "depth_image": depth_base64,
                        "intrinsics": np.reshape(transform["intrinsics"], 9).tolist(),
                        "rotation_matrix":np.reshape(transform["transform"]["rotation_matrix"], 9).tolist(),
                        "translation_vector":np.reshape(transform["transform"]["translation_vector"], 3).tolist(),
                        }

                    # Send serialized data
                    serialized_data = json.dumps(data)
                    conn.sendall(len(serialized_data).to_bytes(4, 'big') + serialized_data.encode('utf-8'))
                    print(f"Frame {frame} sent.")
                    frame += 1
                    time.sleep(0.2)
            except Exception as e:
                print(f"Error sending data: {e}")
                break
            
def save_pointclouds(stop_event, save_interval=10):
    """Process data from the queue and save point clouds."""
    all_points, all_colors = load_pointcloud("final_pointcloud.ply")

    while not stop_event.is_set():
        try:
            data = data_queue.get(block=True)
            points = data["points"]
            colors = data["colors"]
            all_points = np.vstack([all_points, points])
            all_colors = np.vstack([all_colors, colors])

            if len(all_points) % save_interval == 0:
                save_pointcloud(all_points, all_colors, "final_pointcloud.ply")
        except Exception as e:
            print(f"Error saving point cloud: {e}")


def capture_image_thread(hostname="138.16.161.21"):
    """Main function to capture images and process point clouds."""
    sdk = bosdyn.client.create_standard_sdk("SpotGreyscalePointCloudClient")
    robot = sdk.create_robot(hostname)
    bosdyn.client.util.authenticate(robot)
    robot.sync_with_directory()
    robot.time_sync.wait_for_sync(timeout_sec=3.0)

    stop_event = threading.Event()

    receiver_thread = threading.Thread(target=receive_images, args=(robot, stop_event, True), daemon=True)
    saver_thread = threading.Thread(target=save_pointclouds, args=(stop_event,), daemon=True)

    receiver_thread.start()
    saver_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down gracefully...")
        stop_event.set()
        receiver_thread.join()
        saver_thread.join()
        print("All threads stopped.")

def capture_image_socket(hostname="127.0.0.1"):
    """Main function to capture images and send data."""
    sdk = bosdyn.client.create_standard_sdk("SpotGreyscalePointCloudClient")
    robot = sdk.create_robot(hostname)
    bosdyn.client.util.authenticate(robot)
    robot.sync_with_directory()
    robot.time_sync.wait_for_sync(timeout_sec=3.0)

    send_data(robot)
    
def capture_image(hostname="138.16.161.21", interval = 1, save_image=True, 
                  save_folder_path = "spot_util/"):
    sdk = bosdyn.client.create_standard_sdk("SpotGreyscalePointCloudClient")
    robot = sdk.create_robot(hostname)
    bosdyn.client.util.authenticate(robot)
    robot.sync_with_directory()
    robot.time_sync.wait_for_sync(timeout_sec=3.0)
    
    """Receive images, depth images, and transformation matrices from Spot."""
    frame = 0
    while True:
        try:
            greyscale_image, depth_image, intrinsics, transform, points, colors = get_img_and_ply(robot, frame)
            if save_image:
                
                # cv2.imwrite('spot_util/hand_color_image.png', greyscale_image)
                # cv2.imwrite('spot_util/hand_depth_image.png', depth_image.astype(np.uint16),  [cv2.IMWRITE_PNG_COMPRESSION, 0])
                # save_intrinsics_and_transform(intrinsics, transform, "spot_util/hand_intrinsics_and_transform.json")
                root_dir = "/home/xhe71/Desktop/LEGS_eval/system_belief_update/gouger/blue_white_binary_cube_0207/arm_approach_white/"
                cv2.imwrite(root_dir+f'/hand_color_image_{frame}.png', greyscale_image)
                cv2.imwrite(root_dir+f'/hand_depth_image_{frame}.png', depth_image.astype(np.uint16),  [cv2.IMWRITE_PNG_COMPRESSION, 0])
                save_intrinsics_and_transform(intrinsics, transform, root_dir+f"/hand_intrinsics_and_transform_{frame}.json")
                
            frame += 1
            time.sleep(interval)
        except Exception as e:
            print(f"Error receiving data: {e}")
            
            
if __name__ == "__main__":
    capture_image()