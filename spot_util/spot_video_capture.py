import sys
import cv2
import os
import numpy as np
import signal
from bosdyn.api import image_pb2
import bosdyn.client
import bosdyn.client.robot_command
import bosdyn.client.util
import bosdyn.client.lease
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder
from bosdyn.client.robot_state import RobotStateClient
from scipy import ndimage
import time

image_sources = ['back_depth', 'back_depth_in_visual_frame', 'back_fisheye_image', 'frontleft_depth',
                 'frontleft_depth_in_visual_frame', 'frontleft_fisheye_image', 'frontright_depth',
                 'frontright_depth_in_visual_frame', 'frontright_fisheye_image', 'hand_color_image',
                 'hand_color_in_hand_depth_frame', 'hand_depth', 'hand_depth_in_hand_color_frame', 'hand_image',
                 'left_depth', 'left_depth_in_visual_frame', 'left_fisheye_image', 'right_depth',
                 'right_depth_in_visual_frame', 'right_fisheye_image']  # Replace with your desired image sources

ROTATION_ANGLE = {
    'hand_color_image': 0,
    'back_fisheye_image': 0,
    'frontleft_fisheye_image': -78,
    'frontright_fisheye_image': -102,
    'left_fisheye_image': 0,
    'right_fisheye_image': 180,
    'hand_image': -90
}

def pixel_format_string_to_enum(enum_string):
    return dict(image_pb2.Image.PixelFormat.items()).get(enum_string)

def signal_handler(sig, frame):
    print("\nExiting...")
    sys.exit(0)

def main():
    hostname = '138.16.161.24'  # Replace with your robot's hostname or IP address
    image_sources = ['hand_color_image', 'hand_depth_in_hand_color_frame']  # Modify image sources
    image_service = ImageClient.default_service_name  # Default image service
    pixel_format = 'RGBA_U8'  # Desired pixel format
    output_directory = '/Users/ivy/Desktop/gesture_eval_test_2/'  # Directory to save images
    output_video_path = '/Users/ivy/Desktop/gesture_eval_test_2/video.mp4'  # Path to save the final video
    
    frame_rate = 20  # Initial video frame rate
    wait_interval = 1 / frame_rate
    auto_rotate = False  
    duration = 60  # change seconds 

    # Create robot object with an image client.
    sdk = bosdyn.client.create_standard_sdk('imageClient')
    robot = sdk.create_robot(hostname)
    bosdyn.client.util.authenticate(robot)
    robot.sync_with_directory()
    robot.time_sync.wait_for_sync()

    # Lease management
    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    try:
        lease = lease_client.acquire()
    except bosdyn.client.lease.ResourceAlreadyClaimedError:
        lease = lease_client.take()

    # Keep the lease alive during operations
    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
        image_client = robot.ensure_client(image_service)
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)

        # Power on the robot
        robot.logger.info("Powering on robot... This may take several seconds.")
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), "Robot power on failed."
        robot.logger.info("Robot powered on.")
        
        # Command robot to open gripper
        gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(1.0)
        cmd_id = command_client.robot_command(gripper_command)
        
        # Start timing the loop
        start_time = time.time()
        
        # Ensure output directory exists
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # List to store captured images
        image_list = []
        frame_count = 0

        # Handle keyboard interruption (Ctrl+C) gracefully
        signal.signal(signal.SIGINT, signal_handler)
        
        # Start capturing images in a while loop
        print("Starting image capture. Press 'q' to exit the loop and compile images into a video.")
        while True:
            loop_start_time = time.time()  # Start time of loop for dynamic frame rate
            pixel_format_enum = pixel_format_string_to_enum(pixel_format)
            image_request = [
                build_image_request(source, pixel_format=pixel_format_enum)
                for source in image_sources
            ]
            image_responses = image_client.get_image(image_request)

            for image in image_responses:
                num_bytes = 1  # Assume a default of 1 byte encoding
                if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
                    dtype = np.uint16
                    extension = ".png"
                else:
                    if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
                        num_bytes = 3
                    elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGBA_U8:
                        num_bytes = 4
                    dtype = np.uint8
                    extension = ".jpg"

                img = np.frombuffer(image.shot.image.data, dtype=dtype)
                if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
                    try:
                        img = img.reshape((image.shot.image.rows, image.shot.image.cols, num_bytes))
                    except ValueError:
                        img = cv2.imdecode(img, -1)
                else:
                    img = cv2.imdecode(img, -1)

                if auto_rotate:
                    img = ndimage.rotate(img, ROTATION_ANGLE[image.source.name])
                
                # Save the image to the current directory
                image_filename = os.path.join(output_directory, f"{image.source.name}_{frame_count:04d}{extension}")
                print(f"Saving image {image_filename}")
                cv2.imwrite(image_filename, img)

                # Add the image to the list
                image_list.append(image_filename)

            frame_count += 1
            loop_end_time = time.time()  # End time of loop

            # Calculate dynamic frame rate based on loop speed
            loop_duration = loop_end_time - loop_start_time
            if loop_duration > 0:
                dynamic_frame_rate = 1 / loop_duration
                print(f"Loop Frame Rate: {dynamic_frame_rate:.2f} FPS")
                frame_rate = min(frame_rate, dynamic_frame_rate)  # Adjust frame rate dynamically

            # Exit the loop if 'q' is pressed or after duration expires
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting image capture loop.")
                break
            if time.time() - start_time > duration:
                print("Exiting after duration.")
                break
        time.sleep(wait_interval)
        
        # Command robot to close gripper
        gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(0.0)
        cmd_id = command_client.robot_command(gripper_command)
        
        # Compile the saved images into a video
        if image_list:
            print(f"Compiling {len(image_list)} images into a video...")
            frame = cv2.imread(image_list[0])
            height, width, layers = frame.shape
            video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'MP4V'), frame_rate, (width, height))

            for image_path in image_list:
                frame = cv2.imread(image_path)
                video_writer.write(frame)

            video_writer.release()
            print(f"Video saved to {output_video_path}")

    return True

if __name__ == "__main__":
    if not main():
        sys.exit()