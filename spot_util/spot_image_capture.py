import sys
from bosdyn.api import image_pb2
import bosdyn.client
import bosdyn.client.robot_command
import bosdyn.client.util
import bosdyn.client.lease
import bosdyn.api.gripper_command_pb2
from bosdyn.client.image import ImageClient, build_image_request

from bosdyn.client.robot_command import RobotCommandClient, blocking_stand, RobotCommandBuilder, blocking_sit
from bosdyn.client.robot_state import RobotStateClient
import cv2
import numpy as np
from scipy import ndimage
import os
import time



ROTATION_ANGLE = {
    'hand_color_image': 0,
    'back_fisheye_image': 0,
    'frontleft_fisheye_image': -78,
    'frontright_fisheye_image': -102,
    'left_fisheye_image': 0,
    'right_fisheye_image': 180,
    'hand_image': -90
}

def pixel_format_type_strings():
    names = image_pb2.Image.PixelFormat.keys()
    return names[1:]

def pixel_format_string_to_enum(enum_string):
    return dict(image_pb2.Image.PixelFormat.items()).get(enum_string)

def capture_image(hostname='138.16.161.21', ):
    # Hardcoded parameters
    list_images = False         # Set to True if you want to list image sources
    auto_rotate = False         # Set to True to auto-rotate images
    image_sources = ['hand_color_image',  'hand_depth_in_hand_color_frame']  # Modify image sources
    image_service = ImageClient.default_service_name  # Default image service
    pixel_format = 'RGBA_U8'     # Set the desired pixel format, e.g., 'RGB_U8', 'GREYSCALE_U8', etc.
    output_directory = os.path.dirname(os.path.realpath(__file__))  # Directory to save images


    
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


        # Power on the robot. This call will block until the power is on. 
        robot.logger.info("Powering on robot... This may take a several seconds.")
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), "Robot power on failed."
        robot.logger.info("Robot powered on.")
        
        # # Command robot to open gripper
        gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(1.0)
        cmd_id = command_client.robot_command(gripper_command)
        
        image_source = image_client.get_image_from_sources(['hand_color_image'])
        intrinsics = image_source[0].source.pinhole.intrinsics
        print(intrinsics)
        # Command robot to stand
        print("Commanding robot to stand...")
        blocking_stand(command_client, timeout_sec=10)
        time.sleep(1)
        print("Robot is standing.")

        # List image sources on robot if specified.
        if list_images:
            image_sources_list = image_client.list_image_sources()
            print("Image sources:")
            for source in image_sources_list:
                print("\t" + source.name)
            return True

        # Ensure output directory exists
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        # Capture and save images to disk
        if image_sources:
            pixel_format_enum = pixel_format_string_to_enum(pixel_format)
            image_request = [
                build_image_request(source, pixel_format=pixel_format_enum)
                for source in image_sources
            ]
            image_responses = image_client.get_image(image_request)

            for image in image_responses:
                num_bytes = 1  # Assume a default of 1 byte encodings.
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

                # Save the image
                image_saved_path = os.path.join(output_directory, f"{image.source.name.replace('/', '')}_{image.shot.acquisition_time.seconds}{extension}")
                print("Saving image:", image_saved_path)
                cv2.imwrite(image_saved_path, img)

        # Command robot to sit down after capturing the images
        print("Commanding robot to sit...")
        blocking_sit(command_client, timeout_sec=10)
        print("Robot is now sitting.")

    return True

def main():
    capture_image()

if __name__ == "__main__":
    if not main():
        sys.exit(1)
        

# gouger 21
# snouter 24
