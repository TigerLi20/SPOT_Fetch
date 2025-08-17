# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Interactive Arm Grasp and WASD Control: Grasp an object, then move the arm with WASD controls while holding it."""

import argparse # Library thats alows for smooth command line argument parsing, making it easier to place different inputs when running the program. 
import curses
import logging # Library that provides a way to log messages, such as debug, info, warning, error, and critical messages. It is used to track events that happen when the program runs.
import os
import signal
import sys # Library that gives the user access to conventional system functions in the Python library, letting them interact with the operating system and runtime environment. 
import threading
import time # Library that provides various time-related functions, such as sleep, time, and date functions. It is used to control the timing of events in the program, such as waiting for a certain amount of time before executing a command or checking the status of the robot.

import cv2 # Library used to help us process and manipulate the image, specifically displaying the image, allowing users to click on the image, and drawing line on the image. 
import numpy as np # Main library that helps us process and manipulate the image data we obtain from the robot's cameras. 

import bosdyn.client # Library for the BD client module, containing the main functions needed to communicate with and control the Boston Dynamics robots.
import bosdyn.client.estop # Library that provides the E-stop client for the robot, in this case used to checkthe E-stop status of the robot.
import bosdyn.client.lease # Library that provides the LeaseClient, allowing us to manage the lease on the robot, which is required to use the robot.
import bosdyn.client.util # Library that helps with command line arguments, terminal logging, and safe robot connection.
from bosdyn.api import estop_pb2, geometry_pb2, image_pb2, manipulation_api_pb2 # Imports the necessary protocol buffers for the E-stop, geometry, image, and manipulation API messages used in the program, allowing for effective and platform neutral communication with the robot.
from bosdyn.client.estop import EstopClient # Library that provides the E-stop client for the robot, in this case used to checkthe E-stop status of the robot.
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, get_vision_tform_body, math_helpers
from bosdyn.client.image import ImageClient # Library that provides the ImageClient, allowing us to connect and use the robot's cameras.
from bosdyn.client.manipulation_api_client import ManipulationApiClient # Library that provides the ManipulationClient, allowing us to manipulate the robot physically. 
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand # Library that provides the RobotCommandClient, allowing us to send commands to the robot, such as moving the arm or gripper.
from bosdyn.client.robot_state import RobotStateClient # Library that provides the RobotStateClient allowing us obtain information on the robots state, including its position and orientation.

LOGGER = logging.getLogger()
VELOCITY_CMD_DURATION = 0.5
COMMAND_INPUT_RATE = 0.1
VELOCITY_HAND_NORMALIZED = 0.5
VELOCITY_ANGULAR_HAND = 1.0

g_image_click = None
g_image_display = None

def verify_estop(robot):
    client = robot.ensure_client(EstopClient.default_service_name) # Creates an E-Stop client to check for the e-stop status of the robot.
    if client.get_status().stop_level != estop_pb2.ESTOP_LEVEL_NONE: # Checks if the robot's e-stop level is anything other than "none".
        error_message = 'Robot is estopped. Please use an external E-Stop client, such as the estop SDK example, to configure E-Stop.' 
        robot.logger.error(error_message) # If it isn't none, then it prints an error message to the robots logger and throws an exception.
        raise Exception(error_message)

def cv_mouse_callback(event, x, y, flags, param):
    global g_image_click, g_image_display # The same global variables
    clone = g_image_display.copy() # Makes a clone of the g_inage_display so it can be used for editing while preserving the original.
    if event == cv2.EVENT_LBUTTONUP: # If the image is clicked, the coordinates it was click at are stored in g_image_click.
        g_image_click = (x, y)
    else: # If it hasn't been clicked yet, then a pair of horizontal and vertical crosshair lines are drawn on the image a the current mousse position, showing the potontial grasp point.
        color = (30, 30, 30)
        thickness = 2
        image_title = 'Click to grasp'
        height = clone.shape[0]
        width = clone.shape[1]
        cv2.line(clone, (0, y), (width, y), color, thickness)
        cv2.line(clone, (x, 0), (x, height), color, thickness)
        cv2.imshow(image_title, clone) 

def add_grasp_constraint(config, grasp, robot_state_client): # Executes any grasp constriants that the user may have specified in the command line arguments.
    use_vector_constraint = config.force_top_down_grasp or config.force_horizontal_grasp
    grasp.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME
    if use_vector_constraint:
        if config.force_top_down_grasp:
            axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)
            axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=-1)
        if config.force_horizontal_grasp:
            axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=0, y=1, z=0)
            axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=1)
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(axis_on_gripper_ewrt_gripper)
        constraint.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(axis_to_align_with_ewrt_vo)
        constraint.vector_alignment_with_tolerance.threshold_radians = 0.17
    elif config.force_45_angle_grasp:
        robot_state = robot_state_client.get_robot_state()
        vision_T_body = get_vision_tform_body(robot_state.kinematic_state.transforms_snapshot)
        body_Q_grasp = math_helpers.Quat.from_pitch(0.785398)
        vision_Q_grasp = vision_T_body.rotation * body_Q_grasp
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.rotation_with_tolerance.rotation_ewrt_frame.CopyFrom(vision_Q_grasp.to_proto())
        constraint.rotation_with_tolerance.threshold_radians = 0.17
    elif config.force_squeeze_grasp:
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.squeeze_grasp.SetInParent()

def grasp_then_wasd(config):
    bosdyn.client.util.setup_logging(config.verbose) # Sets up logging for the program, printing debug, warning, error messages to the terminal; config.verbose controls how "verbose" the messages are.
    sdk = bosdyn.client.create_standard_sdk('ArmGraspWASDClient') # Creates a stand SDK client for connecting and issuing commands to the SPOT robot.
    robot = sdk.create_robot(config.hostname) # Creates a robot object that represents your physical robot and connects to that robot via the IP address specified in config.hostname.
    bosdyn.client.util.authenticate(robot) # Checks that the robot is authenticated via username and password (command line arguments or env variables).
    robot.time_sync.wait_for_sync() # Waits for the robot and the controling computer ti be properly time synchronized.
    assert robot.has_arm(), 'Robot requires an arm to run this program.' # Stops the program is no arm is detected n the robot.
    verify_estop(robot) # Calls the verify_estop function to ensure the robot is not estopped. 
    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name) # Creates or uses an existing LeaseClient to manage the robot's lease, which is required to use the robot. 
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name) # Creates or retrieves an existing RobotStateClient to allow us to obtain information on the robots state, such as its position and orientation. 
    image_client = robot.ensure_client(ImageClient.default_service_name) #  Creates or retrieves an existing ImageClient to allow us to obtain images from the robots cameras. 
    manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name) # Creates or retrieves an existing ManipulationClient to allow us to manipulate the robot's arm and gripper. (more specific, complicated movements)
    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True): # Creates a context manager, keepin the lease on the robot alive while we are running the program. It also requires the lease to be acquired before running the program, and returns the lease at the end of the program.
        robot.logger.info('Powering on robot...')
        robot.power_on(timeout_sec=20) # Powers on robot, within 20 seconds.
        assert robot.is_powered_on(), 'Robot power on failed.'
        robot.logger.info('Robot powered on.') 
        robot.logger.info('Commanding robot to stand...')
        command_client = robot.ensure_client(RobotCommandClient.default_service_name) # Creates or retrieves an existing RobotCommandClient, allowing us to send commands to the robot. (more general, basic movements)
        blocking_stand(command_client, timeout_sec=10) # Commands the robot to stand, within 10 seconds.
        robot.logger.info('Robot standing.') 
        # Grasp phase
        robot.logger.info('Getting an image from: %s', config.image_source) 
        image_responses = image_client.get_image_from_sources([config.image_source]) # Gets an image from the camera specified in config.image_source, back from the command lin argument.
        if len(image_responses) != 1: # Checks if the number of images returned is not equal to 1, which is the expected number of images.
            print(f'Got invalid number of images: {len(image_responses)}')
            print(image_responses)
            assert False # Forces grasp_then_wasd to return False, stopping the program.
        image = image_responses[0] # Stores the first (an only) image returned in the image variable.
        # Checks what data type each pixel in the image uses. First checks if is a 16-bit unsigned integer (depth image), then checks if it is an 8-bit unsigned integer (color image).
        dtype = np.uint16 if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16 else np.uint8
        # Organizes the raw image data into a numpy array uisng our dtype, cutting every 16 bits or 8 bits (2 bytes or 1 byte) into a single pixel.
        img = np.frombuffer(image.shot.image.data, dtype=dtype)
        if image.shot.image.format == image_pb2.Image.FORMAT_RAW: # Checks if the image from the robot is in raw format.
            img = img.reshape(image.shot.image.rows, image.shot.image.cols) # Reshapes the image data into a 2D array, with the number of rows and columns specified in the image metadata, correspnding to the height and width of the image.
        else:
            img = cv2.imdecode(img, -1) # If it compressed (JPEG/PNG), decodes the image data using OpenCV's imdecode function, converting it into a (width, height) or (width, height, channels) array for grayscale or color respectively. 
        robot.logger.info('Click on an object to start grasping...')
        image_title = 'Click to grasp'
        cv2.namedWindow(image_title) # Creates a new windown on the computer to display the image.
        cv2.setMouseCallback(image_title, cv_mouse_callback) # Creates an event handler for mouses interactions on the image, allowing us to click on the image to select a point. Sets up the H/V lines.
        global g_image_click, g_image_display # Declares that these global variables will be used to store the clicked coordinates and the displayed image.
        g_image_display = img # Stores our img data in the global variable g_image_display, so it can be accessed by the mouse callback function.
        cv2.imshow(image_title, g_image_display) # Finally, after all the seup, showing the image on the screen for the user to interact with. Backed up by the cv2.setMouseCallback functionality, which allows us to click on the image and select a point.
        while g_image_click is None: # This loop just waits for the user to click on the image, aka waiting for cv_mouse_callback to assign coordinates to g_image_click.
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'): # If the user presses "q" or "Q", it exits the program.
                print('"q" pressed, exiting.')
                exit(0)
        cv2.destroyAllWindows() # Closes the image window after the user clicks on the coordinates of interest.
        robot.logger.info(f'Picking object at image location ({g_image_click[0]}, {g_image_click[1]})')
        pick_vec = geometry_pb2.Vec2(x=g_image_click[0], y=g_image_click[1]) # Converts the clicked coordinates into a Vec2 message understandable by the robot. 
        grasp = manipulation_api_pb2.PickObjectInImage( # Creates a PickObjectInImage message (a protocol buffer) to send to the robot, specifying the clicked coordinates, robot/camera transforms, camera frame name, and camera intrinsics. 
            pixel_xy=pick_vec, transforms_snapshot_for_camera=image.shot.transforms_snapshot, # Clicked coordinates, snapshot of all the coordinate frame transforms at the time the image was taken (aka the relative position and orientation of arm, camera, body, world).
            frame_name_image_sensor=image.shot.frame_name_image_sensor, # Frame name of the camera that took the image, used to identify which camera the image was taken from. (ex: left_fisheye_image).
            camera_model=image.source.pinhole) # Camera intrinsics
        add_grasp_constraint(config, grasp, robot_state_client) # Excecutes any grasp constraints the user specified in the command line arguments.
        grasp_request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp) # Creates a request for the manipulation API, packaged with the grasp message we ust created, containing all the information to pick the correct object/location.
        cmd_response = manipulation_api_client.manipulation_api_command( # Sends that grasp_request to the manipulation API.
            manipulation_api_request=grasp_request)
        robot.logger.info('Executing grasp...')
        while True: # While loop that checks the status of the grasp command until it succeeds or fails.
            # Creates a feedback request with the specific manipulation command ID we just sent, so we can get the current state of the grasp.
            feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                manipulation_cmd_id=cmd_response.manipulation_cmd_id)
            # Sends that feedback request to the manipulation API and gets a response.
            response = manipulation_api_client.manipulation_api_feedback_command(
                manipulation_api_feedback_request=feedback_request)
            print(f'Current state: {manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state)}') # Prints the status of the grasp to the terminal.
            if response.current_state in [manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED, manipulation_api_pb2.MANIP_STATE_GRASP_FAILED]:  # If the grasp succeeded or failed, it breaks the loop.
                break
            time.sleep(0.25) # Waits for 0.25 seconds before checking the status again, to avoid flooding the manipulation API with requests.
        robot.logger.info('Finished grasp. You can now move the arm using WASD controls.')
        # The following commands are commented out to disable automatic actions after grasping:
        command_client.robot_command(RobotCommandBuilder.claw_gripper_open_command()) # Opens the gripper to reposition the grip on the object of interest.
        time.sleep(0.5)
        command_client.robot_command(RobotCommandBuilder.claw_gripper_close_command()) # Recloses the gripper to strengthen the grip on the object of interest.
        time.sleep(0.5)
        command_client.robot_command(RobotCommandBuilder.arm_stow_command()) # Stows the arm, completing the fetch.
        # time.sleep(0.5)
        # Enter WASD control mode
        wasd_arm_control_loop(robot, command_client, robot_state_client, image_client, manipulation_api_client, config) # Goes to the wasd arm control loop, allowinf for further functionality after the main grasp.

def wasd_arm_control_loop(robot, command_client, robot_state_client=None, image_client=None, manipulation_api_client=None, config=None):
    print('\nWASD Arm Control:')
    print('w/s: move out/in, a/d: rotate ccw/cw, r/f: up/down,') # Moves arm out/in, rotates CCW/CW, moves arm up/down
    print('m: close gripper, n: open gripper, z: stow arm, x: unstow arm, g: grasp again, q: quit')
    # Allows for an interaxctive control loop, allowing the user to enter as many commands as they wish, one at a time.
    while True:
        key = input('Command: ').strip().lower() # Prompts the user to enter a command, and stores the input in key.
        if key == 'q': # If user enters "q", then it exits the program.
            print('Exiting WASD arm control.')
            break
        elif key == 'w': # 
            send_arm_cylindrical_velocity(command_client, v_r=VELOCITY_HAND_NORMALIZED)
        elif key == 's':
            send_arm_cylindrical_velocity(command_client, v_r=-VELOCITY_HAND_NORMALIZED)
        elif key == 'a':
            send_arm_cylindrical_velocity(command_client, v_theta=VELOCITY_HAND_NORMALIZED)
        elif key == 'd':
            send_arm_cylindrical_velocity(command_client, v_theta=-VELOCITY_HAND_NORMALIZED)
        elif key == 'r':
            send_arm_cylindrical_velocity(command_client, v_z=VELOCITY_HAND_NORMALIZED)
        elif key == 'f':
            send_arm_cylindrical_velocity(command_client, v_z=-VELOCITY_HAND_NORMALIZED)
        elif key == 'm': # Commands the gripper to close.
            command_client.robot_command(RobotCommandBuilder.claw_gripper_close_command())
        elif key == 'n': # Commands the gripper to open.
            command_client.robot_command(RobotCommandBuilder.claw_gripper_open_command())
        elif key == 'z': # Commands the arm to stow.
            command_client.robot_command(RobotCommandBuilder.arm_stow_command())
        elif key == 'x': # Commands the arm to unstow and get ready for movement.
            command_client.robot_command(RobotCommandBuilder.arm_ready_command())
        elif key == 'g': # Commands the robot to grasp again, using the same method as before.
            # Same logic as the grasp phase of the grasp_then_wasd function, above.
            if robot_state_client and image_client and manipulation_api_client and config: 
                print('Taking a new image for grasping...')
                image_responses = image_client.get_image_from_sources([config.image_source])
                if len(image_responses) != 1:
                    print(f'Got invalid number of images: {len(image_responses)}')
                    print(image_responses)
                    continue
                image = image_responses[0]
                dtype = np.uint16 if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16 else np.uint8
                img = np.frombuffer(image.shot.image.data, dtype=dtype)
                if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
                    img = img.reshape(image.shot.image.rows, image.shot.image.cols)
                else:
                    img = cv2.imdecode(img, -1)
                image_title = 'Click to grasp'
                cv2.namedWindow(image_title)
                cv2.setMouseCallback(image_title, cv_mouse_callback)
                global g_image_click, g_image_display
                g_image_click = None
                g_image_display = img
                cv2.imshow(image_title, g_image_display)
                while g_image_click is None:
                    key2 = cv2.waitKey(1) & 0xFF
                    if key2 == ord('q') or key2 == ord('Q'):
                        print('"q" pressed, exiting grasp.')
                        cv2.destroyAllWindows()
                        break
                cv2.destroyAllWindows()
                if g_image_click is not None:
                    print(f'Picking object at image location ({g_image_click[0]}, {g_image_click[1]})')
                    pick_vec = geometry_pb2.Vec2(x=g_image_click[0], y=g_image_click[1])
                    grasp = manipulation_api_pb2.PickObjectInImage(
                        pixel_xy=pick_vec, transforms_snapshot_for_camera=image.shot.transforms_snapshot,
                        frame_name_image_sensor=image.shot.frame_name_image_sensor,
                        camera_model=image.source.pinhole)
                    add_grasp_constraint(config, grasp, robot_state_client)
                    grasp_request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)
                    cmd_response = manipulation_api_client.manipulation_api_command(
                        manipulation_api_request=grasp_request)
                    print('Executing grasp...')
                    while True:
                        feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                            manipulation_cmd_id=cmd_response.manipulation_cmd_id)
                        response = manipulation_api_client.manipulation_api_feedback_command(
                            manipulation_api_feedback_request=feedback_request)
                        print(f'Current state: {manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state)}')
                        if response.current_state in [manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED, manipulation_api_pb2.MANIP_STATE_GRASP_FAILED]:
                            break
                        time.sleep(0.25)
                    print('Finished grasp. You can now move the arm using WASD controls.')
            else:
                print('Grasping again is not available in this mode.')
        else:
            print('Unknown command.')
        time.sleep(COMMAND_INPUT_RATE)

def send_arm_cylindrical_velocity(command_client, v_r=0.0, v_theta=0.0, v_z=0.0): # Sends a velocity command to the arm in cylindrical coordinates, based on the radial, theta, and z values.
    from bosdyn.api import arm_command_pb2, robot_command_pb2, geometry_pb2 # Imports the necessary protocol buffers for communication. 
    import bosdyn.client.util
    arm_velocity_cmd = arm_command_pb2.ArmVelocityCommand.Request() # Creates an ArmVelocityCommeand message to send to the robot after being packed up, as below. 
    
    # Create a SE3Velocity message for cartesian velocity
    se3_velocity = geometry_pb2.SE3Velocity() # Creates a SE3Velocity message, compaible with the robots API, to store the cartesian velocity values.
    se3_velocity.linear.x = v_r # Converts radial to x.
    se3_velocity.linear.y = v_theta # Converts theta to y.
    se3_velocity.linear.z = v_z # Keeps z as z.
    # No angular velocity, so all set to 0.
    se3_velocity.angular.x = 0.0 
    se3_velocity.angular.y = 0.0
    se3_velocity.angular.z = 0.0
    
    arm_velocity_cmd.cartesian_velocity.CopyFrom(se3_velocity) #Copies the se3velocity into the cartesian_velocity field of the arm velocity command. 
    arm_velocity_cmd.duration.CopyFrom( # Sets the duration of the velocity command to VELOCITY_CMD_DURATION, defined at the top, globally.
        bosdyn.client.util.seconds_to_duration(VELOCITY_CMD_DURATION)
    )
    robot_cmd = robot_command_pb2.RobotCommand() # Creates a RobotCommand message to send to the robot, which can contain multiple commands, such as arm and gripper commands.
    robot_cmd.synchronized_command.arm_command.arm_velocity_command.CopyFrom(arm_velocity_cmd) # Copies the arm velocity command into the arm command field of the general robot command.
    command_client.robot_command(robot_cmd) # Sends the rbot command to the robot via the command client.

def main():
    parser = argparse.ArgumentParser() # Creates a parser variable that be used to define command line arguments.
    bosdyn.client.util.add_base_arguments(parser) # Provides standard Boston Dynamics command line arguments for connecting with the SPOT robot, such as hostname, username, and password
    parser.add_argument('-i', '--image-source', help='Get image from source', default='frontleft_fisheye_image') # THese are all command line arguments that can change how SPOT functions in the program 
    parser.add_argument('-t', '--force-top-down-grasp', help='Force the robot to use a top-down grasp (vector_alignment demo)', action='store_true') # Ex. camera choice, grasp angle, grasp strength
    parser.add_argument('-f', '--force-horizontal-grasp', help='Force the robot to use a horizontal grasp (vector_alignment demo)', action='store_true')
    parser.add_argument('-r', '--force-45-angle-grasp', help='Force the robot to use a 45 degree angled down grasp (rotation_with_tolerance demo)', action='store_true')
    parser.add_argument('-s', '--force-squeeze-grasp', help='Force the robot to use a squeeze grasp', action='store_true')
    options = parser.parse_args() # Stores all the command line arguments the user entered into an options variable        
    num = 0
    if options.force_top_down_grasp: # Checks if each of the grasp arguments was entered by the user, and if so, num += 1
        num += 1
    if options.force_horizontal_grasp:
        num += 1
    if options.force_45_angle_grasp:
        num += 1
    if options.force_squeeze_grasp:
        num += 1
    if num > 1: # If more than one grasp argumenbt is specified, errors and instructs the user to only use one instead
        print('Error: cannot force more than one type of grasp.  Choose only one.')
        sys.exit(1) # Automatically exits the program
    try:
        grasp_then_wasd(options) # Runs the grasp_then_wasf function, which will perform the main grasp then move to wasd controls
        return True # If everything runs smoothly, it will return True
    except Exception as exc: # If any exception occurs
        logger = bosdyn.client.util.get_logger() # Creates a logger object for logging the error message(s)
        logger.exception('Threw an exception') # Logs the excpetion that was thrown
        return False

if __name__ == '__main__': # Just checks to make sure that you are running the program directly, and not as a module in another script
    if not main(): # Calls the main function, lets it run and return False or True. If it returns False (due to an exception), it will print the error message and exit the program
        sys.exit(1) # Exits the program
