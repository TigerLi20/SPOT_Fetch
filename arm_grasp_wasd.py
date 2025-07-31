# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

# run with normal front-middle camera: python3 arm_grasp_wasd.py 128.148.140.20
# run with top color camera: python arm_grasp_wasd.py -i hand_color_image 128.148.140.20
# note: change the IP address for the desired SPOT robot

"""Interactive Arm Grasp and WASD Control: Grasp an object, then move the arm with WASD controls while holding it."""

import argparse
import curses
import logging
import os
import signal
import sys
import threading
import time

import cv2
import numpy as np

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import estop_pb2, geometry_pb2, image_pb2, manipulation_api_pb2
from bosdyn.client.estop import EstopClient
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, get_vision_tform_body, math_helpers
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand
from bosdyn.client.robot_state import RobotStateClient

LOGGER = logging.getLogger()
VELOCITY_CMD_DURATION = 0.5
COMMAND_INPUT_RATE = 0.1
VELOCITY_HAND_NORMALIZED = 0.5
VELOCITY_ANGULAR_HAND = 1.0

g_image_click = None
g_image_display = None

def verify_estop(robot):
    client = robot.ensure_client(EstopClient.default_service_name)
    if client.get_status().stop_level != estop_pb2.ESTOP_LEVEL_NONE:
        error_message = 'Robot is estopped. Please use an external E-Stop client, such as the estop SDK example, to configure E-Stop.'
        robot.logger.error(error_message)
        raise Exception(error_message)

def cv_mouse_callback(event, x, y, flags, param):
    global g_image_click, g_image_display
    clone = g_image_display.copy()
    if event == cv2.EVENT_LBUTTONUP:
        g_image_click = (x, y)
    else:
        color = (30, 30, 30)
        thickness = 2
        image_title = 'Click to grasp'
        height = clone.shape[0]
        width = clone.shape[1]
        cv2.line(clone, (0, y), (width, y), color, thickness)
        cv2.line(clone, (x, 0), (x, height), color, thickness)
        cv2.imshow(image_title, clone)

def add_grasp_constraint(config, grasp, robot_state_client):
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
    bosdyn.client.util.setup_logging(config.verbose)
    sdk = bosdyn.client.create_standard_sdk('ArmGraspWASDClient')
    robot = sdk.create_robot(config.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()
    assert robot.has_arm(), 'Robot requires an arm to run this example.'
    verify_estop(robot)
    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)
    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        robot.logger.info('Powering on robot...')
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), 'Robot power on failed.'
        robot.logger.info('Robot powered on.')
        robot.logger.info('Commanding robot to stand...')
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        blocking_stand(command_client, timeout_sec=10)
        robot.logger.info('Robot standing.')
        # Grasp phase
        robot.logger.info('Getting an image from: %s', config.image_source)
        image_responses = image_client.get_image_from_sources([config.image_source])
        if len(image_responses) != 1:
            print(f'Got invalid number of images: {len(image_responses)}')
            print(image_responses)
            assert False
        image = image_responses[0]
        dtype = np.uint16 if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16 else np.uint8
        img = np.frombuffer(image.shot.image.data, dtype=dtype)
        if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
            img = img.reshape(image.shot.image.rows, image.shot.image.cols)
        else:
            img = cv2.imdecode(img, -1)
        robot.logger.info('Click on an object to start grasping...')
        image_title = 'Click to grasp'
        cv2.namedWindow(image_title)
        cv2.setMouseCallback(image_title, cv_mouse_callback)
        global g_image_click, g_image_display
        g_image_display = img
        cv2.imshow(image_title, g_image_display)
        while g_image_click is None:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print('"q" pressed, exiting.')
                exit(0)
        cv2.destroyAllWindows()
        robot.logger.info(f'Picking object at image location ({g_image_click[0]}, {g_image_click[1]})')
        pick_vec = geometry_pb2.Vec2(x=g_image_click[0], y=g_image_click[1])
        grasp = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=pick_vec, transforms_snapshot_for_camera=image.shot.transforms_snapshot,
            frame_name_image_sensor=image.shot.frame_name_image_sensor,
            camera_model=image.source.pinhole)
        add_grasp_constraint(config, grasp, robot_state_client)
        grasp_request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)
        cmd_response = manipulation_api_client.manipulation_api_command(
            manipulation_api_request=grasp_request)
        robot.logger.info('Executing grasp...')
        while True:
            feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                manipulation_cmd_id=cmd_response.manipulation_cmd_id)
            response = manipulation_api_client.manipulation_api_feedback_command(
                manipulation_api_feedback_request=feedback_request)
            print(f'Current state: {manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state)}')
            if response.current_state in [manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED, manipulation_api_pb2.MANIP_STATE_GRASP_FAILED]:
                break
            time.sleep(0.25)
        robot.logger.info('Finished grasp. You can now move the arm using WASD controls.')
        # The following commands are commented out to disable automatic actions after grasping:
        command_client.robot_command(RobotCommandBuilder.claw_gripper_open_command())
        time.sleep(0.5)
        command_client.robot_command(RobotCommandBuilder.claw_gripper_close_command())
        time.sleep(0.5)
        # command_client.robot_command(RobotCommandBuilder.arm_stow_command())
        # time.sleep(0.5)
        # Enter WASD control mode
        wasd_arm_control_loop(robot, command_client, robot_state_client, image_client, manipulation_api_client, config)

def wasd_arm_control_loop(robot, command_client, robot_state_client=None, image_client=None, manipulation_api_client=None, config=None):
    print('\nWASD Arm Control:')
    print('w/s: move out/in, a/d: rotate ccw/cw, r/f: up/down,')
    print('u/o: +rx/-rx, i/k: +ry/-ry, j/l: +rz/-rz, m: close gripper, n: open gripper, z: stow arm, x: unstow arm, g: grasp again, q: quit')
    while True:
        key = input('Command: ').strip().lower()
        if key == 'q':
            print('Exiting WASD arm control.')
            break
        elif key == 'w':
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
        elif key == 'u':
            send_arm_angular_velocity(command_client, v_rx=VELOCITY_ANGULAR_HAND)
        elif key == 'o':
            send_arm_angular_velocity(command_client, v_rx=-VELOCITY_ANGULAR_HAND)
        elif key == 'i':
            send_arm_angular_velocity(command_client, v_ry=VELOCITY_ANGULAR_HAND)
        elif key == 'k':
            send_arm_angular_velocity(command_client, v_ry=-VELOCITY_ANGULAR_HAND)
        elif key == 'j':
            send_arm_angular_velocity(command_client, v_rz=VELOCITY_ANGULAR_HAND)
        elif key == 'l':
            send_arm_angular_velocity(command_client, v_rz=-VELOCITY_ANGULAR_HAND)
        elif key == 'm':
            command_client.robot_command(RobotCommandBuilder.claw_gripper_close_command())
        elif key == 'n':
            command_client.robot_command(RobotCommandBuilder.claw_gripper_open_command())
        elif key == 'z':
            command_client.robot_command(RobotCommandBuilder.arm_stow_command())
        elif key == 'x':
            command_client.robot_command(RobotCommandBuilder.arm_ready_command())
        elif key == 'g':
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

def send_arm_cylindrical_velocity(command_client, v_r=0.0, v_theta=0.0, v_z=0.0):
    from bosdyn.api import arm_command_pb2, robot_command_pb2, geometry_pb2
    import bosdyn.client.util
    arm_velocity_cmd = arm_command_pb2.ArmVelocityCommand.Request()
    
    # Create a SE3Velocity message for cartesian velocity
    se3_velocity = geometry_pb2.SE3Velocity()
    se3_velocity.linear.x = v_r
    se3_velocity.linear.y = v_theta
    se3_velocity.linear.z = v_z
    se3_velocity.angular.x = 0.0
    se3_velocity.angular.y = 0.0
    se3_velocity.angular.z = 0.0
    
    arm_velocity_cmd.cartesian_velocity.CopyFrom(se3_velocity)
    arm_velocity_cmd.duration.CopyFrom(
        bosdyn.client.util.seconds_to_duration(VELOCITY_CMD_DURATION)
    )
    robot_cmd = robot_command_pb2.RobotCommand()
    robot_cmd.synchronized_command.arm_command.arm_velocity_command.CopyFrom(arm_velocity_cmd)
    command_client.robot_command(robot_cmd)

def send_arm_angular_velocity(command_client, v_rx=0.0, v_ry=0.0, v_rz=0.0):
    from bosdyn.api import arm_command_pb2, robot_command_pb2, geometry_pb2
    import bosdyn.client.util
    arm_velocity_cmd = arm_command_pb2.ArmVelocityCommand.Request()
    
    # Create a SE3Velocity message for cartesian velocity
    se3_velocity = geometry_pb2.SE3Velocity()
    se3_velocity.linear.x = 0.0
    se3_velocity.linear.y = 0.0
    se3_velocity.linear.z = 0.0
    se3_velocity.angular.x = v_rx
    se3_velocity.angular.y = v_ry
    se3_velocity.angular.z = v_rz
    
    arm_velocity_cmd.cartesian_velocity.CopyFrom(se3_velocity)
    arm_velocity_cmd.duration.CopyFrom(
        bosdyn.client.util.seconds_to_duration(VELOCITY_CMD_DURATION)
    )
    robot_cmd = robot_command_pb2.RobotCommand()
    robot_cmd.synchronized_command.arm_command.arm_velocity_command.CopyFrom(arm_velocity_cmd)
    command_client.robot_command(robot_cmd)

def main():
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('-i', '--image-source', help='Get image from source', default='frontleft_fisheye_image')
    parser.add_argument('-t', '--force-top-down-grasp', help='Force the robot to use a top-down grasp (vector_alignment demo)', action='store_true')
    parser.add_argument('-f', '--force-horizontal-grasp', help='Force the robot to use a horizontal grasp (vector_alignment demo)', action='store_true')
    parser.add_argument('-r', '--force-45-angle-grasp', help='Force the robot to use a 45 degree angled down grasp (rotation_with_tolerance demo)', action='store_true')
    parser.add_argument('-s', '--force-squeeze-grasp', help='Force the robot to use a squeeze grasp', action='store_true')
    options = parser.parse_args()
    num = 0
    if options.force_top_down_grasp:
        num += 1
    if options.force_horizontal_grasp:
        num += 1
    if options.force_45_angle_grasp:
        num += 1
    if options.force_squeeze_grasp:
        num += 1
    if num > 1:
        print('Error: cannot force more than one type of grasp.  Choose only one.')
        sys.exit(1)
    try:
        grasp_then_wasd(options)
        return True
    except Exception as exc:
        logger = bosdyn.client.util.get_logger()
        logger.exception('Threw an exception')
        return False

if __name__ == '__main__':
    if not main():
        sys.exit(1)
