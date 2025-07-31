import socket
import json
import base64
import cv2
import numpy as np
import os
from SoM_segmentation import run_segmentation
from SoM_GPT4 import combined_probability, run_SoM
from SoM_combine_marks import process_marks
import time
from detection_util import read_json_locked, write_json_locked
from pathlib import Path

HOST = "127.0.0.1"  # Server IP
PORT = 65433       # Port to receive data
color_image_path = "spot_util/hand_color_image.png"
depth_image_path = "spot_util/hand_depth_image.png"
transformation_path = "spot_util/hand_transformation_intrinsic.json"

def receive_data():
    """Receive images, depth images, and transformation matrices."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((HOST, PORT))
        print(f"Connected to server at {HOST}:{PORT}")

        while True:
            try:
                # Receive the size of the incoming message
                data_size = int.from_bytes(client_socket.recv(4), 'big')
                data = b""
                while len(data) < data_size:
                    packet = client_socket.recv(data_size - len(data))
                    if not packet:
                        break
                    data += packet

                # Deserialize the data (ensure it's decoded correctly as a string)
                data = json.loads(data.decode('utf-8'))

                # Decode images from Base64
                greyscale_image_bytes = base64.b64decode(data["image"])
                depth_image_bytes = base64.b64decode(data["depth_image"])

                # Convert Base64-decoded bytes into images
                greyscale_image = cv2.imdecode(np.frombuffer(greyscale_image_bytes, np.uint8), cv2.IMREAD_COLOR)
                depth_image = cv2.imdecode(np.frombuffer(depth_image_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
                intrinsics = np.reshape(np.array(data["intrinsics"]), (3, 3))
                rotation_matrix = np.reshape(np.array(data["rotation_matrix"]), (3, 3))
                translation_vector = np.array(data["translation_vector"])

                # Display or process the data
                print(f"Transformation Matrix: {intrinsics} {rotation_matrix} {translation_vector}")
                cv2.imshow("Greyscale Image", greyscale_image)
                cv2.imshow("Depth Image", depth_image)

                # Exit when 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except Exception as e:
                print(f"Error receiving data: {e}")
                break


        
def run_system(color_path = color_image_path, depth_path = depth_image_path, trans_matrix_path = transformation_path):
    
    transformation_matrix_path = './spot_util/hand_intrinsics_and_transform.json'
    
    image_path = "./spot_util/hand_color_image.png"
    depth_image_path =  "./spot_util/hand_depth_image.png"
    SoM_annotated_image_path = "./.tmp/annotated_image.png"
    detection_confience_path = "./.tmp/detection_confidence.json"

    try:
        os.remove(detection_confience_path)
    except FileNotFoundError:
        print("deleted")
        
    gesture_detected = False
    voice_detected = False
    last_processed_time = 0
    while True:
        current_mod_time = os.path.getmtime(image_path)
        if current_mod_time > last_processed_time:
            last_processed_time = current_mod_time
            # marks = run_segmentation(image_path, depth_image_path, transformation_matrix_path)
            marks = read_json_locked("./.tmp/detection_tmp.json")
            if marks is None:
                continue
        
            if not Path(detection_confience_path).exists():
                write_json_locked(marks, detection_confience_path)
                global_marks = marks
            else:
                global_marks = read_json_locked(detection_confience_path)
                
            org_marks_length = len(global_marks)
            new_global_marks = process_marks(global_marks, marks)
            # if len(new_global_marks) - org_marks_length >  8:
            #     resp, marks = run_SoM(SoM_annotated_image_path, marks)
            #     print(resp)
            new_global_marks = process_marks(global_marks, marks)
            write_json_locked(new_global_marks, detection_confience_path)


        time.sleep(1)


if __name__ == "__main__":
    
    run_system()