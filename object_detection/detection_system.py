import os
import time
import json
import gc
from pathlib import Path
import torch
import statistics

from SoM_segmentation import run_segmentation
from SoM_GPT4 import run_SoM
from detection_util import read_json_locked, write_json_locked, process_marks


class ObjectDetectionSystem:
    def __init__(self, color_path, depth_path, trans_matrix_path, detection_conf_path="./.tmp/detection_confidence.json"):
        """
        Initialize the object detection system with paths and configurations.

        Args:
            color_path (str): Path to the color image.
            depth_path (str): Path to the depth image.
            trans_matrix_path (str): Path to the transformation matrix.
            detection_conf_path (str): Path to the JSON file storing detection confidence.
        """
        self.color_path = color_path
        self.depth_path = depth_path
        self.trans_matrix_path = trans_matrix_path
        self.detection_conf_path = detection_conf_path
        self.last_processed_time = 0
        self.global_marks = []
        self.first_frame_flag = True
        self.updated = False  # Flag to track changes
        self.cycle_times = []  # Store cycle times for statistics
        self.ema_cycle_time = None  # Exponential Moving Average of cycle time
        self.alpha = 0.1  # Smoothing factor for EMA

        # Load existing marks or initialize an empty list
        if Path(self.detection_conf_path).exists():
            self.global_marks = read_json_locked(self.detection_conf_path)
        else:
            self.global_marks = []
        
    def process_image_update(self):
        """
        Check if the color image has been updated and process it if necessary.
        """
        annotated_image_path = "./.tmp/annotated_image.png"
        current_mod_time = os.path.getmtime(self.color_path)

        if current_mod_time > self.last_processed_time:
            self.last_processed_time = current_mod_time
            start_time = time.time()  # Start timing

            # Run segmentation
            marks = run_segmentation(self.color_path, self.depth_path, self.trans_matrix_path)
            if marks is None:
                return

            # First frame special case (run SoM on the first set of detections)
            if self.first_frame_flag:
                self.first_frame_flag = False
                resp, marks = run_SoM(annotated_image_path, marks)
                print("First Frame::::Running SoM-->", resp)
                self.global_marks = marks
                self.updated = True  # Mark as updated
            else:
                # Process and merge new detections with existing ones
                org_marks_length = len(self.global_marks)
                new_global_marks = process_marks(self.global_marks, marks)

                # If more than 3 new objects are detected, re-run SoM
                # if len(new_global_marks) - org_marks_length > 3:
                #     resp, marks = run_SoM(annotated_image_path, marks)
                #     print("Running SoM-->", resp)
                #     new_global_marks = process_marks(self.global_marks, marks)
                resp, marks = run_SoM(annotated_image_path, marks)
                print("Running SoM-->", resp)
                new_global_marks = process_marks(self.global_marks, marks)
                # Only update if new marks are different
                if new_global_marks != self.global_marks:
                    self.global_marks = new_global_marks
                    self.updated = True  # Mark as updated

            # Measure cycle time
            cycle_time = time.time() - start_time
            self.update_cycle_time(cycle_time)

            # Clear memory
            gc.collect()
            torch.cuda.empty_cache()
            return marks, self.global_marks
    
    def save_if_updated(self):
        """Write to JSON only if there are changes."""
        if self.updated:
            write_json_locked(self.global_marks, self.detection_conf_path)
            self.updated = False  # Reset flag

    def update_cycle_time(self, cycle_time):
        """Update and log cycle time statistics."""
        self.cycle_times.append(cycle_time)
        
        # Maintain only the last 50 cycle times for statistics
        if len(self.cycle_times) > 50:
            self.cycle_times.pop(0)

        # Compute Exponential Moving Average (EMA) for smoother updates
        if self.ema_cycle_time is None:
            self.ema_cycle_time = cycle_time  # Initialize with first value
        else:
            self.ema_cycle_time = self.alpha * cycle_time + (1 - self.alpha) * self.ema_cycle_time

        avg_cycle_time = statistics.mean(self.cycle_times)
        fps = 1 / avg_cycle_time if avg_cycle_time > 0 else 0
        print(f"🔹 Avg Cycle Time: {avg_cycle_time:.3f} sec | EMA Cycle Time: {self.ema_cycle_time:.3f} sec | FPS: {fps:.2f}")

    def run(self):
        """
        Main loop to continuously monitor and process image updates.
        """
        print("🚀 Starting Object Detection System...")
        while True:
            try:
                self.process_image_update()
                self.save_if_updated()  # Save only if updates occurred
                time.sleep(1)  # Poll every 1 second
            except KeyboardInterrupt:
                print("🛑 Shutting down Object Detection System...")
                self.save_if_updated()  # Ensure final save before exiting
                break


if __name__ == "__main__":
    # Configure paths
    color_image_path = "spot_util/hand_color_image.png"
    depth_image_path = "spot_util/hand_depth_image.png"
    transformation_path = "spot_util/hand_intrinsics_and_transform.json"
    detection_confidence_path = ".tmp/detection_confidence.json"
    try:
        os.remove(detection_confidence_path)
    except FileNotFoundError:
        print("deleted")

    # Create and run the object detection system
    system = ObjectDetectionSystem(
        color_path=color_image_path,
        depth_path=depth_image_path,
        trans_matrix_path=transformation_path,
        detection_conf_path=detection_confidence_path
    )
    system.run()