#!/usr/bin/env python3
import rospy
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'ODCL_Package'))

from std_msgs.msg import String, Bool
from identify_class import inference
from localization import localization
from remove_duplicates import remove_duplicates_in_folder
from sahi import AutoDetectionModel
import piexif
from PIL import Image
import re
import numpy as np
from pathlib import Path
import time
from queue import Queue
import threading
import logging


def _get_file_signature(file_path):
    """Return a tuple uniquely identifying file state or None if not available."""
    try:
        stat_result = os.stat(file_path)
    except (FileNotFoundError, PermissionError):
        return None
    inode = getattr(stat_result, "st_ino", 0)
    size = getattr(stat_result, "st_size", 0)
    mtime_ns = getattr(stat_result, "st_mtime_ns", int(stat_result.st_mtime * 1e9))
    return (int(inode), int(size), int(mtime_ns))


class ObjectDetection:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('Object_Detection_Node', anonymous=True)

        # Publishers
        self.task_command_pub = rospy.Publisher('/task_status', Bool, queue_size=10)

        # Subscribers
        #rospy.Subscriber('/task_command', String, self.task_command_callback)

        # Parameters
        self.image_dir = './camera_imgs'
        self.output_dir = './output_imgs'
        self.duplicate_dir = './duplicate_imgs'
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Load detection model
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type='ultralytics',
            model_path="/home/raft/jakrif_ws/src/object_detection/best.pt",   # Path to YOLO weights file
            confidence_threshold=0.6, # Confidence threshold
            device='cuda:0'           # GPU (or "cpu")
        )

        # Camera and slicing params
        
        self.fov_horizontal = 60
        self.fov_vertical = 47
        self.slice_params = [800, 800, 0.2, 0.2]

        # Constants
        self.CHECK_INTERVAL = 10
        self.MIN_ERROR = 10  # in meters, for duplicate removal

        # Queues & trackers
        self.image_queue = Queue()
        self.seen_files = set()

        # Logging setup
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler("log_file.log"),
                logging.StreamHandler()
            ]
        )
        logging.info("Log Start")

        # Inactivity monitor parameters
        self.inactivity_file = rospy.get_param("~inactivity_file", "log_file.log")
        self.inactivity_seconds = float(rospy.get_param("~inactivity_seconds", 60.0))
        self.inactivity_poll_hz = float(rospy.get_param("~inactivity_poll_hz", 1.0))
        self.print_once_per_event = bool(rospy.get_param("~print_once_per_event", True))

        # Inactivity monitor state
        self._last_signature = _get_file_signature(self.inactivity_file)
        self._last_change_monotonic = time.monotonic()
        self._already_printed_for_current_inactivity = False

        # Start worker thread
        threading.Thread(target=self.worker, daemon=True).start()
        # Start watching folder in main thread
        threading.Thread(target=self.watch_folder, daemon=True).start()
        # Start inactivity monitor thread
        threading.Thread(target=self._inactivity_monitor_loop, daemon=True).start()

        self.complete_task()
        


    # ---------------- Task Completion ----------------
    def complete_task(self):
        """Publish completion status"""
        try:
            rospy.loginfo("Object_detection task finished.")
            self.task_command_pub.publish(True)
        except Exception as e:
            rospy.logwarn(f"Task completion error: {e}")

    # ---------------- Image Processing ----------------
    def process_image(self, image_path):
        """Run detection + duplicate filtering"""
        logging.info(f"[INFO] Processing {image_path} ...")
        try:
            inference(
                self.detection_model,
                image_path,
                self.output_dir,
                self.slice_params,
                [self.fov_horizontal, self.fov_vertical]
            )
            logging.info(f"[DONE] Processed {image_path}")

            # Remove duplicates
            remove_duplicates_in_folder(
                self.output_dir,
                self.duplicate_dir,
                tolerance_m=15.0,
                keep='oldest'
            )
        except Exception as e:
            logging.error(f"[ERROR] Failed to process {image_path}: {e}")

    # ---------------- Worker Thread ----------------
    def worker(self):
        """Thread that processes images from the queue"""
        while not rospy.is_shutdown():
            image_path = self.image_queue.get()
            try:
                self.process_image(image_path)
            finally:
                self.image_queue.task_done()

    # ---------------- Folder Watcher ----------------
    def watch_folder(self):
        """Monitor folder for new images and queue them"""
        while not rospy.is_shutdown():
            script_dir = os.path.dirname(os.path.realpath(__file__))
            self.image_dir = os.path.join(script_dir, 'camera_imgs')
            for filename in os.listdir(self.image_dir):
                filepath = os.path.join(self.image_dir, filename)
                if os.path.isfile(filepath) and filename not in self.seen_files:
                    logging.info(f"[QUEUE] Adding {filename}")
                    self.seen_files.add(filename)
                    self.image_queue.put(filepath)
            time.sleep(self.CHECK_INTERVAL)

    # ---------------- Inactivity Monitor ----------------
    def _inactivity_monitor_loop(self):
        """Monitor the inactivity of the log file and print 'true' after threshold."""
        poll_period = max(0.01, 1.0 / max(self.inactivity_poll_hz, 0.01))
        while not rospy.is_shutdown():
            now_mono = time.monotonic()
            current_signature = _get_file_signature(self.inactivity_file)

            if current_signature is None:
                # If file missing, reset window and wait
                if self._last_signature is not None:
                    self._last_signature = None
                    self._last_change_monotonic = now_mono
                    self._already_printed_for_current_inactivity = False
                else:
                    self._last_change_monotonic = now_mono
                time.sleep(poll_period)
                continue

            if self._last_signature is None:
                # File appeared
                self._last_signature = current_signature
                self._last_change_monotonic = now_mono
                self._already_printed_for_current_inactivity = False
                time.sleep(poll_period)
                continue

            if current_signature != self._last_signature:
                # File changed
                self._last_signature = current_signature
                self._last_change_monotonic = now_mono
                self._already_printed_for_current_inactivity = False
                time.sleep(poll_period)
                continue

            inactive_seconds = now_mono - self._last_change_monotonic
            if inactive_seconds >= self.inactivity_seconds:
                if not self._already_printed_for_current_inactivity:
                    print("true")
                    try:
                        sys.stdout.flush()
                    except Exception:
                        pass
                    if self.print_once_per_event:
                        self._already_printed_for_current_inactivity = True

            time.sleep(poll_period)


# ---------------- Main ----------------
if __name__ == "__main__":
    try:
        detect = ObjectDetection()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

