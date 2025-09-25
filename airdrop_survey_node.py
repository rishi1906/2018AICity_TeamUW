#!/usr/bin/env python3
import os
import time
import math

import cv2
import piexif
import rospy
from geopy.distance import geodesic
from geopy import Point

from std_msgs.msg import String, Bool
from sensor_msgs.msg import NavSatFix
from mavros_msgs.msg import Waypoint, WaypointReached
from mavros_msgs.srv import WaypointPush, WaypointClear, SetMode, WaypointPull


SAVE_DIR = "/home/raft/jakrif_ws/src/object_detection/camera_imgs"


class AirdropSurveyAndImageCapture:
    def __init__(self):
        rospy.init_node("survey_and_geotagging_node", anonymous=True)

        os.makedirs(SAVE_DIR, exist_ok=True)

        # ROS services
        self.set_mode_client = rospy.ServiceProxy("/mavros/set_mode", SetMode)

        # State
        self.cap = None
        self.frame_counter = 0
        self.latest_latitude = None
        self.latest_longitude = None
        self.latest_altitude = None
        self.waypoints_reached = 0
        self.total_waypoints = 0
        self.cmd = None

        # Camera initialization: prefer video file, fallback to camera index 0
        video_path = "/home/raft/Downloads/test_video.mp4"
        if os.path.isfile(video_path):
            tmp_cap = cv2.VideoCapture(video_path)
            if tmp_cap is not None and tmp_cap.isOpened():
                self.cap = tmp_cap
                rospy.loginfo(f"Using video source: {video_path}")
            else:
                rospy.logwarn(f"Failed to open video file: {video_path}")
        if self.cap is None:
            tmp_cap = cv2.VideoCapture(0)
            if tmp_cap is not None and tmp_cap.isOpened():
                self.cap = tmp_cap
                rospy.loginfo("Using default camera index 0 as video source")
            else:
                rospy.logwarn("No camera device available; image capture will be skipped")

        # Publishers / Subscribers
        self.task_command_pub = rospy.Publisher("/task_status", Bool, queue_size=10)

        rospy.Subscriber("/task_command", String, self.task_command_callback)
        rospy.Subscriber("/mavros/global_position/raw/fix", NavSatFix, self.gps_callback)
        rospy.Subscriber("/mavros/mission/reached", WaypointReached, self.waypoint_reached_callback)

        rospy.on_shutdown(self._shutdown)

    # ----------------------------- Public API ----------------------------- #
    def start_survey(self, boundary_coordinates, spacing=20.5, altitude=75.0):
        if not boundary_coordinates or len(boundary_coordinates) < 4:
            rospy.logerr("Boundary must include at least 4 corner points (rectangle). Aborting.")
            return
        if spacing <= 0:
            rospy.logerr("Spacing must be positive. Aborting.")
            return
        if altitude <= 0:
            rospy.logerr("Altitude must be positive. Aborting.")
            return

        waypoint_list = self.generate_zigzag_waypoints(boundary_coordinates[:4], altitude, spacing)
        if not waypoint_list:
            rospy.logerr("No waypoints generated. Aborting survey start.")
            return

        self.total_waypoints = len(waypoint_list)
        rospy.loginfo(f"Generated {self.total_waypoints} waypoints.")

        # Try to enter GUIDED (or equivalent) before uploading/starting mission
        self.set_mode("GUIDED")
        self.create_waypoint(waypoint_list)
        rospy.loginfo("Survey mission uploaded. Switching to AUTO mission mode.")
        time.sleep(0.5)
        # PX4 expects AUTO.MISSION, ArduPilot often accepts AUTO.
        if not self.set_mode("AUTO.MISSION"):
            self.set_mode("AUTO")

    def load_boundary(self, filename):
        waypoints = []
        try:
            with open(filename, "r") as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        lat, lon = line.split(",")
                        waypoints.append([float(lat), float(lon)])
                    except ValueError:
                        rospy.logwarn(f"Skipping invalid line in boundary file: '{line}'")
        except FileNotFoundError:
            rospy.logerr(f"Boundary file not found: {filename}")
        return waypoints

    def set_mode(self, mode, timeout=5.0):
        try:
            rospy.wait_for_service("/mavros/set_mode", timeout=timeout)
            resp = self.set_mode_client(0, mode)
            success = bool(getattr(resp, "mode_sent", False))
            rospy.loginfo(f"Set mode '{mode}' response: {success}")
            return success
        except (rospy.ROSException, rospy.ServiceException) as e:
            rospy.logerr(f"Failed to set mode to {mode}: {e}")
            return False

    # ----------------------------- Callbacks ------------------------------ #
    def task_command_callback(self, msg):
        self.cmd = msg.data
        rospy.loginfo(f"Task command received: {self.cmd}")
        if self.cmd == "survey_and_detection":
            rospy.loginfo("Survey and detection command received")
            boundary_file = "/home/raft/jakrif_ws/src/airdrop_area_survey/jaipur.txt"
            self.boundary_coordinates = self.load_boundary(boundary_file)
            if not self.boundary_coordinates:
                rospy.logerr("No boundary coordinates loaded; aborting.")
                return
            self.start_survey(self.boundary_coordinates)

    def gps_callback(self, msg):
        self.latest_latitude = msg.latitude
        self.latest_longitude = msg.longitude
        self.latest_altitude = msg.altitude

    def waypoint_reached_callback(self, msg):
        if self.cmd != "survey_and_detection":
            return
        # msg.wp_seq is zero-based index of the reached waypoint
        self.waypoints_reached = int(msg.wp_seq) + 1
        rospy.loginfo(
            f"Waypoint index {msg.wp_seq} reached. ({self.waypoints_reached}/{self.total_waypoints})"
        )

        # Capture at every reached waypoint
        self.capture_and_save_image()

        if self.total_waypoints > 0 and self.waypoints_reached >= self.total_waypoints:
            rospy.loginfo("All mission waypoints reached.")
            self.complete_task()
            rospy.signal_shutdown("Task completed")

    # ----------------------------- Imaging -------------------------------- #
    def capture_and_save_image(self, num_frames=5, delay=15):
        if self.cap is None:
            rospy.logwarn("Camera not available. Skipping capture.")
            return

        for _ in range(num_frames):
            # Warm-up reads
            for _ in range(2):
                self.cap.read()

            ret, frame = self.cap.read()
            if not ret or frame is None:
                rospy.logwarn("Failed to capture frame from camera.")
                continue

            file_path = os.path.join(SAVE_DIR, f"frame_{self.frame_counter}.jpg")
            ok = cv2.imwrite(file_path, frame)
            if not ok:
                rospy.logerr(f"Failed to write image to {file_path}. Check directory permissions.")
                continue

            rospy.loginfo(f"Captured and saved image: {file_path}")

            if all(x is not None for x in [self.latest_latitude, self.latest_longitude, self.latest_altitude]):
                try:
                    self.add_geotag_metadata(file_path)
                except Exception as e:
                    rospy.logerr(f"Geotagging failed: {str(e)}")
            else:
                rospy.logwarn("GPS data unavailable; skipping geotagging.")

            self.frame_counter += 1
            rospy.sleep(delay)

    def add_geotag_metadata(self, image_path):
        def decimal_to_dms(decimal_degrees):
            dd = abs(float(decimal_degrees))
            degrees = int(dd)
            minutes_full = (dd - degrees) * 60.0
            minutes = int(minutes_full)
            seconds = round((minutes_full - minutes) * 60.0 * 100.0)
            # Use rational tuples as (numerator, denominator)
            return [(degrees, 1), (minutes, 1), (int(seconds), 100)]

        if self.latest_latitude is None or self.latest_longitude is None:
            raise ValueError("No GPS fix to write into EXIF")

        lat_ref = b"N" if self.latest_latitude >= 0 else b"S"
        lon_ref = b"E" if self.latest_longitude >= 0 else b"W"
        alt_ref = 0 if (self.latest_altitude or 0) >= 0 else 1
        altitude_abs = abs(float(self.latest_altitude or 0.0))

        gps_ifd = {
            piexif.GPSIFD.GPSLatitudeRef: lat_ref,
            piexif.GPSIFD.GPSLatitude: decimal_to_dms(self.latest_latitude),
            piexif.GPSIFD.GPSLongitudeRef: lon_ref,
            piexif.GPSIFD.GPSLongitude: decimal_to_dms(self.latest_longitude),
            piexif.GPSIFD.GPSAltitudeRef: alt_ref,
            piexif.GPSIFD.GPSAltitude: (int(round(altitude_abs * 100)), 100),
        }

        exif_dict = {"GPS": gps_ifd}
        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, image_path)
        rospy.loginfo(f"Geotag metadata added to {image_path}")

    # ----------------------------- Mission -------------------------------- #
    def create_waypoint(self, waypoint_list):
        # Clear any existing mission
        try:
            rospy.wait_for_service("/mavros/mission/clear", timeout=3.0)
            clear_srv = rospy.ServiceProxy("/mavros/mission/clear", WaypointClear)
            cleared = clear_srv()
            rospy.loginfo("Cleared waypoints: %s", getattr(cleared, "success", cleared))
        except (rospy.ROSException, rospy.ServiceException) as e:
            rospy.logwarn(f"Could not clear waypoints: {e}")

        # Build mission waypoints
        waypoints = []
        for idx, (lat, lon, alt) in enumerate(waypoint_list):
            wp = Waypoint()
            wp.frame = 3  # MAV_FRAME_GLOBAL_RELATIVE_ALT
            wp.command = 16  # MAV_CMD_NAV_WAYPOINT
            wp.is_current = idx == 0
            wp.autocontinue = True
            wp.param1 = 0
            wp.param2 = 0
            wp.param3 = 0
            wp.param4 = 0
            wp.x_lat = float(lat)
            wp.y_long = float(lon)
            wp.z_alt = float(alt)
            waypoints.append(wp)

        # Push mission
        try:
            rospy.wait_for_service("/mavros/mission/push", timeout=5.0)
            push_srv = rospy.ServiceProxy("/mavros/mission/push", WaypointPush)
            resp = push_srv(start_index=0, waypoints=waypoints)
            success = bool(getattr(resp, "success", False))
            rospy.loginfo("Waypoint push success: %s", success)
            if success:
                self.total_waypoints = len(waypoints)
            # Optionally pull to verify
            try:
                rospy.wait_for_service("/mavros/mission/pull", timeout=3.0)
                pull_srv = rospy.ServiceProxy("/mavros/mission/pull", WaypointPull)
                pull_resp = pull_srv()
                rospy.loginfo("Pulled waypoints: %s", getattr(pull_resp, "wp_received", "unknown"))
            except (rospy.ROSException, rospy.ServiceException) as e:
                rospy.logwarn(f"Could not pull mission after push: {e}")
        except (rospy.ROSException, rospy.ServiceException) as e:
            rospy.logerr(f"Service call failed during push: {e}")

    # ----------------------------- Utilities ------------------------------ #
    def complete_task(self):
        rospy.loginfo("Survey completed, task finished.")
        try:
            self.task_command_pub.publish(Bool(data=True))
        except Exception:
            # Publishing should not crash node
            pass

    def calculate_initial_bearing(self, point1, point2):
        lat1, lon1 = math.radians(point1[0]), math.radians(point1[1])
        lat2, lon2 = math.radians(point2[0]), math.radians(point2[1])
        d_lon = lon2 - lon1
        x = math.sin(d_lon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
        return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0

    def generate_zigzag_waypoints(self, boundary, altitude, spacing):
        # Expecting rectangle described by 4 points in order (convex)
        if len(boundary) < 4:
            return []

        def is_inside_rectangle(point, rect):
            def cross_product(p1, p2, p3):
                return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

            return all(cross_product(rect[i], rect[(i + 1) % 4], point) >= 0 for i in range(4))

        long_side_bearing = self.calculate_initial_bearing(boundary[0], boundary[1])
        short_side_bearing = (long_side_bearing + 90.0) % 360.0

        waypoints = []
        current_row_start = boundary[0]
        direction = 1

        # Iterate rows until we step outside rectangle
        while True:
            current_row = []
            current_point = current_row_start

            # Step along the long side until outside
            while is_inside_rectangle(current_point, boundary):
                current_row.append((current_point[0], current_point[1], altitude))
                step = geodesic(meters=spacing).destination(Point(current_point[0], current_point[1]), long_side_bearing)
                current_point = (step.latitude, step.longitude)

            if not current_row:
                break

            if direction == 1:
                waypoints.extend(current_row)
            else:
                waypoints.extend(reversed(current_row))
            direction *= -1

            # Move one row along the short side
            step = geodesic(meters=spacing).destination(Point(current_row_start[0], current_row_start[1]), short_side_bearing)
            current_row_start = (step.latitude, step.longitude)

            if not is_inside_rectangle(current_row_start, boundary):
                break

        return waypoints

    def waypoint_clear_client(self):
        try:
            rospy.wait_for_service("/mavros/mission/clear", timeout=3.0)
            clear_srv = rospy.ServiceProxy("/mavros/mission/clear", WaypointClear)
            return bool(clear_srv().success)
        except (rospy.ROSException, rospy.ServiceException) as e:
            rospy.logerr(f"Service call failed: {e}")
            return False

    def _shutdown(self):
        rospy.loginfo("Shutting down: releasing camera and cleaning up.")
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass


if __name__ == "__main__":
    try:
        survey = AirdropSurveyAndImageCapture()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

