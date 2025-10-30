#!/usr/bin/env python3
import os
import time
import math
import threading
from typing import List, Tuple

import rospy
from std_msgs.msg import String, Bool
from sensor_msgs.msg import NavSatFix
from mavros_msgs.msg import Waypoint, WaypointReached
from mavros_msgs.srv import WaypointPush, WaypointClear, SetMode, StreamRate, StreamRateRequest

import cv2
import piexif
from PIL import Image
from geopy.distance import geodesic
from geopy import Point

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
SAVE_DIR = "/home/raft/jakrif_ws/src/object_detection/ODCL/camera_imgs"


class AirdropSurveyAndImageCapture:
    def __init__(self):
        rospy.init_node('survey_and_geotagging_node', anonymous=True)

        os.makedirs(SAVE_DIR, exist_ok=True)

        self.set_mode_client = rospy.ServiceProxy("/mavros/set_mode", SetMode)
        rospy.Subscriber('/task_command', String, self.task_command_callback)

        self.task = " "
        self.frame_counter = 0
        self.latest_latitude = None
        self.latest_longitude = None
        self.latest_altitude = None
        self.home_latitude = None
        self.home_longitude = None
        self.prev_latitude = None
        self.prev_longitude = None
        self.last_direction_deg = 0.0
        self.waypoints_reached = 0
        self.nav_waypoint_count = 0  # Only NAV_WAYPOINT items (excludes RTL)
        self.total_mission_items = 0  # All uploaded items including RTL
        self.cmd = None

        # Camera initialization
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        if not self.cap.isOpened():
            raise Exception("Failed to open camera")
        for _ in range(5):
            self.cap.read()

        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.frame_timer = rospy.Timer(rospy.Duration(0.1), self._poll_camera_frame)

        self.task_command_pub = rospy.Publisher('/task_status', Bool, queue_size=10)

        rospy.Subscriber("/mavros/global_position/raw/fix", NavSatFix, self.gps_callback)
        rospy.Subscriber('/mavros/mission/reached', WaypointReached, self.waypoint_reached_callback)

        self.set_mavros_stream_rate(0, 10, True)

    def set_mavros_stream_rate(self, stream_id, message_rate, on_off):
        rospy.wait_for_service('/mavros/set_stream_rate')
        try:
            set_rate_service = rospy.ServiceProxy('/mavros/set_stream_rate', StreamRate)
            req = StreamRateRequest()
            req.stream_id = stream_id
            req.message_rate = message_rate
            req.on_off = on_off
            set_rate_service(req)
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def _poll_camera_frame(self, _evt):
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if ok and frame is not None:
            with self.frame_lock:
                self.latest_frame = frame

    # -------- CORRECTED SURVEY LOGIC --------
    def start_survey(self, boundary_coordinates: List[List[float]], spacing: float = 15.0):
        # Generate primary zig-zag waypoints
        zigzag_waypoints = self.generate_zigzag_waypoints(boundary_coordinates, altitude=25.0, spacing=spacing)
        if not zigzag_waypoints:
            rospy.logerr("No waypoints generated. Aborting survey start.")
            return

        # Load target waypoint
        try:
            target_lat, target_lon = self.load_target_waypoint("/home/raft/jakrif_ws/src/object_detection/ODCL/payload_coord.txt")
            rospy.loginfo(f"Loaded target waypoint: ({target_lat}, {target_lon})")
        except Exception as e:
            rospy.logwarn(f"Could not load target waypoint: {e}")
            target_lat, target_lon = zigzag_waypoints[-1][0], zigzag_waypoints[-1][1]  # fallback

        # Construct mission sequence per required logic
        mission_waypoints: List[Tuple[float, float, float]] = []

        if self.home_latitude is not None and self.home_longitude is not None:
            mission_waypoints.append((self.home_latitude, self.home_longitude, 50.0))
        else:
            rospy.logwarn("No GPS fix for home position, skipping home waypoint.")

        # Add first zig-zag at 50m and again at 25m
        first_wp = zigzag_waypoints[0]
        mission_waypoints.append((first_wp[0], first_wp[1], 50.0))
        mission_waypoints.append((first_wp[0], first_wp[1], 25.0))

        # Add remaining zig-zag at 25m
        if len(zigzag_waypoints) > 1:
            mission_waypoints.extend([(wp[0], wp[1], 25.0) for wp in zigzag_waypoints[1:]])

        # Add target waypoint at 20m
        mission_waypoints.append((target_lat, target_lon, 20.0))

        # Upload and include RTL as final command
        nav_cnt, total_cnt, success = self.upload_full_mission(mission_waypoints, include_rtl=True)
        if not success:
            rospy.logerr("Mission upload failed. Aborting survey.")
            return

        self.nav_waypoint_count = nav_cnt
        self.total_mission_items = total_cnt
        rospy.loginfo(f"Full mission uploaded: {nav_cnt} NAV waypoints, {total_cnt - nav_cnt} non-NAV items")

        self.task = "image_capture"
        self.set_mode("AUTO")

    def upload_full_mission(self, waypoint_list: List[Tuple[float, float, float]], include_rtl: bool = False):
        """Uploads a mission list with optional RTL command at end.
        Returns: (nav_waypoint_count, total_mission_items, success)
        """
        try:
            rospy.wait_for_service('/mavros/mission/clear', timeout=5.0)
            clear_srv = rospy.ServiceProxy('/mavros/mission/clear', WaypointClear)
            clear_srv()
        except Exception as e:
            rospy.logwarn(f"Could not clear waypoints before upload: {e}")

        waypoints: List[Waypoint] = []
        for idx, (lat, lon, alt) in enumerate(waypoint_list):
            wp = Waypoint()
            wp.frame = 3  # MAV_FRAME_GLOBAL_RELATIVE_ALT
            wp.command = 16  # MAV_CMD_NAV_WAYPOINT
            wp.is_current = (idx == 0)  # Set first as current
            wp.autocontinue = True
            wp.x_lat = float(lat)
            wp.y_long = float(lon)
            wp.z_alt = float(alt)
            waypoints.append(wp)

        if include_rtl:
            rtl_wp = Waypoint()
            rtl_wp.frame = 3
            rtl_wp.command = 20  # MAV_CMD_NAV_RETURN_TO_LAUNCH
            rtl_wp.is_current = False
            rtl_wp.autocontinue = True
            rtl_wp.x_lat = 0.0
            rtl_wp.y_long = 0.0
            rtl_wp.z_alt = 0.0
            waypoints.append(rtl_wp)

        try:
            rospy.wait_for_service('/mavros/mission/push', timeout=10.0)
            push_srv = rospy.ServiceProxy('/mavros/mission/push', WaypointPush)
            resp = push_srv(start_index=0, waypoints=waypoints)
            success = bool(getattr(resp, 'success', False))
            uploaded = int(getattr(resp, 'wp_transfered', 0))
            rospy.loginfo(f"Mission upload success={success}, items_uploaded={uploaded}")
            nav_cnt = sum(1 for w in waypoints if w.command == 16)
            total_cnt = len(waypoints)
            return nav_cnt, total_cnt, success and uploaded == total_cnt
        except (rospy.ROSException, rospy.ServiceException) as e:
            rospy.logerr(f"Mission push failed: {e}")
            return 0, 0, False

    def load_target_waypoint(self, filename):
        with open(filename, 'r') as f:
            line = f.readline().strip()
            lat, lon = map(float, line.split(','))
            return lat, lon

    def waypoint_reached_callback(self, msg: WaypointReached):
        if self.cmd != "survey_and_detection":
            return

        seq = int(msg.wp_seq)
        # Update count based on sequence index; seq is 0-based
        self.waypoints_reached = max(self.waypoints_reached, seq + 1)
        rospy.loginfo(f"Reached mission item seq={seq} (progress {self.waypoints_reached}/{self.total_mission_items})")

        # Capture images on all NAV waypoints except the final NAV waypoint (target)
        if self.task == "image_capture" and seq < (self.nav_waypoint_count - 1):
            self.capture_and_save_image()

        # If we've reached the last NAV waypoint, consider survey complete
        if self.waypoints_reached >= self.nav_waypoint_count:
            rospy.loginfo("Survey NAV waypoints complete. RTL should be active.")
            self.complete_task()

    def gps_callback(self, msg: NavSatFix):
        if self.home_latitude is None:
            self.home_latitude = msg.latitude
            self.home_longitude = msg.longitude
            rospy.loginfo(f"Home position recorded: {self.home_latitude}, {self.home_longitude}")

        self.prev_latitude = self.latest_latitude
        self.prev_longitude = self.latest_longitude
        self.latest_latitude = msg.latitude
        self.latest_longitude = msg.longitude
        self.latest_altitude = msg.altitude

        if self.prev_latitude is not None and self.prev_longitude is not None:
            self.last_direction_deg = self.calculate_initial_bearing(
                (self.prev_latitude, self.prev_longitude),
                (self.latest_latitude, self.latest_longitude)
            )

    def task_command_callback(self, msg: String):
        self.cmd = msg.data
        if self.cmd == "survey_and_detection":
            rospy.loginfo("Survey and detection command received.")
            boundary = self.load_boundary("/home/raft/jakrif_ws/src/airdrop_area_survey/boundary.txt")
            if not boundary:
                rospy.logerr("Boundary not found or invalid.")
                return
            self.start_survey(boundary)

    def complete_task(self):
        rospy.loginfo("Survey + Target sequence completed (RTL in progress).")
        self.task_command_pub.publish(True)

    # --- Image capture and geotagging ---
    def capture_and_save_image(self):
        with self.frame_lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
        if frame is None:
            rospy.logwarn("No camera frame available; skipping capture.")
            return
        os.makedirs(SAVE_DIR, exist_ok=True)
        file_path = os.path.join(SAVE_DIR, f"frame{self.frame_counter:06d}.jpg")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img.save(file_path, format="JPEG", quality=92, subsampling=0, optimize=True)
        rospy.loginfo(f"Captured and saved image: {file_path}")
        if all(x is not None for x in [self.latest_latitude, self.latest_longitude, self.latest_altitude]):
            try:
                self.add_geotag_metadata(file_path)
            except Exception as e:
                rospy.logerr(f"Geotagging failed: {e}")
        self.frame_counter += 1

    def add_geotag_metadata(self, image_path):
        def decimal_to_dms(decimal):
            dd = abs(decimal)
            degrees = int(dd)
            minutes_full = (dd - degrees) * 60
            minutes = int(minutes_full)
            seconds = round((minutes_full - minutes) * 60 * 100)
            return [(degrees, 1), (minutes, 1), (int(seconds), 100)]

        gps_ifd = {
            piexif.GPSIFD.GPSLatitudeRef: b'N' if self.latest_latitude >= 0 else b'S',
            piexif.GPSIFD.GPSLatitude: decimal_to_dms(self.latest_latitude),
            piexif.GPSIFD.GPSLongitudeRef: b'E' if self.latest_longitude >= 0 else b'W',
            piexif.GPSIFD.GPSLongitude: decimal_to_dms(self.latest_longitude),
            piexif.GPSIFD.GPSAltitudeRef: 0,
            piexif.GPSIFD.GPSAltitude: (int(abs(self.latest_altitude * 100)), 100),
            piexif.GPSIFD.GPSImgDirectionRef: b'T',
            piexif.GPSIFD.GPSImgDirection: (int(round(self.last_direction_deg * 100)), 100),
        }
        exif_dict = {"GPS": gps_ifd}
        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, image_path)
        rospy.loginfo(f"Geotag added to {image_path}")

    def load_boundary(self, filename) -> List[List[float]]:
        waypoints = []
        try:
            with open(filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    lat, lon = map(float, line.split(','))
                    waypoints.append([lat, lon])
        except Exception as e:
            rospy.logerr(f"Error loading boundary: {e}")
        if len(waypoints) < 4:
            rospy.logerr("Boundary requires at least 4 points (rectangle).")
            return []
        return waypoints[:4]  # Expect rectangle (first four in order)

    def calculate_initial_bearing(self, p1, p2):
        lat1, lon1 = map(math.radians, p1)
        lat2, lon2 = map(math.radians, p2)
        dlon = lon2 - lon1
        x = math.sin(dlon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        return (math.degrees(math.atan2(x, y)) + 360) % 360

    def _quad_orientation_signs(self, rect: List[List[float]]):
        """Return edge normals orientation signs for convex quad."""
        def cross(p1, p2, p3):
            return (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
        # Determine sign using centroid
        cx = sum(p[0] for p in rect) / 4.0
        cy = sum(p[1] for p in rect) / 4.0
        signs = []
        for i in range(4):
            s = cross(rect[i], rect[(i+1) % 4], [cx, cy])
            signs.append(1 if s >= 0 else -1)
        return signs

    def _point_inside_convex_quad(self, p: Tuple[float, float], rect: List[List[float]], edge_signs: List[int]):
        def cross(p1, p2, p3):
            return (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
        for i in range(4):
            s = cross(rect[i], rect[(i+1) % 4], p)
            if s == 0:
                continue
            if (s > 0 and edge_signs[i] < 0) or (s < 0 and edge_signs[i] > 0):
                return False
        return True

    def generate_zigzag_waypoints(self, boundary: List[List[float]], altitude: float, spacing: float):
        if len(boundary) < 4:
            return []

        # Use first 4 points as rectangle corners in given order
        rect = boundary[:4]

        # Compute edge bearings and decide long/short edge
        edge01_bearing = self.calculate_initial_bearing(rect[0], rect[1])
        edge12_bearing = self.calculate_initial_bearing(rect[1], rect[2])
        # Compute approximate edge lengths in meters
        len01 = geodesic(rect[0], rect[1]).meters
        len12 = geodesic(rect[1], rect[2]).meters

        long_bearing = edge01_bearing if len01 >= len12 else edge12_bearing
        short_bearing = (long_bearing + 90.0) % 360.0

        # Determine inward/outward consistency for point-in-quad test
        edge_signs = self._quad_orientation_signs(rect)

        waypoints: List[Tuple[float, float, float]] = []
        # Start at rect[0], sweep along long edge, then shift by spacing along short edge
        current = (rect[0][0], rect[0][1])
        direction = 1  # 1: long_bearing, -1: long_bearing + 180

        while True:
            row: List[Tuple[float, float, float]] = []
            pt = current
            # Choose direction bearing for this row
            row_bearing = long_bearing if direction == 1 else (long_bearing + 180.0) % 360.0
            # March along the row until we exit
            while self._point_inside_convex_quad((pt[0], pt[1]), rect, edge_signs):
                row.append((pt[0], pt[1], altitude))
                step = geodesic(meters=spacing).destination(Point(pt[0], pt[1]), row_bearing)
                pt = (step.latitude, step.longitude)

            # Add the row in the correct traversal order
            waypoints.extend(row if direction == 1 else list(reversed(row)))
            direction *= -1

            # Shift to the next row
            step_side = geodesic(meters=spacing).destination(Point(current[0], current[1]), short_bearing)
            current = (step_side.latitude, step_side.longitude)
            if not self._point_inside_convex_quad(current, rect, edge_signs):
                break

        # Deduplicate any accidental consecutive duplicates
        deduped: List[Tuple[float, float, float]] = []
        for w in waypoints:
            if not deduped or (abs(deduped[-1][0] - w[0]) > 1e-9 or abs(deduped[-1][1] - w[1]) > 1e-9):
                deduped.append(w)
        return deduped

    def set_mode(self, mode, timeout=5.0):
        rospy.wait_for_service('/mavros/set_mode', timeout=timeout)
        try:
            resp = self.set_mode_client(0, mode)
            rospy.loginfo(f"Set mode {mode}: {resp}")
        except Exception as e:
            rospy.logerr(f"Failed to set mode: {e}")

    def _shutdown(self):
        rospy.loginfo("Shutting down and releasing camera.")
        try:
            self.frame_timer.shutdown()
        except Exception:
            pass
        if self.cap:
            self.cap.release()


if __name__ == "__main__":
    try:
        survey = AirdropSurveyAndImageCapture()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
