#!/usr/bin/env python3
import os
from pathlib import Path
from typing import List, Optional, Tuple

import rospy
from std_msgs.msg import String, Bool
from mavros_msgs.msg import Waypoint, WaypointReached
from mavros_msgs.srv import WaypointPush, WaypointPushRequest, WaypointClear, SetMode


class AirdropController:

    def __init__(self) -> None:
        rospy.init_node("airdrop_controller_node")

        # --- Parameters ---
        self.coords_file: str = rospy.get_param("~coords_file", "/home/raft/jakrif_ws/src/object_detection/coords.txt")
        self.height_h: float = float(rospy.get_param("~height_h", 15.0))         # height h1
        self.height_h2: float = float(rospy.get_param("~height_h2", 20.0))       # height h2 (new param)
        self.hover_seconds: float = float(rospy.get_param("~hover_sec", 15.0))

        # --- Service clients ---
        rospy.wait_for_service("mavros/mission/push")
        rospy.wait_for_service("mavros/mission/clear")
        rospy.wait_for_service("/mavros/set_mode")

        self.waypoint_push_service = rospy.ServiceProxy("mavros/mission/push", WaypointPush)
        self.waypoint_clear_service = rospy.ServiceProxy("mavros/mission/clear", WaypointClear)
        self.set_mode_client = rospy.ServiceProxy("/mavros/set_mode", SetMode)

        # --- Topics ---
        rospy.Subscriber("/task_command", String, self.task_command_callback)
        rospy.Subscriber("/mavros/mission/reached", WaypointReached, self.waypoint_reached_callback)
        self.task_status_pub = rospy.Publisher("/task_status", Bool, queue_size=1)

        # --- State ---
        self.cmd: str = ""
        self.waypoints_reached: int = 0
        self.total_waypoints: int = 0

    # --------------------------- Utilities --------------------------- #
    def _load_coords_from_file(self) -> List[Tuple[float, float]]:
        """Read coordinates from a text file with lines formatted as: lat,long

        - Ignores blank lines and lines starting with '#'
        - Validates latitude and longitude ranges
        """
        coords_path = Path(self.coords_file)
        if not coords_path.exists():
            rospy.logerr(f"Coordinates file not found: {self.coords_file}")
            return []

        coords: List[Tuple[float, float]] = []
        try:
            with coords_path.open("r", encoding="utf-8") as f:
                for idx, raw_line in enumerate(f, start=1):
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    # Expecting 'lat,long'
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) != 2:
                        rospy.logwarn(f"Skipping line {idx}: expected 'lat,long' got '{line}'")
                        continue
                    try:
                        lat = float(parts[0])
                        lon = float(parts[1])
                    except ValueError:
                        rospy.logwarn(f"Skipping line {idx}: could not parse floats in '{line}'")
                        continue

                    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                        rospy.logwarn(f"Skipping line {idx}: out-of-range lat/lon in '{line}'")
                        continue

                    coords.append((lat, lon))
        except Exception as ex:
            rospy.logerr(f"Failed to read coordinates file '{self.coords_file}': {ex}")
            return []

        rospy.loginfo(f"Collected {len(coords)} waypoint coordinates from file '{self.coords_file}'.")
        return coords

    # --------------------------- MAVROS helpers --------------------------- #
    def set_mode(self, mode: str) -> None:
        try:
            result = self.set_mode_client.call(0, mode)
            if not result or not getattr(result, 'mode_sent', False):
                rospy.logwarn(f"Set mode request sent but not confirmed: {mode}")
            else:
                rospy.loginfo(f"Mode set to {mode}.")
        except Exception as e:
            rospy.logerr(f"Failed to set mode: {e}")

    def waypoint_reached_callback(self, msg: WaypointReached) -> None:
        rospy.loginfo(f"Waypoint {msg.wp_seq} reached.")
        self.waypoints_reached += 1
        if self.waypoints_reached == self.total_waypoints and self.cmd == "airdrop":
            self.complete_task()

    def complete_task(self) -> None:
        rospy.loginfo("Mission complete.")
        self.task_status_pub.publish(True)

    def task_command_callback(self, msg: String) -> None:
        self.cmd = msg.data
        if self.cmd == "airdrop":
            rospy.loginfo("Airdrop command received.")
            self.run_mission()

    # --------------------------- Waypoint factories --------------------------- #
    def create_waypoint(self, lat: float, lon: float, alt: float, command: int = 16, hold_time: float = 0.0) -> Waypoint:
        wp = Waypoint()
        wp.frame = 3  # MAV_FRAME_GLOBAL_RELATIVE_ALT
        wp.command = command  # NAV_WAYPOINT by default
        wp.is_current = False
        wp.autocontinue = True
        wp.param1 = float(hold_time)  # Hold time at waypoint (s)
        wp.param2 = 0.0  # Acceptance radius (m)
        wp.param3 = 0.0  # Pass through
        wp.param4 = 0.0
        wp.x_lat = float(lat)
        wp.y_long = float(lon)
        wp.z_alt = float(alt)
        return wp

    def create_rtl_waypoint(self) -> Waypoint:
        # RTL mission item; coordinates ignored by autopilot
        return self.create_waypoint(0.0, 0.0, 0.0, command=20)

    # --------------------------- Mission generation --------------------------- #
    def generate_mission(self) -> List[Waypoint]:
        # Use coordinates collected from coords file
        coords = self._load_coords_from_file()

        if not coords:
            rospy.logerr("No coordinates extracted from file. Aborting mission generation.")
            return []

        mission: List[Waypoint] = []

        # Step 1: Go to waypoint1 at height h1 (no hover)
        first_lat, first_lon = coords[0]
        mission.append(self.create_waypoint(first_lat, first_lon, self.height_h))

        first_lat, first_lon = coords[0]
        mission.append(self.create_waypoint(first_lat, first_lon, self.height_h))

        # Step 2: Waypoint1 at height h2 with hover time 15s
        mission.append(self.create_waypoint(first_lat, first_lon, self.height_h2, hold_time=self.hover_seconds))

        # Step 3: For remaining waypoints, go at height h2 with hover time 15s
        for lat, lon in coords[1:]:
            mission.append(self.create_waypoint(lat, lon, self.height_h2, hold_time=self.hover_seconds))

        # Append RTL
        mission.append(self.create_rtl_waypoint())

        self.total_waypoints = len(mission)
        rospy.loginfo(f"Generated mission with {self.total_waypoints} waypoints (including RTL).")
        return mission

    def upload_mission(self, waypoints: List[Waypoint]) -> bool:
        try:
            self.waypoint_clear_service.call()
        except Exception as e:
            rospy.logerr(f"Failed to clear mission: {e}")
            return False

        try:
            req = WaypointPushRequest()
            req.start_index = 0
            req.waypoints = waypoints
            res = self.waypoint_push_service.call(req)
            if res and res.success:
                rospy.loginfo(f"Uploaded {len(waypoints)} waypoints successfully.")
                return True
            rospy.logerr("Failed to upload waypoints.")
            return False
        except Exception as e:
            rospy.logerr(f"Upload failed: {e}")
            return False

    def run_mission(self) -> None:
        mission = self.generate_mission()
        if not mission:
            return

        self.set_mode("GUIDED")
        if not self.upload_mission(mission):
            return
        self.set_mode("AUTO")
        rospy.loginfo("Mission started.")


if __name__ == "__main__":
    try:
        node = AirdropController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

