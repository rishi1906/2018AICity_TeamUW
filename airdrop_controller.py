#!/usr/bin/env python3
from pathlib import Path
from typing import List, Optional, Tuple

import rospy
from std_msgs.msg import String, Bool
from mavros_msgs.msg import Waypoint, WaypointReached
from mavros_msgs.srv import WaypointPush, WaypointPushRequest, WaypointClear, SetMode

try:
    import piexif  # type: ignore
except Exception:  # pragma: no cover - environment specific
    piexif = None
    import traceback
    traceback.print_exc()


class AirdropController:

    def __init__(self) -> None:
        rospy.init_node("airdrop_controller_node")

        # --- Parameters ---
        self.images_dir: str = rospy.get_param(
            "~images_dir", "/home/raft/jakrif_ws/src/object_detection/output_imgs"
        )
        self.height_h: float = float(rospy.get_param("~height_h", 30.0))         # height h1
        self.height_h2: float = float(rospy.get_param("~height_h2", 50.0))       # height h2 (new param)
        self.hover_seconds: float = float(rospy.get_param("~hover_sec", 15.0))
        self.file_extensions: List[str] = [
            str(ext).lower() for ext in rospy.get_param("~file_extensions", [".jpg", ".jpeg", ".tif", ".tiff"])
        ]

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

        if piexif is None:
            rospy.logwarn(
                "piexif not available. Install with: pip install piexif. Node cannot parse EXIF GPS without it."
            )

    # --------------------------- Utilities --------------------------- #
    @staticmethod
    def _to_float(num: float, den: float) -> float:
        return float(num) / float(den) if den else 0.0

    @classmethod
    def _dms_to_deg(cls, dms: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]) -> float:
        degrees = cls._to_float(*dms[0])
        minutes = cls._to_float(*dms[1])
        seconds = cls._to_float(*dms[2])
        return degrees + (minutes / 60.0) + (seconds / 3600.0)

    def _get_gps_from_image(self, image_path: Path) -> Optional[Tuple[float, float]]:
        """Extract (lat, lon) in decimal degrees from EXIF using piexif.
        Returns None if GPS data missing or invalid.
        """
        if piexif is None:
            return None
        try:
            exif_dict = piexif.load(str(image_path))
            gps_ifd = exif_dict.get("GPS", {})
            if not gps_ifd:
                return None

            lat = gps_ifd.get(piexif.GPSIFD.GPSLatitude)
            lat_ref = gps_ifd.get(piexif.GPSIFD.GPSLatitudeRef)
            lon = gps_ifd.get(piexif.GPSIFD.GPSLongitude)
            lon_ref = gps_ifd.get(piexif.GPSIFD.GPSLongitudeRef)

            if not (lat and lon and lat_ref and lon_ref):
                return None

            lat_deg = self._dms_to_deg(lat)
            lon_deg = self._dms_to_deg(lon)

            # lat_ref/lon_ref may be bytes
            lat_ref_val = lat_ref.decode("ascii") if isinstance(lat_ref, (bytes, bytearray)) else str(lat_ref)
            lon_ref_val = lon_ref.decode("ascii") if isinstance(lon_ref, (bytes, bytearray)) else str(lon_ref)

            if lat_ref_val.upper() == "S":
                lat_deg = -lat_deg
            if lon_ref_val.upper() == "W":
                lon_deg = -lon_deg

            return lat_deg, lon_deg
        except Exception as ex:  # pragma: no cover - depends on EXIF content
            rospy.logwarn(f"Failed to parse EXIF GPS from {image_path.name}: {ex}")
            return None

    def _collect_waypoint_coords(self) -> List[Tuple[float, float]]:
        """Scan images directory and return list of (lat, lon) in sorted order by filename."""
        images_dir_path = Path(self.images_dir)
        if not images_dir_path.exists():
            rospy.logerr(f"Images directory not found: {self.images_dir}")
            return []

        image_paths: List[Path] = [
            p for p in images_dir_path.iterdir()
            if p.is_file() and p.suffix.lower() in self.file_extensions
        ]
        image_paths.sort(key=lambda p: p.name)

        coords: List[Tuple[float, float]] = []
        for img in image_paths:
            gps = self._get_gps_from_image(img)
            if gps:
                coords.append(gps)
            else:
                rospy.logwarn(f"No GPS EXIF in {img.name}; skipping.")

        rospy.loginfo(f"Collected {len(coords)} waypoint coordinates from EXIF.")
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
        self.task_status_pub.publish(Bool(data=True))

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
        wp.param2 = 1.0  # Acceptance radius (m)
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
        # Use coordinates collected from images if available
        coords = self._collect_waypoint_coords()

        if not coords:
            rospy.logerr("No coordinates extracted from images. Aborting mission generation.")
            return []

        mission: List[Waypoint] = []

        # Step 1: Go to waypoint1 at height h1 (no hover)
        first_lat, first_lon = coords[0]
        mission.append(self.create_waypoint(first_lat, first_lon, self.height_h))

        # Step 2: Waypoint1 at height h2 with hover time
        mission.append(self.create_waypoint(first_lat, first_lon, self.height_h2, hold_time=self.hover_seconds))

        # Step 3: For remaining waypoints, go at height h2 with hover time
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
            if res and getattr(res, 'success', False):
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

