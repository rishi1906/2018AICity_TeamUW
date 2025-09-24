#!/usr/bin/env python3

import os
import sys
import time
from typing import Optional, Tuple

import rospy


FileSignature = Tuple[int, int, int]


def get_file_signature(file_path: str) -> Optional[FileSignature]:
    """Return a tuple uniquely identifying file state or None if not available.

    The tuple is (inode, size, mtime_ns). Using nanosecond resolution when available
    ensures we capture rapid updates. If the file does not exist, return None.
    """
    try:
        stat_result = os.stat(file_path)
    except FileNotFoundError:
        return None
    except PermissionError:
        # Treat as not readable; effectively not available
        return None

    inode = getattr(stat_result, "st_ino", 0)
    size = getattr(stat_result, "st_size", 0)
    mtime_ns = getattr(stat_result, "st_mtime_ns", int(stat_result.st_mtime * 1e9))
    return (int(inode), int(size), int(mtime_ns))


class InactivityMonitor:
    def __init__(self) -> None:
        # Parameters
        self.file_path: str = rospy.get_param("~file_path", "log_file.log")
        self.inactivity_seconds: float = float(rospy.get_param("~inactivity_seconds", 60.0))
        self.poll_hz: float = float(rospy.get_param("~poll_hz", 1.0))
        self.print_once_per_event: bool = bool(rospy.get_param("~print_once_per_event", True))

        # State
        self.last_signature: Optional[FileSignature] = None
        self.last_change_monotonic: float = time.monotonic()
        self.already_printed_for_current_inactivity: bool = False

        # Initialize state from current file (if any)
        initial_signature = get_file_signature(self.file_path)
        if initial_signature is not None:
            self.last_signature = initial_signature
            self.last_change_monotonic = time.monotonic()
            self.already_printed_for_current_inactivity = False
            rospy.loginfo("Monitoring file: %s", self.file_path)
        else:
            # File is not present yet; wait for it to appear
            self.last_signature = None
            self.last_change_monotonic = time.monotonic()
            self.already_printed_for_current_inactivity = False
            rospy.loginfo("File not found yet, waiting: %s", self.file_path)

        # Start timer
        period = max(0.01, 1.0 / max(self.poll_hz, 0.01))
        self._timer = rospy.Timer(rospy.Duration.from_sec(period), self._on_timer)

    def _on_timer(self, _event: rospy.timer.TimerEvent) -> None:
        now_mono = time.monotonic()
        current_signature = get_file_signature(self.file_path)

        # Handle file appearance/disappearance and content changes
        if current_signature is None:
            # File missing
            if self.last_signature is not None:
                # It disappeared -> treat as a change and reset inactivity window
                self.last_signature = None
                self.last_change_monotonic = now_mono
                self.already_printed_for_current_inactivity = False
                rospy.loginfo_throttle(30.0, "File disappeared, waiting for it to reappear: %s", self.file_path)
            else:
                # Still missing; keep resetting the clock so we do not print while absent
                self.last_change_monotonic = now_mono
            return

        # current_signature is not None here
        if self.last_signature is None:
            # File appeared -> treat as change
            self.last_signature = current_signature
            self.last_change_monotonic = now_mono
            self.already_printed_for_current_inactivity = False
            rospy.loginfo("File detected: %s", self.file_path)
            return

        if current_signature != self.last_signature:
            # Content or metadata changed -> reset inactivity window
            self.last_signature = current_signature
            self.last_change_monotonic = now_mono
            self.already_printed_for_current_inactivity = False
            return

        # No change detected; check inactivity duration
        inactive_seconds = now_mono - self.last_change_monotonic
        if inactive_seconds >= self.inactivity_seconds:
            if not self.already_printed_for_current_inactivity:
                # Print to stdout as requested and flush immediately
                print("true")
                try:
                    sys.stdout.flush()
                except Exception:
                    pass
                if self.print_once_per_event:
                    self.already_printed_for_current_inactivity = True


def main() -> None:
    rospy.init_node("inactivity_monitor", anonymous=False)
    _monitor = InactivityMonitor()
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass

