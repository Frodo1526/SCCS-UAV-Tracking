"""
sccs_common — shared library for the SCCS Drone project.

Code in this package is imported, unmodified, by more than one independently
deployed subsystem (currently drone_air and tello_poc, and ground_station for
config loading). Anything here must stay platform-agnostic: no hard dependency
on GStreamer, MAVSDK, djitellopy, or any single subsystem's assumptions.
"""

__version__ = "0.1.0"
