"""
JetAcker VLA Pipeline - path_planner.py

Author: Arush Kotta
Starting Date: July 26, 2026
Date of Most Recent Update: August 2, 2026

Description: This module is a CONTINUOUS closed-loop motion controller. It replaces an
             earlier open-loop design that computed one fixed plan per LLM trigger and
             executed it blind -- that approach assumed the car could turn in place then
             drive straight, which an Ackermann-steered car physically can't do.

             How this works instead:
                 1. When the LLM (Call #2) gives a delta, set_target() converts that --
                    which is relative to wherever the car happened to be AT THE MOMENT
                    of the LLM call -- into a fixed point in a persistent "world" frame.
                    This happens ONCE per LLM trigger.
                 2. Between LLM triggers, compute_control() gets called repeatedly (e.g.
                    ~10Hz) using the car's own live odometry (position + heading,
                    updated locally by wheel encoders/IMU -- NOT the LLM). Each call
                    recomputes where that SAME fixed target now sits relative to the
                    car's current position and heading, and outputs a fresh drive
                    command. Like glancing at a friend and adjusting your walking
                    direction continuously, rather than planning the whole walk in
                    advance.
                 3. No LLM calls happen in step 2 -- only arithmetic on odometry data.

             Output is standard (linear_x, angular_z) -- the same "speed + turn rate"
             pair any ROS robot expects via a Twist message. This does NOT compute a raw
             steering angle: the JetAcker's own onboard driver already converts
             (linear_x, angular_z) into real Ackermann wheel speeds + servo steering
             angle internally (confirmed from HiWonder's own motion control docs), and
             it already clamps steering to the car's real max angle for us.

             Real JetAcker numbers (from HiWonder's official docs):
                 wheelbase      = 0.216 m
                 track_width    = 0.195 m
                 wheel_diameter = 0.097 m
                 max steering angle = ~37 degrees (clamped by the robot's own driver)
                 cmd_vel topic = /controller/cmd_vel, geometry_msgs/Twist
                 linear.x range: -0.6 to 0.6 m/s
                 angular.z: positive = left turn, negative = right turn
                 odometry topics: /odom_raw (raw) or /odom (EKF-fused with IMU)

                NEW IN V0.2:
                - Switched output from (speed, steering_angle_deg) to standard
                  (linear_x, angular_z), matching the real /controller/cmd_vel topic.
                - Removed the manual max-steering-angle clamp -- the robot's own driver
                  already enforces this.
"""


#============================== LIBRARY IMPORTS ======================================

#The standard library imports:
import math #This is used throughout for the heading/distance trig


#================================ MOTION CONTROLLER ====================================

class MotionController:

    #This sets up the controller with the car's capabilities and the proportional gain
    def __init__(self, car_capabilities, angular_gain = 1.0):
        """
        car_capabilities: expects at least max_velocity_mps, min_velocity_mps,
                           min_stopping_distance_m.
        angular_gain: proportional gain (Kp) on heading error (radians) -> angular.z
                      (rad/s). This is a reasonable starting guess, NOT a measured
                      value -- tune on the real car once running.
        """
        self.caps = car_capabilities
        self.angular_gain = angular_gain
        self.target_world_xy = None #(x, y) in the persistent world frame


    #This function is called ONCE per LLM trigger. It converts the car-relative delta
    #(dx = forward, dy = left, relative to the car's pose AT THIS MOMENT) into a fixed
    #point in the world frame, using the car's current position + heading.
    def set_target(self, delta_result, car_state):
        if delta_result.get("status") != "moving":
            self.target_world_xy = None
            return False

        dx = delta_result["delta"]["dx"]
        dy = delta_result["delta"]["dy"]
        car_x, car_y = car_state["position_xy"]
        heading_rad = math.radians(car_state["heading_deg"])

        #====== Rotating the car-relative offset into the world frame ======

        world_dx = dx * math.cos(heading_rad) - dy * math.sin(heading_rad)
        world_dy = dx * math.sin(heading_rad) + dy * math.cos(heading_rad)

        self.target_world_xy = (car_x + world_dx, car_y + world_dy)

        return True


    #This function is called repeatedly (e.g. ~10Hz) between LLM triggers, using the
    #car's LIVE odometry-updated car_state. Returns (linear_x_mps, angular_z_radps,
    #status_label) -- the exact pair to publish on /controller/cmd_vel as a Twist.
    def compute_control(self, car_state):
        if self.target_world_xy is None:
            return (0.0, 0.0, "no_target")

        car_x, car_y = car_state["position_xy"]
        heading_deg = car_state["heading_deg"]
        target_x, target_y = self.target_world_xy

        dx = target_x - car_x
        dy = target_y - car_y
        distance = math.hypot(dx, dy)

        min_stop = self.caps.get("min_stopping_distance_m", 0.10)
        if distance <= min_stop:
            self.target_world_xy = None
            return (0.0, 0.0, "arrived")

        bearing_to_target_deg = math.degrees(math.atan2(dy, dx))
        heading_error_deg = self._normalize_angle(bearing_to_target_deg - heading_deg)
        heading_error_rad = math.radians(heading_error_deg)

        #====== Speed -- slows down near the target and while turning hard ======

        max_v = self.caps.get("max_velocity_mps", 0.5)
        min_v = self.caps.get("min_velocity_mps", 0.05)
        distance_factor = min(1.0, distance / 1.0)
        heading_factor = max(0.2, 1.0 - abs(heading_error_deg) / 90.0)
        linear_x = self._clamp(max_v * distance_factor * heading_factor, min_v, max_v)

        #====== Turn rate -- proportional to heading error ======

        #No manual clamp to a max steering angle needed here -- the robot's own driver
        #(ackermann.py) already clamps the resulting steering angle to ~37 degrees.
        angular_z = self.angular_gain * heading_error_rad

        return (round(linear_x, 3), round(angular_z, 3), "driving")


    #This function wraps an angle to [-180, 180]
    @staticmethod
    def _normalize_angle(angle_deg):
        return (angle_deg + 180) % 360 - 180

    #This function clamps a value between lo and hi
    @staticmethod
    def _clamp(value, lo, hi):
        return max(lo, min(hi, value))
