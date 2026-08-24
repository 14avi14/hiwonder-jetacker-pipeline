'''
JetAcker VLA Pipeline - target_finder.py

Description: This recieves instructions from the host server/computer and sends commands 
             to topics in order to control the actual JetAcker. This file, along with 
             a copy of socket_ops.py and path_planner.py must, must be put on the 
             JetAcker's Jetson Nano directly. 
             
             Note: All libraries except for dotenv should already be on the Jetson Nano.
'''

import socket
import json
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy # Needed to specify QoS
from message_filters import Subscriber, ApproximateTimeSynchronizer # Sync the depth and rgb images from different topics
from sensor_msgs.msg import Image, CameraInfo # Images(depth and rgb) and intrinsic camera parameters, respectively
from geometry_msgs.msg import Twist # For controlling the car
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

# Local imports
import socket_ops
import path_planner

#------------------- Following is copied over from run_pipeline.py --------------------
#============================== CAR STATE / CAPABILITIES ==============================

#Starting state -- zeroed since we're not connected to real odometry yet. On the real
#car, replace this with a live read from /odom or /odom_raw each cycle
#(nav_msgs/msg/Odometry -- gives position + heading directly).
STARTING_CAR_STATE = {
    "velocity": 0.0,
    "heading_deg": 0.0,
    "position_xy": [0.0, 0.0]
}

CAR_CAPABILITIES = {
    #====== REAL, from HiWonder's official JetAcker motion control docs ======
    "wheelbase_m": 0.216,
    "track_width_m": 0.195,
    "wheel_diameter_m": 0.097,
    "max_steering_angle_deg": 37,
    "max_velocity_mps": 0.6, #Docs: keep linear.x within -0.6 to 0.6

    #====== CALCULATED from the real numbers above ======
    #Not directly published, but a straightforward derivation:
    #R = wheelbase / tan(max_steering)
    "min_turn_radius_m": round(0.216 / math.tan(math.radians(37)), 3),

    #====== ESTIMATED / our own design choices, NOT from any spec ======
    #Untested on real hardware -- treat these as starting guesses to tune.
    "min_velocity_mps": 0.05, #Arbitrary minimum crawl speed
    "min_stopping_distance_m": 0.10, #Arbitrary safety margin

    #====== PLACEHOLDER, NOT MEASURED -- forward-looking only ======
    #For "can the car fit through this gap / turn in this corridor" reasoning.
    #Currently INERT: nothing in vision.py measures gap or corridor width yet, so even
    #accurate numbers here have nothing to compare against. Two separate todos before
    #this is real:
    #    1. Measure the actual chassis with a tape measure (no official spec sheet
    #       found online) and replace the guesses below.
    #    2. Build the perception logic that outputs a gap/clear-width measurement for
    #       the LLM to compare width_m against.
    "length_m": 0.30, #GUESS, not measured
    "width_m": 0.18,  #GUESS, not measured
    "height_m": 0.15  #GUESS, not measured
}

sim_topics = {
    "odom": "/odom",
    "cmd_vel": "/controller/cmd_vel",
    "depth_cam": "/depth_cam/depth_cam", # This topic returns an rgb image, not depth, as such, depth capabilities will not work in simulation
    "rgb_cam": "/depth_cam/depth_cam",
}


real_topics = {
    "odom": "/odom_raw",
    "cmd_vel": "/controller/cmd_vel",
    "depth_cam": "/depth_cam/depth/image_raw",
    "rgb_cam": "/depth_cam/rgb/image_raw",
}

class RemoteController(Node):
    def __init__(self, is_sim=False):
        super().__init__("remote_controller")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, # Ensure messages are delivered
            history=HistoryPolicy.KEEP_LAST, # Keep last (N) messages, N=depth
            depth=10
        )

        if is_sim:
            topics = sim_topics
        else:
            topics = real_topics

        self.odom_sub = self.create_subscription(Odometry, topics["odom"], self.run_trajectory, qos) # Odometer subscriber

        self.control_pub = self.create_publisher(Twist, topics["cmd_vel"], qos) # Publishes instructions for movement 

        # Subscribing to topics for images
        self.depth_sub = Subscriber(self, Image, topics["depth_cam"])
        self.rgb_sub = Subscriber(self, Image, topics["rgb_cam"])

        self.bridge = CvBridge() # ROS2 Image to cv2 bridge

        # Synchronizing the subscribers such that both the rgb and depth image can be sent to the server
        queue_size = 10
        max_delay = 1
        self.time_sync = ApproximateTimeSynchronizer([self.depth_sub, self.rgb_sub],
                                                  queue_size, max_delay)
        self.time_sync.registerCallback(self.update_target)

        # Socket connection
        self.connection = socket.socket()
        self.connection.connect(socket_ops.SERVER_ADDRESS)

        # Path planner controller
        self.controller = path_planner.MotionController(CAR_CAPABILITIES)
        self.get_instructions = True
        self.car_state = STARTING_CAR_STATE

        self.get_logger().info("Starting controls")

    def update_target(self, depth, rgb):
        if self.get_instructions:
            cv_bgr_image = self.bridge.imgmsg_to_cv2(rgb)[..., ::-1] #Reverse index changes RGB to BGR
            cv_depth_image = self.bridge.imgmsg_to_cv2(depth, desired_encoding="passthrough")

            socket_ops.send_string(self.connection, socket_ops.INITIAL_PREDICTION_MSG) # Notify server that need initial prediction
            socket_ops.send_image(self.connection, cv_bgr_image) # Send the color(in BGR format) image
            socket_ops.send_image(self.connection, cv_depth_image) # Send the depth image
            self.get_logger().info("[MSG SENT] Messages sent to car")

            result = json.loads(socket_ops.recieve_string(self.connection))
            got_target = self.controller.set_target(result, self.car_state)

            if not got_target:
                print(f"No target set (Call #2 status was '{result.get('status')}', not 'moving'). "
                    "Nothing to drive toward -- car stays stopped.")
            else:
                self.get_instructions = False

    def run_trajectory(self, odom):
        position = odom.pose.pose.position
        angle = odom.pose.pose.orientation.z
        linear_vel = odom.twist.twist.linear.x
        angular_vel = odom.twist.twist.angular.z

        self.car_state = {
            "velocity": linear_vel,
            "heading_deg": math.degrees(angle),
            "position_xy": [position.x, position.y]
        }

        linear_x, angular_z, status = self.controller.compute_control(self.car_state)

        self.get_logger().info(f"STATUS: {status}\nCar State: {self.car_state}\nTarget: {self.controller.target_world_xy}")

        if status == "arrived":
            self.get_logger().info("============ FINISHED ============")
        elif status == "no_target":
            pass

        msg = Twist()
        msg.linear.x = linear_x
        msg.linear.y = 0.0
        msg.linear.z = 0.0

        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = angular_z

        self.control_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = RemoteController(is_sim=False) # Change depending on whether in simulation or not
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()