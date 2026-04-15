import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class LineFollowerPID(Node):

    def __init__(self):
        super().__init__('line_follower_pid')

        
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)

        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10)

        self.bridge = CvBridge()

        # PID values (start simple)
        self.Kp = 0.005
        self.Kd = 0.002

        self.prev_error = 0

        self.base_speed = 0.15  # forward speed

    def image_callback(self, msg):

        # Convert ROS image → OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        height, width, _ = frame.shape
        #region of interest is the bottom part of the image where the line is expected to be
        roi = frame[int(height*0.7):height, :]
        #convert to grayscale for easier processing
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        #threshold to get binary image 
         #black line will be white (255) and the rest will be black (0)
        _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

        #center point of white pixels (line) in the binary image
        M = cv2.moments(thresh)

        cmd = Twist()

        """
        M["m00"] → total white area (number of white pixels)
         M["m10"] → sum of x positions
         M["m01"] → sum of y positions
        """

        if M['m00'] > 0:
            #center of the line in the x direction (cx) is calculated by dividing the sum of x positions (M['m10']) by the total white area (M['m00'])
            cx = int(M['m10'] / M['m00'])
           #error is the distance from the center of the image to the center of the line (cx)
            error = (width // 2) - cx

           
            # PD control
           
            derivative = error - self.prev_error

            control = self.Kp * error + self.Kd * derivative

            self.prev_error = error

            
            # Convert to motion
        
            cmd.linear.x = self.base_speed
            cmd.angular.z = float(control)

        else:
            # line lost → rotate slowly
            cmd.linear.x = 0.0
            cmd.angular.z = 0.3

        # Publish command
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = LineFollowerPID()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()