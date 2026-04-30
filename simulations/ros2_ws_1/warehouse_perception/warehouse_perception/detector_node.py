import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

class WarehouseDetector(Node):
    def __init__(self):
        super().__init__('warehouse_detector')
        self.bridge = CvBridge()
        self.model = YOLO("yolov8n.pt")  # Loads the nano model for speed
        # Subscribe to the Gazebo camera topic (standard naming)
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.get_logger().info("Warehouse Perception Node Started")

    def image_callback(self, msg):
        # Convert ROS Image to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        # Run YOLO inference
        results = self.model(cv_image, verbose=False)
        # Log detected objects
        for result in results:
            self.get_logger().info(f"Detected: {result.boxes.cls}")

def main(args=None):
    rclpy.init(args=args)
    node = WarehouseDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
