#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import argparse
import numpy as np
from ultralytics import YOLO



class YoloZedNode(Node):
    def __init__(self, tensorrt_model_path, imgsz=640):
        super().__init__('yolo_zed_node')
        self.get_logger().info("Initializing YOLO ZED node")
        self.subscription = self.create_subscription(
            Image,
            '/zed/zed_node/rgb/image_rect_color',
            self.listener_callback,
            10
        )
        self.bridge = CvBridge()
        self.imgsz = imgsz

        self.net = YOLO("/home/rm/test_ws/src/obj_det/obj_det/atwork.engine")
        self.get_logger().info(f"Loaded TENSORRT model from: {tensorrt_model_path}")

    def listener_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if cv_image is None or cv_image.size == 0:
                self.get_logger().error("Converted image is empty")
                return
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        if len(cv_image.shape) == 3 and cv_image.shape[2] == 4:
            cv_image = cv_image[:, :, :3]

        cv_image = np.ascontiguousarray(cv_image)  
        annotated_img = results[0].plot()
        cv2.imshow("YOLOv8 tensorrt Detection", annotated_img)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser(description="ROS2 Node for YOLO tensorrt inference on ZED camera feed")
    parser.add_argument('--tensorrt_model', type=str, required=True, help="Path to the YOLO tensorrt model file")
    parser.add_argument('--imgsz', type=int, default=640, help="Input image size (default: 640)")
    parsed_args, unknown = parser.parse_known_args()

    node = YoloZedNode(parsed_args.tensorrt_model, imgsz=parsed_args.imgsz)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down YOLO node...")
    finally:
        node.destroy_node()
        if rclpy.ok():  
            rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()