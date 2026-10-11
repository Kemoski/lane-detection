"""
lane_ros_publisher.py
ROS 2 publish katmanı — görüntü işleme mantığı içermez.

Yayınlanan topic'ler:
  /lane/processed_image            sensor_msgs/Image       (BGR8)
  /lane/processed_image/compressed sensor_msgs/CompressedImage (JPEG)
  /lane/data                       std_msgs/String         (JSON)

JSON formatı:
  {
    "timestamp_ns"  : int,
    "algorithm"     : "hough_lines",
    "left_lane"     : [x1, y1, x2, y2] | null,
    "right_lane"    : [x1, y1, x2, y2] | null,
    "lanes_detected": bool,
    "fps"           : float
  }
"""

import json
import cv2

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, CompressedImage
    from std_msgs.msg import String
    from cv_bridge import CvBridge
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False


class LanePublisherNode:
    """
    ROS 2 node wrapper — sadece publish işlemleri.

    Kullanım:
        node = LanePublisherNode()
        node.publish(result_frame, left_line, right_line, fps)
        node.spin_once()
        ...
        node.destroy()
    """

    JPEG_QUALITY = 75

    def __init__(self, node_name="lane_publisher", jpeg_quality=75):
        if not ROS_AVAILABLE:
            raise RuntimeError(
                "ROS 2 (rclpy) bulunamadı. "
                "Çalıştırmadan önce 'source /opt/ros/<distro>/setup.bash' çalıştırın."
            )

        rclpy.init()
        self._node        = Node(node_name)
        self._bridge      = CvBridge()
        self._encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]

        self._pub_image = self._node.create_publisher(
            Image, "/lane/processed_image", 10
        )
        self._pub_compressed = self._node.create_publisher(
            CompressedImage, "/lane/processed_image/compressed", 10
        )
        self._pub_data = self._node.create_publisher(
            String, "/lane/data", 10
        )

        self._node.get_logger().info(
            f"{node_name} başlatıldı  |  "
            f"/lane/processed_image  /lane/processed_image/compressed  /lane/data"
        )

    def publish(self, result_frame, left_line, right_line, fps=0.0):
        """
        result_frame : anotasyonlu BGR numpy array
        left_line    : (x1,y1,x2,y2) veya None
        right_line   : (x1,y1,x2,y2) veya None
        fps          : float
        """
        now   = self._node.get_clock().now()
        stamp = now.to_msg()

        # sensor_msgs/Image (ham)
        img_msg = self._bridge.cv2_to_imgmsg(result_frame, encoding="bgr8")
        img_msg.header.stamp    = stamp
        img_msg.header.frame_id = "lane_camera"
        self._pub_image.publish(img_msg)

        # sensor_msgs/CompressedImage (JPEG)
        ok, buf = cv2.imencode(".jpg", result_frame, self._encode_param)
        if ok:
            comp_msg = CompressedImage()
            comp_msg.header.stamp    = stamp
            comp_msg.header.frame_id = "lane_camera"
            comp_msg.format          = "jpeg"
            comp_msg.data            = buf.tobytes()
            self._pub_compressed.publish(comp_msg)

        # std_msgs/String (JSON)
        payload = {
            "timestamp_ns":  now.nanoseconds,
            "algorithm":     "hough_lines",
            "left_lane":     list(left_line)  if left_line  else None,
            "right_lane":    list(right_line) if right_line else None,
            "lanes_detected": bool(left_line and right_line),
            "fps":           round(fps, 1),
        }
        str_msg      = String()
        str_msg.data = json.dumps(payload)
        self._pub_data.publish(str_msg)

    def spin_once(self):
        """Bekleyen ROS callback'lerini işle (non-blocking)."""
        rclpy.spin_once(self._node, timeout_sec=0)

    def destroy(self):
        self._node.destroy_node()
        rclpy.shutdown()
