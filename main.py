import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError

import cv2
import numpy as np

class DetermineColor(Node):
    def __init__(self):
        super().__init__('color_detector')
        self.image_sub = self.create_subscription(Image, '/camera/color/image_raw', self.callback, 10)
        self.color_pub = self.create_publisher(Header, '/rotate_cmd', 10)
        self.bridge = CvBridge()

    def callback(self, data):
        try:
            # 이미지 수신 및 변환
            image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            H, W = gray.shape
            cx, cy = W // 2, H // 2
            best_score = -1
            selected_contour = None

            # 밝기 threshold를 25~45까지 5 간격으로 시도
            for thresh in range(25, 46, 5):
                _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    x, y, w, h = cv2.boundingRect(cnt)

                    # 경계에 닿는 contour 제외
                    if x <= 0 or y <= 0 or (x + w) >= W or (y + h) >= H:
                        continue

                    M = cv2.moments(cnt)
                    if M["m00"] == 0:
                        continue
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    center_dist = np.hypot(cx - cX, cy - cY)
                    score = area - 2.0 * center_dist

                    if score > best_score:
                        best_score = score
                        selected_contour = cnt

            # 마스크 생성
            mask = np.zeros_like(gray)
            if selected_contour is not None:
                cv2.drawContours(mask, [selected_contour], -1, 255, -1)

            # HSV 색상 분석
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            R_mask = (((hsv[:, :, 0] > 160) | (hsv[:, :, 0] < 20)) &
                      (hsv[:, :, 1] > 50) & (hsv[:, :, 2] > 50) & (mask == 255))
            G_mask = ((hsv[:, :, 0] > 40) & (hsv[:, :, 0] < 80) &
                      (hsv[:, :, 1] > 50) & (hsv[:, :, 2] > 50) & (mask == 255))
            B_mask = ((hsv[:, :, 0] > 100) & (hsv[:, :, 0] < 140) &
                      (hsv[:, :, 1] > 50) & (hsv[:, :, 2] > 50) & (mask == 255))

            # 픽셀 수 계산
            R_count = np.sum(R_mask)
            G_count = np.sum(G_mask)
            B_count = np.sum(B_mask)

            counts = {'Red': R_count, 'Green': G_count, 'Blue': B_count}
            dominant = max(counts, key=counts.get)

            # 퍼블리시할 메시지 준비
            msg = Header()
            msg = data.header
            msg.frame_id = '0'

            # dominant color에 따라 회전 명령 부여
            if dominant == 'Blue':
                msg.frame_id = '+1'  # CCW
            elif dominant == 'Red':
                msg.frame_id = '-1'  # CW
            else:
                msg.frame_id = '0'   # STOP

            self.color_pub.publish(msg)

        except CvBridgeError as e:
            self.get_logger().error('Failed to convert image: %s' % e)

if __name__ == '__main__':
    rclpy.init()
    detector = DetermineColor()
    rclpy.spin(detector)
    detector.destroy_node()
    rclpy.shutdown()
