"""차선 세그멘테이션(lane_1 / lane_2) 검출 결과 -> 주행용 차선 중심점(LaneInfo).

minicar_sim(gwakminji/minicar_sim)의 camera_perception_pkg/lane_info_extractor_node.py를
가져와서, 하드코딩돼 있던 튜닝값들을 ROS 파라미터로 뺀 버전이다.
(값은 sensor_fusion_bringup/config/params.yaml 에서 관리)

동작 순서
  1. /detections 에서 lane_1 / lane_2 마스크를 받아 "내가 지금 몇 차선인지" 판단
  2. /lidar_obstacle_info(Point32: x=거리[m], y=이미지상 중심 x[px], z=감지플래그)를 보고
     장애물이 내 차선 bbox 안에 있으면 옆 차선 쪽으로 목표 오프셋을 준다
  3. 추종할 차선 마스크의 edge를 BEV로 펴고, 높이별 차선 중심 x를 뽑아 LaneInfo로 발행
"""

import cv2
import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point32
from interfaces_pkg.msg import TargetPoint, LaneInfo, DetectionArray, PathPlanningResult
from .lib import camera_perception_func_lib as CPFL

#---------------Constant Variables---------------
SUB_TOPIC_NAME = "/detections"
SUB_OBSTACLE_TOPIC = "/lidar_obstacle_info"
PUB_TOPIC_NAME = "/yolov8_lane_info"
ROI_IMAGE_TOPIC_NAME = "/roi_image"
SHOW_IMAGE = True
LANE_WIDTH_PIXEL = 200      # 차선 변경 시 옆 차선까지의 BEV 픽셀 거리 (실차ws에서는 280을 씀)
AVOIDANCE_TRIGGER_DIST = 2.4  # 이 거리[m]보다 가까운 장애물만 회피 대상 (실차ws에서는 1.8)
SHIFT_SPEED = 20.0          # 프레임당 오프셋 변화량(px). 클수록 차선 변경이 급해짐
IMAGE_CENTER_X = 320
LANE_1_FAR_LEFT_THRESHOLD = 180
LANE_2_FAR_RIGHT_THRESHOLD = 460
# 차선 상태(1차선/2차선) 전환 디바운싱: 새 상태가 이만큼 연속으로 잡혀야 실제로 전환한다.
# 1이면 minicar_sim 원본과 동일(즉시 전환), 실차에서는 마스크가 튀어서 15 정도가 안정적.
LANE_CHANGE_THRESHOLD_COUNT = 15

# BEV 변환용 원본 이미지 좌표 4점 (좌상, 우상, 우하, 좌하) - 640x480 기준
SRC_POINTS = [154.0, 298.0, 486.0, 298.0, 614.0, 470.0, 26.0, 470.0]
ROI_CUTTING_IDX = 300       # BEV 이미지에서 아래쪽으로 잘라낼 픽셀 (차 앞쪽만 남김)
TARGET_Y_START = 5          # 타겟 포인트를 뽑을 y 시작/끝/간격 (ROI 이미지 좌표)
TARGET_Y_END = 155
TARGET_Y_STEP = 30
LANE_WIDTH_FOR_CENTER = 300  # get_lane_center가 한쪽 선만 보일 때 가정하는 차선 폭(px)
#----------------------------------------------

class Yolov8InfoExtractor(Node):
    def __init__(self):
        super().__init__('lane_info_extractor_node')
        self.sub_topic = self.declare_parameter('sub_detection_topic', SUB_TOPIC_NAME).value
        self.pub_topic = self.declare_parameter('pub_topic', PUB_TOPIC_NAME).value
        self.sub_obstacle_topic = self.declare_parameter('sub_lidar_obstacle_topic', SUB_OBSTACLE_TOPIC).value
        self.roi_image_topic = self.declare_parameter('roi_image_topic', ROI_IMAGE_TOPIC_NAME).value
        self.show_image = bool(self.declare_parameter('show_image', SHOW_IMAGE).value)

        self.lane_width_pixel = float(self.declare_parameter('lane_width_pixel', float(LANE_WIDTH_PIXEL)).value)
        self.avoidance_trigger_dist = float(
            self.declare_parameter('avoidance_trigger_dist', AVOIDANCE_TRIGGER_DIST).value)
        self.shift_speed = float(self.declare_parameter('shift_speed', SHIFT_SPEED).value)
        self.image_center_x = int(self.declare_parameter('image_center_x', IMAGE_CENTER_X).value)
        # BEV ROI에서 차량이 맞추려는 기준 x. 디버그 화면에도 수직선으로 표시한다.
        self.tracking_center_x = int(
            self.declare_parameter('tracking_center_x', self.image_center_x).value)
        self.lane_1_far_left_threshold = float(
            self.declare_parameter('lane_1_far_left_threshold', float(LANE_1_FAR_LEFT_THRESHOLD)).value)
        self.lane_2_far_right_threshold = float(
            self.declare_parameter('lane_2_far_right_threshold', float(LANE_2_FAR_RIGHT_THRESHOLD)).value)
        self.lane_change_threshold_count = int(
            self.declare_parameter('lane_change_threshold_count', LANE_CHANGE_THRESHOLD_COUNT).value)

        src_flat = list(self.declare_parameter('src_points', SRC_POINTS).value)
        self.src_mat = [[int(round(src_flat[i])), int(round(src_flat[i + 1]))] for i in range(0, 8, 2)]
        self.roi_cutting_idx = int(self.declare_parameter('roi_cutting_idx', ROI_CUTTING_IDX).value)
        self.target_y_start = int(self.declare_parameter('target_y_start', TARGET_Y_START).value)
        self.target_y_end = int(self.declare_parameter('target_y_end', TARGET_Y_END).value)
        self.target_y_step = int(self.declare_parameter('target_y_step', TARGET_Y_STEP).value)
        self.lane_width_for_center = int(
            self.declare_parameter('lane_width_for_center', LANE_WIDTH_FOR_CENTER).value)

        self.cv_bridge = CvBridge()
        self.qos_profile = qos_profile_sensor_data
        self.subscriber = self.create_subscription(DetectionArray, self.sub_topic, self.yolov8_detections_callback, self.qos_profile)
        self.obstacle_sub = self.create_subscription(Point32, self.sub_obstacle_topic, self.obstacle_callback, self.qos_profile)
        self.path_sub = self.create_subscription(
            PathPlanningResult, 'path_planning_result', self.path_callback, self.qos_profile)
        self.publisher = self.create_publisher(LaneInfo, self.pub_topic, 10)
        self.roi_image_publisher = self.create_publisher(Image, self.roi_image_topic, 10)

        # 추종할 차선을 고정한다('lane_1' | 'lane_2'). 빈 문자열이면 기존 상태머신 사용.
        # 실차에서는 한쪽 선만 보이는 구간이 대부분이라 상태머신이 흔들리고,
        # 전환될 때마다 타겟이 lane_width_pixel만큼 튄다.
        self.fixed_lane_class = str(
            self.declare_parameter('fixed_lane_class', '').value).strip()
        if self.fixed_lane_class not in ('', 'lane_1', 'lane_2'):
            self.get_logger().warn(
                f"fixed_lane_class='{self.fixed_lane_class}' 는 올바르지 않음. 자동 판단으로 진행")
            self.fixed_lane_class = ''
        if self.fixed_lane_class:
            self.get_logger().info(f"차선 추종 고정: {self.fixed_lane_class}")

        self.current_lane_state = self.fixed_lane_class or 'lane_2'
        self.current_offset = 0.0
        self.target_offset = 0.0
        self.obstacle_detected = False
        self.obstacle_dist = 999.0
        self.obstacle_pixel_x = -1.0
        self.latest_path = []

        # 차선 상태 전환 디바운싱용
        self.lane_change_counter = 0
        self.potential_next_state = None

        self.get_logger().info("Method B: BBox Overlap Logic with Debouncing Ready.")

    def obstacle_callback(self, msg: Point32):
        if msg.z == 1.0:
            self.obstacle_detected = True
            self.obstacle_dist = msg.x
            self.obstacle_pixel_x = msg.y # ★ 필수
        else:
            self.obstacle_detected = False
            self.obstacle_dist = 999.0
            self.obstacle_pixel_x = -1.0

    def path_callback(self, msg: PathPlanningResult):
        self.latest_path = list(zip(msg.x_points, msg.y_points))

    def yolov8_detections_callback(self, detection_msg: DetectionArray):
        if len(detection_msg.detections) == 0: return

        # 차선 정보 추출 (Localization용)
        lane_1_box = None
        lane_2_box = None
        lane_1_cx, lane_2_cx = -1, -1
        has_lane_1, has_lane_2 = False, False

        for d in detection_msg.detections:
            if d.class_name == 'lane_1':
                lane_1_cx = d.bbox.center.position.x
                lane_1_box = d # 박스 정보 저장
                has_lane_1 = True
            elif d.class_name == 'lane_2':
                lane_2_cx = d.bbox.center.position.x
                lane_2_box = d # 박스 정보 저장
                has_lane_2 = True

        # 내 차선 판단 (이번 프레임의 '임시' 상태)
        detected_state = self.current_lane_state

        # fixed_lane_class가 지정되면 상태 판단/전환을 아예 하지 않고 그 차선만 추종한다.
        # 실측(주행 20초, 191프레임): lane_2 단독 91.1%, 둘 다 8.9%, lane_1 단독 0%.
        # 이 상태에서 상태머신을 돌리면 lane_2_far_right_threshold 근처에서 판정이
        # 흔들리고, 전환될 때마다 lane_width_pixel(280)만큼 타겟이 통째로 이동한다.
        if self.fixed_lane_class:
            detected_state = self.fixed_lane_class
            self.current_lane_state = self.fixed_lane_class
            self.lane_change_counter = 0
            self.potential_next_state = None
        elif has_lane_1 and has_lane_2:
            dist_1 = abs(lane_1_cx - self.image_center_x)
            dist_2 = abs(lane_2_cx - self.image_center_x)
            detected_state = 'lane_1' if dist_1 < dist_2 else 'lane_2'
        elif has_lane_2 and not has_lane_1:
            detected_state = 'lane_1' if lane_2_cx > self.lane_2_far_right_threshold else 'lane_2'
        elif has_lane_1 and not has_lane_2:
            detected_state = 'lane_2' if lane_1_cx < self.lane_1_far_left_threshold else 'lane_1'

        # 상태 변경 디바운싱: 같은 새 상태가 연속으로 lane_change_threshold_count번 잡혀야 전환
        if detected_state != self.current_lane_state:
            if detected_state == self.potential_next_state:
                self.lane_change_counter += 1
            else:
                self.potential_next_state = detected_state
                self.lane_change_counter = 1

            if self.lane_change_counter >= self.lane_change_threshold_count:
                self.get_logger().info(f"🔄 Lane State Changed: {self.current_lane_state} -> {detected_state}")
                self.current_lane_state = detected_state
                self.lane_change_counter = 0
        else:
            self.lane_change_counter = 0
            self.potential_next_state = None

        # ---------------------------------------------------
        # 2. [방식 B] BBox Overlap Check (겹침 확인)
        # ---------------------------------------------------
        self.target_offset = 0.0
        tracking_class = self.current_lane_state

        # 장애물이 어디 있는지 동적으로 판단
        obstacle_in_lane_1 = False
        obstacle_in_lane_2 = False

        if self.obstacle_detected:
            # 1차선 박스 안에 장애물 중심(Pixel X)이 들어가는가?
            if has_lane_1:
                l1_min = lane_1_box.bbox.center.position.x - (lane_1_box.bbox.size.x / 2)
                l1_max = lane_1_box.bbox.center.position.x + (lane_1_box.bbox.size.x / 2)
                if l1_min < self.obstacle_pixel_x < l1_max:
                    obstacle_in_lane_1 = True

            # 2차선 박스 안에 장애물 중심(Pixel X)이 들어가는가?
            if has_lane_2:
                l2_min = lane_2_box.bbox.center.position.x - (lane_2_box.bbox.size.x / 2)
                l2_max = lane_2_box.bbox.center.position.x + (lane_2_box.bbox.size.x / 2)
                if l2_min < self.obstacle_pixel_x < l2_max:
                    obstacle_in_lane_2 = True

            # (만약 박스가 안 잡혔다면 픽셀 기준으로 대체)
            if not has_lane_1 and not has_lane_2:
                if self.obstacle_pixel_x < self.image_center_x: obstacle_in_lane_1 = True
                else: obstacle_in_lane_2 = True

        # 전략 수립
        if self.obstacle_detected and self.obstacle_dist < self.avoidance_trigger_dist:
            if self.current_lane_state == 'lane_2':
                # 내가 2차선인데 2차선에 장애물이 '확실히' 있다 -> 피함
                if obstacle_in_lane_2:
                    self.target_offset = -self.lane_width_pixel
                    self.get_logger().warn(f"🚧 Obs Inside Lane 2 Box -> Dodge LEFT")
                else:
                    # 1차선에만 있거나 어디에도 안 겹침 -> 직진
                    self.target_offset = 0.0

            elif self.current_lane_state == 'lane_1':
                # 내가 1차선인데 1차선에 장애물이 '확실히' 있다 -> 피함
                if obstacle_in_lane_1:
                    self.target_offset = self.lane_width_pixel
                    self.get_logger().warn(f"🚧 Obs Inside Lane 1 Box -> Dodge RIGHT")
                else:
                    self.target_offset = 0.0

        # 추종할 차선 마스크가 안 보이면 반대쪽 차선을 대신 추종하고 오프셋으로 보정
        final_tracking_class = tracking_class
        final_offset_modifier = 0.0

        if tracking_class == 'lane_1':
            if has_lane_1: final_tracking_class = 'lane_1'; final_offset_modifier = 0.0
            elif has_lane_2: final_tracking_class = 'lane_2'; final_offset_modifier = -self.lane_width_pixel
        elif tracking_class == 'lane_2':
            if has_lane_2: final_tracking_class = 'lane_2'; final_offset_modifier = 0.0
            elif has_lane_1: final_tracking_class = 'lane_1'; final_offset_modifier = self.lane_width_pixel

        real_target_offset = self.target_offset + final_offset_modifier

        if self.current_offset < real_target_offset: self.current_offset = min(self.current_offset + self.shift_speed, real_target_offset)
        elif self.current_offset > real_target_offset: self.current_offset = max(self.current_offset - self.shift_speed, real_target_offset)

        # CPFL.draw_edges()는 detections[0].mask로 이미지 크기를 잡는다. 우리 쪽 yolov8_node는
        # cone/car_back(detect 모델, 마스크 없음)과 lane_seg를 합쳐서 발행하므로, 첫 검출이
        # 콘이면 크기가 0인 이미지가 만들어진다. 마스크가 있는 차선 검출만 따로 넘긴다.
        lane_msg = DetectionArray()
        lane_msg.header = detection_msg.header
        lane_msg.detections = [d for d in detection_msg.detections
                               if d.class_name in ('lane_1', 'lane_2') and d.mask.height > 0]
        if not lane_msg.detections:
            return

        try:
            edge_image = CPFL.draw_edges(lane_msg, cls_name=final_tracking_class, color=255)
            (h, w) = (edge_image.shape[0], edge_image.shape[1])
            dst_mat = [[round(w * 0.2), round(h * 0.0)], [round(w * 0.8), round(h * 0.0)], [round(w * 0.8), h], [round(w * 0.2), h]]

            bird_image_raw = CPFL.bird_convert(edge_image, srcmat=self.src_mat, dstmat=dst_mat)
            bird_image = cv2.convertScaleAbs(bird_image_raw)
            roi_image = CPFL.roi_rectangle_below(bird_image, cutting_idx=self.roi_cutting_idx)

        except Exception: return

        grad = CPFL.dominant_gradient(roi_image, theta_limit=70)
        target_points = []
        for target_point_y in range(self.target_y_start, self.target_y_end, self.target_y_step):
            # lane_1은 좌측선, lane_2는 우측선이다. 어느 쪽인지 알려주면 한쪽 선만
            # 보일 때 중심을 어느 방향으로 밀지 기울기 부호로 추측하지 않아도 된다.
            target_point_x = CPFL.get_lane_center(roi_image, detection_height=target_point_y,
                                                  detection_thickness=10, road_gradient=grad,
                                                  lane_width=self.lane_width_for_center,
                                                  line_side=('left' if final_tracking_class == 'lane_1'
                                                             else 'right'))
            if target_point_x != -1:
                final_x = target_point_x + self.current_offset
                final_x = max(0, min(640, final_x))
            else: final_x = -1
            tp = TargetPoint(); tp.target_x = round(final_x); tp.target_y = round(target_point_y); target_points.append(tp)

        if self.show_image:
            debug_img = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
            # 노란선: 카메라/차량의 추종 기준, 초록점: 모델 마스크로 계산한 차선 중심점.
            cv2.line(debug_img, (self.tracking_center_x, 0),
                     (self.tracking_center_x, debug_img.shape[0] - 1), (0, 255, 255), 2)
            valid_x = []
            for point in target_points:
                if point.target_x < 0:
                    continue
                valid_x.append(point.target_x)
                cv2.circle(debug_img, (point.target_x, point.target_y), 6, (0, 255, 0), -1)
                cv2.line(debug_img, (self.tracking_center_x, point.target_y),
                         (point.target_x, point.target_y), (255, 0, 255), 1)

            # 파란선: path_planner가 실제 motion_planner로 보내는 최종 경로.
            path_pixels = np.array([
                [round(x), round(y)] for x, y in self.latest_path
                if 0 <= x < debug_img.shape[1] and 0 <= y < debug_img.shape[0]
            ], dtype=np.int32)
            if len(path_pixels) >= 2:
                cv2.polylines(debug_img, [path_pixels], False, (255, 0, 0), 2,
                              cv2.LINE_AA)

            cv2.putText(debug_img, f"Tracking: {self.current_lane_state}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if valid_x:
                mean_x = sum(valid_x) / len(valid_x)
                error_x = mean_x - self.tracking_center_x
                cv2.putText(debug_img,
                            f"center={self.tracking_center_x} target={mean_x:.1f} error={error_x:+.1f}px",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            else:
                cv2.putText(debug_img, "NO VALID LANE CENTER", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            if self.obstacle_detected:
                obs_info = "L1" if obstacle_in_lane_1 else ("L2" if obstacle_in_lane_2 else "None")
                color = ((0, 0, 255) if
                         (self.current_lane_state == 'lane_1' and obstacle_in_lane_1) or
                         (self.current_lane_state == 'lane_2' and obstacle_in_lane_2)
                         else (200, 200, 200))
                cv2.putText(debug_img, f"Obs In: {obs_info}", (10, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.imshow('Lane Info (ROI)', debug_img)
            cv2.waitKey(1)

        lane = LaneInfo(); lane.slope = grad; lane.target_points = target_points
        self.publisher.publish(lane)
        try: self.roi_image_publisher.publish(self.cv_bridge.cv2_to_imgmsg(cv2.convertScaleAbs(roi_image), encoding="mono8"))
        except: pass

def main(args=None):
    rclpy.init(args=args); node = Yolov8InfoExtractor()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); cv2.destroyAllWindows(); rclpy.shutdown()
if __name__ == '__main__': main()
