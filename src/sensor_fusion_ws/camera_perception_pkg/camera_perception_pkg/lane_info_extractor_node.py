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
AVOIDANCE_REARM_SEC = 2.0    # 첫 장애물을 두 번째 장애물로 재인식하지 않을 최소 시간
AVOIDANCE_CLEAR_SEC = 0.5    # 원래 차선이 연속으로 비어 있어야 하는 시간
NEW_OBSTACLE_JUMP_DIST = 0.4  # 다음 콘으로 대상이 바뀌었다고 볼 거리 증가량[m]
NEW_OBSTACLE_PIXEL_JUMP_PX = 80.0  # 서로 다른 차선의 다음 콘으로 볼 화면 x 변화량
LANE_CHANGE_DURATION_SEC = 0.8  # 현재 차선에서 회피 차선 목표까지 이동할 시간
LANE_CHANGE_HOLD_SEC = 0.7  # 목표 오프셋 도달 후 회피 조향을 유지할 시간
DRIVING_DIRECTION = 'clockwise'  # clockwise=오른쪽 회피, counterclockwise=왼쪽 회피
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
LANE_CENTER_BIAS_PX = 0.0  # 최종 추종 경로의 좌우 평행 이동(+오른쪽 / -왼쪽)
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
        self.avoidance_rearm_sec = float(
            self.declare_parameter('avoidance_rearm_sec', AVOIDANCE_REARM_SEC).value)
        self.avoidance_clear_sec = float(
            self.declare_parameter('avoidance_clear_sec', AVOIDANCE_CLEAR_SEC).value)
        self.new_obstacle_jump_dist = float(
            self.declare_parameter(
                'new_obstacle_jump_dist', NEW_OBSTACLE_JUMP_DIST).value)
        self.new_obstacle_pixel_jump_px = float(
            self.declare_parameter(
                'new_obstacle_pixel_jump_px', NEW_OBSTACLE_PIXEL_JUMP_PX).value)
        self.lane_change_duration_sec = max(0.05, float(
            self.declare_parameter(
                'lane_change_duration_sec', LANE_CHANGE_DURATION_SEC).value))
        self.lane_change_hold_sec = max(0.0, float(
            self.declare_parameter(
                'lane_change_hold_sec', LANE_CHANGE_HOLD_SEC).value))
        self.driving_direction = str(
            self.declare_parameter('driving_direction', DRIVING_DIRECTION).value
        ).strip().lower()
        if self.driving_direction not in ('clockwise', 'counterclockwise'):
            self.get_logger().warn(
                f"driving_direction='{self.driving_direction}'는 올바르지 않음; clockwise 사용")
            self.driving_direction = 'clockwise'
        # BEV x축은 오른쪽이 양수다. 시계방향은 오른쪽, 반시계방향은 왼쪽으로 회피한다.
        self.avoidance_offset_sign = (
            1.0 if self.driving_direction == 'clockwise' else -1.0
        )
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
        self.lane_center_bias_px = float(
            self.declare_parameter('lane_center_bias_px', LANE_CENTER_BIAS_PX).value)

        self.cv_bridge = CvBridge()
        self.qos_profile = qos_profile_sensor_data
        self.subscriber = self.create_subscription(DetectionArray, self.sub_topic, self.yolov8_detections_callback, self.qos_profile)
        self.obstacle_sub = self.create_subscription(Point32, self.sub_obstacle_topic, self.obstacle_callback, self.qos_profile)
        self.path_sub = self.create_subscription(
            PathPlanningResult, 'path_planning_result', self.path_callback, self.qos_profile)
        self.publisher = self.create_publisher(LaneInfo, self.pub_topic, 10)
        self.roi_image_publisher = self.create_publisher(Image, self.roi_image_topic, 10)

        # 영상 속 라벨 위치로 현재 차선을 재판정하지 않는다. 시작 차선에서 출발해
        # 장애물 회피가 실제로 완료됐을 때만 lane_1 <-> lane_2 상태를 바꾼다.
        self.initial_lane_class = str(
            self.declare_parameter('initial_lane_class', 'lane_2').value).strip()
        if self.initial_lane_class not in ('lane_1', 'lane_2'):
            self.get_logger().warn(
                f"initial_lane_class='{self.initial_lane_class}' 는 올바르지 않음; lane_2 사용")
            self.initial_lane_class = 'lane_2'
        self.get_logger().info(f"초기 주행 차선: {self.initial_lane_class}")
        self.get_logger().info(
            f"주행 방향: {self.driving_direction} "
            f"(회피 방향: {'오른쪽' if self.avoidance_offset_sign > 0 else '왼쪽'})")

        self.current_lane_state = self.initial_lane_class
        self.pending_lane_state = self.current_lane_state
        self.current_offset = 0.0
        self.target_offset = 0.0
        self.obstacle_detected = False
        self.obstacle_dist = 999.0
        self.obstacle_pixel_x = -1.0
        # 차선 번호가 아니라 기준 차선(0) / 반대 차선(1)만 내부 상태로 관리한다.
        # 새 장애물을 만날 때마다 토글하고, 장애물이 없으면 현재 차선을 계속 유지한다.
        self.active_lane_index = 0
        self.lane_change_in_progress = False
        self.lane_change_direction_sign = 0.0
        self.lane_change_target_reached_sec = None
        self.obstacle_armed = True
        self.last_lane_change_sec = 0.0
        self.tracked_obstacle_min_dist = float('inf')
        self.tracked_obstacle_pixel_x = -1.0
        self.obstacle_clear_since_sec = None
        self.last_offset_update_sec = self.get_clock().now().nanoseconds / 1e9
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

        # 중요: lane_1/lane_2 검출이 순간적으로 뒤바뀌어도 current_lane_state는
        # 여기서 변경하지 않는다. 상태 변경 권한은 아래 장애물 회피 상태머신에만 있다.

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

        # lane_1/lane_2 검출 라벨은 회피 상태로 사용하지 않는다. 새로운 장애물을 만날 때마다
        # 현재 차선과 반대 차선을 토글하고, 장애물이 없을 때는 현재 차선을 계속 유지한다.
        now_sec = self.get_clock().now().nanoseconds / 1e9
        obstacle_close = (
            self.obstacle_detected
            and self.obstacle_dist < self.avoidance_trigger_dist
        )

        if self.obstacle_armed and obstacle_close:
            self.active_lane_index = 1 - self.active_lane_index
            self.pending_lane_state = (
                'lane_1' if self.current_lane_state == 'lane_2' else 'lane_2'
            )
            # 오프셋은 차선 변경 중에만 사용한다. 새 차선에 들어간 뒤에는 0으로
            # 되돌려 카메라에 보이는 새 차선의 중앙을 그대로 추종한다.
            self.lane_change_direction_sign = (
                self.avoidance_offset_sign
                if self.active_lane_index == 1 else -self.avoidance_offset_sign
            )
            self.lane_change_in_progress = True
            self.lane_change_target_reached_sec = None
            self.obstacle_armed = False
            self.last_lane_change_sec = now_sec
            self.obstacle_clear_since_sec = None
            self.tracked_obstacle_min_dist = self.obstacle_dist
            self.tracked_obstacle_pixel_x = self.obstacle_pixel_x
            move_to_other = self.active_lane_index == 1
            if move_to_other:
                move_direction = '오른쪽' if self.avoidance_offset_sign > 0 else '왼쪽'
            else:
                move_direction = '왼쪽' if self.avoidance_offset_sign > 0 else '오른쪽'
            self.get_logger().warn(
                f"새 장애물 {self.obstacle_dist:.2f}m: {move_direction}으로 차선 전환 "
                f"({self.current_lane_state} -> {self.pending_lane_state})")

        distance_jump = False
        pixel_jump = False
        if not self.obstacle_armed:
            if self.obstacle_detected:
                self.obstacle_clear_since_sec = None
                distance_jump = (
                    self.obstacle_dist
                    >= self.tracked_obstacle_min_dist + self.new_obstacle_jump_dist
                )
                self.tracked_obstacle_min_dist = min(
                    self.tracked_obstacle_min_dist, self.obstacle_dist)
                pixel_jump = (
                    self.tracked_obstacle_pixel_x >= 0.0
                    and abs(self.obstacle_pixel_x - self.tracked_obstacle_pixel_x)
                    >= self.new_obstacle_pixel_jump_px
                )
                # 시작 위치와 누적 비교하면 회피 중 같은 콘이 화면을 가로질러도 새 콘으로
                # 오인한다. 매 프레임 갱신해 순간적인 bbox 중심 전환만 잡는다.
                self.tracked_obstacle_pixel_x = self.obstacle_pixel_x
            elif self.obstacle_clear_since_sec is None:
                self.obstacle_clear_since_sec = now_sec
                distance_jump = False
            else:
                distance_jump = False

        self.target_offset = (
            self.lane_change_direction_sign * self.lane_width_pixel
            if self.lane_change_in_progress else 0.0
        )

        # 추종할 라벨이 잠깐 안 보이면 다른 라벨의 마스크를 영상 형상용으로만 사용한다.
        # 여기서 lane_width_pixel 보정을 넣으면 YOLO 라벨이 한 프레임 뒤집힐 때마다
        # 장애물이 없어도 가짜 차선 변경 명령(±280 px)이 만들어진다.
        final_tracking_class = tracking_class

        if tracking_class == 'lane_1':
            if has_lane_1: final_tracking_class = 'lane_1'
            elif has_lane_2: final_tracking_class = 'lane_2'
        elif tracking_class == 'lane_2':
            if has_lane_2: final_tracking_class = 'lane_2'
            elif has_lane_1: final_tracking_class = 'lane_1'

        real_target_offset = self.target_offset

        # 추론 FPS에 따라 차선 변경 시간이 달라지지 않도록 실제 경과시간 기준으로 이동한다.
        # 콜백 간격이 duration보다 길면 이번 프레임에서 목표 오프셋을 즉시 확정한다.
        elapsed_sec = max(0.0, now_sec - self.last_offset_update_sec)
        self.last_offset_update_sec = now_sec
        offset_step = self.lane_width_pixel * elapsed_sec / self.lane_change_duration_sec
        if self.current_offset < real_target_offset:
            self.current_offset = min(self.current_offset + offset_step, real_target_offset)
        elif self.current_offset > real_target_offset:
            self.current_offset = max(self.current_offset - offset_step, real_target_offset)

        # 목표 오프셋에 도달한 뒤에도 짧게 유지해야 실제 차가 차선을 끝까지 바꾼다.
        # duration과 완료 시각을 같게 두면 최대 오프셋에 닿자마자 0으로 풀려서
        # "차선을 바꾸다가 마는" 현상이 생긴다.
        change_elapsed = now_sec - self.last_lane_change_sec
        change_target_reached = abs(self.current_offset - real_target_offset) <= 5.0
        if self.lane_change_in_progress and change_target_reached:
            if self.lane_change_target_reached_sec is None:
                self.lane_change_target_reached_sec = now_sec
        elif self.lane_change_in_progress:
            self.lane_change_target_reached_sec = None

        hold_done = (
            self.lane_change_target_reached_sec is not None
            and now_sec - self.lane_change_target_reached_sec >= self.lane_change_hold_sec
        )
        if self.lane_change_in_progress and hold_done:
            self.lane_change_in_progress = False
            self.lane_change_direction_sign = 0.0
            self.lane_change_target_reached_sec = None
            self.target_offset = 0.0
            self.current_offset = 0.0
            previous_lane_state = self.current_lane_state
            self.current_lane_state = self.pending_lane_state
            self.get_logger().info(
                f"차선 전환 완료: {previous_lane_state} -> {self.current_lane_state}, "
                "새 차선 중심 추종 시작")

        if not self.obstacle_armed:
            rearm_done = change_elapsed >= self.avoidance_rearm_sec
            clear_done = (
                self.obstacle_clear_since_sec is not None
                and now_sec - self.obstacle_clear_since_sec >= self.avoidance_clear_sec
            )
            if (rearm_done and not self.lane_change_in_progress
                    and (clear_done or distance_jump or pixel_jump)):
                self.obstacle_armed = True
                self.obstacle_clear_since_sec = None
                self.tracked_obstacle_min_dist = float('inf')
                self.tracked_obstacle_pixel_x = -1.0
                if pixel_jump:
                    rearm_reason = '콘 x좌표 변경'
                elif distance_jump:
                    rearm_reason = '거리 점프'
                else:
                    rearm_reason = '검출 clear'
                self.get_logger().info(
                    f"현재 차선 {self.active_lane_index} 유지, 다음 장애물 감지 준비 "
                    f"({rearm_reason})")

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
            # 검출 라벨이 순간적으로 바뀌어도 좌/우 의미는 명령된 주행 차선 상태를 쓴다.
            # 그래야 fallback 마스크 때문에 중심 계산 방향까지 반전되지 않는다.
            target_point_x = CPFL.get_lane_center(roi_image, detection_height=target_point_y,
                                                  detection_thickness=10, road_gradient=grad,
                                                  lane_width=self.lane_width_for_center,
                                                  line_side=('left' if tracking_class == 'lane_1'
                                                             else 'right'))
            if target_point_x != -1:
                # 차체/카메라 기준점은 그대로 두고 주행 경로만 평행 이동한다.
                # 이미지/BEV x축은 오른쪽이 양수다.
                final_x = (target_point_x + self.current_offset
                           + self.lane_center_bias_px)
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
            cv2.putText(debug_img,
                        f"Lane: {self.active_lane_index} "
                        f"({'CHANGE' if self.lane_change_in_progress else ('ARMED' if self.obstacle_armed else 'WAIT_CLEAR')})",
                        (330, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
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
