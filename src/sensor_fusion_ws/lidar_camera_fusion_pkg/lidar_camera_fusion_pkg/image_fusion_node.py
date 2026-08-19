#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import cv2
from typing import Tuple, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import TransformStamped, Point32
from tf2_ros import Buffer, TransformListener

from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
from std_msgs.msg import Header

from interfaces_pkg.msg import DetectionArray

# 투영 수식은 calibration_node(정렬 도구)와 공유한다 - 한쪽만 고쳐서
# 두 화면이 다르게 보이는 일을 막기 위함
from lidar_camera_fusion_pkg.calibration_utils import (
    apply_calibration,
    build_fallback_extrinsic,
    project_camera_points,
    quaternion_to_matrix,
)


WINDOW_NAME = 'Fusion Visualizer'

# 숫자 키: 주 화면 전환 / q,w,e,r,t 키: 보조 화면(분할뷰) 전환
MODE_KEYS = {
    ord('1'): 'raw',
    ord('2'): 'lidar',
    ord('3'): 'boxes',
    ord('4'): 'bev',
    ord('5'): 'bev_roi',
}
SECONDARY_KEYS = {
    ord('q'): 'raw',
    ord('w'): 'lidar',
    ord('e'): 'boxes',
    ord('r'): 'bev',
    ord('t'): 'bev_roi',
}

# 라이다 투영 정렬(캘리브레이션) 실시간 보정 키.
# (키, 보정항목, 부호) - 부호는 "화면에서 점이 움직이는 방향" 기준으로 직관적이게 맞춰둠
CALIB_KEYS = {
    ord('i'): ('pitch', +1.0),   # 점을 위로
    ord('k'): ('pitch', -1.0),   # 점을 아래로
    ord('j'): ('yaw',   -1.0),   # 점을 왼쪽으로
    ord('l'): ('yaw',   +1.0),   # 점을 오른쪽으로
    ord('u'): ('roll',  -1.0),
    ord('o'): ('roll',  +1.0),
    ord('n'): ('height', -1.0),  # 카메라가 실제로 더 낮게 달림 -> 점이 위로
    ord('m'): ('height', +1.0),  # 카메라가 실제로 더 높게 달림 -> 점이 아래로
}
MODE_LABELS = {
    'raw': 'Raw Camera',
    'lidar': 'LiDAR Points',
    'boxes': 'Bounding Boxes',
    'bev': "Bird's Eye View",
    'bev_roi': "Bird's Eye ROI (Original)",
}


class FusionVisualizerNode(Node):
    """
    - 동기화(message_filters) 제거: 이미지 콜백이 오면 무조건 화면 출력
    - 최신 LaserScan / DetectionArray가 있으면 오버레이
    - QoS는 sensor_data(BEST_EFFORT)로 고정 -> 카메라/라이다와 호환성 최대화
    - 창 1개 + 숫자키(1~5)로 화면 모드 전환 (raw / lidar / boxes / bev / bev_roi)
    - v: 분할뷰 토글, q/w/e/r/t: 분할뷰의 보조 화면 선택
    """

    def __init__(self):
        super().__init__('fusion_visualizer_node')

        # -------------------------
        # Topics
        # -------------------------
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('det_topic', '/detections')
        self.declare_parameter('bird_eye_topic', '/bird_eye/image')
        self.declare_parameter('bird_eye_roi_topic', '/bird_eye/roi')

        self.declare_parameter('publish_annotated', False)
        self.declare_parameter('annotated_topic', '/fusion/annotated_image')
        self.declare_parameter('display', True)

        # 판단(decision_making) 쪽에 넘길 장애물 정보.
        # Point32(x=가장 가까운 장애물까지 거리[m], y=그 박스의 이미지상 중심 x[px], z=감지 플래그)
        # 포맷으로 /lidar_obstacle_info를 발행한다 (minicar_sim의 box_lidar_match_node와 동일 포맷).
        self.declare_parameter('publish_obstacle_info', False)
        self.declare_parameter('obstacle_topic', '/lidar_obstacle_info')
        # 장애물로 치지 않을 클래스. 차선 세그멘테이션(lane_1/lane_2)은 박스가 화면을 가득 채워서
        # 그대로 두면 "코앞에 장애물이 있다"고 잘못 나간다.
        self.declare_parameter('obstacle_class_exclude', 'lane_1,lane_2')
        # 'boxes' 화면에서 박스를 그리지 않을 클래스 (차선 마스크 박스는 화면을 다 가림)
        self.declare_parameter('box_class_exclude', 'lane_1,lane_2')

        self.declare_parameter('window_width', 960)
        self.declare_parameter('window_height', 720)

        # -------------------------
        # Intrinsic
        # -------------------------
        self.declare_parameter('fx', 478.681350)
        self.declare_parameter('fy', 480.893055)
        self.declare_parameter('cx', 314.853795)
        self.declare_parameter('cy', 259.235816)
        self.declare_parameter('distortion', [0.019110, -0.134271, 0.007227, -0.003467, 0.0])

        # -------------------------
        # Extrinsic (LiDAR -> Camera)
        # -------------------------
        # TF(URDF)를 못 쓸 때만 사용하는 폴백 값. URDF(unita_minicar.urdf)의 실제 장착 위치와
        # 맞춰둔다 - 예전 기본값(0.032 / 0.0)은 URDF와 전혀 달라서, TF가 안 뜨면 투영이
        # 조용히 엉뚱한 곳으로 나갔다.
        #   laser:  base_link 기준 x=+0.500, z=0.119
        #   camera: base_link 기준 x=-0.230, z=0.669
        #   -> 라이다 원점은 카메라보다 0.730 m 앞, 0.550 m 아래
        # (둘 다 "카메라에서 본 라이다 원점의 위치"이므로 양수)
        self.declare_parameter('cam_x_offset', 0.730)
        self.declare_parameter('cam_height', 0.550)
        # 폴백 경로에서만 쓰는 카메라 다운틸트(도). TF 경로에서는 static TF가 이미 반영함.
        self.declare_parameter('cam_pitch_deg', 14.0)
        self.declare_parameter('use_urdf_extrinsic', True)
        self.declare_parameter('lidar_frame_id', 'laser')
        self.declare_parameter('camera_frame_id', 'camera_link')

        # -------------------------
        # Extrinsic 미세보정 (실차에서 창을 보며 키로 맞춘 뒤 params.yaml에 적어두는 값)
        # 부호는 "화면에서 라이다 점이 움직이는 방향" 기준:
        #   pitch +  -> 점이 위로 / yaw +  -> 점이 오른쪽으로
        #   height + -> 카메라가 실제로 더 높이 달려있다는 뜻(가까운 점이 아래로)
        # 각도 1도 ≈ 화면상 약 10 px (fy≈567 기준) 이므로, 몇 도만 틀려도 눈에 띈다.
        # -------------------------
        self.declare_parameter('calib_pitch_deg', 0.0)
        self.declare_parameter('calib_yaw_deg', 0.0)
        self.declare_parameter('calib_roll_deg', 0.0)
        self.declare_parameter('calib_height_m', 0.0)
        self.declare_parameter('calib_step_deg', 0.25)    # 키 한 번당 각도 변화량
        self.declare_parameter('calib_step_m', 0.01)      # 키 한 번당 높이 변화량
        self.declare_parameter('show_calib_hud', True)    # 화면에 현재 보정값 표시

        # -------------------------
        # LiDAR filtering / projection
        # -------------------------
        self.declare_parameter('max_range', 10.0)
        self.declare_parameter('min_range', 0.1)
        self.declare_parameter('min_cam_z', 0.1)

        self.declare_parameter('enable_fov_filter', True)
        self.declare_parameter('cam_fov_deg', 55.0)
        self.declare_parameter('front_angle_deg', 180.0)

        self.declare_parameter('point_stride', 1)       # 라이다 점 샘플링 간격 (1이면 전부 표시)
        self.declare_parameter('draw_all_points', True) # 카메라 이미지 위에 라이다 포인트를 전부 표시할지 여부
        self.declare_parameter('display_mode', 'boxes')  # 시작 화면 모드: raw / lidar / boxes / bev / bev_roi (창에서 숫자키 1~5로 전환)
        self.declare_parameter('distance_method', 'center')  # min / p20 / median / center
        self.declare_parameter('distance_tolerance', 0.6)    # 거리 오차 허용 범위 [m]

        # 최신 데이터 유효 시간(초): 너무 오래된 scan/det는 무시
        self.declare_parameter('max_age_scan', 0.5)
        # CPU에서 2개 모델을 순차 추론하다 보니 프레임당 지연이 들쭉날쭉할 수 있음.
        # 너무 짧으면 추론이 살짝 느려질 때마다 박스가 깜빡이므로 여유 있게 잡음.
        self.declare_parameter('max_age_det', 1.5)
        # 빈 탐지 프레임이 연속 이 횟수만큼 오면 박스를 즉시 지움 (오탐이 오래 남는 것 방지)
        self.declare_parameter('det_clear_after_empty', 3)
        self.declare_parameter('max_age_bev', 1.5)
        self.declare_parameter('max_age_bev_roi', 1.5)

        # Load params
        self.image_topic = self.get_parameter('image_topic').value
        self.scan_topic = self.get_parameter('scan_topic').value
        self.det_topic = self.get_parameter('det_topic').value
        self.bird_eye_topic = self.get_parameter('bird_eye_topic').value
        self.bird_eye_roi_topic = self.get_parameter('bird_eye_roi_topic').value

        self.publish_annotated = bool(self.get_parameter('publish_annotated').value)
        self.annotated_topic = self.get_parameter('annotated_topic').value
        self.display = bool(self.get_parameter('display').value)

        self.publish_obstacle_info = bool(self.get_parameter('publish_obstacle_info').value)
        self.obstacle_topic = str(self.get_parameter('obstacle_topic').value)
        self.obstacle_class_exclude = {
            c.strip() for c in str(self.get_parameter('obstacle_class_exclude').value).split(',') if c.strip()
        }
        self.box_class_exclude = {
            c.strip() for c in str(self.get_parameter('box_class_exclude').value).split(',') if c.strip()
        }

        self.window_width = int(self.get_parameter('window_width').value)
        self.window_height = int(self.get_parameter('window_height').value)

        fx = float(self.get_parameter('fx').value)
        fy = float(self.get_parameter('fy').value)
        cx = float(self.get_parameter('cx').value)
        cy = float(self.get_parameter('cy').value)
        self.K = np.array([[fx, 0.0, cx],
                           [0.0, fy, cy],
                           [0.0, 0.0, 1.0]], dtype=np.float64)
        self.distortion = np.asarray(
            self.get_parameter('distortion').value, dtype=np.float64)

        self.use_urdf_extrinsic = bool(self.get_parameter('use_urdf_extrinsic').value)
        self.lidar_frame_id = str(self.get_parameter('lidar_frame_id').value)
        self.camera_frame_id = str(self.get_parameter('camera_frame_id').value)

        self.front_angle_deg = float(self.get_parameter('front_angle_deg').value)

        dist = float(self.get_parameter('cam_x_offset').value)
        height = float(self.get_parameter('cam_height').value)
        cam_pitch = float(self.get_parameter('cam_pitch_deg').value)
        # TF에서 읽어온(또는 폴백으로 만든) 보정 전 원본 extrinsic
        self.base_extrinsic_mat = self._init_extrinsic(
            dist, height, self.front_angle_deg, cam_pitch)
        self.tf_ok = False
        self.tf_warned = False

        self.calib_pitch_deg = float(self.get_parameter('calib_pitch_deg').value)
        self.calib_yaw_deg = float(self.get_parameter('calib_yaw_deg').value)
        self.calib_roll_deg = float(self.get_parameter('calib_roll_deg').value)
        self.calib_height_m = float(self.get_parameter('calib_height_m').value)
        self.calib_step_deg = float(self.get_parameter('calib_step_deg').value)
        self.calib_step_m = float(self.get_parameter('calib_step_m').value)
        self.show_calib_hud = bool(self.get_parameter('show_calib_hud').value)

        self.extrinsic_mat = self._apply_calibration(self.base_extrinsic_mat)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.max_range = float(self.get_parameter('max_range').value)
        self.min_range = float(self.get_parameter('min_range').value)
        self.min_cam_z = float(self.get_parameter('min_cam_z').value)

        self.enable_fov_filter = bool(self.get_parameter('enable_fov_filter').value)
        self.fov_deg = float(self.get_parameter('cam_fov_deg').value)
        self.fov_center_rad = math.radians(self.front_angle_deg)

        self.point_stride = int(self.get_parameter('point_stride').value)
        self.draw_all_points = bool(self.get_parameter('draw_all_points').value)
        self.mode = str(self.get_parameter('display_mode').value).lower().strip()
        if self.mode not in MODE_LABELS:
            self.get_logger().warn(f"Unknown display_mode '{self.mode}', falling back to 'boxes'")
            self.mode = 'boxes'
        self.secondary = 'lidar' if self.mode != 'lidar' else 'boxes'
        self.split_view = False
        self.distance_method = str(self.get_parameter('distance_method').value).lower().strip()
        self.distance_tolerance = float(self.get_parameter('distance_tolerance').value)

        self.max_age_scan = float(self.get_parameter('max_age_scan').value)
        self.max_age_det = float(self.get_parameter('max_age_det').value)
        self.det_clear_after_empty = int(self.get_parameter('det_clear_after_empty').value)
        self.max_age_bev = float(self.get_parameter('max_age_bev').value)
        self.max_age_bev_roi = float(self.get_parameter('max_age_bev_roi').value)

        self.bridge = CvBridge()

        # 최신 메시지 버퍼
        self.last_scan: Optional[LaserScan] = None
        self.last_scan_time = None

        self.last_det: Optional[DetectionArray] = None
        self.last_det_time = None
        self.empty_det_count = 0

        self.last_bev: Optional[np.ndarray] = None
        self.last_bev_time = None

        self.last_bev_roi: Optional[np.ndarray] = None
        self.last_bev_roi_time = None

        self.last_img_time = None
        self._cur_scan_age = None

        # Subscribers (QoS: sensor_data로 고정)
        self.create_subscription(Image, self.image_topic, self.image_cb, qos_profile_sensor_data)
        self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, qos_profile_sensor_data)
        self.create_subscription(DetectionArray, self.det_topic, self.det_cb, 10)  # det는 reliable여도 수신 가능
        self.create_subscription(Image, self.bird_eye_topic, self.bev_cb, qos_profile_sensor_data)
        self.create_subscription(Image, self.bird_eye_roi_topic, self.bev_roi_cb, qos_profile_sensor_data)

        # Publisher
        if self.publish_annotated:
            self.pub_img = self.create_publisher(Image, self.annotated_topic, 10)
        else:
            self.pub_img = None

        if self.publish_obstacle_info:
            self.pub_obstacle = self.create_publisher(Point32, self.obstacle_topic, 10)
        else:
            self.pub_obstacle = None

        if self.display:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, self.window_width, self.window_height)

        # 디버그 타이머: 데이터 미수신 상태를 주기적으로 알려줌
        self.create_timer(1.0, self.debug_timer)

        self.get_logger().info(
            "1:raw 2:lidar 3:boxes 4:bev 5:bev_roi (주화면) | q/w/e/r/t: 보조화면 | "
            "v: 분할보기 토글 - 창을 클릭한 뒤 키 입력\n"
            "[정렬 보정] i/k:pitch(점 위/아래) j/l:yaw(점 왼/오른쪽) u/o:roll n/m:height "
            "[/]:스텝 0:초기화 p:현재값 출력\n"
            f"FusionVisualizerNode started.\n"
            f"  image_topic={self.image_topic}\n"
            f"  scan_topic={self.scan_topic}\n"
            f"  det_topic={self.det_topic}\n"
            f"  bird_eye_topic={self.bird_eye_topic}\n"
            f"  bird_eye_roi_topic={self.bird_eye_roi_topic}\n"
            f"  publish_annotated={self.publish_annotated} ({self.annotated_topic})\n"
        )

    def _try_update_extrinsic_from_tf(self) -> None:
        if not self.use_urdf_extrinsic:
            self.extrinsic_mat = self._apply_calibration(self.base_extrinsic_mat)
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                self.camera_frame_id,
                self.lidar_frame_id,
                rclpy.time.Time()
            )
        except Exception as e:
            # 예전에는 여기서 조용히 return 해버려서, TF가 안 뜨면 URDF와 전혀 다른
            # 폴백 extrinsic으로 계속 투영되는데도 아무 경고가 없었다.
            if not self.tf_warned:
                self.get_logger().warn(
                    f"TF '{self.lidar_frame_id}' -> '{self.camera_frame_id}' 조회 실패 ({e}). "
                    f"URDF 대신 폴백 extrinsic(cam_x_offset/cam_height)으로 투영 중 - 정렬이 어긋납니다. "
                    f"unita_minicar_description의 description.launch.py(robot_state_publisher + "
                    f"camera tilt static TF)가 떠 있는지 확인하세요."
                )
                self.tf_warned = True
            self.tf_ok = False
            self.extrinsic_mat = self._apply_calibration(self.base_extrinsic_mat)
            return

        if not self.tf_ok:
            self.get_logger().info(
                f"TF extrinsic 적용: '{self.lidar_frame_id}' -> '{self.camera_frame_id}'")
        self.tf_ok = True
        self.tf_warned = False

        translation = transform.transform.translation
        rotation = transform.transform.rotation

        quat = [rotation.x, rotation.y, rotation.z, rotation.w]
        rot_mat = self._quaternion_to_matrix(quat)
        t_vec = np.array([translation.x, translation.y, translation.z], dtype=np.float64)

        ext = np.eye(4, dtype=np.float64)
        ext[:3, :3] = rot_mat
        ext[:3, 3] = t_vec

        self.base_extrinsic_mat = ext
        self.extrinsic_mat = self._apply_calibration(ext)

    def _apply_calibration(self, ext: np.ndarray) -> np.ndarray:
        """실측 미세보정을 덧씌운다 (수식은 calibration_utils와 공유)."""
        return apply_calibration(ext, self.calib_pitch_deg, self.calib_yaw_deg,
                                 self.calib_roll_deg, self.calib_height_m)

    def _bump_calibration(self, field: str, sign: float) -> None:
        if field == 'pitch':
            self.calib_pitch_deg += sign * self.calib_step_deg
        elif field == 'yaw':
            self.calib_yaw_deg += sign * self.calib_step_deg
        elif field == 'roll':
            self.calib_roll_deg += sign * self.calib_step_deg
        elif field == 'height':
            self.calib_height_m += sign * self.calib_step_m
        self.extrinsic_mat = self._apply_calibration(self.base_extrinsic_mat)
        self.get_logger().info(
            f"보정: pitch={self.calib_pitch_deg:+.2f}deg yaw={self.calib_yaw_deg:+.2f}deg "
            f"roll={self.calib_roll_deg:+.2f}deg height={self.calib_height_m:+.3f}m")

    def _log_calibration_block(self) -> None:
        """현재 보정값을 params.yaml에 그대로 붙여넣을 수 있는 형태로 출력."""
        self.get_logger().info(
            "\n===== params.yaml의 image_fusion_node: ros__parameters: 아래에 붙여넣기 =====\n"
            f"    calib_pitch_deg: {self.calib_pitch_deg:.2f}\n"
            f"    calib_yaw_deg: {self.calib_yaw_deg:.2f}\n"
            f"    calib_roll_deg: {self.calib_roll_deg:.2f}\n"
            f"    calib_height_m: {self.calib_height_m:.3f}\n"
            "==========================================================================")

    @staticmethod
    def _quaternion_to_matrix(q) -> np.ndarray:
        return quaternion_to_matrix(q)

    @staticmethod
    def _init_extrinsic(dist: float, height: float, front_angle_deg: float,
                        cam_pitch_deg: float = 0.0) -> np.ndarray:
        """TF를 못 쓸 때의 폴백 extrinsic (수식은 calibration_utils와 공유)."""
        return build_fallback_extrinsic(dist, height, front_angle_deg, cam_pitch_deg)

    def debug_timer(self):
        now = self.get_clock().now()
        def age(t):
            if t is None:
                return None
            return (now - t).nanoseconds / 1e9

        img_age = age(self.last_img_time)
        scan_age = age(self.last_scan_time)
        det_age = age(self.last_det_time)

        # 이미지가 아예 안 들어오면 검정 화면 고정입니다.
        if img_age is None or img_age > 1.0:
            self.get_logger().warn(
                f"[No Image] image_topic='{self.image_topic}' is not arriving. "
                f"scan_age={scan_age}, det_age={det_age}"
            )

    def scan_cb(self, msg: LaserScan):
        self.last_scan = msg
        self.last_scan_time = self.get_clock().now()

    def det_cb(self, msg: DetectionArray):
        # yolov8_node는 탐지가 하나도 없는 프레임에도 빈 DetectionArray를 계속 발행한다.
        # 매 프레임 그대로 반영하면 박스가 깜빡이므로 빈 메시지를 한 번은 무시하되,
        # 예전처럼 max_age_det(1.5 s)까지 통째로 버티게 두면 오탐 박스 하나가 1.5초씩
        # 화면에 남아버린다. 빈 프레임이 연속 N번 오면 바로 지운다.
        if len(msg.detections) == 0:
            self.empty_det_count += 1
            if self.empty_det_count >= self.det_clear_after_empty:
                self.last_det = None
                self.last_det_time = None
            return

        self.empty_det_count = 0
        self.last_det = msg
        self.last_det_time = self.get_clock().now()

    def bev_cb(self, img_msg: Image):
        try:
            self.last_bev = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"bev imgmsg_to_cv2 failed: {e}")
            return
        self.last_bev_time = self.get_clock().now()

    def bev_roi_cb(self, img_msg: Image):
        try:
            self.last_bev_roi = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"bev_roi imgmsg_to_cv2 failed: {e}")
            return
        self.last_bev_roi_time = self.get_clock().now()

    def image_cb(self, img_msg: Image):
        self.last_img_time = self.get_clock().now()

        # 1) image to cv2
        try:
            img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"imgmsg_to_cv2 failed: {e}")
            return

        h, w = img.shape[:2]

        self._try_update_extrinsic_from_tf()

        # 2) 최신 scan/det가 유효한지 확인
        now = self.get_clock().now()

        scan_ok = False
        det_ok = False
        bev_ok = False
        bev_roi_ok = False

        if self.last_scan is not None and self.last_scan_time is not None:
            scan_age = (now - self.last_scan_time).nanoseconds / 1e9
            scan_ok = (scan_age <= self.max_age_scan)
            # 오래된 스캔을 현재 프레임에 그대로 겹치면, 차가 움직일 때 그만큼 점이 밀린다.
            # (라이다 10 Hz 기준 0.1 s만 밀려도 회전 중에는 눈에 띄게 어긋남)
            self._cur_scan_age = scan_age

        if self.last_det is not None and self.last_det_time is not None:
            det_age = (now - self.last_det_time).nanoseconds / 1e9
            det_ok = (det_age <= self.max_age_det)

        if self.last_bev is not None and self.last_bev_time is not None:
            bev_age = (now - self.last_bev_time).nanoseconds / 1e9
            bev_ok = (bev_age <= self.max_age_bev)

        if self.last_bev_roi is not None and self.last_bev_roi_time is not None:
            bev_roi_age = (now - self.last_bev_roi_time).nanoseconds / 1e9
            bev_roi_ok = (bev_roi_age <= self.max_age_bev_roi)

        # 3) scan -> projected points (lidar/boxes 모드에서 사용)
        needed_modes = {self.mode} | ({self.secondary} if self.split_view else set())

        u_pix = np.array([], dtype=np.int32)
        v_pix = np.array([], dtype=np.int32)
        ranges = np.array([], dtype=np.float64)

        # 화면에 안 그리는 모드여도 장애물 정보를 발행해야 하면 투영은 해야 한다
        if scan_ok and ((needed_modes & {'lidar', 'boxes'}) or self.publish_obstacle_info):
            try:
                u_pix, v_pix, ranges = self.project_scan_to_image(self.last_scan, w, h)
            except Exception as exc:
                self.get_logger().error(
                    f'라이다-카메라 투영 실패: {type(exc).__name__}: {exc}',
                    throttle_duration_sec=2.0)

        if self.publish_obstacle_info:
            self._publish_obstacle_info(det_ok, scan_ok, u_pix, v_pix, ranges, w, h)

        # 4) 화면 모드별 프레임 생성 (분할뷰면 주+보조 2장, 아니면 주화면 1장)
        display = self._render_mode(
            self.mode, img, w, h, scan_ok, det_ok, bev_ok, bev_roi_ok, u_pix, v_pix, ranges)

        if self.split_view:
            secondary_frame = self._render_mode(
                self.secondary, img, w, h, scan_ok, det_ok, bev_ok, bev_roi_ok, u_pix, v_pix, ranges)
            display = cv2.hconcat([display, secondary_frame])

        # 5) show / publish
        if self.display:
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key in MODE_KEYS:
                self.mode = MODE_KEYS[key]
                self.get_logger().info(f'주화면: {self.mode}')
            elif key in SECONDARY_KEYS:
                self.secondary = SECONDARY_KEYS[key]
                self.get_logger().info(f'보조화면: {self.secondary}')
            elif key in CALIB_KEYS:
                field, sign = CALIB_KEYS[key]
                self._bump_calibration(field, sign)
            elif key == ord('v'):
                self.split_view = not self.split_view
                self.get_logger().info(f'분할보기: {self.split_view}')
            elif key == ord('p'):
                self._log_calibration_block()
            elif key == ord('0'):
                self.calib_pitch_deg = 0.0
                self.calib_yaw_deg = 0.0
                self.calib_roll_deg = 0.0
                self.calib_height_m = 0.0
                self.extrinsic_mat = self._apply_calibration(self.base_extrinsic_mat)
                self.get_logger().info('보정값 초기화')
            elif key == ord('['):
                self.calib_step_deg = max(0.05, self.calib_step_deg / 2.0)
                self.calib_step_m = max(0.002, self.calib_step_m / 2.0)
                self.get_logger().info(
                    f'보정 스텝: {self.calib_step_deg:.2f}deg / {self.calib_step_m:.3f}m')
            elif key == ord(']'):
                self.calib_step_deg = min(5.0, self.calib_step_deg * 2.0)
                self.calib_step_m = min(0.2, self.calib_step_m * 2.0)
                self.get_logger().info(
                    f'보정 스텝: {self.calib_step_deg:.2f}deg / {self.calib_step_m:.3f}m')

        if self.pub_img is not None:
            out_msg = self.bridge.cv2_to_imgmsg(display, encoding='bgr8')
            out_msg.header = Header()
            out_msg.header.stamp = img_msg.header.stamp
            out_msg.header.frame_id = img_msg.header.frame_id
            self.pub_img.publish(out_msg)

    def _publish_obstacle_info(self, det_ok, scan_ok, u_pix, v_pix, ranges, w, h):
        """가장 가까운 장애물을 Point32(x=거리[m], y=이미지상 중심 x[px], z=감지 플래그)로 발행.

        박스별 거리는 화면 표시와 똑같이 estimate_distance_in_bbox()로 구하므로,
        'boxes' 화면에 찍히는 거리와 판단 노드가 받는 거리가 항상 같다.
        obstacle_class_exclude에 든 클래스(기본: 차선)는 장애물로 치지 않는다.
        """
        closest_dist = None
        closest_cx = -1.0

        if det_ok and scan_ok and len(ranges) > 0:
            for det in self.last_det.detections:
                if str(getattr(det, 'class_name', '')) in self.obstacle_class_exclude:
                    continue

                bbox = det.bbox
                box_cx = float(bbox.center.position.x)
                box_cy = float(bbox.center.position.y)
                bw = float(bbox.size.x)
                bh = float(bbox.size.y)

                x1 = int(box_cx - bw / 2.0)
                y1 = int(box_cy - bh / 2.0)
                x2 = int(box_cx + bw / 2.0)
                y2 = int(box_cy + bh / 2.0)

                x1c = max(0, min(w - 1, x1))
                y1c = max(0, min(h - 1, y1))
                x2c = max(0, min(w - 1, x2))
                y2c = max(0, min(h - 1, y2))

                dist_m, _ = self.estimate_distance_in_bbox(u_pix, v_pix, ranges, x1c, y1c, x2c, y2c)
                if dist_m is None:
                    continue

                if closest_dist is None or dist_m < closest_dist:
                    closest_dist = dist_m
                    closest_cx = (x1c + x2c) / 2.0

        obs_msg = Point32()
        if closest_dist is not None:
            obs_msg.x = float(closest_dist)
            obs_msg.y = float(closest_cx)
            obs_msg.z = 1.0
        else:
            obs_msg.x = -1.0
            obs_msg.y = -1.0
            obs_msg.z = 0.0

        self.pub_obstacle.publish(obs_msg)

    def _render_mode(self, mode, img, w, h, scan_ok, det_ok, bev_ok, bev_roi_ok, u_pix, v_pix, ranges) -> np.ndarray:
        if mode == 'raw':
            frame = img.copy()

        elif mode == 'lidar':
            frame = img.copy()
            if scan_ok:
                self.draw_projected_points(frame, u_pix, v_pix, ranges)

        elif mode == 'boxes':
            frame = img.copy()
            if det_ok:
                for det in self.last_det.detections:
                    class_name = getattr(det, 'class_name', 'Unknown')
                    # 차선 세그멘테이션 박스는 화면을 가려서 거리 표시가 안 보이므로 그리지 않음
                    if str(class_name) in self.box_class_exclude:
                        continue
                    score = float(getattr(det, 'score', 0.0))

                    bbox = det.bbox
                    box_cx = float(bbox.center.position.x)
                    box_cy = float(bbox.center.position.y)
                    bw = float(bbox.size.x)
                    bh = float(bbox.size.y)

                    x1 = int(box_cx - bw / 2.0)
                    y1 = int(box_cy - bh / 2.0)
                    x2 = int(box_cx + bw / 2.0)
                    y2 = int(box_cy + bh / 2.0)

                    x1c = max(0, min(w - 1, x1))
                    y1c = max(0, min(h - 1, y1))
                    x2c = max(0, min(w - 1, x2))
                    y2c = max(0, min(h - 1, y2))

                    dist_m, best_uv = (None, None)
                    if scan_ok and len(ranges) > 0:
                        dist_m, best_uv = self.estimate_distance_in_bbox(
                            u_pix, v_pix, ranges, x1c, y1c, x2c, y2c
                        )

                    color = (0, 255, 0)

                    if dist_m is None:
                        text = f"{class_name} {score:.2f}  dist:N/A"
                    else:
                        text = f"{class_name} {score:.2f}  dist:{dist_m:.2f}m"

                    self._draw_box_with_label(frame, x1c, y1c, x2c, y2c, color, text, best_uv)

        elif mode == 'bev':  # bird_eye_node(camera_perception_pkg)의 버드아이뷰 출력
            if bev_ok and self.last_bev is not None:
                frame = cv2.resize(self.last_bev, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                frame = self._bev_placeholder(img.shape, "Bird's Eye View - No Signal")

        else:  # bev_roi - bird_eye_node의 ROI 표시용 원본 출력
            if bev_roi_ok and self.last_bev_roi is not None:
                frame = cv2.resize(self.last_bev_roi, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                frame = self._bev_placeholder(img.shape, "Bird's Eye ROI - No Signal")

        cv2.putText(frame, MODE_LABELS[mode], (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if self.show_calib_hud and mode in ('lidar', 'boxes'):
            self._draw_calib_hud(frame, scan_ok)
        return frame

    def _draw_calib_hud(self, frame, scan_ok: bool) -> None:
        """정렬을 맞추는 동안 현재 보정값/스캔 지연을 화면에 표시."""
        h = frame.shape[0]
        src = 'TF' if (self.use_urdf_extrinsic and self.tf_ok) else 'FALLBACK'
        lines = [
            f"calib  pitch {self.calib_pitch_deg:+.2f}  yaw {self.calib_yaw_deg:+.2f}"
            f"  roll {self.calib_roll_deg:+.2f}  h {self.calib_height_m:+.3f}",
            f"step {self.calib_step_deg:.2f}deg/{self.calib_step_m:.3f}m   ext:{src}",
            "i/k pitch  j/l yaw  u/o roll  n/m height  [/] step  0 reset  p print",
        ]
        if scan_ok and self._cur_scan_age is not None:
            warn = "  <-- 지연 큼(움직이면 어긋남)" if self._cur_scan_age > 0.15 else ""
            lines.append(f"scan age {self._cur_scan_age * 1000:.0f} ms{warn}")

        y = h - 8 - 18 * (len(lines) - 1)
        for line in lines:
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 255, 255), 1, cv2.LINE_AA)
            y += 18

    @staticmethod
    def _bev_placeholder(shape, text: str) -> np.ndarray:
        h, w = shape[:2]
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(frame, text, (30, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        return frame

    def draw_projected_points(self, img, u_pix, v_pix, ranges=None):
        if len(u_pix) == 0:
            return

        stride = max(1, self.point_stride)
        for idx in range(0, len(u_pix), stride):
            if not self.draw_all_points and idx != 0:
                continue

            u = int(u_pix[idx])
            v = int(v_pix[idx])
            # 거리별로 색을 달리해서, 어떤 점이 어떤 물체인지 눈으로 대응시키기 쉽게 함
            # (정렬 보정할 때 콘 위에 어느 점이 찍혀야 하는지 판단하는 용도)
            if ranges is not None and idx < len(ranges):
                t = float(np.clip(ranges[idx] / max(self.max_range, 1e-3), 0.0, 1.0))
                color = (int(255 * t), 80, int(255 * (1.0 - t)))  # 가까움=빨강, 멂=파랑
            else:
                color = (0, 0, 255)
            cv2.circle(img, (u, v), 2, color, -1)

    def _draw_box_with_label(self, img, x1, y1, x2, y2, color, text, best_uv):
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_x, label_y = self._compute_distance_label_position(
            x1, y1, x2, y2, text_size[0], text_size[1], img.shape[1], img.shape[0]
        )

        cv2.rectangle(
            img,
            (label_x - 2, label_y - text_size[1] - 2),
            (label_x + text_size[0] + 2, label_y + 2),
            (0, 0, 0),
            -1,
        )
        cv2.putText(img, text, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        if best_uv is not None:
            cv2.circle(img, best_uv, 4, (255, 255, 255), -1)

    def _compute_distance_label_position(self, x1, y1, x2, y2, text_w, text_h, img_w, img_h):
        margin_x = 6
        margin_y = 8

        # label_y is the text baseline; the background box spans
        # [label_y - text_h - 2, label_y + 2], so offset by text_h to keep
        # the label inside the bbox (below its top edge) instead of above it.
        label_y = y1 + margin_y + text_h + 2
        label_y = max(text_h + 2, min(label_y, img_h - 3))

        label_x = x2 - text_w - margin_x
        label_x = max(margin_x, min(label_x, img_w - text_w - margin_x))

        return label_x, label_y

    def project_scan_to_image(self, scan_msg: LaserScan, img_w: int, img_h: int):
        ranges = np.asarray(scan_msg.ranges, dtype=np.float64)
        n = ranges.shape[0]
        angles = scan_msg.angle_min + np.arange(n, dtype=np.float64) * scan_msg.angle_increment

        finite = np.isfinite(ranges)
        valid = finite & (ranges >= self.min_range) & (ranges <= min(float(scan_msg.range_max), self.max_range))

        if self.enable_fov_filter:
            half_fov = math.radians(self.fov_deg / 2.0)
            angle_diff = np.abs(np.arctan2(np.sin(angles - self.fov_center_rad), np.cos(angles - self.fov_center_rad)))
            valid = valid & (angle_diff <= half_fov)

        if np.count_nonzero(valid) == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float64)

        ranges_v = ranges[valid]
        angles_v = angles[valid]

        x = ranges_v * np.cos(angles_v)
        y = ranges_v * np.sin(angles_v)
        z = np.zeros_like(x)
        ones = np.ones_like(x)

        pts_lidar = np.vstack([x, y, z, ones])  # 4xN
        pts_cam = self.extrinsic_mat @ pts_lidar

        front = pts_cam[2, :] > self.min_cam_z
        pts_cam = pts_cam[:, front]
        ranges_v = ranges_v[front]

        if pts_cam.shape[1] == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float64)

        u, v = project_camera_points(
            pts_cam[:3, :], self.K, self.distortion)

        # astype은 0쪽으로 버림이라 좌우가 비대칭으로 밀린다. 반올림으로 교체.
        u_i = np.rint(u).astype(np.int32)
        v_i = np.rint(v).astype(np.int32)

        inside = (u_i >= 0) & (u_i < img_w) & (v_i >= 0) & (v_i < img_h)
        return u_i[inside], v_i[inside], ranges_v[inside]

    def estimate_distance_in_bbox(self, u, v, ranges, x1, y1, x2, y2) -> Tuple[Optional[float], Optional[Tuple[int, int]]]:
        mask = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
        if np.count_nonzero(mask) == 0:
            return None, None

        r = ranges[mask]
        uu = u[mask]
        vv = v[mask]

        filtered = self._filter_front_points(r, uu, vv, x1, y1, x2, y2)
        if filtered is None:
            return None, None

        candidate_r, candidate_u, candidate_v = filtered
        if len(candidate_r) == 0:
            return None, None

        if self.distance_method == 'min':
            idx = int(np.argmin(candidate_r))
            dist = float(candidate_r[idx])
            best_uv = (int(candidate_u[idx]), int(candidate_v[idx]))
            return dist, best_uv

        if self.distance_method == 'median':
            dist = float(np.median(candidate_r))
            idx = int(np.argmin(np.abs(candidate_r - dist)))
            best_uv = (int(candidate_u[idx]), int(candidate_v[idx]))
            return dist, best_uv

        if self.distance_method == 'p20':
            dist = float(np.percentile(candidate_r, 20))
            idx = int(np.argmin(np.abs(candidate_r - dist)))
            best_uv = (int(candidate_u[idx]), int(candidate_v[idx]))
            return dist, best_uv

        # center: bbox 중심에 가깝고, 앞쪽(더 짧은 거리) 포인트들을 우선적으로 사용
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        center_deltas = np.sqrt((candidate_u - cx) ** 2 + (candidate_v - cy) ** 2)
        center_order = np.argsort(center_deltas)
        sorted_r = candidate_r[center_order]
        sorted_u = candidate_u[center_order]
        sorted_v = candidate_v[center_order]

        # 오차 허용 범위 안에 있는 포인트들을 묶어서 평균 사용
        if len(sorted_r) >= 1:
            base_dist = float(sorted_r[0])
            tolerance = max(self.distance_tolerance, 1e-3)
            close_mask = np.abs(sorted_r - base_dist) <= tolerance
            if np.count_nonzero(close_mask) == 0:
                close_mask = np.ones_like(sorted_r, dtype=bool)

            selected_r = sorted_r[close_mask]
            selected_u = sorted_u[close_mask]
            selected_v = sorted_v[close_mask]

            if len(selected_r) >= 1:
                dist = float(np.mean(selected_r))
                best_uv = (int(np.mean(selected_u)), int(np.mean(selected_v)))
                return dist, best_uv

        idx = int(np.argmin(sorted_r))
        dist = float(sorted_r[idx])
        best_uv = (int(sorted_u[idx]), int(sorted_v[idx]))
        return dist, best_uv

    def _filter_front_points(self, r, u, v, x1, y1, x2, y2):
        if len(r) == 0:
            return None

        valid_mask = np.isfinite(r) & (r >= self.min_range) & (r <= self.max_range)
        if np.count_nonzero(valid_mask) == 0:
            return None

        r_valid = r[valid_mask]
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]

        if len(r_valid) == 0:
            return None

        # 박스 중심 기준으로 앞쪽에 있는 포인트만 남김
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        center_deltas = np.sqrt((u_valid - cx) ** 2 + (v_valid - cy) ** 2)
        center_order = np.argsort(center_deltas)

        ordered_r = r_valid[center_order]
        ordered_u = u_valid[center_order]
        ordered_v = v_valid[center_order]

        # 가까운 포인트를 우선적으로 사용하되, 너무 멀리 있는 배경 포인트는 제외
        base_dist = float(np.min(ordered_r))
        tolerance = max(self.distance_tolerance, 1e-3)
        close_mask = np.abs(ordered_r - base_dist) <= tolerance
        if np.count_nonzero(close_mask) == 0:
            close_mask = np.ones_like(ordered_r, dtype=bool)

        return ordered_r[close_mask], ordered_u[close_mask], ordered_v[close_mask]


def main(args=None):
    rclpy.init(args=args)
    node = FusionVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.display:
            cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
