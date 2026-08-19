#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""라이다-카메라 정렬 캘리브레이션 도구.

카메라 다운틸트 각도를 자로 정확히 재는 건 사실상 불가능하다(1~2도만 틀려도
화면에서 10~20 px가 밀린다). 그래서 각도를 재는 대신, **화면을 보면서 키로
라이다 점을 물체 위에 맞춘 뒤 한 키로 저장**하는 방식으로 잡는다.

사용법
------
1. 콘(또는 벽/박스)을 차 앞 1~3 m에 두고 이 도구를 실행한다.
2. 화면에 카메라 영상 + 라이다 점이 겹쳐 보인다.
3. i/k(위아래), j/l(좌우) 키로 라이다 점이 실제 물체 위에 오도록 맞춘다.
4. **s 키를 누르면 현재 값이 params.yaml에 바로 저장된다.**
5. 창을 닫고 평소대로 full_bringup을 실행하면 그 값이 적용된 상태로 뜬다.

키 정리
-------
  i / k    pitch  - 점을 위 / 아래로
  j / l    yaw    - 점을 왼쪽 / 오른쪽으로
  u / o    roll   - 점을 반시계 / 시계로
  n / m    height - 점을 위 / 아래로 (가까운 점일수록 크게 움직임)
  [ / ]    조정 스텝 절반 / 두 배
  0        보정값 전부 0으로
  z        마지막 조정 취소 (undo)
  space    화면 정지/해제 (정지 상태에서도 보정은 그대로 반영됨)
  g        조준용 격자/중심선 표시 토글
  s        ★ 저장 - params.yaml에 현재 값 기록
  r        저장된 값 다시 불러오기
  q / ESC  종료
"""

import math
import os
from typing import Optional

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from tf2_ros import Buffer, TransformListener

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, LaserScan

from lidar_camera_fusion_pkg.calibration_utils import (
    apply_calibration,
    build_fallback_extrinsic,
    project_scan,
    quaternion_to_matrix,
)
from lidar_camera_fusion_pkg.params_writer import find_params_file, update_node_params


WINDOW_NAME = 'LiDAR-Camera Calibration'

# (키, 항목, 부호) - 부호는 "화면에서 점이 움직이는 방향" 기준
ADJUST_KEYS = {
    ord('i'): ('pitch', +1.0),
    ord('k'): ('pitch', -1.0),
    ord('j'): ('yaw', -1.0),
    ord('l'): ('yaw', +1.0),
    ord('u'): ('roll', -1.0),
    ord('o'): ('roll', +1.0),
    ord('n'): ('height', -1.0),
    ord('m'): ('height', +1.0),
}

FIELDS = ('pitch', 'yaw', 'roll', 'height')


class CalibrationNode(Node):

    def __init__(self):
        super().__init__('lidar_camera_calibration_node')

        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('scan_topic', '/scan')

        # 내부 파라미터 (params.yaml의 image_fusion_node 값과 같아야 함)
        self.declare_parameter('fx', 478.681350)
        self.declare_parameter('fy', 480.893055)
        self.declare_parameter('cx', 314.853795)
        self.declare_parameter('cy', 259.235816)
        self.declare_parameter('distortion', [0.019110, -0.134271, 0.007227, -0.003467, 0.0])

        # extrinsic 원본
        self.declare_parameter('use_urdf_extrinsic', True)
        self.declare_parameter('lidar_frame_id', 'laser')
        self.declare_parameter('camera_frame_id', 'camera_optical_frame_tilted')
        self.declare_parameter('cam_x_offset', 0.730)
        self.declare_parameter('cam_height', 0.550)
        self.declare_parameter('cam_pitch_deg', 14.0)
        self.declare_parameter('front_angle_deg', -180.0)

        # 스캔 필터
        self.declare_parameter('max_range', 10.0)
        self.declare_parameter('min_range', 0.1)
        self.declare_parameter('min_cam_z', 0.1)
        self.declare_parameter('enable_fov_filter', True)
        self.declare_parameter('cam_fov_deg', 55.0)
        self.declare_parameter('max_age_scan', 0.5)

        # 시작 보정값
        self.declare_parameter('calib_pitch_deg', 0.0)
        self.declare_parameter('calib_yaw_deg', 0.0)
        self.declare_parameter('calib_roll_deg', 0.0)
        self.declare_parameter('calib_height_m', 0.0)
        self.declare_parameter('calib_step_deg', 0.25)
        self.declare_parameter('calib_step_m', 0.01)

        # 저장 대상. 비우면 소스 트리의 params.yaml을 자동으로 찾는다
        self.declare_parameter('params_file', '')

        self.declare_parameter('window_width', 1100)
        self.declare_parameter('window_height', 760)

        gp = self.get_parameter
        self.image_topic = gp('image_topic').value
        self.scan_topic = gp('scan_topic').value

        self.K = np.array([[float(gp('fx').value), 0.0, float(gp('cx').value)],
                           [0.0, float(gp('fy').value), float(gp('cy').value)],
                           [0.0, 0.0, 1.0]], dtype=np.float64)
        self.distortion = np.asarray(gp('distortion').value, dtype=np.float64)

        self.use_urdf_extrinsic = bool(gp('use_urdf_extrinsic').value)
        self.lidar_frame_id = str(gp('lidar_frame_id').value)
        self.camera_frame_id = str(gp('camera_frame_id').value)
        self.front_angle_deg = float(gp('front_angle_deg').value)

        self.max_range = float(gp('max_range').value)
        self.min_range = float(gp('min_range').value)
        self.min_cam_z = float(gp('min_cam_z').value)
        self.enable_fov_filter = bool(gp('enable_fov_filter').value)
        self.fov_deg = float(gp('cam_fov_deg').value)
        self.fov_center_rad = math.radians(self.front_angle_deg)
        self.max_age_scan = float(gp('max_age_scan').value)

        self.calib = {
            'pitch': float(gp('calib_pitch_deg').value),
            'yaw': float(gp('calib_yaw_deg').value),
            'roll': float(gp('calib_roll_deg').value),
            'height': float(gp('calib_height_m').value),
        }
        self.step_deg = float(gp('calib_step_deg').value)
        self.step_m = float(gp('calib_step_m').value)
        self.history = []

        self.base_extrinsic = build_fallback_extrinsic(
            float(gp('cam_x_offset').value),
            float(gp('cam_height').value),
            self.front_angle_deg,
            float(gp('cam_pitch_deg').value),
        )
        self.tf_ok = False
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.params_file = find_params_file(str(gp('params_file').value))

        self.bridge = CvBridge()
        self.last_scan: Optional[LaserScan] = None
        self.last_scan_time = None
        self.frozen_frame: Optional[np.ndarray] = None
        self.frozen_scan: Optional[LaserScan] = None
        self.show_grid = True
        self.status_msg = ''
        self.status_ticks = 0

        self.create_subscription(Image, self.image_topic, self.image_cb, qos_profile_sensor_data)
        self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, qos_profile_sensor_data)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, int(gp('window_width').value),
                         int(gp('window_height').value))

        self.get_logger().info(
            '\n'
            '==================== 라이다-카메라 정렬 도구 ====================\n'
            ' 콘을 차 앞 1~3 m에 두고, 라이다 점이 콘 위에 오도록 키로 맞추세요.\n'
            '   i/k  점을 위/아래   (pitch)\n'
            '   j/l  점을 왼/오른쪽 (yaw)\n'
            '   u/o  회전           (roll)\n'
            '   n/m  높이           (height)\n'
            '   [/]  스텝 절반/두배    0  전부 초기화    z  실행취소\n'
            '   space 화면정지      g  격자표시\n'
            '   s    ★저장(params.yaml)   r  저장값 불러오기   q/ESC 종료\n'
            f' 저장 대상: {self.params_file or "(찾지 못함 - params_file 파라미터로 지정)"}\n'
            '================================================================')

    # ------------------------------------------------------------------ 콜백

    def scan_cb(self, msg: LaserScan):
        self.last_scan = msg
        self.last_scan_time = self.get_clock().now()

    def image_cb(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'imgmsg_to_cv2 실패: {e}')
            return

        self._update_extrinsic_from_tf()

        if self.frozen_frame is not None:
            img = self.frozen_frame
            scan = self.frozen_scan
            scan_ok = scan is not None
        else:
            scan = self.last_scan
            scan_ok = False
            if scan is not None and self.last_scan_time is not None:
                age = (self.get_clock().now() - self.last_scan_time).nanoseconds / 1e9
                scan_ok = age <= self.max_age_scan

        frame = img.copy()
        h, w = frame.shape[:2]

        if self.show_grid:
            self._draw_grid(frame, w, h)

        n_points = 0
        if scan_ok and scan is not None:
            u, v, r = self._project(scan, w, h)
            n_points = len(u)
            self._draw_points(frame, u, v, r)

        self._draw_panel(frame, w, h, scan_ok, n_points)

        cv2.imshow(WINDOW_NAME, frame)
        self._handle_key(cv2.waitKey(1) & 0xFF, img, scan)

    # ------------------------------------------------------------------ 기하

    def _update_extrinsic_from_tf(self):
        if not self.use_urdf_extrinsic:
            return
        try:
            tf = self.tf_buffer.lookup_transform(
                self.camera_frame_id, self.lidar_frame_id, rclpy.time.Time())
        except Exception:
            return

        t = tf.transform.translation
        q = tf.transform.rotation
        ext = np.eye(4, dtype=np.float64)
        ext[:3, :3] = quaternion_to_matrix([q.x, q.y, q.z, q.w])
        ext[:3, 3] = [t.x, t.y, t.z]
        self.base_extrinsic = ext
        if not self.tf_ok:
            self.get_logger().info(
                f"TF extrinsic 사용: {self.lidar_frame_id} -> {self.camera_frame_id}")
        self.tf_ok = True

    def _current_extrinsic(self) -> np.ndarray:
        return apply_calibration(self.base_extrinsic, self.calib['pitch'],
                                 self.calib['yaw'], self.calib['roll'],
                                 self.calib['height'])

    def _project(self, scan: LaserScan, w: int, h: int):
        ranges = np.asarray(scan.ranges, dtype=np.float64)
        n = ranges.shape[0]
        angles = scan.angle_min + np.arange(n, dtype=np.float64) * scan.angle_increment

        valid = (np.isfinite(ranges) & (ranges >= self.min_range)
                 & (ranges <= min(float(scan.range_max), self.max_range)))

        if self.enable_fov_filter:
            half = math.radians(self.fov_deg / 2.0)
            diff = np.abs(np.arctan2(np.sin(angles - self.fov_center_rad),
                                     np.cos(angles - self.fov_center_rad)))
            valid = valid & (diff <= half)

        if not np.any(valid):
            empty = np.array([], dtype=np.int32)
            return empty, empty, np.array([], dtype=np.float64)

        return project_scan(ranges[valid], angles[valid], self._current_extrinsic(),
                            self.K, w, h, self.min_cam_z, self.distortion)

    # ------------------------------------------------------------------ 그리기

    @staticmethod
    def _draw_grid(frame, w, h):
        """조준용 중심선. 콘을 화면 정중앙에 두고 맞추면 yaw를 잡기 쉽다."""
        cv2.line(frame, (w // 2, 0), (w // 2, h), (70, 70, 70), 1)
        cv2.line(frame, (0, h // 2), (w, h // 2), (70, 70, 70), 1)
        for frac in (0.25, 0.75):
            cv2.line(frame, (int(w * frac), 0), (int(w * frac), h), (45, 45, 45), 1)
            cv2.line(frame, (0, int(h * frac)), (w, int(h * frac)), (45, 45, 45), 1)

    def _draw_points(self, frame, u, v, r):
        if len(u) == 0:
            return
        # 가장 가까운 점을 크게 표시 - 보통 그게 맞추려는 콘이다
        nearest = int(np.argmin(r)) if len(r) else -1
        for idx in range(len(u)):
            t = float(np.clip(r[idx] / max(self.max_range, 1e-3), 0.0, 1.0))
            color = (int(255 * t), 80, int(255 * (1.0 - t)))  # 가까움=빨강, 멂=파랑
            cv2.circle(frame, (int(u[idx]), int(v[idx])), 2, color, -1)

        if nearest >= 0:
            pt = (int(u[nearest]), int(v[nearest]))
            cv2.circle(frame, pt, 9, (255, 255, 255), 2)
            cv2.putText(frame, f'{r[nearest]:.2f}m', (pt[0] + 12, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_panel(self, frame, w, h, scan_ok, n_points):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 118), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        def put(text, x, y, color=(255, 255, 255), scale=0.55, thick=1):
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                        color, thick, cv2.LINE_AA)

        put('LiDAR-Camera Calibration', 10, 24, (0, 255, 255), 0.7, 2)

        put(f"pitch {self.calib['pitch']:+6.2f} deg", 10, 50, (120, 255, 120))
        put(f"yaw   {self.calib['yaw']:+6.2f} deg", 190, 50, (120, 255, 120))
        put(f"roll  {self.calib['roll']:+6.2f} deg", 370, 50, (120, 255, 120))
        put(f"height {self.calib['height']:+6.3f} m", 545, 50, (120, 255, 120))

        src = 'TF(URDF)' if (self.use_urdf_extrinsic and self.tf_ok) else 'FALLBACK'
        scan_txt = f'{n_points} pts' if scan_ok else 'NO SCAN'
        scan_col = (200, 200, 200) if scan_ok else (0, 165, 255)
        put(f'step {self.step_deg:.2f}deg / {self.step_m:.3f}m', 10, 72)
        put(f'ext: {src}', 190, 72)
        put(f'scan: {scan_txt}', 370, 72, scan_col)
        if self.frozen_frame is not None:
            put('[FROZEN]', 545, 72, (0, 165, 255))

        put('i/k pitch  j/l yaw  u/o roll  n/m height  [/] step  0 reset  z undo',
            10, 94, (180, 180, 180), 0.5)
        put('space freeze   g grid   s SAVE to params.yaml   r reload   q quit',
            10, 112, (180, 180, 180), 0.5)

        if self.status_ticks > 0:
            self.status_ticks -= 1
            color = (0, 0, 255) if self.status_msg.startswith('!') else (0, 255, 0)
            cv2.rectangle(frame, (0, h - 34), (w, h), (0, 0, 0), -1)
            put(self.status_msg, 10, h - 11, color, 0.6, 2)

    def _set_status(self, msg: str, ticks: int = 90):
        self.status_msg = msg
        self.status_ticks = ticks
        self.get_logger().info(msg)

    # ------------------------------------------------------------------ 입력

    def _handle_key(self, key, img, scan):
        if key == 255:
            return

        if key in ADJUST_KEYS:
            field, sign = ADJUST_KEYS[key]
            self.history.append(dict(self.calib))
            if len(self.history) > 200:
                self.history.pop(0)
            step = self.step_m if field == 'height' else self.step_deg
            self.calib[field] += sign * step

        elif key == ord('['):
            self.step_deg = max(0.05, self.step_deg / 2.0)
            self.step_m = max(0.002, self.step_m / 2.0)
            self._set_status(f'스텝: {self.step_deg:.2f}deg / {self.step_m:.3f}m', 45)

        elif key == ord(']'):
            self.step_deg = min(5.0, self.step_deg * 2.0)
            self.step_m = min(0.2, self.step_m * 2.0)
            self._set_status(f'스텝: {self.step_deg:.2f}deg / {self.step_m:.3f}m', 45)

        elif key == ord('0'):
            self.history.append(dict(self.calib))
            self.calib = {k: 0.0 for k in FIELDS}
            self._set_status('보정값 초기화')

        elif key == ord('z'):
            if self.history:
                self.calib = self.history.pop()
            else:
                self._set_status('되돌릴 기록이 없습니다', 45)

        elif key == ord(' '):
            if self.frozen_frame is None:
                self.frozen_frame = img.copy()
                self.frozen_scan = scan
                self._set_status('화면 정지 - 천천히 맞추세요 (space로 해제)')
            else:
                self.frozen_frame = None
                self.frozen_scan = None
                self._set_status('화면 정지 해제')

        elif key == ord('g'):
            self.show_grid = not self.show_grid

        elif key == ord('s'):
            self._save()

        elif key == ord('r'):
            self._reload()

        elif key in (ord('q'), 27):
            self._set_status('종료')
            raise KeyboardInterrupt

    # ------------------------------------------------------------------ 저장

    def _save(self):
        if not self.params_file:
            self._set_status('! params.yaml을 찾지 못함 - params_file 파라미터로 경로 지정', 150)
            return

        values = {
            'calib_pitch_deg': round(self.calib['pitch'], 3),
            'calib_yaw_deg': round(self.calib['yaw'], 3),
            'calib_roll_deg': round(self.calib['roll'], 3),
            'calib_height_m': round(self.calib['height'], 4),
        }
        try:
            updated, added = update_node_params(
                self.params_file, 'image_fusion_node', values)
        except Exception as e:
            self._set_status(f'! 저장 실패: {e}', 150)
            return

        self._set_status(f'저장됨 -> {os.path.basename(self.params_file)} '
                         f'(갱신 {len(updated)}, 추가 {len(added)})', 150)
        self.get_logger().info(
            f'저장 위치: {self.params_file}\n'
            f'  calib_pitch_deg: {values["calib_pitch_deg"]}\n'
            f'  calib_yaw_deg:   {values["calib_yaw_deg"]}\n'
            f'  calib_roll_deg:  {values["calib_roll_deg"]}\n'
            f'  calib_height_m:  {values["calib_height_m"]}\n'
            '  (백업: 같은 경로에 .bak)\n'
            '  적용하려면: colcon build --packages-select sensor_fusion_bringup')

    def _reload(self):
        if not self.params_file:
            self._set_status('! params.yaml 경로를 모릅니다', 90)
            return
        try:
            import yaml
            with open(self.params_file, encoding='utf-8') as f:
                data = yaml.safe_load(f)
            p = data['image_fusion_node']['ros__parameters']
        except Exception as e:
            self._set_status(f'! 불러오기 실패: {e}', 150)
            return

        self.history.append(dict(self.calib))
        self.calib = {
            'pitch': float(p.get('calib_pitch_deg', 0.0)),
            'yaw': float(p.get('calib_yaw_deg', 0.0)),
            'roll': float(p.get('calib_roll_deg', 0.0)),
            'height': float(p.get('calib_height_m', 0.0)),
        }
        self._set_status('params.yaml에서 값을 다시 불러왔습니다')


def main(args=None):
    rclpy.init(args=args)
    node = CalibrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
