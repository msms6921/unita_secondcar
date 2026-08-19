#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""라이다->카메라 투영에 쓰는 기하 계산 모음.

image_fusion_node(주행용 시각화)와 calibration_node(정렬 맞추는 도구)가
똑같은 수식을 써야 하므로 여기로 모아둔다. 한쪽만 고쳐서 두 화면이 다르게
보이는 일을 막기 위함.

좌표계 약속
-----------
- laser 프레임: RPLIDAR 스캔 좌표계 (x 앞, y 왼쪽, z 위)
- 카메라 광학 좌표계: X 오른쪽, Y 아래, Z 정면(깊이)
- extrinsic 4x4는 항상 "laser -> 카메라 광학" 변환
"""

import math

import numpy as np


def quaternion_to_matrix(q) -> np.ndarray:
    """[x, y, z, w] 쿼터니언을 3x3 회전행렬로."""
    x, y, z, w = q
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def build_fallback_extrinsic(dist: float, height: float, front_angle_deg: float,
                             cam_pitch_deg: float = 0.0) -> np.ndarray:
    """TF(URDF)를 못 쓸 때의 폴백 extrinsic.

    dist/height는 "카메라에서 본 라이다 원점"의 앞(+Z)/아래(+Y) 거리다.
    cam_pitch_deg는 카메라 다운틸트로, 예전엔 이 경로에 아예 반영이 안 돼 있어서
    TF가 없을 때 투영이 크게 어긋났다.
    """
    t_vec = np.array([0.0, height, dist], dtype=np.float64).reshape(3, 1)

    r_axis_swap = np.array([
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ], dtype=np.float64)

    # front_angle_deg 하나로 yaw 회전을 만든다 (FOV 필터와 같은 기준을 공유)
    theta = math.radians(front_angle_deg)
    r_yaw = np.array([
        [math.cos(theta), -math.sin(theta), 0.0],
        [math.sin(theta), math.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # 다운틸트: 광학 좌표계 X축(오른쪽) 기준 회전
    p = math.radians(cam_pitch_deg)
    r_pitch = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(p), -math.sin(p)],
        [0.0, math.sin(p), math.cos(p)],
    ], dtype=np.float64)

    ext = np.eye(4, dtype=np.float64)
    ext[:3, :3] = r_pitch @ r_axis_swap @ r_yaw
    # 평행이동도 기울어진 광학 좌표계 기준으로 회전시켜야 한다
    ext[:3, 3] = (r_pitch @ t_vec).flatten()
    return ext


def apply_calibration(ext: np.ndarray, pitch_deg: float, yaw_deg: float,
                      roll_deg: float, height_m: float) -> np.ndarray:
    """URDF/TF에서 얻은 extrinsic 위에 실측 미세보정을 덧씌운다.

    보정은 카메라 광학 좌표계 기준이고, 부호는 "화면에서 점이 움직이는 방향"으로 잡았다:

    - pitch  + : 점이 화면 **위**로   (카메라가 실제로 더 아래를 보고 있었다는 뜻)
    - yaw    + : 점이 화면 **오른쪽**으로
    - roll   + : 이미지 롤
    - height + : 카메라가 실제로 더 **높이** 달려있다 -> 가까운 점이 아래로

    각도 1도가 화면에서 약 10 px(fy≈567)이라, 손으로 잰 다운틸트가 1~2도만 틀려도
    눈에 띄게 어긋난다. 그 잔차를 여기서 걷어낸다.
    """
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    r = math.radians(roll_deg)

    # X축(오른쪽) 회전: +면 점이 화면 위로
    rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, math.cos(p), -math.sin(p)],
                   [0.0, math.sin(p), math.cos(p)]], dtype=np.float64)
    # Y축(아래) 회전: +면 점이 화면 오른쪽으로
    ry = np.array([[math.cos(y), 0.0, math.sin(y)],
                   [0.0, 1.0, 0.0],
                   [-math.sin(y), 0.0, math.cos(y)]], dtype=np.float64)
    # Z축(정면) 회전: 이미지 롤
    rz = np.array([[math.cos(r), -math.sin(r), 0.0],
                   [math.sin(r), math.cos(r), 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float64)

    corr = np.eye(4, dtype=np.float64)
    corr[:3, :3] = rz @ ry @ rx
    corr[1, 3] = height_m  # 광학 Y = 아래 방향

    return corr @ ext


def project_camera_points(points_cam: np.ndarray, k_mat: np.ndarray,
                          distortion: np.ndarray | None = None):
    """카메라 좌표 3xN 점을 Brown-Conrady 왜곡을 적용해 픽셀로 변환."""
    x = points_cam[0] / points_cam[2]
    y = points_cam[1] / points_cam[2]

    coeffs = np.zeros(5, dtype=np.float64)
    if distortion is not None:
        supplied = np.asarray(distortion, dtype=np.float64).reshape(-1)
        coeffs[:min(5, supplied.size)] = supplied[:5]
    k1, k2, p1, p2, k3 = coeffs

    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    x_distorted = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_distorted = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

    u = k_mat[0, 0] * x_distorted + k_mat[0, 2]
    v = k_mat[1, 1] * y_distorted + k_mat[1, 2]
    return u, v


def project_scan(ranges: np.ndarray, angles: np.ndarray, extrinsic: np.ndarray,
                 k_mat: np.ndarray, img_w: int, img_h: int, min_cam_z: float = 0.1,
                 distortion: np.ndarray | None = None):
    """유효한 (range, angle) 배열을 이미지 픽셀로 투영.

    반환: (u, v, range) - 모두 이미지 안에 들어온 점만.
    """
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    z = np.zeros_like(x)
    ones = np.ones_like(x)

    pts_cam = extrinsic @ np.vstack([x, y, z, ones])

    front = pts_cam[2, :] > min_cam_z
    pts_cam = pts_cam[:, front]
    ranges_v = ranges[front]

    empty = (np.array([], dtype=np.int32),
             np.array([], dtype=np.int32),
             np.array([], dtype=np.float64))
    if pts_cam.shape[1] == 0:
        return empty

    u, v = project_camera_points(pts_cam[:3, :], k_mat, distortion)
    # astype은 0쪽으로 버림이라 좌우가 비대칭으로 밀린다. 반올림으로.
    u_i = np.rint(u).astype(np.int32)
    v_i = np.rint(v).astype(np.int32)

    inside = (u_i >= 0) & (u_i < img_w) & (v_i >= 0) & (v_i < img_h)
    return u_i[inside], v_i[inside], ranges_v[inside]
