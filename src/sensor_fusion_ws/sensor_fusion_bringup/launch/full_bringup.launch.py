import os

import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # 모든 기본값은 config/params.yaml 하나로 관리. 값을 바꾸려면 그 파일을 수정할 것
    # (launch 인자로 임시 override도 여전히 가능).
    config_file = os.path.join(
        get_package_share_directory('sensor_fusion_bringup'),
        'config', 'params.yaml'
    )
    with open(config_file) as f:
        params = yaml.safe_load(f)

    def section(name):
        return params.get(name, {}).get('ros__parameters', {})

    rplidar_p = section('rplidar_node')
    cam_p = section('image_publisher_node')
    yolo_p = section('yolov8_node')
    fusion_p = section('image_fusion_node')
    l_shape_p = section('l_shape_node')

    # params.yaml에서 항목이 빠져도 launch 전체가 죽지 않도록 폴백을 둔다
    # (실제로 l_shape_node의 launch_rviz가 빠지면서 KeyError로 launch가 통째로 죽은 적 있음)
    _FALLBACK = {
        'serial_port': '/dev/ttyUSB0', 'serial_baudrate': 460800, 'frame_id': 'laser',
        'device': 'cuda:0', 'cam_num': 0,
        'fx': 478.681350, 'cx': 314.853795, 'front_angle_deg': -180.0,
        'display_mode': 'boxes', 'distance_tolerance': 0.6, 'draw_all_points': True,
        'use_urdf_extrinsic': True, 'lidar_frame_id': 'laser',
        'camera_frame_id': 'camera_optical_frame_tilted', 'cam_pitch_deg': 14.0,
        'fov_deg': 150.0, 'launch_rviz': False,
    }

    def val(sec, key):
        return str(sec.get(key, _FALLBACK[key]))

    def vbool(sec, key):
        return val(sec, key).lower()

    # --- fusion_bringup.launch.py 인자 (그대로 전달) ---
    serial_port = LaunchConfiguration('serial_port', default=val(rplidar_p, 'serial_port'))
    serial_baudrate = LaunchConfiguration('serial_baudrate', default=val(rplidar_p, 'serial_baudrate'))
    frame_id = LaunchConfiguration('frame_id', default=val(rplidar_p, 'frame_id'))
    device = LaunchConfiguration('device', default=val(yolo_p, 'device'))
    fx = LaunchConfiguration('fx', default=val(fusion_p, 'fx'))
    cx = LaunchConfiguration('cx', default=val(fusion_p, 'cx'))
    lidar_front_offset_deg = LaunchConfiguration(
        'lidar_front_offset_deg', default=val(fusion_p, 'front_angle_deg'))
    cam_num = LaunchConfiguration('cam_num', default=val(cam_p, 'cam_num'))
    display_mode = LaunchConfiguration(
        'display_mode', default=val(fusion_p, 'display_mode'))
    distance_tolerance = LaunchConfiguration(
        'distance_tolerance', default=val(fusion_p, 'distance_tolerance'))
    draw_all_points = LaunchConfiguration(
        'draw_all_points', default=vbool(fusion_p, 'draw_all_points'))
    use_urdf_extrinsic = LaunchConfiguration(
        'use_urdf_extrinsic', default=vbool(fusion_p, 'use_urdf_extrinsic'))
    lidar_frame_id = LaunchConfiguration('lidar_frame_id', default=val(fusion_p, 'lidar_frame_id'))
    camera_frame_id = LaunchConfiguration('camera_frame_id', default=val(fusion_p, 'camera_frame_id'))

    # 카메라 다운틸트(도). description.launch.py로 전달돼 camera_link_tilted TF를 만든다.
    # image_fusion_node의 폴백 extrinsic이 쓰는 cam_pitch_deg와 같은 값이어야 한다.
    camera_pitch_deg = LaunchConfiguration(
        'camera_pitch_deg', default=val(fusion_p, 'cam_pitch_deg'))

    # bird_eye_node(보기용 차선 BEV)를 띄울지 여부. drive.launch.py는 false로 넘겨서
    # lane_seg 추론이 yolov8_node 한 곳에서만 돌게 한다.
    enable_bird_eye = LaunchConfiguration('enable_bird_eye', default='true')
    bird_eye_preview = LaunchConfiguration('bird_eye_preview', default='false')

    # --- l_shape_node 전용 인자 ---
    fov_deg = LaunchConfiguration('fov_deg', default=val(l_shape_p, 'fov_deg'))
    launch_rviz = LaunchConfiguration('launch_rviz', default=vbool(l_shape_p, 'launch_rviz'))

    fusion_launch_path = os.path.join(
        get_package_share_directory('lidar_camera_fusion_pkg'), 'launch', 'fusion_bringup.launch.py'
    )
    description_launch_path = os.path.join(
        get_package_share_directory('unita_minicar_description'), 'launch', 'description.launch.py'
    )

    return LaunchDescription([
        DeclareLaunchArgument('serial_port', default_value=serial_port,
                               description='RPLIDAR USB serial port'),
        DeclareLaunchArgument('serial_baudrate', default_value=serial_baudrate,
                               description='RPLIDAR serial baudrate (C1: 460800)'),
        DeclareLaunchArgument('frame_id', default_value=frame_id,
                               description='RPLIDAR scan frame_id'),
        DeclareLaunchArgument('device', default_value=device,
                               description='YOLO inference device (cpu / cuda:0)'),
        DeclareLaunchArgument('fx', default_value=fx,
                               description='카메라 초점거리 fx(px), 캘리브레이션 결과값'),
        DeclareLaunchArgument('cx', default_value=cx,
                               description='카메라 광학 중심 cx(px), 캘리브레이션 결과값'),
        DeclareLaunchArgument('lidar_front_offset_deg', default_value=lidar_front_offset_deg,
                               description='LiDAR 0-angle vs camera forward direction offset in degrees '
                                           '(정반대 마운트면 180). l_shape_node의 front_angle_deg에도 '
                                           '동일하게 적용됨'),
        DeclareLaunchArgument('cam_num', default_value=cam_num,
                               description='카메라 장치 번호 (ls /dev/video* 로 확인)'),
        DeclareLaunchArgument('display_mode', default_value=display_mode,
                               description='Fusion Visualizer 시작 화면 모드: raw / lidar / boxes / bev '
                                           '(창에서 숫자키 1~4로 실행 중 전환 가능)'),
        DeclareLaunchArgument('draw_all_points', default_value=draw_all_points,
                               description='카메라 위에 라이다 포인트를 전부 그릴지 여부'),
        DeclareLaunchArgument('distance_tolerance', default_value=distance_tolerance,
                               description='bbox 거리 계산 시 허용할 거리 오차 범위 [m]'),
        DeclareLaunchArgument('use_urdf_extrinsic', default_value=use_urdf_extrinsic,
                               description='URDF/TF 기반 외인척 변환을 사용할지 여부'),
        DeclareLaunchArgument('lidar_frame_id', default_value=lidar_frame_id,
                               description='LiDAR frame id'),
        DeclareLaunchArgument('camera_frame_id', default_value=camera_frame_id,
                               description='Camera frame id'),
        DeclareLaunchArgument('camera_pitch_deg', default_value=camera_pitch_deg,
                               description='카메라 다운틸트(도). 이 값이 실제와 다르면 라이다 '
                                           '투영이 위/아래로 밀린다 (1도 ≈ 10 px). 잔차는 '
                                           'image_fusion_node의 calib_pitch_deg로 보정'),
        DeclareLaunchArgument('bird_eye_preview', default_value=bird_eye_preview,
                               description='bird_eye_node 자체 미리보기 창을 띄울지 여부'),
        DeclareLaunchArgument('enable_bird_eye', default_value=enable_bird_eye,
                               description='bird_eye_node(보기용 차선 BEV) 실행 여부. '
                                           '주행 중에는 yolov8_node가 이미 lane_seg를 '
                                           '돌리므로 false 권장'),
        DeclareLaunchArgument('fov_deg', default_value=fov_deg,
                               description='l_shape_node: front_angle_deg를 중심으로 남길 전체 시야각(도)'),
        DeclareLaunchArgument('launch_rviz', default_value=launch_rviz,
                               description='l_shape_node가 시작될 때 rviz2를 자동으로 띄울지 여부'),

        # 로봇 URDF 기반 TF (base_link -> laser / camera_link 등 고정 변환).
        # image_fusion_node를 use_urdf_extrinsic:=true로 띄우면 이 TF를 읽어서 외인척 변환에 사용함
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(description_launch_path),
            launch_arguments={'camera_pitch_deg': camera_pitch_deg}.items(),
        ),

        # 라이다 드라이버 + 카메라 + YOLO + 퓨전 (rplidar_node는 여기서 한 번만 실행됨)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(fusion_launch_path),
            launch_arguments={
                'serial_port': serial_port,
                'serial_baudrate': serial_baudrate,
                'frame_id': frame_id,
                'device': device,
                'fx': fx,
                'cx': cx,
                'lidar_front_offset_deg': lidar_front_offset_deg,
                'cam_num': cam_num,
                'display_mode': display_mode,
                'distance_tolerance': distance_tolerance,
                'draw_all_points': draw_all_points,
                'use_urdf_extrinsic': use_urdf_extrinsic,
                'lidar_frame_id': lidar_frame_id,
                'camera_frame_id': camera_frame_id,
                'enable_bird_eye': enable_bird_eye,
                'bird_eye_preview': bird_eye_preview,
            }.items(),
        ),

        # L-shape fitting (같은 /scan을 구독만 하므로 별도 rplidar_node 없음)
        # config/params.yaml 전체를 로드하고, launch 인자로만 일부 값을 override
        Node(
            package='lidar_cluster_pkg',
            executable='l_shape_node',
            name='l_shape_node',
            output='screen',
            parameters=[config_file, {
                'frame_id': frame_id,
                'front_angle_deg': lidar_front_offset_deg,
                'fov_deg': fov_deg,
                'launch_rviz': launch_rviz,
            }],
        ),
    ])
