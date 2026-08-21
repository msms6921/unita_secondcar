import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

import yaml


# 하드코딩 폴백 (sensor_fusion_bringup이 아직 빌드 안 된 경우에만 사용)
_FALLBACK = {
    'rplidar_node': {'serial_port': '/dev/ttyUSB0', 'serial_baudrate': 460800,
                     'frame_id': 'laser'},
    'image_publisher_node': {'cam_num': 0},
    'yolov8_node': {'device': 'cuda:0'},
    'image_fusion_node': {
        'fx': 478.681350, 'cx': 314.853795, 'front_angle_deg': -180.0,
        'display_mode': 'boxes', 'distance_tolerance': 0.6, 'draw_all_points': True,
        'use_urdf_extrinsic': True, 'lidar_frame_id': 'laser',
        'camera_frame_id': 'camera_optical_frame_tilted',
    },
}


def _load_params():
    """모든 기본값의 단일 출처인 sensor_fusion_bringup/config/params.yaml을 읽는다.

    예전에는 이 파일이 자체 하드코딩 기본값을 들고 있어서 params.yaml과 갈렸다.
    특히 use_urdf_extrinsic이 여기선 false, params.yaml에선 true라서, 이 launch를
    단독 실행하면 URDF TF 대신 폴백 extrinsic으로 투영돼 정렬이 조용히 어긋났다.
    """
    try:
        config_file = os.path.join(
            get_package_share_directory('sensor_fusion_bringup'), 'config', 'params.yaml')
        with open(config_file) as f:
            raw = yaml.safe_load(f)
        params = {k: v['ros__parameters'] for k, v in raw.items() if 'ros__parameters' in v}
        return config_file, params
    except Exception:
        return None, _FALLBACK


def generate_launch_description():
    config_file, params = _load_params()

    def p(node, key):
        return str(params.get(node, {}).get(key, _FALLBACK.get(node, {}).get(key, '')))

    def pbool(node, key):
        return p(node, key).lower()

    # params.yaml 전체를 각 노드에 넘겨서, 새 파라미터를 추가할 때 launch를 안 고쳐도 되게 한다.
    # (아래 명시적 launch 인자는 이 위에 덮어쓰기로 적용됨)
    base = [config_file] if config_file else []

    serial_port = LaunchConfiguration('serial_port', default=p('rplidar_node', 'serial_port'))
    serial_baudrate = LaunchConfiguration('serial_baudrate', default=p('rplidar_node', 'serial_baudrate'))
    frame_id = LaunchConfiguration('frame_id', default=p('rplidar_node', 'frame_id'))
    device = LaunchConfiguration('device', default=p('yolov8_node', 'device'))
    fx = LaunchConfiguration('fx', default=p('image_fusion_node', 'fx'))
    cx = LaunchConfiguration('cx', default=p('image_fusion_node', 'cx'))
    lidar_front_offset_deg = LaunchConfiguration(
        'lidar_front_offset_deg', default=p('image_fusion_node', 'front_angle_deg'))
    cam_num = LaunchConfiguration('cam_num', default=p('image_publisher_node', 'cam_num'))
    display_mode = LaunchConfiguration('display_mode', default=p('image_fusion_node', 'display_mode'))
    distance_tolerance = LaunchConfiguration(
        'distance_tolerance', default=p('image_fusion_node', 'distance_tolerance'))
    draw_all_points = LaunchConfiguration(
        'draw_all_points', default=pbool('image_fusion_node', 'draw_all_points'))
    use_urdf_extrinsic = LaunchConfiguration(
        'use_urdf_extrinsic', default=pbool('image_fusion_node', 'use_urdf_extrinsic'))
    lidar_frame_id = LaunchConfiguration(
        'lidar_frame_id', default=p('image_fusion_node', 'lidar_frame_id'))
    camera_frame_id = LaunchConfiguration(
        'camera_frame_id', default=p('image_fusion_node', 'camera_frame_id'))

    # 콘/드럼 검출과 차선 세그멘테이션을 한 YOLO 노드에서 순차 실행해 결과를 합친다.
    # best_cone.pt: cone/drum bbox, lane_seg.pt: lane_1/lane_2 mask.
    camera_models_dir = os.path.join(
        get_package_share_directory('camera_perception_pkg'), 'models'
    )
    model_path = ','.join([
        os.path.join(camera_models_dir, 'best_cone.pt'),
        os.path.join(camera_models_dir, 'car_back.pt'),
        os.path.join(camera_models_dir, 'lane_seg.pt'),
    ])

    # bird_eye_node는 lane.pt를 한 번 더 추론하는 '보기용' 노드라, 주행(drive.launch.py)
    # 때는 꺼서 Jetson GPU를 아낀다.
    enable_bird_eye = LaunchConfiguration('enable_bird_eye', default='true')
    # bird_eye_node 자체 미리보기 창(원본+버드아이뷰 나란히). Fusion Visualizer의 4/5번 화면과
    # 별개로 창을 하나 더 띄운다.
    bird_eye_preview = ParameterValue(
        LaunchConfiguration('bird_eye_preview', default='false'), value_type=bool)

    return LaunchDescription([
        DeclareLaunchArgument(
            'serial_port', default_value=serial_port,
            description='RPLIDAR USB serial port'),
        DeclareLaunchArgument(
            'serial_baudrate', default_value=serial_baudrate,
            description='RPLIDAR serial baudrate (C1: 460800)'),
        DeclareLaunchArgument(
            'frame_id', default_value=frame_id,
            description='RPLIDAR scan frame_id'),
        DeclareLaunchArgument(
            'device', default_value=device,
            description='YOLO inference device (cpu / cuda:0)'),
        DeclareLaunchArgument(
            'fx', default_value=fx,
            description='카메라 초점거리 fx(px), 캘리브레이션 결과값'),
        DeclareLaunchArgument(
            'cx', default_value=cx,
            description='카메라 광학 중심 cx(px), 캘리브레이션 결과값'),
        DeclareLaunchArgument(
            'lidar_front_offset_deg', default_value=lidar_front_offset_deg,
            description='LiDAR 0-angle vs camera forward direction offset in degrees '
                        '(정반대 마운트면 180)'),
        DeclareLaunchArgument(
            'cam_num', default_value=cam_num,
            description='카메라 장치 번호 (ls /dev/video* 로 확인)'),
        DeclareLaunchArgument(
            'display_mode', default_value=display_mode,
            description='Fusion Visualizer 시작 화면 모드: raw / lidar / boxes / bev '
                        '(창에서 숫자키 1~4로 실행 중 전환 가능)'),
        DeclareLaunchArgument(
            'draw_all_points', default_value=draw_all_points,
            description='카메라 위에 라이다 포인트를 전부 그릴지 여부'),
        DeclareLaunchArgument(
            'distance_tolerance', default_value=distance_tolerance,
            description='bbox 거리 계산 시 허용할 거리 오차 범위 [m]'),
        DeclareLaunchArgument(
            'use_urdf_extrinsic', default_value=use_urdf_extrinsic,
            description='URDF/TF 기반 외인척 변환을 사용할지 여부'),
        DeclareLaunchArgument(
            'lidar_frame_id', default_value=lidar_frame_id,
            description='LiDAR frame id'),
        DeclareLaunchArgument(
            'camera_frame_id', default_value=camera_frame_id,
            description='Camera frame id'),
        DeclareLaunchArgument(
            'bird_eye_preview', default_value='false',
            description='bird_eye_node 자체 미리보기 창(원본+버드아이뷰)을 띄울지 여부'),
        DeclareLaunchArgument(
            'enable_bird_eye', default_value=enable_bird_eye,
            description="bird_eye_node(보기용 차선 BEV) 실행 여부. 주행 중에는 "
                        'yolov8_node가 이미 lane_seg를 돌리므로 false를 권장'),

        # LiDAR
        Node(
            package='rplidar_ros',
            executable='rplidar_node',
            name='rplidar_node',
            output='screen',
            parameters=base + [{
                'channel_type': 'serial',
                'serial_port': serial_port,
                'serial_baudrate': serial_baudrate,
                'frame_id': frame_id,
                'inverted': False,
                'angle_compensate': True,
                'scan_mode': 'Standard',
            }],
        ),

        # Camera
        # (해상도 / 오토포커스 / 자동노출 등 나머지 설정은 params.yaml에서 들어옴)
        Node(
            package='camera_perception_pkg',
            executable='image_publisher_node',
            name='image_publisher_node',
            output='screen',
            parameters=base + [{
                'cam_num': cam_num,
                'logger': False,
            }],
        ),

        # YOLO detection
        # (threshold / 오탐 억제 필터는 params.yaml에서 들어옴)
        Node(
            package='camera_perception_pkg',
            executable='yolov8_node',
            name='yolov8_node',
            output='screen',
            parameters=base + [{
                'model': model_path,
                'device': device,
            }],
        ),

        # Bird's-eye lane view: bird_eye/image(버드아이뷰)는 Fusion Visualizer 4번,
        # bird_eye/roi(ROI 표시된 원본)는 5번 화면으로 각각 전달됨.
        # 자체 미리보기 창은 중복이라 꺼둠 (show_preview).
        Node(
            package='camera_perception_pkg',
            executable='bird_eye_node',
            name='bird_eye_node',
            output='screen',
            condition=IfCondition(enable_bird_eye),
            parameters=base + [{
                # 퓨전(image_fusion_node)이 쓰는 카메라와 같은 토픽을 본다
                'input_topic': 'image_raw',
                'device': device,
                'show_preview': bird_eye_preview,
            }],
        ),

        # LiDAR-Camera fusion (distance overlay)
        Node(
            package='lidar_camera_fusion_pkg',
            executable='image_fusion_node',
            name='image_fusion_node',
            output='screen',
            # fy/cy와 calib_* 미세보정 값은 params.yaml에서 들어옴
            # (예전에는 fx/cx만 여기서 넘기고 fy/cy는 소스 하드코딩 기본값이 조용히 쓰였다)
            parameters=base + [{
                'fx': fx,
                'cx': cx,
                'front_angle_deg': lidar_front_offset_deg,
                'display_mode': display_mode,
                'draw_all_points': draw_all_points,
                'distance_tolerance': distance_tolerance,
                'use_urdf_extrinsic': use_urdf_extrinsic,
                'lidar_frame_id': lidar_frame_id,
                'camera_frame_id': camera_frame_id,
                'display': True,
                'publish_annotated': False,
            }],
        ),
    ])
