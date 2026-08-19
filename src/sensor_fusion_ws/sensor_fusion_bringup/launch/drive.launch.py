"""자율주행 전체 실행: 센서/인지(full_bringup) + 판단(decision_making) + 제어(serial).

체인:
  카메라 -> yolov8_node(cone/car_back/lane_seg) -> /detections
    -> lane_info_extractor_node  -> /yolov8_lane_info   (차선 중심점)
    -> image_fusion_node         -> /lidar_obstacle_info (가장 가까운 장애물 거리/픽셀x)
    -> path_planner_node(lattice) -> /path_planning_result
    -> motion_planner_node(pure pursuit + PD) -> /topic_control_signal (MotionCommand)
    -> serial_sender_node -> 아두이노("C,<조향 -1.0~1.0>,<후륜 PWM>")

모든 파라미터는 config/params.yaml에서 관리한다.
라이다와 Arduino 포트는 각각 lidar_serial_port / arduino_serial_port로 분리한다.

예)
  ros2 launch sensor_fusion_bringup drive.launch.py cam_num:=1
  ros2 launch sensor_fusion_bringup drive.launch.py lidar_serial_port:=/dev/ttyUSB0 arduino_serial_port:=/dev/ttyACM0
  ros2 launch sensor_fusion_bringup drive.launch.py enable_serial:=false   # 바퀴 안 굴리고 확인만
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config_file = os.path.join(
        get_package_share_directory('sensor_fusion_bringup'),
        'config', 'params.yaml'
    )
    full_bringup_path = os.path.join(
        get_package_share_directory('sensor_fusion_bringup'),
        'launch', 'full_bringup.launch.py'
    )

    # 아두이노로 실제 명령을 내보낼지 여부. false면 /topic_control_signal까지만 돌아서
    # (ros2 topic echo /topic_control_signal) 바퀴를 안 굴리고 조향/속도를 확인할 수 있다.
    enable_serial = LaunchConfiguration('enable_serial', default='true')
    lidar_serial_port = LaunchConfiguration('lidar_serial_port')
    arduino_serial_port = LaunchConfiguration('arduino_serial_port')
    driving_direction = LaunchConfiguration('driving_direction')
    # 판단 노드들을 센서/YOLO가 뜬 뒤에 올리기 위한 지연[s]
    decision_start_delay = LaunchConfiguration('decision_start_delay', default='5.0')

    # 버드아이뷰(bird_eye_node) - 퓨전과 같은 카메라(/image_raw)를 보고 차선을 그린다.
    # 결과는 Fusion Visualizer의 4(bev)/5(bev_roi) 화면으로 들어간다.
    # 주행 경로는 yolov8_node의 마스크를 직접 사용한다. 이 노드는 보기용으로 같은 lane.pt를
    # 한 번 더 추론하므로 CPU 주행에서는 기본으로 꺼서 추론 중복을 없앤다.
    enable_bird_eye = LaunchConfiguration('enable_bird_eye', default='false')
    # 자체 미리보기 창('Lane bird-eye | original')은 Fusion Visualizer와 중복이라 기본 꺼둠
    bird_eye_preview = LaunchConfiguration('bird_eye_preview', default='false')

    decision_nodes = [
        # 차선 마스크 -> 주행 목표점
        Node(
            package='camera_perception_pkg',
            executable='lane_info_extractor_node',
            name='lane_info_extractor_node',
            output='screen',
            parameters=[config_file, {'driving_direction': driving_direction}],
        ),

        # 목표점 + 장애물 -> lattice 경로
        Node(
            package='decision_making_pkg',
            executable='path_planner_node',
            name='path_planner_node',
            output='screen',
            parameters=[config_file],
        ),

        # 경로 -> 조향/속도 명령
        Node(
            package='decision_making_pkg',
            executable='motion_planner_node',
            name='motion_planner_node',
            output='screen',
            parameters=[config_file],
        ),

        # 명령 -> 아두이노 시리얼
        Node(
            package='serial_communication_pkg',
            executable='serial_sender_node',
            name='serial_sender_node',
            output='screen',
            condition=IfCondition(enable_serial),
            parameters=[config_file, {'port': arduino_serial_port}],
        ),
    ]

    return LaunchDescription([
        DeclareLaunchArgument(
            'lidar_serial_port', default_value='/dev/ttyUSB0',
            description='RPLIDAR 시리얼 포트'),
        DeclareLaunchArgument(
            'arduino_serial_port', default_value='/dev/ttyACM0',
            description='Arduino 제어 시리얼 포트'),
        DeclareLaunchArgument(
            'driving_direction', default_value='clockwise',
            description='주행 방향: clockwise 또는 counterclockwise'),
        DeclareLaunchArgument(
            'enable_serial', default_value=enable_serial,
            description='아두이노로 실제 제어 명령을 보낼지 여부. false면 명령 토픽까지만 확인'),
        DeclareLaunchArgument(
            'decision_start_delay', default_value=decision_start_delay,
            description='센서/YOLO가 뜬 뒤 판단 노드를 올리기까지의 지연[s]'),
        DeclareLaunchArgument(
            'enable_bird_eye', default_value=enable_bird_eye,
            description='버드아이뷰(bird_eye_node) 실행 여부. GPU가 모자라면 false'),
        DeclareLaunchArgument(
            'bird_eye_preview', default_value=bird_eye_preview,
            description='bird_eye_node 자체 미리보기 창(원본+버드아이뷰) 표시 여부'),

        # 센서 + 인지 + 퓨전 + 버드아이뷰
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(full_bringup_path),
            launch_arguments={
                'serial_port': lidar_serial_port,
                'enable_bird_eye': enable_bird_eye,
                'bird_eye_preview': bird_eye_preview,
            }.items(),
        ),

        TimerAction(period=decision_start_delay, actions=decision_nodes),
    ])
