# unita_minicar_master

UNITA 미니카 자율주행 전체 코드. **ROS2(Humble) 워크스페이스 + 아두이노 펌웨어**를 함께 담는다.

카메라(YOLOv8)로 차선과 장애물을 인식하고, 라이다(RPLIDAR C1)를 퓨전해 거리를 재고,
lattice 경로계획 + pure pursuit으로 조향을 만들어 아두이노로 내보낸다.

```
카메라 ─ yolov8_node ─┬─ lane_info_extractor_node ─┐
                      │   (차선 마스크 → 목표점)     │
                      └─ image_fusion_node ────────┤   (장애물 거리/픽셀x)
라이다 ───────────────────────────────────────────┤
                                                   ▼
                          path_planner_node (lattice 경로)
                                                   ▼
                    motion_planner_node (pure pursuit + PD)
                                                   ▼
                       serial_sender_node ─ "C,<조향>,<후륜PWM>" ─ 아두이노
                                                   ▼
                        firmware/autonomous_mega (조향 폐루프 + 구동)
```

| 디렉토리 | 내용 |
|---|---|
| `src/sensor_fusion_ws/` | ROS2 패키지 전체 (인식·퓨전·판단·제어) |
| `firmware/autonomous_mega/` | 아두이노 Mega 펌웨어 (PlatformIO). 조향 폐루프·구동·초음파 |
| `firmware/tools/` | 조향 캘리브레이션 측정 스크립트 |

**하드웨어 캘리브레이션 값과 그 근거는 [9번](#9-조향-캘리브레이션-실측-기준)에 정리돼 있다.**
조향이 이상하게 동작하면 그것부터 볼 것.

---

패키지별 상세 설명은 각 패키지의 README 참고:
[`sensor_fusion_bringup`](src/sensor_fusion_ws/sensor_fusion_bringup/README.md) ·
[`interfaces_pkg`](src/sensor_fusion_ws/interfaces_pkg/README.md) ·
[`camera_perception_pkg`](src/sensor_fusion_ws/camera_perception_pkg/README.md) ·
[`lidar_camera_fusion_pkg`](src/sensor_fusion_ws/lidar_camera_fusion_pkg/README.md) ·
[`lidar_cluster_pkg`](src/sensor_fusion_ws/lidar_cluster_pkg/README.md) ·
[`rplidar_ros`](src/sensor_fusion_ws/rplidar_ros/README.md)

자율주행(판단·제어) 패키지는 [minicar_sim](https://github.com/gwakminji/minicar_sim)에서 가져왔다:
`decision_making_pkg`(lattice 경로계획 + pure pursuit 조향), `serial_communication_pkg`(아두이노 시리얼 송신),
`camera_perception_pkg/lane_info_extractor_node`(차선 마스크 → 주행 목표점). 실행은 [3-1](#3-1-자율주행-실행-drivelaunchpy) 참고.

## 1. 카메라·라이다 연결 및 포트 확인

### 1-1. 연결 전

USB 포트에 아직 아무것도 꽂지 않은 상태에서 기준선을 확인해둔다.

```bash
ls /dev/video* 2>/dev/null    # 현재 잡혀있는 비디오 장치 (다른 카메라가 있으면 미리 보임)
ls /dev/ttyUSB* 2>/dev/null   # 현재 잡혀있는 USB 시리얼 장치
```

### 1-2. 웹캠 연결 후 장치 번호 확인

```bash
ls /dev/video*
```

카메라 1대에 `/dev/video0`, `/dev/video1`처럼 번호가 2개 잡히는 경우가 흔한데(영상용 노드 + 메타데이터 노드),
보통 더 작은 번호가 실제 영상 장치다. 어떤 번호가 맞는지 확실히 하려면 실제로 프레임을 읽어본다:

```bash
python3 -c "
import cv2
for i in range(4):
    cap = cv2.VideoCapture(i)
    ok, _ = cap.read()
    print(i, 'OK' if ok else 'fail')
    cap.release()
"
```

`OK`가 뜨는 가장 작은 번호를 카메라 번호로 쓰면 된다.

### 1-3. 라이다(RPLIDAR C1) 연결 후 포트 확인

```bash
ls /dev/ttyUSB*
```

연결 전엔 없다가 연결 직후 새로 나타난 번호(`/dev/ttyUSB0` 등)가 라이다다. 여러 개의 USB-시리얼 장치가
동시에 꽂혀있어서 헷갈리면:

```bash
dmesg | tail -20   # 방금 연결한 장치의 커널 로그 (ttyUSB 번호가 어떤 장치인지 보임)
```

권한도 같이 확인 (`dialout` 그룹이 없으면 포트를 열 수 없음):

```bash
groups   # 목록에 dialout 이 있어야 함
# 없으면: sudo usermod -aG dialout $USER   (실행 후 재로그인 필요)
```

### 1-4. 확인한 번호를 실행 시 반영

기본값과 다르면, **소스를 고칠 필요 없이** launch 인자로 그때그때 넘기거나
[`config/params.yaml`](src/sensor_fusion_ws/sensor_fusion_bringup/config/params.yaml)의 값을 고쳐서 영구적으로 반영할 수 있다
(단, `config/params.yaml`을 고친 뒤에는 `colcon build --packages-select sensor_fusion_bringup`을 다시 해야 `install/`에 반영됨 —
symlink 설치가 아니라 파일을 복사하는 방식이라서다).

```bash
ros2 launch sensor_fusion_bringup full_bringup.launch.py \
  cam_num:=1 \
  serial_port:=/dev/ttyUSB0
```

ROS2 Humble이 설치되어 있어야 한다 (`/opt/ros/humble`).

## 2. 빌드

```bash
cd ~/sensor_fusion_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

매번 새 터미널을 열 때마다 `/opt/ros/humble/setup.bash`와 `install/setup.bash` 두 개를 순서대로 source 해야 한다.

### 2-1. CUDA 확인 (GPU 사용 시)

`yolov8_node`/`bird_eye_node`의 추론 디바이스 기본값은 `cuda:0`다 (Jetson Orin Nano 등 GPU 대상).
실행 전에 CUDA를 실제로 쓸 수 있는지 먼저 확인한다.

```bash
python3 -c "import torch; print('cuda available:', torch.cuda.is_available())"
```

`True`가 아니면:

- Jetson은 일반 `pip install torch`(x86/CUDA 데스크톱용)로는 GPU를 못 잡는다. JetPack 버전에 맞는
  NVIDIA 전용 torch/torchvision wheel(또는 그걸 반영한 `ultralytics`)을 설치해야 한다.
- GPU가 아예 없는 환경(개발 PC 등)에서는 `device:=cpu`를 launch 인자로 넘겨서 CPU로 돌리면 된다:
  ```bash
  ros2 launch sensor_fusion_bringup full_bringup.launch.py device:=cpu
  ```

## 3. 전체 파이프라인 실행

라이다 + 카메라 + YOLO + 퓨전(거리 측정) + L-shape fitting까지 한 번에 띄운다 (권장 진입점).
값들은 전부 [`config/params.yaml`](src/sensor_fusion_ws/sensor_fusion_bringup/config/params.yaml)
하나로 관리되고, 필요하면 launch 인자로 그때그때 override할 수 있다.

```bash
ros2 launch sensor_fusion_bringup full_bringup.launch.py
```

라이다·카메라·YOLO·퓨전만 필요하면 (L-shape fitting/rviz 없이) 아래처럼 한 단계 아래 launch 파일을
직접 실행해도 된다. 단 이 경우 기본값은 `config/params.yaml`이 아니라 그 파일 자체에 하드코딩된 값이다.

```bash
ros2 launch lidar_camera_fusion_pkg fusion_bringup.launch.py
```

### 화면 (Fusion Visualizer)

- 정상 동작하면 `Fusion Visualizer`라는 이름의 OpenCV 창 1개가 뜬다. 창을 한 번 클릭해서
  포커스를 준 뒤 키보드로 화면 모드를 전환할 수 있다.

| 키 | 동작 |
|---|---|
| `1` | 주화면: 기본 카메라 이미지 (오버레이 없음) |
| `2` | 주화면: 라이다 포인트만 표시 |
| `3` | 주화면: YOLO 바운딩박스 + 거리 표시 (기본 시작 모드) |
| `4` | 주화면: 버드아이뷰 (`bird_eye_node`의 차선 검출 결과) |
| `5` | 주화면: 버드아이뷰 ROI 원본 (버드아이뷰 변환에 쓰는 사다리꼴을 원본 위에 표시) |
| `q`/`w`/`e`/`r`/`t` | 보조화면을 각각 raw/lidar/boxes/bev/bev_roi로 선택 |
| `v` | 분할보기 토글 — 켜면 주화면+보조화면을 가로로 나란히 표시 |

- YOLO는 cone/drum 탐지 모델(`best_cone.pt`), 차량 후면 탐지 모델(`car_back.pt`),
  차선 세그멘테이션 모델(`lane.pt`, 클래스 `lane_1`/`lane_2`) 세 개를 동시에 돌려서
  결과를 하나로 합쳐 발행한다 (`camera_perception_pkg/models/`, `yolov8_node`의 `model` 파라미터에
  콤마로 구분해서 넘기면 여러 모델을 함께 로딩함).
- 차선 마스크는 박스가 화면을 거의 다 덮기 때문에, Fusion Visualizer의 박스 표시와
  `/lidar_obstacle_info`(장애물 거리) 계산에서는 제외된다
  (`image_fusion_node`의 `box_class_exclude` / `obstacle_class_exclude` 파라미터).

### launch 인자 (필요할 때만 덮어쓰기)

`full_bringup.launch.py`의 인자는 `fusion_bringup.launch.py`의 모든 인자에 `fov_deg`, `launch_rviz`가
추가된 것과 같다 (자세한 표는 [`sensor_fusion_bringup` README](src/sensor_fusion_ws/sensor_fusion_bringup/README.md) 참고).
자주 바꾸는 것 위주로 추리면:

| 인자 | 기본값 | 설명 |
|---|---|---|
| `cam_num` | `config/params.yaml` 참고 | 카메라 장치 번호, 1번에서 확인한 값 |
| `serial_port` | `/dev/ttyUSB0` | 라이다가 다른 포트로 잡히면 변경 |
| `serial_baudrate` | `460800` | RPLIDAR C1 기준값 |
| `frame_id` | `laser` | 라이다 스캔 좌표계 이름 |
| `device` | `cuda:0` | YOLO/버드아이뷰 추론 디바이스 (`cuda:0` / `cpu`, GPU 없으면 `cpu`로 override) |
| `fx`, `cx` | `565.529459`, `337.983746` | 카메라 초점거리/광학중심(px), 캘리브레이션 결과값 |
| `lidar_front_offset_deg` | `-180.0` | 라이다 0도 방향과 카메라 정면 방향의 차이(도) |
| `display_mode` | `boxes` | Fusion Visualizer 시작 화면 모드 (`raw`/`lidar`/`boxes`/`bev`/`bev_roi`, 실행 중엔 키로 전환) |

예:
```bash
ros2 launch sensor_fusion_bringup full_bringup.launch.py serial_port:=/dev/ttyUSB1 device:=cuda:0
```

## 3-1. 자율주행 실행 (drive.launch.py)

위의 인지(카메라·라이다·YOLO·퓨전)에 **판단·제어**를 붙여서 실제로 차를 굴리는 진입점.
판단/제어 코드는 [minicar_sim](https://github.com/gwakminji/minicar_sim)에서 가져와 이 워크스페이스에
맞춘 것이고, 마지막 시리얼 송신 노드는 실차용 `serial_communication_pkg`다.

```
카메라 → yolov8_node (cone / car_back / lane_seg) → /detections
   ├→ lane_info_extractor_node → /yolov8_lane_info      (BEV로 편 차선의 중심점들)
   └→ image_fusion_node        → /lidar_obstacle_info   (가장 가까운 장애물 거리[m] + 화면상 x[px])
        → path_planner_node (lattice)      → /path_planning_result
        → motion_planner_node (pure pursuit + PD) → /topic_control_signal  (MotionCommand)
        → serial_sender_node → 아두이노 시리얼 "C,<조향 -1.0~1.0>,<후륜 PWM>"
```

```bash
# 바퀴를 굴리지 않고 명령만 확인 (처음엔 반드시 이걸로 먼저)
ros2 launch sensor_fusion_bringup drive.launch.py enable_serial:=false

# 실제 주행 (차를 들어올리거나 넓은 곳에서, 전원 차단 준비하고)
ros2 launch sensor_fusion_bringup drive.launch.py cam_num:=2 serial_port:=/dev/ttyUSB1
```

`full_bringup.launch.py`의 인자(`cam_num`, `serial_port`, `device` …)는 여기서도 그대로 먹는다.
추가 인자는 아래 두 개다.

| 인자 | 기본값 | 설명 |
|---|---|---|
| `enable_serial` | `true` | 아두이노로 실제 명령을 보낼지 여부. `false`면 `/topic_control_signal`까지만 돌아서 바퀴가 안 움직인다 |
| `decision_start_delay` | `5.0` | 센서·YOLO가 뜬 뒤 판단 노드를 올리기까지의 지연[s] |

주행 파라미터(차선 폭, 회피 거리, lattice 가중치, 조향/속도 이득, 시리얼 포트 등)는 전부
[`config/params.yaml`](src/sensor_fusion_ws/sensor_fusion_bringup/config/params.yaml)의
`lane_info_extractor_node` / `path_planner_node` / `motion_planner_node` / `serial_sender_node`
항목에 모여 있다. 고친 뒤에는 `colcon build --packages-select sensor_fusion_bringup`을 다시 해야
`install/`에 반영된다.

### 확인 순서

```bash
ros2 topic hz /yolov8_lane_info      # 차선이 보이면 카메라 프레임레이트 근처로 나와야 함
ros2 topic hz /path_planning_result  # 유효 차선점이 3개 이상일 때만 발행됨
ros2 topic echo /topic_control_signal  # steering(-9~9), left_speed/right_speed
```

- `/yolov8_lane_info`가 안 나오면 → 차선이 YOLO에 안 잡히는 것. `lane_info_extractor_node`가 띄우는
  `Lane Info (ROI)` 창에서 흰 선이 보이는지 확인하고, 안 보이면 `params.yaml`의 `src_points`(BEV 사다리꼴)를
  이 카메라 장착 각도에 맞게 다시 잡아야 한다.
- `steering` 부호가 반대로 먹으면 `serial_sender_node`의 `steer_invert: true`.
- 처음 굴릴 때는 `motion_planner_node`의 `base_speed`(기본 120)를 더 낮춰서 시작할 것.
  `max_steer_cmd`는 `motion_planner_node`와 `serial_sender_node` 양쪽이 같은 값이어야 한다.

## 4. 정상 동작 확인 (다른 터미널에서)

```bash
source /opt/ros/humble/setup.bash
source ~/sensor_fusion_ws/install/setup.bash

ros2 topic list
# /image_raw, /scan, /detections 가 보여야 함

ros2 topic hz /image_raw     # ~30Hz 근처로 나오면 카메라 정상
ros2 topic hz /scan          # 라이다가 정상 발행 중이면 수 Hz~10Hz대로 나옴
ros2 topic hz /detections    # YOLO가 프레임마다 검출 결과를 발행 중인지 확인 (물체가 없어도 빈 배열은 발행됨)

ros2 topic echo /detections --once   # 검출 결과 1개 내용 확인 (class_name, score, bbox 등)
```

각 토픽이 하나라도 안 뜨면 3번 launch를 실행한 터미널의 로그에서 해당 노드가 에러를 내고 있는지 확인.

## 5. 문제가 생겼을 때 개별 노드로 나눠서 확인

한 번에 다 띄우지 않고 노드를 하나씩 켜보면 어느 단계에서 막히는지 좁힐 수 있다.

```bash
# 라이다만
ros2 run rplidar_ros rplidar_node --ros-args -p serial_port:=/dev/ttyUSB0 -p serial_baudrate:=460800 -p frame_id:=laser

# 카메라만
ros2 run camera_perception_pkg image_publisher_node

# YOLO만 (카메라가 먼저 떠 있어야 함, 모델 경로는 설치 경로 기준)
# model 파라미터에 콤마로 여러 개를 넘기면 각 모델을 모두 돌려서 결과를 합쳐 발행한다
MODELS_DIR=$(ros2 pkg prefix camera_perception_pkg)/share/camera_perception_pkg/models
ros2 run camera_perception_pkg yolov8_node --ros-args \
  -p model:="$MODELS_DIR/best_cone.pt,$MODELS_DIR/car_back.pt" \
  -p device:=cuda:0

# 퓨전만 (카메라·라이다·YOLO가 먼저 떠 있어야 함) — 실제 launch 파일이 쓰는 노드
ros2 run lidar_camera_fusion_pkg image_fusion_node --ros-args \
  -p fx:=565.529459 -p cx:=337.983746 -p front_angle_deg:=-180.0
```

> `lidar_camera_fusion_pkg`에는 `sensor_fusion_node`(동축 마운트 가정, 픽셀 각도 기반)라는 더 단순한
> 대안 노드도 있지만, 현재 launch 파일들은 `image_fusion_node`(3D 투영 기반)를 사용한다.
> 자세한 차이는 [`lidar_camera_fusion_pkg` README](src/sensor_fusion_ws/lidar_camera_fusion_pkg/README.md) 참고.

### 자주 나는 에러

- `can't open camera by index` — 다른 프로세스가 이미 해당 `/dev/video*`를 잡고 있거나(`fuser /dev/video0`로 확인 후
  `kill -9 <pid>`), 1-2번에서 확인한 것과 다른 `cam_num`을 쓰고 있는 경우.
- `serial_port` 관련 permission denied — `groups`에 `dialout` 없으면
  `sudo usermod -aG dialout $USER` 후 재로그인.
- YOLO가 뜨긴 하는데 거리 텍스트가 항상 `N/A` — `/scan` 토픽이 안 들어오고 있거나
  (라이다 미연결/포트 오류), `lidar_front_offset_deg`가 실제 마운트 방향과 안 맞는 경우.

## 6. 라이다 점이 실제와 틀어져 보일 때 — 정렬 캘리브레이션 도구

라이다 점이 물체보다 살짝 위/아래/옆으로 밀려 보이는 가장 흔한 원인은 **카메라 다운틸트 각도**다.
이 각도는 자로 정확히 재는 게 사실상 불가능한데, **1도만 틀려도 화면에서 약 10 px가 밀린다**
(`fy≈567` 기준). 그래서 각도를 재는 대신 **화면을 보면서 키로 맞추고 한 키로 저장**하는
전용 도구를 쓴다.

```bash
ros2 launch lidar_camera_fusion_pkg calibration.launch.py
```

YOLO/버드아이뷰 없이 라이다 + 카메라 + 정렬 화면만 띄운다 (GPU를 안 쓰므로 가볍다).

### 맞추는 순서

1. **콘(또는 상자·벽)을 차 앞 1~3 m 정면에 둔다.** 화면 중앙선(`g`키로 켜고 끔)에 맞춰 놓으면
   좌우(yaw)를 판단하기 쉽다.
2. 창을 한 번 클릭해 포커스를 준다. 라이다 점이 색으로 표시되고, **가장 가까운 점은 흰 원과
   거리 값**으로 강조된다 — 보통 그게 맞추려는 콘이다.
3. `space`로 화면을 정지시키면 천천히 맞출 수 있다 (정지 중에도 보정은 실시간 반영됨).
4. 아래 키로 **라이다 점이 실제 콘 위에 오도록** 움직인다.
5. **`s` 키를 누르면 현재 값이 `config/params.yaml`에 바로 저장된다.** (원본은 `.bak`으로 백업)
6. 창을 닫고 아래를 실행해 주행 파이프라인에 반영한다:
   ```bash
   colcon build --packages-select sensor_fusion_bringup
   ```

### 키

| 키 | 동작 |
|---|---|
| `i` / `k` | pitch — 점을 **위 / 아래**로 |
| `j` / `l` | yaw — 점을 **왼쪽 / 오른쪽**으로 |
| `u` / `o` | roll — 점을 반시계 / 시계로 |
| `n` / `m` | height — 점을 위 / 아래로 (가까운 점일수록 크게 움직임) |
| `[` / `]` | 조정 스텝 절반 / 두 배 (기본 0.25도) |
| `0` | 보정값 전부 0으로 |
| `z` | 마지막 조정 취소 (undo) |
| `space` | 화면 정지 / 해제 |
| `g` | 조준용 중심선·격자 표시 토글 |
| **`s`** | **저장 — `params.yaml`에 기록** |
| `r` | 저장된 값 다시 불러오기 |
| `q` / `ESC` | 종료 |

> 팁: 회전(pitch/yaw)은 **거리와 상관없이 항상 같은 픽셀만큼** 점을 움직이고, height는
> 가까운 점일수록 크게 움직인다. 그러니 가까운 콘과 먼 콘이 **둘 다 같은 방향으로** 밀려 있으면
> pitch/yaw를, **가까운 것만** 많이 밀려 있으면 height를 건드리면 된다.

주행 중에 쓰는 `Fusion Visualizer`(3번 화면) 창에서도 같은 키(`i/k/j/l/u/o/n/m`)로 즉석 보정이
가능하고, `p`키를 누르면 `params.yaml`에 붙여넣을 값이 로그로 출력된다. 다만 저장(`s`)은
위 캘리브레이션 도구에만 있다.

### 그래도 안 맞으면

- **차가 움직일 때만** 어긋난다면 정렬이 아니라 **시간 지연** 문제다. 화면 하단 HUD의
  `scan age`를 보고, 150 ms를 넘으면 라이다/카메라 발행이 밀리고 있는 것이다.
- 화면 전체가 **좌우로 늘어난 느낌**이면 카메라가 요청 해상도(640x480)를 안 주고 있을 수 있다.
  `image_publisher_node` 로그에 해상도 경고가 뜨는지 확인할 것 (캘리브레이션을 뽑은 해상도와
  실제 해상도가 다르면 `fx/fy/cx/cy`가 통째로 안 맞는다).
- TF가 안 뜨면 `image_fusion_node`가 폴백 extrinsic으로 투영하며 경고를 남긴다. 그 경고가
  보이면 `description.launch.py`(robot_state_publisher + static TF)가 떠 있는지 확인할 것.

## 7. YOLO가 콘 대신 멀리 있는 엉뚱한 걸 잡을 때

**손바닥으로 렌즈를 가렸다 떼면 콘을 잘 잡는데 가만히 두면 이상한 걸 잡는다** — 이건 거의 항상
웹캠의 **오토포커스(AF) / 자동노출(AE)** 때문이다. 가만히 두면 AF가 초점을 찾아 헤매다 먼 배경에
락이 걸리고, 그러면 가까운 콘은 흐려져서 놓치고 선명해진 먼 배경에서 오탐이 난다. 손으로 가렸다
떼는 순간 AF/AE가 재수렴하면서 가까운 콘에 초점이 잡히는 것.

`config/params.yaml`의 `image_publisher_node`에서 고정한다 (기본값은 이미 AF/AE 끔):

```yaml
image_publisher_node:
  ros__parameters:
    autofocus: false
    focus: -1          # 0~255. 콘이 선명해지는 값으로 고정 (아래 방법으로 찾음)
    auto_exposure: false
    exposure: -1       # 화면이 너무 어두우면 올릴 것
```

**초점값 찾는 법** — 콘을 주행 시 보이는 거리에 두고, 카메라를 연결한 채 터미널에서:

```bash
# 이 카메라가 지원하는 항목과 범위 확인 (이름은 드라이버마다 조금씩 다름)
v4l2-ctl -d /dev/video0 --list-ctrls

# 오토포커스 끄고 초점을 바꿔가며 가장 선명한 값 찾기
v4l2-ctl -d /dev/video0 -c focus_automatic_continuous=0
for f in 0 20 40 60 80 100 140 180 255; do
  v4l2-ctl -d /dev/video0 -c focus_absolute=$f
  echo "focus=$f"; sleep 1        # 화면 보면서 가장 선명한 값 기록
done
```

가장 선명했던 값을 `focus:`에 적고 `colcon build --packages-select sensor_fusion_bringup`.
노드 시작 시 로그에 실제 반영된 AF/AE 값이 찍히니, 안 먹었으면 위 `v4l2-ctl` 명령으로 직접 잡으면 된다.

### 그래도 먼 오탐이 남으면

`yolov8_node`의 크기·위치 필터로 거른다 (`config/params.yaml`):

```yaml
yolov8_node:
  ros__parameters:
    threshold: 0.5
    min_box_area_px: 576        # 24x24 px보다 작은 검출은 버림 (멀어서 의미 없음)
    min_box_height_px: 20
    min_box_bottom_ratio: 0.30  # 박스 아래변이 화면 위 30% 안에서 끝나면 버림(=지평선 위)
    class_thresholds: ""        # 예: "cone:0.60,drum:0.55" 클래스별로 다르게
```

너무 많이 걸러지면 로그에 "필터로 버린 검출 N개"가 주기적으로 찍히니 그걸 보고 값을 낮추면 된다.

## 8. 캘리브레이션이 바뀌면

- 카메라를 바꾸거나 재캘리브레이션하면 `fx`, `fy`, `cx`, `cy`를 새 값으로 갱신할 것
  (4개 모두 `config/params.yaml`에서 관리된다). 고친 뒤
  `colcon build --packages-select sensor_fusion_bringup`.
- 카메라 장착 각도를 바꾸면 위 6번의 캘리브레이션 도구로 다시 맞출 것. 다운틸트 자체를 아예
  새로 넣고 싶으면 `cam_pitch_deg`(또는 launch 인자 `camera_pitch_deg`)를 고친 뒤 잔차만 도구로 잡는다.
- 라이다/카메라 장착 방향을 바꾸면 `lidar_front_offset_deg` 재확인 (자세한 내용은
  [`lidar_camera_fusion_pkg` README](src/sensor_fusion_ws/lidar_camera_fusion_pkg/README.md) 참고)
- 새 YOLO 모델(`.pt`)을 받으면 `camera_perception_pkg/models/`에 넣고, `yolov8_node`의 `model`
  파라미터(콤마로 여러 개 지정 가능)를 그 파일명으로 맞춘 뒤 `camera_perception_pkg`를 재빌드할 것.

---

## 9. 조향 캘리브레이션 (실측 기준)

조향은 아두이노가 포텐셔미터로 위치를 되먹임하는 **폐루프**로 돈다. ROS는 각도를 직접 주지 않고
`-1.0 ~ +1.0` 정규화 값만 보내고, 펌웨어가 그걸 pot 목표값으로 바꾼다. 그래서 아래 값들이
틀어지면 인식이 아무리 정확해도 차는 엉뚱하게 간다.

### 9-1. 시리얼 프로토콜

`firmware/autonomous_mega/src/comm.ino`가 받는 형식은 **필드 2개**다.

```
C,<조향 -1.0~1.0>,<후륜 PWM -255~255>\n
```

`handleLine()`은 쉼표가 3개 이상이면 **줄 전체를 버린다**(`idx3 != -1` 검사). 뒤에 필드를
덧붙이면 명령이 통째로 무시되고 워치독(`DRIVE_WD_MS`, 300ms)이 모터를 0으로 내린다.
호스트 쪽 생성기는 `serial_communication_pkg/lib/protocol_convert_func_lib.py`.

조향 필드는 **PWM이 아니라 정규화 값**이다. 펌웨어가 `normalizedToSteeringPosition()`으로
pot 목표값으로 변환한다.

### 9-2. 펌웨어 상수 (`firmware/autonomous_mega/src/firmware_main.ino`)

| 상수 | 값 | 근거 |
|---|---|---|
| `steeringPotCenter` | 510 | 앞바퀴를 자로 재서 직진에 맞춘 뒤 A0 실측 (121샘플, 509~511) |
| `steeringPotMin` | 440 | 실측 기계적 한계 434 안쪽으로 여유 |
| `steeringPotMax` | 616 | 실측 기계적 한계 622 안쪽으로 여유 |
| `STEERING_DEADBAND` | 6 | 조향 분해능을 직접 결정한다. 아래 주의 참고 |
| `STEERING_MIN_PWM` | 55 | 목표 근처 접근용. 낮으면 지면에서 안 움직이고, 높으면 지나친다 |
| `STEERING_MAX_PWM` | 160 | 지면 정지 조향(dry steering)을 이길 만큼 |
| `STEERING_SLOWDOWN_RANGE` | 50 | 감속구간. 편측 가동폭보다 넓으면 PWM이 최대치에 도달하지 못한다 |

**중앙값은 반드시 Min~Max 사이여야 한다.** 벗어나면 `normalizedToSteeringPosition()`의
`constrain(target, Min, Max)`에 걸려 조향 목표가 한쪽 끝에 고정된다. 직진 명령(0.0)조차
풀락이 되고, 차선을 정확히 잡아도 계속 한쪽으로만 간다.

**데드밴드는 조향 분해능이다.** ROS는 `steering`을 `int32`로 `±max_steer_cmd`(=9) 정수로
보내므로 한 단계가 편측 가동폭의 1/9다. 데드밴드가 그보다 크면 명령 대부분이 삼켜져
조향이 걸리지 않는다.

가동범위는 좌 70 / 우 106으로 **비대칭**이다(링키지 자체가 치우쳐 있다. 중앙 510인데
기계적 중점은 528). 같은 크기의 명령이어도 우회전이 더 급하게 들어간다.

### 9-3. 조향 방향

`steer_invert`(`params.yaml`의 `serial_sender_node`)로 뒤집는다. **반드시 실측으로 확인할 것.**
부호가 반대면 제어기가 보정할수록 반대로 밀려서, 정지 상태는 멀쩡한데 굴리면 한쪽으로
계속 흘러간다.

확인 방법은 조향을 `+1`/`-1`로 번갈아 크게 물리고 앞바퀴를 직접 보는 것이다.

```bash
python3 firmware/tools/drive_straight.py 3 30   # 조향 중립 고정 + 직진 (기계적 트림 확인)
```

### 9-4. 펌웨어 빌드·업로드

```bash
cd firmware/autonomous_mega
~/.platformio/penv/bin/pio run --target upload
```

`/dev/ttyACM0`를 ROS 노드가 잡고 있으면 업로드가 실패한다. 런치를 먼저 내릴 것.

### 9-5. 측정 도구

```bash
python3 firmware/tools/steer_pwm_sweep.py
```

조향 최소 구동 PWM과 실제 기계적 가동범위를 단계별로 잰다. 매 측정 전에 중앙으로 복귀시키므로
이동량이 항상 중앙 기준이다. 각 단계마다 Enter를 기다리고, Ctrl+C로 즉시 정지한다
(명령 스트림이 끊기면 펌웨어 워치독도 300ms 안에 모터를 멈춘다).

> **앞바퀴를 띄우고 잴 것.** 지면에 닿은 정지 조향은 부하가 가장 커서, 기계적 한계가 아니라
> 마찰에 막힌 지점을 한계로 잘못 기록하게 된다. 반대로 최소 구동 PWM은 띄우면 실제보다
> 낮게 나오므로, 그 값은 하한으로만 쓰고 여유를 얹어야 한다.

## 10. 차선 중심 추정 (BEV)

`camera_perception_func_lib.py`의 `get_lane_center()`가 BEV ROI에서 차선 중심 x를 뽑는다.
실패하면 `-1`을 반환하고 호출측이 걸러낸다.

`draw_edges()`는 **추종 중인 클래스 하나만** 그린다. 그래서 두 차선이 다 검출돼도 이 함수가
보는 건 항상 선 1개이고, 중심은 `선위치 ± lane_width_for_center/2`로 추정된다.
**즉 `lane_width_for_center`는 일부 구간이 아니라 사실상 모든 프레임에 적용된다.**

| 파라미터 | 값 | 의미 |
|---|---|---|
| `fixed_lane_class` | `lane_2` | 추종 차선 고정. 빈 값이면 상태머신이 자동 판단 |
| `lane_width_for_center` | 216 | BEV에서 가정하는 차선 폭(px) |
| `car_center_x` | 320 | BEV ROI에서의 추종 기준선. 현재는 카메라 영상 정중앙을 사용 |
| `target_y_end` | 95 | 목표점을 뽑을 행 범위 상한 (5/35/65 세 행만 사용) |

현재 `car_center_x`는 카메라 영상 중심인 320이다. 카메라가 차체 중심축에서 벗어나 장착됐다면
카메라는 차선 중앙을 따르더라도 차체는 치우칠 수 있다. 차체 중심을 기준으로 다시 잡으려면
앞바퀴에서 좌우 차선까지 거리를 **자로 재서** 맞춘 뒤 `target_x`를 측정한다.

`lane_width_for_center`는 주행 중 `target_x` 평균과 `car_center_x`의 차이로 역산한다.
`중심 = 선위치 − 폭/2`이므로 **`폭 보정량 = 2 × 편차`**다. 조향이 한쪽으로 쏠려 있으면
이 관계로 한 번에 맞출 수 있다.

### 알려진 한계

- **행별 폭 미보정.** BEV는 원근 보정이라 실제 차선 폭이 행마다 다르다(근거리 ~125,
  원거리 ~222). 지금은 모든 행에 같은 `lane_width_for_center`를 써서, 차량에 가까운
  행(y=95, 125)은 먼 행과 70~80px 어긋난다. 그래서 `target_y_end`를 95로 두어 먼 행 3개만
  쓴다. 경로 점이 3개뿐이라 S자 곡선에서 중간 형상을 읽는 능력이 떨어진다.
  행별 폭을 쓰도록 고치면 5개로 늘릴 수 있다.
- **조향 해상도.** `MotionCommand.steering`이 `int32`라 `±max_steer_cmd`(=9) 정수 19단계다.
  더 세밀하게 하려면 `max_steer_cmd`를 키우거나(양쪽 노드 값을 같이) 메시지를 float로 바꿔야 한다.

## 11. 문제 해결 순서 (조향·주행)

증상별로 확인할 곳이 다르다. 위에서부터 순서대로 좁힌다.

1. **차가 아예 안 움직인다** — 시리얼 프로토콜(9-1). 아두이노가 `FRONT:0,REAR:0`을
   되돌려 보내는지 확인. 명령이 폐기되고 있으면 워치독이 계속 0으로 내린다.
2. **삐 소리만 나고 안 굴러간다** — 구동 PWM이 정지마찰보다 낮다. `base_speed`를 올린다.
   펌웨어 `setMotor()`에는 구동 모터 PWM 하한 보정이 없다.
3. **조향이 안 걸린다** — 데드밴드가 명령 한 단계보다 큰지 확인(9-2).
4. **조향은 되는데 방향이 반대로 간다** — `steer_invert`(9-3).
5. **직진 명령으로 굴렸을 때 한쪽으로 간다** — 기계적 트림. `steeringPotCenter`를 옮긴다.
   `drive_straight.py`로 확인.
6. **직진은 되는데 자율주행에서 한쪽으로 치우친다** — 인식 기준점. `car_center_x`와
   `lane_width_for_center`(10번).
7. **곡선에서 못 따라간다** — `lookahead_distance`(작을수록 민감). 경로 점 개수도 함께 볼 것.
