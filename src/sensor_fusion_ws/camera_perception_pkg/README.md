# camera_perception_pkg

카메라 입력을 발행하고, YOLOv8로 객체를 검출/세그멘테이션하는 패키지. `image_publisher_node`(카메라 입력),
`yolov8_node`(객체 검출), `bird_eye_node`(차선 세그멘테이션 기반 버드아이뷰) 세 노드가 들어있다.

## 노드

### `image_publisher_node`

OpenCV(`cv2.VideoCapture`)로 카메라(혹은 이미지/동영상 파일)를 읽어 `sensor_msgs/Image`로 발행한다.
usb_cam 드라이버 패키지 없이 웹캠을 바로 쏠 수 있어서, 이 워크스페이스의 실제 카메라 입력원으로 쓴다.

- 발행 토픽: `image_raw` (기본값, 파라미터로 변경 가능)
- 파라미터
  - `data_source`: `camera` / `image` / `video` 중 택1 (기본 `camera`)
  - `cam_num`: 카메라 장치 번호, `ls /dev/video*`로 확인 (기본 `0`)
  - `img_dir`, `video_path`: `data_source`가 `image`/`video`일 때 쓰는 경로
  - `pub_topic`: 발행 토픽 이름 (기본 `image_raw`)
  - `logger`: `True`면 `cv2.imshow`로 화면에 미리보기 (기본 `True`)
  - `timer`: 발행 주기(초) (기본 `0.03` ≈ 33Hz)

### `yolov8_node`

`ultralytics` YOLOv8 모델로 `image_raw`를 받아 추론하고, 결과를 `interfaces_pkg/DetectionArray`로 발행하는
lifecycle 노드. bbox, segmentation mask, pose keypoint를 모두 지원(모델 종류에 따라).

- 구독 토픽: `image_raw`
- 발행 토픽: `detections` (`interfaces_pkg/DetectionArray`)
- 서비스: `enable` (`std_srvs/SetBool`) — 런타임에 추론 on/off
- 파라미터
  - `model`: 가중치(.pt) 경로. **콤마(`,`)로 여러 개를 넘기면 각 모델을 모두 로딩해서 프레임마다 순차
    추론하고, 결과를 하나의 `DetectionArray`로 합쳐서 발행한다** (예: `best_cone.pt,car_back.pt`).
    기본값은 파일명만(`best.pt`)이라, 실제 실행 시에는 launch 파일에서 `models/` 아래 절대경로를
    넣어줘야 한다 (`fusion_bringup.launch.py` 참고 — 현재는 `best_cone.pt` + `car_back.pt`를 함께 로딩).
  - `device`: `cpu` 또는 `cuda:0`. 노드 자체 기본값은 `cpu`지만, `fusion_bringup.launch.py`/
    `full_bringup.launch.py`로 실행하면 `cuda:0`가 기본값으로 전달된다 (Jetson Orin Nano 타깃).
  - `threshold`: confidence threshold (기본 `0.5`)
  - `iou`: NMS IoU threshold (기본 `0.45`)
  - `enable`: 시작 시 추론 활성화 여부 (기본 `True`)

### `bird_eye_node`

한 대의 카메라 입력에서 차선을 세그멘테이션(YOLOv8-seg)한 뒤, 원근 변환(homography)으로
버드아이뷰로 펼쳐서 발행하는 노드. `lidar_camera_fusion_pkg`의 Fusion Visualizer 4번(`bev`)/5번
(`bev_roi`) 화면에서 이 노드의 출력을 보여준다.

- 구독 토픽: `input_topic` (기본 `image_raw`)
- 발행 토픽
  - `output_topic`(기본 `bird_eye/image`): 버드아이뷰로 워핑된 차선 세그멘테이션 결과
  - `roi_topic`(기본 `bird_eye/roi`): 버드아이뷰 변환에 쓰는 사다리꼴 ROI를 원본 이미지 위에 표시
  - `comparison_topic`(기본 `bird_eye/comparison`): 위 둘을 나란히 붙인 비교용 이미지
- 파라미터
  - `model`: 세그멘테이션 가중치 경로. 비워두면 패키지에 포함된 `models/lane.pt`를 사용
  - `device`: `cpu` 또는 `cuda:0`. `fusion_bringup.launch.py`에서는 `yolov8_node`와 동일한 `device`
    launch 인자를 공유 (기본 `cuda:0`)
  - `src_points`(정규화 좌표 4쌍) / `normalized_src_points`: 도로면으로 볼 사다리꼴 ROI 꼭짓점
    (좌상/우상/우하/좌하). 카메라 설치 각도·높이가 바뀌면 재조정 필요
  - `output_width`, `output_height`: 버드아이뷰 출력 해상도 (기본 `640x640`)
  - `confidence_threshold`(기본 `0.50`), `iou_threshold`(기본 `0.45`)
  - `show_preview`: 노드 자체 OpenCV 미리보기 창(`Lane bird-eye | original`) 표시 여부. `fusion_bringup.launch.py`에서는
    Fusion Visualizer와 창이 중복되므로 `false`로 꺼져 있음

## models/

`setup.py`에서 `models/*.pt`를 전부 `share/camera_perception_pkg/models/`로 설치한다.

| 파일 | 용도 |
|---|---|
| `best_cone.pt` | 콘/드럼 검출 (YOLOv8 detect), `yolov8_node`가 로딩 |
| `car_back.pt` | 차량 후면 검출 (YOLOv8 detect), `yolov8_node`가 로딩 |
| `lane.pt` | 차선 세그멘테이션 (YOLOv8-seg), `bird_eye_node`가 로딩 |
| `best.pt` | 이전 단일 모델(레거시). 현재 launch 파일들은 쓰지 않음 |
