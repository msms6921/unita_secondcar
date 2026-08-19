// ===== Drive motor pin definition =====
// Front steering motor (left/right)
const int FRONT_EN  = 5;
const int FRONT_IN1 = 22;
const int FRONT_IN2 = 23;

// Rear drive motor (forward/reverse)
const int REAR_EN  = 6;
const int REAR_IN1 = 24;
const int REAR_IN2 = 25;

// ===== Steering position potentiometer =====
constexpr uint8_t STEERING_POT_PIN = A0;
constexpr unsigned long STEERING_REPORT_INTERVAL_MS = 50UL;
// 데드밴드는 조향 분해능을 직접 결정한다. ROS는 steering을 int32로 -max_steer_cmd..
// +max_steer_cmd(=9) 정수로 보내므로, 한 단계가 편측 가동폭 70의 1/9 ≈ 7.8 counts다.
// 데드밴드가 이보다 크면 명령 대부분이 삼켜져 조향이 아예 안 걸린다.
// (실측: 노드 구동 중 |steering|>=3 인 샘플이 2.3%뿐 -> 사실상 조향 정지)
// 한때 18로 넓혔던 건 앞바퀴를 띄운 상태의 헌팅 때문이었는데, 지면에 닿으면
// 마찰이 관성을 잡아주므로 좁혀도 된다. 한 단계(7.8)보다 작게 유지할 것.
constexpr int STEERING_DEADBAND = 6;
// 앞바퀴를 띄우고 실측: PWM 40에서도 0.5초에 43 counts 움직였다(모터 정상).
// 하지만 바닥에 닿은 정지 상태(dry steering)에서는 PWM 50으로 10초를 밀어도
// 꿈쩍하지 않았다. 지면 마찰을 이기도록 올리되, 목표 근처 접근용 MIN은
// 관성으로 튀어나가지 않을 만큼만 둔다.
constexpr int STEERING_MIN_PWM = 55;
constexpr int STEERING_MAX_PWM = 160;
// 예전 값 150은 전체 가동폭(편측 70)보다 넓어서 감속구간을 벗어나는 경우가 없었다.
// 그래서 map()이 항상 낮은 PWM만 뱉었다(실측 49~54). 목표 근처에서만 감속하되,
// 30은 너무 늦게 줄여서 오버슈트를 키웠다.
constexpr int STEERING_SLOWDOWN_RANGE = 50;
constexpr int STEERING_DIRECTION = 1;

// Steering potentiometer calibration values.
// steeringPotCenter는 반드시 Min~Max 사이여야 한다. 예전 값 5는 Min(429)보다 작아서
// normalizedToSteeringPosition()의 constrain(target, Min, Max)에 걸려,
// 정규화 조향값이 0.676 미만이면(직진 0.0 포함) 목표가 전부 429로 눌렸다.
// 그 결과 차선을 중앙으로 잡아도 앞바퀴가 한쪽 끝에 물린 채 계속 오른쪽으로 갔다.
// 새 센서 실측 범위는 오른쪽 끝 433, 왼쪽 끝 627이다.
// 코드의 Min/Max는 물리 방향명이 아니라 ADC 숫자의 최소/최대 순서여야 한다.
//
// Min/Max도 실측으로 재보정했다. 앞바퀴를 띄우고 PWM 60/80/120으로 끝까지 밀었을 때
// 실제 도달한 기계적 한계는 434 / 622였다(예전 값 429/632는 도달 불가능한 값이라,
// 풀조향 시 오차가 데드밴드에 못 들어와 모터가 스토퍼를 계속 미는 상태가 된다).
//
// 실측 한계 기준이면 좌 76 / 우 112로 비대칭이다. 한때 pure pursuit의 대칭 가정에
// 맞추려고 ±70으로 잘라뒀는데, 그러면 우측 조향 능력의 38%를 안 쓰게 되고
// 차선을 못 따라가고 벗어난다. 실측 한계(434/622) 안쪽으로 여유만 두고 전부 쓴다.
// 좌 70 / 우 106으로 비대칭이므로 같은 크기의 명령이어도 우회전이 더 급하다.
// 기계적 끝을 계속 밀지 않도록 양 끝에서 약 7 counts 안쪽으로 제한한다.
// Center 530은 두 끝의 중간값으로 잡은 초기값이며 직진 실측 후 보정한다.
int steeringPotMin = 440;
int steeringPotCenter = 530;
int steeringPotMax = 620;
unsigned long lastSteeringReport = 0;

// 워치독(구동 명령 끊기면 정지)
unsigned long lastDriveMs = 0;
const unsigned long DRIVE_WD_MS = 300;

// 명령 변수 (전역으로 공유됨)
int rear_pwm_cmd = 0;  // drive motor: -255..255
int targetSteering = steeringPotCenter; // converted normalized steering target
int steeringPwmOutput = 0;
bool driveCommandActive = false;
bool frontTestActive = false;
int frontTestPwm = 0;


// ===== Ultrasonic (HC-SR04 x4) =====
// Pins 22-25 are reserved for the front/rear drive motors.
// Remaining wiring (TRIG/ECHO): 26/27, 28/29, 30/31, 32/33
const uint8_t N_US = 4;
const uint8_t TRIG_PINS[N_US] = {26, 28, 30, 32};
const uint8_t ECHO_PINS[N_US] = {27, 29, 31, 33};

const unsigned long ECHO_TIMEOUT_US = 30000UL; // 30ms ≈ 5m
const uint16_t BETWEEN_SENSORS_MS = 25;        // 간섭 방지(필요하면 40~60)
const uint16_t ULTRA_PUB_MS = 100;             // 초음파 송신 주기(10Hz 느낌)

float us_cm[N_US] = {-1, -1, -1, -1};
uint8_t us_idx = 0;
unsigned long lastUsMs = 0;
unsigned long lastPubMs = 0;


void setup() {
  Serial.begin(115200);

  // 모터 핀 초기화(motor_control.ino)
  initMotors();

  lastDriveMs = millis();

  for (uint8_t i = 0; i < N_US; i++) {
    pinMode(TRIG_PINS[i], OUTPUT);
    pinMode(ECHO_PINS[i], INPUT);
    digitalWrite(TRIG_PINS[i], LOW);
  }
  lastUsMs = millis();
  lastPubMs = millis();
  lastSteeringReport = millis();
}

void loop() {
  // 통신 확인(comm.ino)
  checkSerial();
  reportSteeringPosition();
  ultrasonicStep();              // 초음파 1개 측정
  ultrasonicPublishIfDue();      
  
  // 워치독 (안전장치)
  if (millis() - lastDriveMs > DRIVE_WD_MS) {
    rear_pwm_cmd = 0;
    driveCommandActive = false;
    frontTestActive = false;
    frontTestPwm = 0;
  }

  // F 명령은 보정 전 측정용 개루프 제어, C 명령은 일반 폐루프 제어다.
  if (frontTestActive) {
    steeringPwmOutput = frontTestPwm;
    setMotor(FRONT_EN, FRONT_IN1, FRONT_IN2, steeringPwmOutput);
  } else {
    updateSteeringControl();
  }
  setMotor(REAR_EN, REAR_IN1, REAR_IN2, rear_pwm_cmd);
}
