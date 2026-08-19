#!/usr/bin/env python3
"""조향 가변저항(A0)의 최솟값/최댓값을 키보드로 조향하며 측정한다.

펌웨어가 50 ms마다 출력하는 ``S <ADC>`` 줄을 읽는다. ``a``는 바퀴를 왼쪽,
``l``은 오른쪽으로 직접 구동하며 가변저항의 현재값과 누적 최솟값/최댓값을
표시한다. ``s`` 또는 Space를 누르면 즉시 정지한다.

사용법:
  python3 firmware/tools/measure_steering_pot.py
  python3 firmware/tools/measure_steering_pot.py --port /dev/ttyACM1
  python3 firmware/tools/measure_steering_pot.py --duration 20
"""

import argparse
import select
import sys
import termios
import time
import tty

import serial


def parse_args():
    parser = argparse.ArgumentParser(
        description="Arduino가 출력하는 조향 가변저항 A0의 범위를 측정합니다."
    )
    parser.add_argument("--port", default="/dev/ttyACM0", help="Arduino 시리얼 포트")
    parser.add_argument("--baudrate", type=int, default=115200, help="시리얼 속도")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="측정 시간(초). 0이면 Ctrl+C를 누를 때까지 측정",
    )
    parser.add_argument(
        "--pwm", type=int, default=100, help="측정용 앞모터 PWM (기본값: 100)"
    )
    parser.add_argument(
        "--invert-directions",
        action="store_true",
        help="a/l 조향 방향이 실제 바퀴와 반대일 때 명령 부호를 뒤집음",
    )
    return parser.parse_args()


def steering_adc(line):
    """'S 510' 형식만 ADC 값으로 변환하고 초음파 출력은 무시한다."""
    if not line.startswith("S ") or ":" in line:
        return None
    try:
        value = int(line[2:].strip())
    except ValueError:
        return None
    return value if 0 <= value <= 1023 else None


def print_result(minimum, maximum, count):
    print("\n\n측정 결과")
    if count == 0:
        print("  유효한 A0 값을 받지 못했습니다.")
        print("  현재 펌웨어가 업로드됐는지와 포트/baudrate를 확인하세요.")
        return 1

    midpoint = (minimum + maximum) / 2.0
    span = maximum - minimum
    print(f"  샘플 수 : {count}")
    print(f"  최솟값  : {minimum}")
    print(f"  최댓값  : {maximum}")
    print(f"  중간값  : {midpoint:.1f}  (기계적 중앙값 참고용)")
    print(f"  전체 폭 : {span}")
    print("\n펌웨어 입력 예시(기계적 끝에 약간의 여유를 두고 결정):")
    print(f"  int steeringPotMin = {minimum};")
    print(f"  int steeringPotCenter = {round(midpoint)};")
    print(f"  int steeringPotMax = {maximum};")

    if span < 50:
        print("\n경고: 측정 폭이 50보다 작습니다. 배선, 장착 연동 또는 가변저항을 확인하세요.")
    if minimum <= 2 or maximum >= 1021:
        print("\n경고: 값이 ADC 끝값(0/1023)에 가깝습니다. 신호선 단선이나 전원 배선을 확인하세요.")
    return 0


def read_key():
    """터미널에 대기 중인 키 하나를 반환한다."""
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    return sys.stdin.read(1).lower() if ready else None


def send_front_pwm(device, pwm):
    """측정용 개루프 앞모터 명령을 보낸다."""
    device.write(f"F,{pwm}\n".encode("ascii"))


def adc_bar(value, width=50):
    """0~1023 ADC 위치를 고정 폭의 선형 막대로 표시한다."""
    position = round(value * (width - 1) / 1023)
    cells = ["-"] * width
    cells[position] = "|"
    return "".join(cells)


def main():
    args = parse_args()
    minimum = 1023
    maximum = 0
    count = 0
    started = time.monotonic()
    old_terminal = None
    front_pwm = None
    left_pwm = -args.pwm if args.invert_directions else args.pwm

    try:
        with serial.Serial(args.port, args.baudrate, timeout=0.05) as device:
            print(f"Arduino 재시작 대기 중: {args.port} @ {args.baudrate}")
            time.sleep(2.5)
            device.reset_input_buffer()
            print("안전을 위해 앞바퀴를 띄우세요.")
            print("조작: a=왼쪽, l=오른쪽, s/Space=정지, q/Ctrl+C=종료")
            if args.duration > 0:
                print(f"{args.duration:g}초 동안 측정합니다.")

            if sys.stdin.isatty():
                old_terminal = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
            else:
                print("경고: 터미널 입력이 아니므로 키보드 조향은 사용할 수 없습니다.")

            started = time.monotonic()
            while args.duration <= 0 or time.monotonic() - started < args.duration:
                key = read_key() if old_terminal is not None else None
                if key == "q":
                    break
                if key == "a":
                    front_pwm = left_pwm
                elif key == "l":
                    front_pwm = -left_pwm
                elif key in ("s", " "):
                    front_pwm = None
                    send_front_pwm(device, 0)

                # 키를 선택한 동안만 명령을 반복한다. None이면 워치독이 모터를 정지시킨다.
                if front_pwm is not None:
                    send_front_pwm(device, front_pwm)
                raw = device.readline().decode("ascii", errors="ignore").strip()
                value = steering_adc(raw)
                if value is None:
                    continue

                count += 1
                minimum = min(minimum, value)
                maximum = max(maximum, value)
                print(
                    f"\r0 [{adc_bar(value)}] 1023 "
                    f"현재:{value:4d} 최소:{minimum:4d} 최대:{maximum:4d} "
                    f"모터:{'정지' if front_pwm is None else f'{front_pwm:+d} PWM'}",
                    end="",
                    flush=True,
                )
    except KeyboardInterrupt:
        pass
    except serial.SerialException as exc:
        print(f"시리얼 포트를 열 수 없습니다: {exc}", file=sys.stderr)
        return 2
    finally:
        # 정상 종료와 예외 모두 즉시 정지하고, 전송 실패 시에는 펌웨어 워치독에 맡긴다.
        try:
            if "device" in locals() and device.is_open:
                send_front_pwm(device, 0)
        except serial.SerialException:
            pass
        if old_terminal is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_terminal)

    return print_result(minimum, maximum, count)


if __name__ == "__main__":
    raise SystemExit(main())
