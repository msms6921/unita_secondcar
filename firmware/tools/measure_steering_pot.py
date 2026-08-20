#!/usr/bin/env python3
"""조향 가변저항(A0)의 최솟값/최댓값을 수동으로 측정한다.

펌웨어가 50 ms마다 출력하는 ``S <ADC>`` 줄을 읽는다. 이 도구는 모터 명령을
보내지 않으므로, 바퀴를 띄운 뒤 조향 장치를 손으로 천천히 좌우 끝까지 움직인다.

사용법:
  python3 firmware/tools/measure_steering_pot.py
  python3 firmware/tools/measure_steering_pot.py --port /dev/ttyACM1
  python3 firmware/tools/measure_steering_pot.py --duration 20
"""

import argparse
import sys
import time

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


def main():
    args = parse_args()
    minimum = 1023
    maximum = 0
    count = 0
    started = time.monotonic()

    try:
        with serial.Serial(args.port, args.baudrate, timeout=0.3) as device:
            print(f"Arduino 재시작 대기 중: {args.port} @ {args.baudrate}")
            time.sleep(2.5)
            device.reset_input_buffer()
            print("앞바퀴를 띄우고 조향을 손으로 천천히 좌우 끝까지 움직이세요.")
            print("종료: Ctrl+C" if args.duration <= 0 else f"{args.duration:g}초 동안 측정합니다.")

            started = time.monotonic()
            while args.duration <= 0 or time.monotonic() - started < args.duration:
                raw = device.readline().decode("ascii", errors="ignore").strip()
                value = steering_adc(raw)
                if value is None:
                    continue

                count += 1
                minimum = min(minimum, value)
                maximum = max(maximum, value)
                print(
                    f"\r현재: {value:4d} | 최소: {minimum:4d} | 최대: {maximum:4d}",
                    end="",
                    flush=True,
                )
    except KeyboardInterrupt:
        pass
    except serial.SerialException as exc:
        print(f"시리얼 포트를 열 수 없습니다: {exc}", file=sys.stderr)
        return 2

    return print_result(minimum, maximum, count)


if __name__ == "__main__":
    raise SystemExit(main())
