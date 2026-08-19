#!/usr/bin/env python3
"""조향을 중립에 고정한 채 직진만 시킨다 (기계적 트림 확인용).

  python3 drive_straight.py [주행초] [PWM]
  예) python3 drive_straight.py 3 30

Ctrl+C로 즉시 정지. 아두이노 워치독(300ms)도 별도로 걸려 있다.
"""
import sys, time, serial

sec = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
pwm = int(sys.argv[2]) if len(sys.argv) > 2 else 30

s = serial.Serial('/dev/ttyACM0', 115200, timeout=0.2)
time.sleep(2.5); s.reset_input_buffer()

def send(n, rear, dur):
    t0 = time.time(); pot = None
    while time.time() - t0 < dur:
        s.write(f"C,{n:.4f},{rear}\n".encode()); time.sleep(0.05)
        while s.in_waiting:
            l = s.readline().strip()
            if l.startswith(b'S ') and b':' not in l:
                try: pot = int(l[2:])
                except ValueError: pass
    return pot

try:
    print("조향 중립 정렬 중...")
    p = send(0.0, 0, 3.0)
    print(f"  정렬 완료: pot={p} (현재 펌웨어의 steeringPotCenter와 비교)\n")
    print(f"직진 {sec}초 (rear PWM {pwm}, 조향 고정 0)")
    p = send(0.0, pwm, sec)
    print(f"  주행 중 pot={p}")
finally:
    for _ in range(6):
        s.write(b"C,0.0,0\n"); time.sleep(0.05)
    s.close()
    print("\n정지 완료")
