# Steering potentiometer reader

조향 모터를 구동하지 않고 A0 가변저항의 원시 ADC 값(0~1023)만 50 ms마다 출력한다.

```bash
cd firmware/steering_pot_reader
pio run --target upload --upload-port /dev/ttyACM0
pio device monitor --port /dev/ttyACM0 --baud 115200
```

시리얼 모니터 종료는 `Ctrl+C`다. 측정 후 차량을 사용하려면
`firmware/autonomous_mega` 펌웨어를 다시 업로드해야 한다.
