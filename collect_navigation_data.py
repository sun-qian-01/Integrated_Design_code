"""Collect first-person phone frames and manual action labels; never sends touch."""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import time

import cv2

from phone_direction_control import PhoneController, find_adb
from vision_navigation import capture


LABELS = {
    'w': 'forward',
    'a': 'up_left',
    'd': 'up_right',
    'b': 'backward',
    's': 'stop',
    'x': 'obstacle',
    'n': 'uncertain',
}


def main():
    parser = argparse.ArgumentParser(description='采集第一视角手动驾驶样本，不发送任何触控。')
    parser.add_argument('--adb')
    parser.add_argument('--serial')
    parser.add_argument('--output', type=Path, default=Path('navigation_dataset'))
    parser.add_argument('--interval', type=float, default=0.25)
    args = parser.parse_args()
    if args.interval < 0.1:
        parser.error('--interval 不能小于 0.1 秒。')
    controller = PhoneController(find_adb(args.adb), args.serial)
    controller.connect()
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / 'labels.csv'
    new_file = not csv_path.exists()
    with csv_path.open('a', newline='', encoding='utf-8') as records:
        writer = csv.writer(records)
        if new_file:
            writer.writerow(['filename', 'label', 'width', 'height', 'capture_seconds', 'timestamp'])
        cv2.namedWindow('Dataset collector | W/A/D/S/X/N | Q quit', cv2.WINDOW_NORMAL)
        print('只采集画面，不发送触控。用 scrcpy 或手机手动驾驶；按 W/A/D/S/X/N 标注最近画面。')
        try:
            while True:
                frame = capture(controller)
                preview = frame.image.copy()
                cv2.putText(preview, 'W forward | A left | D right | B reverse | S stop | X obstacle | N uncertain | Q quit',
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.imshow('Dataset collector | W/A/D/S/X/N | Q quit', preview)
                key = cv2.waitKey(max(1, round(args.interval * 1000))) & 0xFF
                if key in (ord('q'), 27):
                    break
                if key in {ord(name) for name in LABELS}:
                    label = LABELS[chr(key)]
                    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
                    filename = f'{timestamp}_{label}.jpg'
                    path = args.output / filename
                    cv2.imwrite(str(path), frame.image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    writer.writerow([filename, label, frame.width, frame.height,
                                     round(frame.finished - frame.started, 4), timestamp])
                    records.flush()
                    print(f'{label}: {filename}')
        finally:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
