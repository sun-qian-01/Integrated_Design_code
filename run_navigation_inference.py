"""Run model inference on live phone frames; observe-only by design."""

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from phone_direction_control import PhoneController, find_adb
from train_navigation_model import features
from vision_navigation import capture


def main():
    parser = argparse.ArgumentParser(description='第一视角模型实时推理，不发送触控。')
    parser.add_argument('--model', type=Path, default=Path('navigation_model.json'))
    parser.add_argument('--adb')
    parser.add_argument('--serial')
    args = parser.parse_args()
    model = json.loads(args.model.read_text(encoding='utf-8'))
    means = {label: np.array(vector, dtype=np.float32) for label, vector in model['means'].items()}
    controller = PhoneController(find_adb(args.adb), args.serial)
    controller.connect()
    window = 'Inference observe-only | Q quit'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        while True:
            frame = capture(controller)
            vector = features(frame.image)
            distances = {label: float(np.linalg.norm(vector - mean)) for label, mean in means.items()}
            label = min(distances, key=distances.get)
            preview = frame.image.copy()
            cv2.putText(preview, f'PREDICT: {label}  distance={distances[label]:.3f}  NO CONTROL',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            cv2.imshow(window, preview)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
            time.sleep(0.05)
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
