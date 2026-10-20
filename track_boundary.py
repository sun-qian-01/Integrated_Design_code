"""Train a color road-surface model from explicit polygons on historical frames."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


MODEL = Path('navigation_track_model.json')
ANNOTATIONS = Path('navigation_dataset/track_annotations.json')


def read_image(path):
    image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'无法读取图片：{path}')
    return image


def polygon_mask(shape, polygons):
    height, width = shape[:2]
    mask = np.zeros((height, width), np.uint8)
    for polygon in polygons:
        points = np.rint(np.asarray(polygon) * [width - 1, height - 1]).astype(np.int32)
        if len(points) < 3 or not np.isfinite(points).all():
            raise ValueError('标注多边形至少需要三个有效顶点。')
        cv2.fillPoly(mask, [points], 255)
    return mask


def color_features(image):
    return cv2.cvtColor(cv2.GaussianBlur(image, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.float32)


def distances(pixels, centers):
    best = np.full(pixels.shape[0], np.inf, np.float32)
    for center in centers:
        best = np.minimum(best, np.sum((pixels - center) ** 2, axis=1))
    return np.sqrt(best)


def train(dataset, annotations, destination):
    records = json.loads(annotations.read_text(encoding='utf-8'))['images']
    random = np.random.default_rng(7)
    samples = {'road': [], 'background': []}
    held = []
    files = []
    splits = {}
    for record in records:
        split = record.get('split', 'train')
        if split not in ('train', 'validation') or record['filename'] in splits:
            raise ValueError('每张图片只能出现一次，并指定 train 或 validation。')
        splits[record['filename']] = split
        image = read_image(dataset / record['filename'])
        colors = color_features(image)
        masks = {label: polygon_mask(image.shape, record[label]) for label in samples}
        if np.any((masks['road'] > 0) & (masks['background'] > 0)):
            raise ValueError(f'赛道/背景标注重叠：{record["filename"]}')
        for label, mask in masks.items():
            pixels = colors[mask > 0]
            if not len(pixels):
                raise ValueError(f'缺少 {label} 标注：{record["filename"]}')
            pixels = pixels[random.choice(len(pixels), min(len(pixels), 4000), replace=False)]
            if split == 'validation':
                held.append((label, pixels))
            else:
                samples[label].append(pixels)
        files.append({'filename': record['filename'], 'split': split})
    cv2.setRNGSeed(7)
    model = {'version': 1, 'space': 'blurred-Lab', 'centers': {}, 'radius': {}, 'source_images': files}
    for label, groups in samples.items():
        if not groups:
            raise ValueError('训练图像不足，不能全部设为验证集。')
        pixels = np.concatenate(groups)
        _, _, centers = cv2.kmeans(pixels, min(16, len(pixels)), None,
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.2),
                                   3, cv2.KMEANS_PP_CENTERS)
        model['centers'][label] = centers.tolist()
        model['radius'][label] = float(np.clip(np.quantile(distances(pixels, centers), 0.98) * 1.3, 12, 30))
    classifier = TrackBoundary(model)
    report = {}
    for label in samples:
        pixels = [pixels for actual, pixels in held if actual == label]
        if pixels:
            predictions = classifier.classify(np.concatenate(pixels))
            report[label] = {name: float(np.mean(predictions == value))
                             for name, value in [('road', 1), ('background', 0), ('unknown', -1)]}
    model['validation_polygons'] = report
    destination.write_text(json.dumps(model, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'model': str(destination), 'images': files, 'validation_polygons': report},
                     ensure_ascii=False, indent=2))
    return model


class TrackBoundary:
    def __init__(self, model):
        if model.get('version') != 1 or model.get('space') != 'blurred-Lab':
            raise ValueError('赛道模型版本不兼容，请重新训练。')
        self.centers = {label: np.asarray(model['centers'][label], np.float32) for label in ('road', 'background')}
        self.radius = model['radius']
        for label, centers in self.centers.items():
            if centers.ndim != 2 or centers.shape[1] != 3 or not np.isfinite(centers).all():
                raise ValueError('赛道模型颜色中心无效。')
            if not 0 < self.radius[label] <= 100:
                raise ValueError('赛道模型颜色半径无效。')

    @classmethod
    def load(cls, path):
        return cls(json.loads(path.read_text(encoding='utf-8')))

    def classify(self, pixels):
        road = distances(pixels, self.centers['road'])
        background = distances(pixels, self.centers['background'])
        result = np.full(len(pixels), -1, np.int8)
        result[(road <= self.radius['road']) & (road + 2.5 < background)] = 1
        result[(background <= self.radius['background']) & (background + 2.5 < road)] = 0
        return result

    def analyze(self, image):
        small = cv2.resize(image, (320, 144))
        labels = self.classify(color_features(small).reshape(-1, 3)).reshape(144, 320)
        valid = np.zeros((144, 320), np.uint8)
        valid[78:135, 6:314] = 255
        for left, top, right, bottom in [(0.30, 0.86, 0.40, 1), (0.60, 0.86, 0.67, 1),
                                          (0.73, 0.30, 0.94, 0.76), (0.625, 0.3, 0.645, 0.85)]:
            valid[int(top * 144):int(bottom * 144), int(left * 320):int(right * 320)] = 0
        road = ((labels == 1) & (valid > 0)).astype(np.uint8) * 255
        road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, np.ones((5, 9), np.uint8))
        road[valid == 0] = 0
        bands = []
        for top, bottom in [(111, 125), (91, 105)]:
            coverage = np.sum(road[top:bottom] > 0, axis=0) / np.maximum(1, np.sum(valid[top:bottom] > 0, axis=0))
            present = (coverage > 0.55).astype(np.uint8)
            present = cv2.morphologyEx(present[None, :], cv2.MORPH_CLOSE, np.ones((1, 15), np.uint8))[0]
            transitions = np.diff(np.r_[0, present, 0].astype(np.int16))
            runs = [(int(left), int(right)) for left, right in zip(np.where(transitions == 1)[0], np.where(transitions == -1)[0])
                    if right - left >= 30]
            selected = min(runs, key=lambda run: abs((run[0] + run[1]) / 2 - 160)) if runs else None
            bands.append(selected)
        near, far = bands
        known = float(np.mean(labels[valid > 0] >= 0))
        center = None if far is None else (far[0] + far[1]) / 640
        near_safe = near is not None and near[0] + 12 <= 160 <= near[1] - 12
        usable = known >= 0.45 and near_safe and center is not None
        offset = None if center is None else center - 0.5
        suggestion = 'stop' if not usable else 'up_left' if offset < -0.10 else 'up_right' if offset > 0.10 else 'forward'
        corridors = {}
        for action, left, right in [('forward', 0.44, 0.56), ('up_left', 0.30, 0.44), ('up_right', 0.56, 0.70)]:
            region = labels[100:119, int(left * 320):int(right * 320)]
            visible = valid[100:119, int(left * 320):int(right * 320)] > 0
            corridors[action] = {'background': float(np.mean(region[visible] == 0)) if visible.any() else 0.0,
                                 'road': float(np.mean(region[visible] == 1)) if visible.any() else 0.0}
        return {'usable': usable, 'suggestion': suggestion, 'offset': offset, 'known_fraction': known,
                'near': near, 'far': far, 'corridors': corridors, 'labels': labels, 'valid': valid}

    def constrain(self, image, proposed, pulse_ms, turn_ms):
        result = self.analyze(image)
        action = proposed
        if proposed != 'stop':
            if not result['usable']:
                action = 'stop'
            elif proposed in result['corridors'] and result['corridors'][proposed]['background'] > 0.65:
                alternative = result['suggestion']
                corridor = result['corridors'].get(alternative, {})
                action = alternative if corridor.get('road', 0) > 0.45 and corridor.get('background', 1) < 0.2 else 'stop'
        duration = min(pulse_ms, turn_ms if action in ('up_left', 'up_right', 'backward') else 250)
        details = {key: value for key, value in result.items() if key not in ('labels', 'valid')}
        details.update(proposed=proposed, executed=action, pulse_ms=duration)
        return action, duration, details

    def overlay(self, image):
        result = self.analyze(image)
        colors = np.zeros((144, 320, 3), np.uint8)
        colors[result['labels'] == 1] = (40, 210, 40)
        colors[result['labels'] == 0] = (50, 50, 220)
        colors[result['labels'] == -1] = (0, 180, 240)
        mask = cv2.resize(result['valid'], (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        blended = cv2.addWeighted(image, 0.65, cv2.resize(colors, (image.shape[1], image.shape[0])), 0.35, 0)
        output = image.copy()
        output[mask] = blended[mask]
        cv2.putText(output, f'TRACK: {result["suggestion"]} known={result["known_fraction"]:.2f}',
                    (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        return output


def annotate(dataset, destination):
    records = json.loads(destination.read_text(encoding='utf-8')) if destination.exists() else {'images': []}
    with (dataset / 'labels.csv').open(encoding='utf-8-sig', newline='') as stream:
        names = list(dict.fromkeys(row['filename'] for row in csv.DictReader(stream)))
    points = []

    def click(event, horizontal, vertical, flags, userdata):
        if event == cv2.EVENT_LBUTTONUP:
            points.append([horizontal / 959, vertical / 431])

    window = 'Track labels | R road B background | U undo | N next | V validation | S save Q quit'
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, click)
    try:
        for filename in names:
            previous = next((item for item in records['images'] if item['filename'] == filename), None)
            record = previous or {'filename': filename, 'road': [], 'background': [], 'split': 'train'}
            original = cv2.resize(read_image(dataset / filename), (960, 432))
            while True:
                preview = original.copy()
                for label, color in [('road', (0, 255, 0)), ('background', (0, 0, 255))]:
                    mask = polygon_mask(preview.shape, record[label]) > 0
                    preview[mask] = (preview[mask] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
                if points:
                    vertices = np.rint(np.array(points) * [959, 431]).astype(np.int32)
                    cv2.polylines(preview, [vertices], False, (0, 255, 255), 2)
                cv2.putText(preview, f'{filename[:55]} | {record["split"]} | click polygon then R/B',
                            (5, 20), 0, 0.48, (0, 255, 255), 1)
                cv2.imshow(window, preview)
                key = cv2.waitKey(30) & 0xFF
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    key = ord('q')
                if key in (ord('r'), ord('b')) and len(points) >= 3:
                    record['road' if key == ord('r') else 'background'].append(points.copy())
                    points.clear()
                if key == ord('u'):
                    if points:
                        points.pop()
                    else:
                        record['road'].clear()
                        record['background'].clear()
                if key == ord('v'):
                    record['split'] = 'validation' if record['split'] == 'train' else 'train'
                if key in (ord('s'), ord('n'), ord('q'), 27):
                    if record['road'] and record['background'] and previous is None:
                        records['images'].append(record)
                    destination.write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
                    points.clear()
                    if key in (ord('q'), 27):
                        return
                    if key == ord('n'):
                        break
    finally:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='使用历史图片标注、训练并离线检查赛道，不连接或控制手机。')
    parser.add_argument('command', choices=['annotate', 'train', 'preview'])
    parser.add_argument('--dataset', type=Path, default=Path('navigation_dataset'))
    parser.add_argument('--annotations', type=Path, default=ANNOTATIONS)
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--output', type=Path, default=Path('navigation_rl_logs/track_preview'))
    args = parser.parse_args()
    if args.command == 'annotate':
        annotate(args.dataset, args.annotations)
    elif args.command == 'train':
        train(args.dataset, args.annotations, args.model)
    else:
        model = TrackBoundary.load(args.model)
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.dataset / 'labels.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        report = []
        for row in rows:
            image = read_image(args.dataset / row['filename'])
            result = model.analyze(image)
            report.append({'filename': row['filename'], 'human_action': row['label'],
                           **{key: value for key, value in result.items() if key not in ('labels', 'valid')}})
            success, data = cv2.imencode('.jpg', model.overlay(image))
            if not success:
                raise ValueError('预览编码失败。')
            data.tofile(args.output / row['filename'])
        (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'已生成 {len(rows)} 张离线预览：{args.output.resolve()}')


if __name__ == '__main__':
    main()
