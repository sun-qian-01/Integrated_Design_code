"""Train a small, interpretable first-person action classifier from collected images."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


LABELS = ('forward', 'up_left', 'up_right', 'stop', 'obstacle', 'uncertain')


def features(image):
    image = cv2.resize(image, (32, 18))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32) / 255
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    edges = cv2.Canny((gray * 255).astype(np.uint8), 60, 140).astype(np.float32) / 255
    return np.concatenate([lab.reshape(-1), edges.reshape(-1)])


def load_dataset(directory):
    labels_path = directory / 'labels.csv'
    if not labels_path.exists():
        raise ValueError(f'找不到 {labels_path}，请先运行采集程序。')
    rows, vectors = [], []
    with labels_path.open(encoding='utf-8', newline='') as records:
        for row in csv.DictReader(records):
            path = directory / row['filename']
            image = cv2.imread(str(path))
            if image is None:
                continue
            rows.append(row)
            vectors.append(features(image))
    if len(rows) < 12:
        raise ValueError(f'只有 {len(rows)} 个有效样本，至少采集 12 个；每个动作建议 30～100 个。')
    return rows, np.array(vectors, dtype=np.float32)


def train(directory, output):
    rows, vectors = load_dataset(directory)
    classes = sorted({row['label'] for row in rows})
    if len(classes) < 2:
        raise ValueError('至少需要两类动作样本。')
    means, counts = {}, {}
    for label in classes:
        selected = vectors[[row['label'] == label for row in rows]]
        means[label] = selected.mean(axis=0).tolist()
        counts[label] = len(selected)
    model = {'version': 1, 'feature_size': vectors.shape[1], 'classes': classes,
             'means': means, 'counts': counts,
             'warning': 'nearest-centroid baseline; observe-only until validated'}
    output.write_text(json.dumps(model, ensure_ascii=False), encoding='utf-8')
    print(f'已保存 {output}')
    print('样本数量：' + ', '.join(f'{label}={counts[label]}' for label in classes))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default=Path('navigation_dataset'))
    parser.add_argument('--output', type=Path, default=Path('navigation_model.json'))
    args = parser.parse_args()
    train(args.dataset, args.output)


if __name__ == '__main__':
    main()
