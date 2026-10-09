"""Bounded visual Q-learning with human rewards and a demonstration prior."""

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import queue
import threading
import time

import cv2
import numpy as np

from phone_direction_control import CONFIG_PATH, PhoneController, find_adb, load_points, pixel_point
from train_navigation_model import features
from vision_navigation import capture


ACTIONS = ('forward', 'up_left', 'up_right', 'stop')
BUTTONS = {'forward': 'up', 'up_left': 'up_left', 'up_right': 'up_right'}
LABELS = set(ACTIONS) | {'obstacle', 'uncertain'}
WINDOW = 'Visual RL | G start | SPACE pause | Q quit | 1 good 2 bad 3 goal 0 neutral'


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read_image(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


@dataclass
class Observation:
    state: str
    label: str
    allowed: bool
    reason: str
    distance: float
    margin: float


class Demonstrations:
    def __init__(self, directory):
        vectors, labels, filenames = [], [], []
        digest = hashlib.sha256(b'rl-features-v1-full-image-32x18')
        with (directory / 'labels.csv').open(encoding='utf-8-sig', newline='') as records:
            for row in csv.DictReader(records):
                if row['label'] not in LABELS:
                    raise ValueError(f"未知样本标签：{row['label']}")
                path = directory / row['filename']
                image = read_image(path)
                if image is None:
                    raise ValueError(f'图片无法读取：{path}')
                vector = features(image)
                vectors.append(vector)
                labels.append(row['label'])
                filenames.append(row['filename'])
                digest.update(vector.tobytes())
                digest.update(row['label'].encode('utf-8'))
        if not vectors:
            raise ValueError('训练集为空。')
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.labels = np.array(labels)
        self.filenames = filenames
        self.fingerprint = digest.hexdigest()
        self.counts = {label: labels.count(label) for label in sorted(set(labels))}
        self.centroids = {label: self.vectors[self.labels == label].mean(axis=0)
                          for label in self.counts}
        self.radii = {}
        for label in self.counts:
            samples = self.vectors[self.labels == label]
            if len(samples) < 3:
                self.radii[label] = 0.0
                continue
            distances = np.linalg.norm(samples[:, None] - samples[None, :], axis=2)
            np.fill_diagonal(distances, np.inf)
            self.radii[label] = float(max(0.15, np.quantile(distances.min(axis=1), 0.95) * 1.25))

    def observe(self, image, previous='stop', minimum_margin=0.12):
        vector = features(image)
        distances = np.linalg.norm(self.vectors - vector, axis=1)
        index = int(np.argmin(distances))
        label = str(self.labels[index])
        best = float(distances[index])
        competing = distances[self.labels != label]
        second = float(competing.min()) if len(competing) else best
        margin = (second - best) / max(second, 1e-6)
        centroid_label = min(self.centroids, key=lambda name: float(np.linalg.norm(vector - self.centroids[name])))
        reason = 'supported by demonstrations'
        if label not in BUTTONS:
            reason = f'label={label}'
        elif self.counts[label] < 5:
            reason = 'too few movement samples'
        elif best > self.radii[label]:
            reason = 'outside demonstration coverage'
        elif margin < minimum_margin:
            reason = 'ambiguous labels'
        elif centroid_label != label:
            reason = 'nearest/centroid disagreement'
        allowed = reason == 'supported by demonstrations'
        return Observation(f'{index}:{previous}', label, allowed, reason, best, margin)

    def report(self):
        predictions = []
        for index, vector in enumerate(self.vectors):
            distances = np.linalg.norm(self.vectors - vector, axis=1)
            distances[index] = np.inf
            predictions.append(str(self.labels[int(np.argmin(distances))]) if len(distances) > 1 else 'none')
        confusion = {label: {} for label in self.counts}
        for actual, predicted in zip(self.labels, predictions):
            confusion[actual][predicted] = confusion[actual].get(predicted, 0) + 1
        return {'counts': self.counts, 'coverage_radii': self.radii,
                'leave_one_image_out_confusion': confusion,
                'fingerprint': self.fingerprint,
                'limitations': 'Descriptive only. Adjacent frames may leak across this check; no independent track evaluation. '
                              'Image similarity gates do not detect collisions, free space, distance or goals.'}


class QPolicy:
    def __init__(self, path, fingerprint, seed=7):
        self.path = path
        self.fingerprint = fingerprint
        self.values = {}
        self.updates = 0
        self.random = np.random.default_rng(seed)
        if path.exists():
            stored = json.loads(path.read_text(encoding='utf-8'))
            if stored.get('fingerprint') != fingerprint or stored.get('actions') != list(ACTIONS):
                raise ValueError('样本或特征已改变；请用 --policy 指定新的策略文件，避免旧状态编号混用。')
            self.values = stored['q_values']
            self.updates = stored['updates']
            if any(len(values) != len(ACTIONS) or not np.isfinite(values).all() for values in self.values.values()):
                raise ValueError('策略文件数值无效。')

    def row(self, observation):
        if observation.state not in self.values:
            values = [0.0] * len(ACTIONS)
            action = observation.label if observation.label in BUTTONS else 'stop'
            values[ACTIONS.index(action)] = 0.2
            self.values[observation.state] = values
        return self.values[observation.state]

    def choose(self, observation, epsilon=0.0):
        if not observation.allowed:
            return 'stop'
        if self.random.random() < epsilon:
            return str(self.random.choice(ACTIONS))
        return ACTIONS[int(np.argmax(self.row(observation)))]

    def update(self, before, action, reward, after, terminal, alpha=0.25, gamma=0.9):
        row = self.row(before)
        future = 0.0 if terminal or not after.allowed else max(self.row(after))
        index = ACTIONS.index(action)
        row[index] += alpha * (reward + gamma * future - row[index])
        self.updates += 1
        self.save()

    def save(self):
        save_json(self.path, {'version': 1, 'fingerprint': self.fingerprint,
                             'actions': list(ACTIONS), 'updates': self.updates, 'q_values': self.values})


class FrameGuard:
    def __init__(self, max_age=1.5, freeze_seconds=3.0):
        self.max_age = max_age
        self.freeze_seconds = freeze_seconds
        self.shape = None
        self.last = None
        self.changed_at = None
        self.previous_label = None
        self.stable = 0

    def inspect(self, frame, observation, now):
        shape = (frame.width, frame.height)
        if self.shape is None:
            self.shape = shape
        reason = None
        if shape != self.shape or frame.width <= frame.height:
            reason = 'screen geometry changed'
        elif now - frame.started > self.max_age:
            reason = 'slow/stale capture'
        elif not observation.allowed:
            reason = observation.reason
        height, width = frame.image.shape[:2]
        roi = frame.image[int(height * 0.55):int(height * 0.83), :int(width * 0.72)]
        gray = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (96, 32)).astype(float) / 255.0
        if self.last is None or np.mean(np.abs(gray - self.last)) > 0.008:
            self.changed_at = now
        self.last = gray
        if self.changed_at is not None and now - self.changed_at > self.freeze_seconds:
            reason = 'unchanged view: check live video'
        if reason:
            self.stable = 0
            self.previous_label = None
            return False, reason
        self.stable = self.stable + 1 if observation.label == self.previous_label else 1
        self.previous_label = observation.label
        return self.stable >= 3, 'ready' if self.stable >= 3 else 'need 3 consistent fresh frames'


def write_frame(path, image):
    success, data = cv2.imencode('.jpg', image)
    if not success:
        raise ValueError('保存图像失败。')
    data.tofile(path)


def run_session(args, demonstrations, policy, controller):
    if not args.points.is_file():
        raise ValueError('缺少已校准的方向坐标文件。')
    points = load_points(args.points)
    output = args.logs / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    output.mkdir(parents=True)
    messages = queue.Queue()
    guard = FrameGuard(args.max_age)
    busy = False
    task_kind = None
    armed = False
    halted = False
    previous = 'stop'
    latest = None
    observation = None
    allowed = False
    pending = None
    steps = 0
    turn_steps = 0
    started = None
    status = 'Observe first. G starts a bounded episode.'
    canvas = np.zeros((560, 960, 3), dtype=np.uint8)

    def submit(kind, operation):
        nonlocal busy, task_kind
        busy = True
        task_kind = kind

        def worker():
            try:
                messages.put((kind, operation(), None))
            except Exception as error:
                messages.put((kind, None, str(error)))

        threading.Thread(target=worker, daemon=True).start()

    def perform(action, target):
        controller.press(target, args.pulse_ms)
        time.sleep(args.settle)
        return capture(controller)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.imshow(WINDOW, canvas)
    record_path = output / 'transitions.jsonl'
    with record_path.open('a', encoding='utf-8') as records:
        def record(data):
            records.write(json.dumps({'time': time.time(), **data}, ensure_ascii=False) + '\n')
            records.flush()

        def finish_feedback(reward, terminal):
            nonlocal pending, armed, status, guard
            before, action, after, prefix = pending
            policy.update(before, action, reward, after, terminal)
            record({'event': 'reward', 'id': prefix, 'state': before.state, 'action': action,
                    'next_state': after.state, 'reward': reward, 'terminal': terminal,
                    'source': 'human', 'updates': policy.updates})
            pending = None
            guard.stable = 0
            if terminal or not after.allowed:
                armed = False
            status = f'Reward {reward:+g} saved. ' + ('Paused; reposition then G.' if not armed else 'Continuing.')

        try:
            while True:
                try:
                    kind, result, error = messages.get_nowait()
                except queue.Empty:
                    kind = None
                if kind:
                    busy = False
                    if error:
                        armed = False
                        halted = True
                        status = f'ERROR: {error}. Check car stopped; restart program.'
                        record({'event': 'error', 'operation': kind, 'message': error})
                    else:
                        latest = result
                        observation = demonstrations.observe(latest.image, previous, args.margin)
                        allowed, reason = guard.inspect(latest, observation, time.monotonic())
                        if kind == 'action':
                            after = observation
                            prefix, before, action = active_transition
                            write_frame(output / f'{prefix}_after.jpg', latest.image)
                            record({'event': 'action_completed', 'id': prefix, 'action': action,
                                    'next_label': after.label, 'allowed': after.allowed, 'guard': reason})
                            guard.stable = 0
                            allowed = False
                            if args.mode == 'train':
                                pending = (before, action, after, prefix)
                                status = 'Stopped for reward: 1 good / 2 bad+end / 3 goal+end / 0 neutral'
                            else:
                                status = 'Pulse released; evaluating new frames.'
                        elif pending is None:
                            status = reason
                        if not observation.allowed or reason in ('screen geometry changed', 'slow/stale capture', 'unchanged view: check live video'):
                            armed = False
                        canvas = latest.image.copy()
                now = time.monotonic()
                if armed and started is not None and (now - started >= args.seconds or steps >= args.steps):
                    armed = False
                    status = 'Episode limit reached. Paused.'
                age_ok = latest is not None and now - latest.started <= args.max_age
                if not age_ok:
                    allowed = False
                    if latest is not None and not busy:
                        armed = False
                panel = canvas.copy()
                cv2.rectangle(panel, (0, 0), (panel.shape[1], 116), (20, 20, 20), -1)
                texts = [f'{args.mode.upper()} | {"RUNNING" if armed else "PAUSED"} | steps {steps}/{args.steps} | Q updates {policy.updates}',
                         status[:115],
                         'G start | SPACE pause | Q quit | 1 good, 2 bad/end, 3 goal/end, 0 neutral']
                if observation:
                    texts.append(f'label={observation.label} distance={observation.distance:.3f} margin={observation.margin:.2f} (NOT probability)')
                for index, text in enumerate(texts):
                    cv2.putText(panel, text, (8, 22 + index * 27), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 240, 255), 1)
                cv2.imshow(WINDOW, panel)
                key = cv2.waitKey(25) & 0xFF
                if key in (27, ord('q')) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    armed = False
                    break
                if key == 32:
                    armed = False
                    status = 'Paused. An in-flight short pulse still completes and releases.'
                if key == ord('g'):
                    if args.mode != 'observe' and not halted and not busy and pending is None and allowed and age_ok:
                        armed = True
                        started = now
                        steps = 0
                        turn_steps = 0
                        status = 'Bounded episode started.'
                    else:
                        status = 'Cannot start: observe mode / pending reward / frame not ready.'
                if pending is not None and key in (ord('1'), ord('2'), ord('3'), ord('0')):
                    reward, terminal = {ord('1'): (1.0, False), ord('2'): (-2.0, True),
                                        ord('3'): (5.0, True), ord('0'): (0.0, False)}[key]
                    finish_feedback(reward, terminal)
                    allowed = False
                if not busy and not halted and pending is None:
                    if armed and allowed and age_ok:
                        action = policy.choose(observation, args.epsilon if args.mode == 'train' else 0.0)
                        if action == 'stop':
                            armed = False
                            record({'event': 'policy_stop', 'state': observation.state})
                            if args.mode == 'train':
                                pending = (observation, action, observation, f'{steps:04d}_stop')
                            status = 'Policy STOP. No touch sent.'
                            continue
                        turn_steps = turn_steps + 1 if action != 'forward' else 0
                        if turn_steps > 3:
                            armed = False
                            status = 'Three consecutive turns reached; inspect/reposition.'
                            continue
                        steps += 1
                        prefix = f'{time.time_ns()}_{steps:04d}'
                        before = observation
                        write_frame(output / f'{prefix}_before.jpg', latest.image)
                        active_transition = (prefix, before, action)
                        target = pixel_point(points[BUTTONS[action]], latest.width, latest.height)
                        record({'event': 'action_submitted', 'id': prefix, 'state': before.state,
                                'action': action, 'point': target, 'pulse_ms': args.pulse_ms})
                        previous = action
                        allowed = False
                        submit('action', lambda action=action, target=target: perform(action, target))
                    else:
                        submit('capture', lambda: capture(controller))
        finally:
            if busy and task_kind == 'action':
                print('等待短按释放。若连接异常，请现场确认小车已停。')
                try:
                    _, _, error = messages.get(timeout=45)
                    record({'event': 'exit_during_action', 'error': error, 'learned': False})
                except queue.Empty:
                    print('无法确认释放，请人工停车。')
            if pending is not None:
                record({'event': 'unrated_transition', 'id': pending[3], 'learned': False})
            policy.save()
            cv2.destroyAllWindows()
    print(f'日志：{output}；累计强化学习更新：{policy.updates}')


def main():
    parser = argparse.ArgumentParser(description='第一视角、人类奖励 Q-learning；默认只观察。')
    parser.add_argument('--mode', choices=('observe', 'train', 'drive'), default='observe')
    parser.add_argument('--dataset', type=Path, default=Path('navigation_dataset'))
    parser.add_argument('--policy', type=Path, default=Path('navigation_q_policy.json'))
    parser.add_argument('--points', type=Path, default=CONFIG_PATH)
    parser.add_argument('--logs', type=Path, default=Path('navigation_rl_logs'))
    parser.add_argument('--adb')
    parser.add_argument('--serial')
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--pulse-ms', type=int, default=100)
    parser.add_argument('--settle', type=float, default=0.5)
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--margin', type=float, default=0.12)
    parser.add_argument('--max-age', type=float, default=1.5)
    parser.add_argument('--audit', action='store_true', help='只分析数据，不连接手机')
    parser.add_argument('--check', action='store_true', help='分析一次手机画面，不发送动作')
    args = parser.parse_args()
    if not (1 <= args.steps <= 50 and 1 <= args.seconds <= 180 and 50 <= args.pulse_ms <= 200
            and 0.2 <= args.settle <= 5 and 0 <= args.epsilon <= 0.2
            and 0.05 <= args.margin <= 0.9 and 0.1 <= args.max_age <= 3):
        parser.error('参数超出允许范围；steps 1..50, seconds 1..180, pulse 50..200, epsilon 0..0.2。')
    try:
        demonstrations = Demonstrations(args.dataset)
        if args.audit:
            report = demonstrations.report()
            save_json(args.logs / 'dataset_audit.json', report)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return
        controller = PhoneController(find_adb(args.adb), args.serial)
        controller.connect()
        if args.check:
            frame = capture(controller)
            observation = demonstrations.observe(frame.image, minimum_margin=args.margin)
            print(json.dumps({**vars(observation), 'capture_seconds': frame.finished - frame.started,
                              'frame_fresh': time.monotonic() - frame.started <= args.max_age,
                              'touch_sent': False}, ensure_ascii=False, indent=2))
            return
        policy = QPolicy(args.policy, demonstrations.fingerprint)
        run_session(args, demonstrations, policy, controller)
    except Exception as error:
        parser.exit(1, f'错误：{error}\n')


if __name__ == '__main__':
    main()
