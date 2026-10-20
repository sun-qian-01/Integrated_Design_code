"""Bounded visual Q-learning with human rewards and a demonstration prior."""

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import queue
import shutil
import threading
import time

import cv2
import numpy as np

from phone_direction_control import CONFIG_PATH, PhoneController, find_adb, load_points, pixel_point
from train_navigation_model import features
from vision_navigation import capture


ACTIONS = ('forward', 'up_left', 'up_right', 'stop', 'backward')
BUTTONS = {'forward': 'up', 'up_left': 'up_left', 'up_right': 'up_right', 'backward': 'down'}
LABELS = set(ACTIONS) | {'obstacle', 'uncertain'}
WINDOW = 'Visual RL | G start | B reverse | SPACE pause | Q quit | 1 good 2 bad 3 goal 0 neutral'


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
    competing_label: str = ''
    nearest_image: str = ''
    required_margin: float = 0.12


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
        other_indices = np.flatnonzero(self.labels != label)
        competing_label = str(self.labels[other_indices[np.argmin(distances[other_indices])]]) if len(other_indices) else label
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
        return Observation(f'{index}:{previous}', label, allowed, reason, best, margin,
                           competing_label, self.filenames[index], minimum_margin)

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
            legacy = stored.get('actions') == list(ACTIONS[:-1])
            if stored.get('fingerprint') != fingerprint or not (legacy or stored.get('actions') == list(ACTIONS)):
                raise ValueError('样本或特征已改变；请用 --policy 指定新的策略文件，避免旧状态编号混用。')
            self.values = stored['q_values']
            if legacy:
                self.values = {state: values + [0.0] for state, values in self.values.items()}
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

    def choose(self, observation, epsilon=0.0, allow_backward=False):
        if not observation.allowed:
            return 'stop'
        candidates = list(ACTIONS if allow_backward else ACTIONS[:-1])
        if self.random.random() < epsilon:
            return str(self.random.choice(candidates))
        return max(candidates, key=lambda action: self.row(observation)[ACTIONS.index(action)])

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
    def __init__(self, max_age=1.5, freeze_seconds=3.0, stable_frames=3):
        self.max_age = max_age
        self.freeze_seconds = freeze_seconds
        self.shape = None
        self.last = None
        self.changed_at = None
        self.previous_label = None
        self.stable = 0
        self.stable_frames = stable_frames

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
        return self.stable >= self.stable_frames, 'ready' if self.stable >= self.stable_frames else 'need consistent fresh frames'


def write_frame(path, image):
    success, data = cv2.imencode('.jpg', image)
    if not success:
        raise ValueError('保存图像失败。')
    data.tofile(path)


def execute_logged(controller, target, action, before, observation, output, prefix,
                   pulse_ms, settle, interval_ms, source):
    metadata = {'id': prefix, 'action': action, 'source': source, 'point': target,
                'button': BUTTONS[action], 'pulse_ms': pulse_ms, 'target_interval_ms': interval_ms,
                'before_image': f'{prefix}_before.jpg', 'after_image': f'{prefix}_after.jpg',
                'before_capture_started': before.started, 'before_capture_finished': before.finished,
                'predicted_label': observation.label, 'status': 'prepared',
                'corrected_action': '', 'notes': '', 'wall_time': time.time()}
    path = output / f'{prefix}.json'
    write_frame(output / metadata['before_image'], before.image)
    save_json(path, metadata)
    try:
        metadata.update(status='dispatching', command_started=time.monotonic())
        save_json(path, metadata)
        controller.press(target, pulse_ms)
        metadata['command_finished'] = time.monotonic()
        metadata['command_elapsed_ms'] = 1000 * (metadata['command_finished'] - metadata['command_started'])
        deadline = max(before.started + interval_ms / 1000, metadata['command_finished'] + settle)
        time.sleep(max(0, deadline - time.monotonic()))
        after = capture(controller)
        write_frame(output / metadata['after_image'], after.image)
        metadata.update(status='completed', after_capture_started=after.started,
                        after_capture_finished=after.finished,
                        actual_capture_interval_ms=1000 * (after.started - before.started))
        return after
    except Exception as error:
        metadata.update(status='error', error=str(error))
        raise
    finally:
        save_json(path, metadata)
        review = output / 'review.csv'
        exists = review.exists()
        fields = ['id', 'action', 'source', 'before_image', 'after_image', 'pulse_ms',
                  'actual_capture_interval_ms', 'command_elapsed_ms', 'status', 'corrected_action', 'notes']
        with review.open('a', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            if not exists:
                writer.writeheader()
            writer.writerow(metadata)


def learn_correction(directory, model_path, output, prefix, action, before, old):
    if action not in ACTIONS:
        raise ValueError('人工动作标签无效。')
    filename = f'correction_{prefix}_{action}.jpg'
    labels_path = directory / 'labels.csv'
    with labels_path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    if not fields or not {'filename', 'label'}.issubset(fields):
        raise ValueError('标签表缺少 filename/label。')
    if not any(row['filename'] == filename for row in rows):
        shutil.copyfile(output / f'{prefix}_before.jpg', directory / filename)
        row = {'filename': filename, 'label': action, 'width': before.width,
               'height': before.height, 'capture_seconds': before.finished - before.started,
               'timestamp': datetime.now().isoformat()}
        with labels_path.open('a', encoding='utf-8', newline='') as stream:
            csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore').writerow(row)
    updated = Demonstrations(directory)
    length = len(old.filenames)
    if (updated.filenames[:length] != old.filenames or
            not np.array_equal(updated.labels[:length], old.labels) or
            not np.array_equal(updated.vectors[:length], old.vectors)):
        raise ValueError('原始样本被修改，不能在线迁移 Q 状态，请重启并使用新策略。')
    save_json(model_path, {'version': 1, 'feature_size': updated.vectors.shape[1],
                          'classes': sorted(updated.counts),
                          'means': {label: mean.tolist() for label, mean in updated.centroids.items()},
                          'counts': updated.counts, 'warning': 'human corrections included; not independently validated'})
    save_json(output / f'{prefix}_learning.json', {'filename': filename, 'label': action,
              'source': 'manual_command', 'fingerprint': updated.fingerprint,
              'model': str(model_path.resolve())})
    return updated


def run_session(args, demonstrations, policy, controller):
    if not args.points.is_file():
        raise ValueError('缺少已校准的方向坐标文件。')
    points = load_points(args.points)
    output = args.logs / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    output.mkdir(parents=True)
    messages = queue.Queue()
    guard = FrameGuard(args.max_age, stable_frames=args.stable_frames)
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
    manual_action = None
    start_requested = False
    last_guard_reason = None
    next_capture = 0.0

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

    def perform_and_learn(action, target, prefix, before_frame, before, source):
        after = execute_logged(controller, target, action, before_frame, before, output, prefix,
                               args.pulse_ms, args.settle, args.interval_ms, source)
        if source != 'manual':
            return after
        updated = learn_correction(args.dataset, args.model_output, output, prefix,
                                   action, before_frame, demonstrations)
        return after, updated

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
                        if isinstance(result, tuple):
                            result, updated = result
                            demonstrations = updated
                            policy.fingerprint = updated.fingerprint
                            policy.save()
                            print(f'人工示范已加入训练并生效；共 {len(updated.filenames)} 张，'
                                  f'模型已保存到 {args.model_output}', flush=True)
                            record({'event': 'manual_learning', 'id': active_transition[0],
                                    'action': active_transition[2], 'samples': len(updated.filenames),
                                    'fingerprint': updated.fingerprint})
                        latest = result
                        observation = demonstrations.observe(latest.image, previous, args.margin)
                        allowed, reason = guard.inspect(latest, observation, time.monotonic())
                        if reason != last_guard_reason:
                            record({'event': 'decision_gate', 'allowed': allowed, 'reason': reason,
                                    'label': observation.label, 'margin': observation.margin,
                                    'capture_seconds': latest.finished - latest.started,
                                    'competing_label': observation.competing_label,
                                    'nearest_image': observation.nearest_image,
                                    'required_margin': observation.required_margin})
                            print(f'判断：{observation.label}；竞争标签：{observation.competing_label}；'
                                  f'区分度 {observation.margin:.3f}/{observation.required_margin:.3f}；'
                                  f'状态：{reason}；参考样本：{observation.nearest_image}；'
                                  f'截图 {latest.finished - latest.started:.2f}s', flush=True)
                            last_guard_reason = reason
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
                         'G auto | W/A/D/B manual pulse | SPACE cancel | Q quit | 1/2/3/0 reward']
                if observation:
                    texts.append(f'label={observation.label} distance={observation.distance:.3f} margin={observation.margin:.2f} (NOT probability)')
                for index, text in enumerate(texts):
                    cv2.putText(panel, text, (8, 22 + index * 27), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 240, 255), 1)
                cv2.imshow(WINDOW, panel)
                key = cv2.waitKey(25) & 0xFF
                if ord('A') <= key <= ord('Z'):
                    key += 32
                if key in (27, ord('q')) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    armed = False
                    break
                if key == 32:
                    armed = False
                    manual_action = None
                    start_requested = False
                    status = 'Paused. An in-flight short pulse still completes and releases.'
                if key in (ord('w'), ord('a'), ord('d'), ord('b')):
                    if args.mode != 'observe' and not halted and pending is None and not (busy and task_kind == 'action'):
                        manual_action = {ord('w'): 'forward', ord('a'): 'up_left', ord('d'): 'up_right', ord('b'): 'backward'}[key]
                        start_requested = False
                        armed = False
                        status = f'Manual {manual_action} requested; waiting for capture.'
                        record({'event': 'manual_requested', 'action': manual_action})
                    else:
                        status = 'Manual unavailable: observe mode / pending reward / action in flight.'
                    print(status, flush=True)
                if key == ord('g'):
                    if args.mode != 'observe' and not halted and pending is None:
                        start_requested = True
                        manual_action = None
                        status = 'Start requested; waiting for capture and decision gate.'
                        record({'event': 'start_requested'})
                    else:
                        status = 'Cannot start: observe mode / pending reward / error.'
                    print(status, flush=True)
                if pending is not None and key in (ord('1'), ord('2'), ord('3'), ord('0')):
                    reward, terminal = {ord('1'): (1.0, False), ord('2'): (-2.0, True),
                                        ord('3'): (5.0, True), ord('0'): (0.0, False)}[key]
                    finish_feedback(reward, terminal)
                    allowed = False
                if not busy and not halted and pending is None:
                    if start_requested and latest is not None:
                        start_requested = False
                        if allowed and age_ok:
                            armed = True
                            started = now
                            steps = 0
                            turn_steps = 0
                            status = 'Bounded episode started.'
                        else:
                            status = f'Start blocked: {last_guard_reason}; use W/A/D/B for manual correction.'
                            record({'event': 'start_blocked', 'reason': last_guard_reason, 'frame_fresh': age_ok})
                        print(status, flush=True)
                    if manual_action is not None and (not age_ok or latest is None or (latest.width, latest.height) != guard.shape):
                        manual_action = None
                        print('手动动作取消：截图过期或尺寸改变，请重新请求。', flush=True)
                    if (armed and allowed and age_ok) or manual_action is not None:
                        source = 'manual' if manual_action is not None else 'policy'
                        action = manual_action if manual_action is not None else policy.choose(
                            observation, args.epsilon if args.mode == 'train' else 0.0,
                            allow_backward=args.allow_backward)
                        manual_action = None
                        if action == 'stop':
                            armed = False
                            stop_id = f'{time.time_ns()}_stop'
                            write_frame(output / f'{stop_id}_before.jpg', latest.image)
                            record({'event': 'policy_stop', 'id': stop_id, 'state': observation.state,
                                    'action': 'stop', 'before_image': f'{stop_id}_before.jpg', 'touch_sent': False})
                            if args.mode == 'train':
                                pending = (observation, action, observation, stop_id)
                            status = 'Policy STOP. No touch sent.'
                            continue
                        turn_steps = turn_steps + 1 if action in ('up_left', 'up_right') else 0
                        if source == 'policy' and turn_steps > 3:
                            armed = False
                            status = 'Three consecutive turns reached; inspect/reposition.'
                            continue
                        steps += 1
                        prefix = f'{time.time_ns()}_{steps:04d}'
                        before = observation
                        active_transition = (prefix, before, action)
                        target = pixel_point(points[BUTTONS[action]], latest.width, latest.height)
                        record({'event': 'action_submitted', 'id': prefix, 'state': before.state,
                                'action': action, 'point': target, 'pulse_ms': args.pulse_ms})
                        print(f'发送指令：{action} → {BUTTONS[action]}，{args.pulse_ms} ms，坐标 {target}', flush=True)
                        previous = action
                        allowed = False
                        before_frame = latest
                        submit('action', lambda action=action, target=target, prefix=prefix,
                               before_frame=before_frame, before=before, source=source: perform_and_learn(
                                   action, target, prefix, before_frame, before, source))
                    elif now >= next_capture:
                        next_capture = now + args.interval_ms / 1000
                        submit('capture', lambda: capture(controller))
        finally:
            if busy and task_kind == 'action':
                print('等待短按释放。若连接异常，请现场确认小车已停。')
                try:
                    _, result, error = messages.get(timeout=45)
                    if error is None and isinstance(result, tuple):
                        policy.fingerprint = result[1].fingerprint
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
    parser.add_argument('--mode', choices=('observe', 'train', 'drive', 'trial', 'showcase'), default='observe')
    parser.add_argument('--trial-policy', type=Path, default=Path('navigation_trial_policy.json'))
    parser.add_argument('--track-model', type=Path, default=Path('navigation_track_model.json'), help='trial/showcase 赛道像素模型')
    parser.add_argument('--no-track-guard', action='store_true', help='禁用赛道边界约束，保留原策略')
    parser.add_argument('--turn-ms', type=int, default=150, help='启用赛道约束时转弯/后退按压上限，默认 150ms')
    parser.add_argument('--dataset', type=Path, default=Path('navigation_dataset'))
    parser.add_argument('--model-output', type=Path, default=Path('navigation_model.json'))
    parser.add_argument('--policy', type=Path, default=Path('navigation_q_policy.json'))
    parser.add_argument('--points', type=Path, default=CONFIG_PATH)
    parser.add_argument('--logs', type=Path, default=Path('navigation_rl_logs'))
    parser.add_argument('--adb')
    parser.add_argument('--serial')
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--pulse-ms', type=int, default=500)
    parser.add_argument('--settle', type=float, default=0.0)
    parser.add_argument('--interval-ms', type=int, help='目标截图起始间隔；默认等于 pulse-ms，不排队补帧')
    parser.add_argument('--stable-frames', type=int, default=1)
    parser.add_argument('--allow-backward', action='store_true', help='允许策略自动选择后退；B 单步后退不需要此选项')
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--margin', type=float, default=0.12)
    parser.add_argument('--max-age', type=float, default=1.5)
    parser.add_argument('--audit', action='store_true', help='只分析数据，不连接手机')
    parser.add_argument('--check', action='store_true', help='分析一次手机画面，不发送动作')
    args = parser.parse_args()
    if not 50 <= args.turn_ms <= 500:
        parser.error('turn-ms 应为 50..500。')
    if args.interval_ms is None:
        args.interval_ms = args.pulse_ms
    if not 50 <= args.interval_ms <= 5000 or not 1 <= args.stable_frames <= 10:
        parser.error('interval-ms 应为 50..5000，stable-frames 应为 1..10。')
    if not (1 <= args.steps <= 50 and 1 <= args.seconds <= 180 and 50 <= args.pulse_ms <= 500
            and 0 <= args.settle <= 5 and 0 <= args.epsilon <= 0.2
            and 0.05 <= args.margin <= 0.9 and 0.1 <= args.max_age <= 3):
        parser.error('参数超出允许范围；steps 1..50, seconds 1..180, pulse 50..500, epsilon 0..0.2。')
    try:
        demonstrations = Demonstrations(args.dataset)
        print(f'读取标注数据：{args.dataset.resolve()}；类别数量：{demonstrations.counts}', flush=True)
        print(f'按压 {args.pulse_ms} ms；目标采样间隔 {args.interval_ms} ms。'
              '当前使用样本近邻和重新计算的类别中心，不加载 navigation_model.json。', flush=True)
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
        if args.mode in ('trial', 'showcase'):
            from trial_navigation import run_trial
            run_trial(args, demonstrations, controller)
            return
        policy = QPolicy(args.policy, demonstrations.fingerprint)
        print(f'Q 策略：{args.policy.resolve()}；奖励更新次数：{policy.updates}；'
              f'已记录状态：{len(policy.values)}。人工 W/A/D/B 指令会加入示范训练；自动指令不会自我标注。', flush=True)
        run_session(args, demonstrations, policy, controller)
    except Exception as error:
        parser.exit(1, f'错误：{error}\n')


if __name__ == '__main__':
    main()
