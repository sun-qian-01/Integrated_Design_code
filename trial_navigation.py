"""One physical trial per human feedback, with persistent anchored Q states."""

from dataclasses import dataclass
from datetime import datetime
import json
import queue
import threading
import time

import cv2
import numpy as np

from phone_direction_control import load_points, pixel_point
from rl_navigation import ACTIONS, BUTTONS, execute_logged, learn_correction, save_json, write_frame
from train_navigation_model import features
from vision_navigation import capture


WINDOW = 'Trial learning | G start | 1 good 2 bad 3 goal 0 neutral | W/A/D/B/S correct | SPACE pause Q quit'
CORRECTIONS = {'w': 'forward', 'a': 'up_left', 'd': 'up_right', 'b': 'backward', 's': 'stop'}


class TrialPolicy:
    def __init__(self, path, demonstrations):
        self.path = path
        self.random = np.random.default_rng()
        self.values = {}
        self.updates = 0
        self.corrections = 0
        self.anchors = demonstrations.vectors.copy()
        if path.exists():
            stored = json.loads(path.read_text(encoding='utf-8'))
            if stored.get('version') != 1 or stored.get('actions') != list(ACTIONS):
                raise ValueError('试走策略版本不兼容，请另指定 --trial-policy。')
            self.anchors = np.asarray(stored['anchors'], dtype=np.float32)
            self.values = stored['q_values']
            self.updates = stored['updates']
            self.corrections = stored['corrections']
        if self.anchors.ndim != 2 or self.anchors.shape[1] != demonstrations.vectors.shape[1] or not np.isfinite(self.anchors).all():
            raise ValueError('试走策略的固定图像特征无效。')
        if any(len(row) != len(ACTIONS) or not np.isfinite(row).all() for row in self.values.values()):
            raise ValueError('试走策略 Q 值无效。')

    def state(self, image):
        return str(int(np.argmin(np.linalg.norm(self.anchors - features(image), axis=1))))

    def row(self, state, suggestion, remember=True):
        if state not in self.values:
            values = [0.0] * len(ACTIONS)
            proposed = suggestion if suggestion in ACTIONS else 'forward'
            values[ACTIONS.index(proposed)] = 0.2
            if not remember:
                return values
            self.values[state] = values
        return self.values[state]

    def choose(self, state, observation, epsilon, allow_backward, remember=True):
        row = self.row(state, observation.label, remember)
        if observation.label in ('stop', 'obstacle'):
            return 'stop', 'explicit stop/obstacle label; request feedback'
        candidates = list(ACTIONS if allow_backward else ACTIONS[:-1])
        if self.random.random() < epsilon:
            return str(self.random.choice(candidates)), 'bounded epsilon exploration'
        return max(candidates, key=lambda action: row[ACTIONS.index(action)]), 'highest Q with demonstration prior'

    def update(self, state, action, reward, next_state, terminal, next_suggestion):
        row = self.row(state, action)
        future = 0.0 if terminal else max(self.row(next_state, next_suggestion))
        row[ACTIONS.index(action)] += 0.25 * (reward + 0.9 * future - row[ACTIONS.index(action)])
        self.updates += 1
        self.save()

    def correct(self, state, attempted, corrected):
        row = self.row(state, attempted)
        if corrected != attempted:
            row[ACTIONS.index(attempted)] -= 0.5
        row[ACTIONS.index(corrected)] = max(row) + 0.25
        self.corrections += 1
        self.save()

    def save(self):
        save_json(self.path, {'version': 1, 'actions': list(ACTIONS), 'anchors': self.anchors.tolist(),
                             'q_values': self.values, 'updates': self.updates, 'corrections': self.corrections})


@dataclass
class Trial:
    identifier: str
    before: object
    observation: object
    state: str
    action: str
    after: object = None


class TrialCycle:
    def __init__(self, max_age):
        self.max_age = max_age
        self.shape = None
        self.pending = None

    def ready(self, frame, now):
        if self.pending is not None:
            return False, 'feedback required'
        if frame is None:
            return False, 'waiting for frame'
        shape = (frame.width, frame.height)
        if frame.width <= frame.height:
            return False, 'phone must be landscape'
        if self.shape is None:
            self.shape = shape
        if shape != self.shape:
            return False, 'screen geometry changed; restart'
        if now - frame.started > self.max_age:
            return False, 'stale frame; waiting for new capture'
        return True, 'fresh frame; ambiguity allowed in trial mode'

    def begin(self, trial, now):
        ready, reason = self.ready(trial.before, now)
        if not ready:
            raise ValueError(reason)
        self.pending = trial

    def complete(self, frame):
        self.pending.after = frame

    def consume(self):
        if self.pending is None or self.pending.after is None:
            raise ValueError('没有已完成、等待评价的动作。')
        result = self.pending
        self.pending = None
        return result


def run_trial(args, demonstrations, controller):
    if not args.points.is_file():
        raise ValueError('需要已校准的方向坐标。')
    points = load_points(args.points)
    policy = TrialPolicy(args.trial_policy, demonstrations)
    track = None
    track_path = getattr(args, 'track_model', None)
    if track_path is not None and not getattr(args, 'no_track_guard', False):
        from track_boundary import TrackBoundary
        if not track_path.is_file():
            raise ValueError('缺少赛道模型，请先运行 track_boundary.py train，或用 --no-track-guard 运行原策略。')
        track = TrackBoundary.load(track_path)
        print(f'已启用赛道边界约束：{track_path}；转弯最多 {args.turn_ms} ms，直行最多 250 ms。', flush=True)
    autonomous = getattr(args, 'mode', 'trial') == 'showcase'
    persist_training = not autonomous
    window = 'Navigation | AUTO start | TRIAL start | PAUSE | Q quit'
    clicks = queue.Queue()

    def mouse_click(event, horizontal, vertical, flags, userdata):
        if event == cv2.EVENT_LBUTTONUP and button_top[0] <= vertical < button_top[0] + 42:
            if 10 <= horizontal < 270:
                clicks.put('auto')
            elif 280 <= horizontal < 540:
                clicks.put('trial')
            elif 550 <= horizontal < 810:
                clicks.put('pause')

    button_top = [0]
    cycle = TrialCycle(args.max_age)
    output = args.logs / (('showcase-' if autonomous else 'trial-') + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    output.mkdir(parents=True)
    messages = queue.Queue()
    busy = None
    active = False
    failed = False
    steps = 0
    episode_started = None
    latest = None
    next_capture = 0.0
    status = 'Click AUTO / TRIAL to start. G starts the selected mode.'
    print('AUTO 展示：使用已训练策略连续决策，无随机探索、不等待评分、不更新训练文件。'
          'TRIAL：每步等待评价或纠正。底部按钮开始，空格暂停。', flush=True)
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.imshow(window, np.zeros((540, 960, 3), dtype=np.uint8))
    cv2.setMouseCallback(window, mouse_click)

    def submit(kind, operation):
        nonlocal busy
        busy = kind

        def worker():
            try:
                messages.put((kind, operation(), None))
            except Exception as error:
                messages.put((kind, None, str(error)))

        threading.Thread(target=worker, daemon=True).start()

    def correction_task(trial, action):
        updated = learn_correction(args.dataset, args.model_output, output, trial.identifier,
                                   action, trial.before, demonstrations)
        return updated, trial, action

    with (output / 'transitions.jsonl').open('a', encoding='utf-8') as stream:
        def record(event, **data):
            stream.write(json.dumps({'time': time.time(), 'event': event, **data}, ensure_ascii=False) + '\n')
            stream.flush()

        def apply_correction(result):
            nonlocal demonstrations
            updated, trial, corrected = result
            demonstrations = updated
            policy.correct(trial.state, trial.action, corrected)
            save_json(output / f'{trial.identifier}_correction.json', {
                'before_image': f'{trial.identifier}_before.jpg', 'attempted': trial.action,
                'corrected_action': corrected, 'state': trial.state})
            record('correction', id=trial.identifier, action=trial.action, corrected_action=corrected,
                   corrections=policy.corrections, samples=len(updated.filenames))

        try:
            while True:
                try:
                    kind, result, error = messages.get_nowait()
                except queue.Empty:
                    kind = None
                if kind:
                    busy = None
                    if error:
                        active = False
                        failed = True
                        status = f'ERROR: {error}; inspect vehicle, then restart.'
                        record('error', operation=kind, message=error)
                        print(status, flush=True)
                    elif kind == 'correction':
                        apply_correction(result)
                        cycle.consume()
                        latest = None
                        status = 'Correction learned. Taking NEW image before next choice.'
                        print(status, flush=True)
                    else:
                        latest = result
                        if kind == 'action':
                            cycle.complete(latest)
                            status = 'Released. Rate: 1 good / 2 bad / 3 goal / 0 neutral; W/A/D/B/S correct BEFORE image.'
                            record('action_completed', id=cycle.pending.identifier, action=cycle.pending.action)
                            if autonomous:
                                completed = cycle.consume()
                                record('showcase_step', id=completed.identifier, action=completed.action, learned=False)
                                status = f'AUTO: {completed.action} completed; deciding from the new frame.'
                            print(status, flush=True)
                now = time.monotonic()
                if active and (steps >= args.steps or now - episode_started >= args.seconds):
                    active = False
                    record('episode_limit', steps=steps)
                    status = 'Episode limit reached. Paused. Start again for a new bounded run.'
                panel = latest.image.copy() if latest is not None else np.zeros((540, 960, 3), dtype=np.uint8)
                if track is not None and latest is not None:
                    panel = track.overlay(panel)
                waiting = cycle.pending is not None and cycle.pending.after is not None
                if waiting:
                    before_image = cycle.pending.before.image
                    after_image = cycle.pending.after.image
                    before_view = cv2.resize(before_image, (480, 216))
                    after_view = cv2.resize(after_image, (480, 216))
                    panel = np.zeros((380, 960, 3), dtype=np.uint8)
                    panel[150:366, :480] = before_view
                    panel[150:366, 480:] = after_view
                    cv2.putText(panel, f'BEFORE: correct this decision ({cycle.pending.action})',
                                (8, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                    cv2.putText(panel, 'AFTER: evaluate the result', (488, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                cv2.rectangle(panel, (0, 0), (panel.shape[1], 118), (15, 15, 15), -1)
                lines = [f'{"AUTO SHOWCASE" if autonomous else "TRIAL"} | {"WAIT FEEDBACK" if waiting else "ACTIVE" if active else "PAUSED"} | steps {steps}/{args.steps}',
                         status[:115], f'Q rewards={policy.updates} corrections={policy.corrections} | pulse cap={args.pulse_ms}ms | track={track is not None}',
                         'G start | 1/2/3/0 reward | W/A/D/B/S correct previous decision | SPACE pause | Q quit']
                for index, text in enumerate(lines):
                    cv2.putText(panel, text, (8, 22 + index * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 240, 255), 1)
                button_top[0] = panel.shape[0] + 5
                display = np.zeros((panel.shape[0] + 54, 960, 3), dtype=np.uint8)
                display[:panel.shape[0], :panel.shape[1]] = panel
                for left, text, color in ((10, 'AUTO: start showcase', (65, 125, 40)),
                                          (280, 'TRIAL: start training', (130, 85, 35)),
                                          (550, 'PAUSE', (45, 45, 145))):
                    cv2.rectangle(display, (left, button_top[0]), (left + 260, button_top[0] + 41), color, -1)
                    cv2.putText(display, text, (left + 10, button_top[0] + 27),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1)
                cv2.imshow(window, display)
                key = cv2.waitKey(25) & 0xFF
                requested_mode = None
                try:
                    click = clicks.get_nowait()
                    if click == 'pause':
                        key = 32
                    else:
                        requested_mode = click == 'auto'
                        key = ord('g')
                except queue.Empty:
                    pass
                if ord('A') <= key <= ord('Z'):
                    key += 32
                if key in (27, ord('q')) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    active = False
                    break
                if key == 32:
                    active = False
                    status = 'Paused; current pulse finishes and releases. Feedback still accepted.'
                if key == ord('g') and not failed and cycle.pending is None and busy != 'correction':
                    if requested_mode is not None:
                        autonomous = requested_mode
                        persist_training = persist_training or not autonomous
                    active = True
                    steps = 0
                    episode_started = now
                    record('episode_start', mode='showcase' if autonomous else 'trial',
                           updates=policy.updates, corrections=policy.corrections)
                    status = 'Start queued; waiting for a fresh image.'
                elif key == ord('g') and cycle.pending is not None:
                    status = 'Finish feedback / current action before starting another mode.'
                if waiting and busy != 'correction':
                    trial = cycle.pending
                    if key in tuple(map(ord, '1230')):
                        reward, terminal = {'1': (1.0, False), '2': (-2.0, True),
                                            '3': (5.0, True), '0': (0.0, False)}[chr(key)]
                        after_observation = demonstrations.observe(trial.after.image, trial.action, args.margin)
                        terminal = terminal or after_observation.label in ('stop', 'obstacle')
                        policy.update(trial.state, trial.action, reward, policy.state(trial.after.image),
                                      terminal, after_observation.label)
                        record('reward', id=trial.identifier, action=trial.action, reward=reward,
                               terminal=terminal, updates=policy.updates, source='human')
                        cycle.consume()
                        latest = None
                        if terminal:
                            active = False
                        status = f'Reward {reward:+g} learned. New image before next decision.'
                        print(status, flush=True)
                    elif chr(key) in CORRECTIONS:
                        corrected = CORRECTIONS[chr(key)]
                        status = f'Learning {corrected} for the BEFORE image. No immediate movement.'
                        submit('correction', lambda trial=trial, corrected=corrected: correction_task(trial, corrected))
                if not busy and not failed and cycle.pending is None:
                    ready, reason = cycle.ready(latest, time.monotonic())
                    if active and ready:
                        observation = demonstrations.observe(latest.image, minimum_margin=args.margin)
                        state = policy.state(latest.image)
                        action, choice_reason = policy.choose(state, observation, 0.0 if autonomous else args.epsilon,
                                                              args.allow_backward, remember=not autonomous)
                        pulse_ms = args.pulse_ms
                        track_decision = None
                        if track is not None:
                            action, pulse_ms, track_decision = track.constrain(latest.image, action, pulse_ms, args.turn_ms)
                            choice_reason += f'; track={track_decision["suggestion"]}, executed={action}, pulse={pulse_ms}ms'
                        trial = Trial(str(time.time_ns()), latest, observation, state, action)
                        cycle.begin(trial, time.monotonic())
                        steps += 1
                        record('proposal', id=trial.identifier, state=state, action=action,
                               suggested=observation.label, ambiguity=observation.reason,
                               reason=choice_reason, q_values=list(policy.row(state, observation.label, remember=not autonomous)),
                               track=track_decision, pulse_ms=pulse_ms,
                               mode='showcase' if autonomous else 'trial')
                        print(f'自主选择：{action}；原因：{choice_reason}；原分类：{observation.reason}', flush=True)
                        if action == 'stop':
                            write_frame(output / f'{trial.identifier}_before.jpg', latest.image)
                            cycle.complete(latest)
                            record('policy_stop', id=trial.identifier, touch_sent=False)
                            status = 'Policy stop: rate it or W/A/D/B/S correct it; no touch sent.'
                            if autonomous:
                                cycle.consume()
                                active = False
                                status = 'AUTO stopped by policy/obstacle label. Inspect scene before restarting.'
                        else:
                            target = pixel_point(points[BUTTONS[action]], latest.width, latest.height)
                            record('action_submitted', id=trial.identifier, action=action,
                                   point=target, pulse_ms=pulse_ms, source='policy')
                            submit('action', lambda trial=trial, target=target, pulse_ms=pulse_ms: execute_logged(
                                controller, target, trial.action, trial.before, trial.observation,
                                output, trial.identifier, pulse_ms, args.settle, min(args.interval_ms, pulse_ms), 'policy'))
                    elif now >= next_capture:
                        if active:
                            status = reason
                        next_capture = now + args.interval_ms / 1000
                        submit('capture', lambda: capture(controller))
        finally:
            if busy in ('action', 'correction'):
                print('等待当前任务完成；不会追加运动指令。', flush=True)
                try:
                    kind, result, error = messages.get(timeout=45)
                    if not error and kind == 'correction':
                        apply_correction(result)
                        cycle.consume()
                    record('exit_during_task', operation=kind, error=error)
                except queue.Empty:
                    record('exit_timeout', operation=busy)
            if cycle.pending is not None:
                record('unrated_transition', id=cycle.pending.identifier, learned=False)
            if persist_training:
                policy.save()
            cv2.destroyAllWindows()
    print(f'试走记录：{output}；Q 奖励更新 {policy.updates}，人工纠正 {policy.corrections}', flush=True)
