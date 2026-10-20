"""Calibrated floor segmentation and supervised, bounded local navigation."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import queue
import subprocess
import threading
import time

import cv2
import numpy as np

from phone_direction_control import CONFIG_PATH, DIRECTIONS, PhoneController, find_adb, load_points, pixel_point


PROFILE_PATH = Path(__file__).with_name('vision_profile.json')
VIEW_WIDTH = 960


@dataclass
class Frame:
    image: np.ndarray
    width: int
    height: int
    started: float
    finished: float


def capture(controller):
    video = controller.video_frame()
    width, height = video.width, video.height
    if width <= height:
        raise ValueError('当前手机不是横屏，请先打开小车控制界面。')
    image = video.image
    image = cv2.resize(image, (VIEW_WIDTH, round(height * VIEW_WIDTH / width)))
    return Frame(image, width, height, video.received, time.monotonic())


def crop(image, rectangle):
    horizontal, vertical, width, height = map(int, rectangle)
    return image[vertical:vertical + height, horizontal:horizontal + width]


def select_rectangle(title, image):
    rectangle = tuple(map(int, cv2.selectROI(title, image, showCrosshair=True)))
    cv2.destroyWindow(title)
    if rectangle[2] < 10 or rectangle[3] < 10:
        raise ValueError('选区过小或已取消，请重新校准。')
    return rectangle


def calibrate(controller, destination):
    frame = capture(controller)
    road = select_rectangle('1 Select road ROI: drag, ENTER accept, ESC cancel', frame.image)
    scene = crop(frame.image, road)
    exclusions = cv2.selectROIs('2 Mask UI inside ROI: drag ENTER each, ESC finish', scene)
    cv2.destroyAllWindows()
    sample = select_rectangle('3 Select a clean FLOOR patch (not obstacle)', scene)
    anchor = select_rectangle('4 Select fixed APP icon outside live video', frame.image)
    lab = cv2.cvtColor(crop(scene, sample), cv2.COLOR_BGR2LAB).astype(np.float32)
    median = np.median(lab.reshape(-1, 3), axis=0)
    profile = {
        'screen_size': [frame.width, frame.height],
        'road_roi': list(road),
        'exclude_roi': [list(map(int, rectangle)) for rectangle in exclusions],
        'floor_lab': median.tolist(),
        'color_tolerance': 28.0,
        'anchor_roi': list(anchor),
        'anchor_gray': cv2.cvtColor(crop(frame.image, anchor), cv2.COLOR_BGR2GRAY).tolist(),
        'anchor_error_limit': 16.0,
        'clear_ratio': 0.88,
        'near_clear_ratio': 0.95,
        'max_frame_age_s': 1.5,
        'freeze_s': 3.0,
        'pulse_ms': 100,
        'settle_s': 0.5,
        'max_steps': 20,
        'max_turns': 3,
        'controls_verified': False,
        'release_stops_verified': False,
        'actions': {'forward': 'up', 'turn_left': 'up_left', 'turn_right': 'up_right'},
    }
    destination.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'已保存 {destination}。预填上/左上/右上；需确认前进和松开停车行为，自动执行关闭。')


def validate_profile(profile):
    for key in ('road_roi', 'anchor_roi'):
        rectangle = profile[key]
        if len(rectangle) != 4 or any(type(value) is not int for value in rectangle):
            raise ValueError(f'{key} 必须为四个整数。')
        if min(rectangle[:2]) < 0 or min(rectangle[2:]) < 10:
            raise ValueError(f'{key} 选区无效。')
    if len(profile['floor_lab']) != 3 or not all(0 <= value <= 255 for value in profile['floor_lab']):
        raise ValueError('地面颜色无效。')
    limits = {
        'color_tolerance': (1, 100), 'clear_ratio': (0.5, 1),
        'near_clear_ratio': (0.5, 1), 'max_frame_age_s': (0.1, 3),
        'freeze_s': (0.5, 10), 'pulse_ms': (50, 200),
        'settle_s': (0.2, 5), 'max_steps': (1, 100), 'max_turns': (1, 5),
        'anchor_error_limit': (0, 40),
    }
    for key, (lower, upper) in limits.items():
        if not lower <= profile[key] <= upper:
            raise ValueError(f'{key} 必须在 {lower}～{upper} 之间。')


def validate_drive(profile, points_path):
    if not points_path.is_file():
        raise ValueError('请先用八方向面板校准并保存触控坐标。')
    if profile['controls_verified'] is not True or profile['release_stops_verified'] is not True:
        raise ValueError('需要先验证方向含义和松开即停车，再更新 profile 中对应字段。')
    actions = profile['actions']
    if any(actions.get(name) not in DIRECTIONS for name in ('forward', 'turn_left', 'turn_right')):
        raise ValueError('请配置 forward、turn_left、turn_right 的真实按钮映射。')
    if len(set(actions.values())) != 3:
        raise ValueError('前进、左转、右转不能映射到同一按钮。')


def analyze(image, profile):
    scene = crop(image, profile['road_roi'])
    height, width = scene.shape[:2]
    expected_width, expected_height = profile['road_roi'][2:]
    if (width, height) != (expected_width, expected_height):
        raise ValueError('道路选区超出画面，请重新校准。')
    valid = np.full((height, width), 255, dtype=np.uint8)
    for rectangle in profile['exclude_roi']:
        crop(valid, rectangle)[:] = 0
    lab = cv2.cvtColor(scene, cv2.COLOR_BGR2LAB).astype(np.float32)
    distance = np.linalg.norm(lab - np.array(profile['floor_lab'], dtype=np.float32), axis=2)
    floor = ((distance <= profile['color_tolerance']) & (valid > 0)).astype(np.uint8)
    floor = cv2.morphologyEx(floor, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels = cv2.connectedComponents(floor, connectivity=8)
    seeds = np.unique(labels[max(0, height - 8):, width // 3:2 * width // 3])
    seeds = seeds[seeds != 0]
    connected = np.isin(labels, seeds) if count > 1 and seeds.size else np.zeros_like(floor, dtype=bool)
    scores = {}
    near_scores = {}
    for index, name in enumerate(('left', 'center', 'right')):
        start, finish = index * width // 3, (index + 1) * width // 3
        band_valid = valid[:, start:finish] > 0
        band_floor = connected[:, start:finish]
        scores[name] = float(band_floor.sum() / max(1, band_valid.sum())) if band_valid.mean() >= 0.7 else 0.0
        near_valid = band_valid[height * 2 // 3:]
        near_floor = band_floor[height * 2 // 3:]
        near_scores[name] = float(near_floor.sum() / max(1, near_valid.sum())) if near_valid.mean() >= 0.7 else 0.0
    anchor = cv2.cvtColor(crop(image, profile['anchor_roi']), cv2.COLOR_BGR2GRAY)
    reference = np.array(profile['anchor_gray'], dtype=np.uint8)
    anchor_error = float(np.mean(np.abs(anchor.astype(float) - reference.astype(float)))) if anchor.shape == reference.shape else float('inf')
    return scene, connected, valid, scores, near_scores, anchor_error


def propose(scores, near_scores, profile):
    if near_scores['center'] < profile['near_clear_ratio']:
        return 'stop', 'near field uncertain/blocked'
    if scores['center'] >= profile['clear_ratio']:
        return 'forward', 'center resembles floor'
    choices = [name for name in ('left', 'right') if scores[name] >= profile['clear_ratio']
               and near_scores[name] >= profile['near_clear_ratio']]
    if not choices:
        return 'stop', 'no clear corridor'
    if len(choices) == 2 and abs(scores['left'] - scores['right']) < 0.05:
        return 'stop', 'ambiguous turn'
    direction = max(choices, key=lambda name: scores[name])
    return f'turn_{direction}', 'side corridor resembles floor'


class DecisionGate:
    def __init__(self):
        self.last_gray = None
        self.last_change = None
        self.candidate = None
        self.stable = 0

    def reset_consensus(self):
        self.candidate = None
        self.stable = 0

    def evaluate(self, frame, scene, action, anchor_error, profile, now):
        if (frame.width, frame.height) != tuple(profile['screen_size']):
            self.reset_consensus()
            return 'stop', 'screen size changed'
        if now - frame.started > profile['max_frame_age_s']:
            self.reset_consensus()
            return 'stop', 'capture too old/slow'
        if anchor_error > profile['anchor_error_limit']:
            self.reset_consensus()
            return 'stop', 'app anchor changed'
        gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
        changed = self.last_gray is None or np.mean(np.abs(gray.astype(float) - self.last_gray.astype(float))) > 0.08
        self.last_gray = gray
        if changed:
            self.last_change = now
        if self.last_change is not None and now - self.last_change > profile['freeze_s']:
            self.reset_consensus()
            return 'stop', 'scene unchanged: possible frozen feed'
        self.stable = self.stable + 1 if action == self.candidate else 1
        self.candidate = action
        if action == 'stop':
            return 'stop', 'blocked/uncertain'
        if self.stable < 3:
            return 'wait', 'need 3 consistent frames'
        return action, 'consistent local proposal'


class FrameSource:
    def __init__(self, controller):
        self.controller = controller
        self.messages = queue.Queue(maxsize=1)
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self.worker, daemon=True)
        self.thread.start()

    def worker(self):
        while not self.closed.is_set():
            try:
                result = capture(self.controller)
            except Exception as error:
                result = error
            try:
                self.messages.get_nowait()
            except queue.Empty:
                pass
            self.messages.put(result)
            if isinstance(result, Exception):
                return
            self.closed.wait(0.05)


def monitor(controller, profile, points_path, drive, log_path):
    if drive:
        validate_drive(profile, points_path)
    points = load_points(points_path)
    gate = DecisionGate()
    source = FrameSource(controller)
    actions_done = queue.Queue()
    armed = False
    busy = False
    steps = 0
    turns = 0
    fresh_after = 0.0
    latest_receipt = time.monotonic()
    latest_action = 'wait'
    failed = False
    panel = None
    window = 'Local navigation | G arm | SPACE pause | Q quit (observe by default)'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.imshow(window, np.zeros((500, 960, 3), dtype=np.uint8))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def execute(target):
        try:
            controller.press(target, int(profile['pulse_ms']))
            actions_done.put(None)
        except Exception as error:
            actions_done.put(str(error))

    try:
        with log_path.open('a', encoding='utf-8') as log:
            while True:
                try:
                    error = actions_done.get_nowait()
                    busy = False
                    fresh_after = time.monotonic() + profile['settle_s']
                    gate.reset_consensus()
                    if error:
                        armed = False
                        failed = True
                        print(f'触控失败，已暂停；请人工确认停车：{error}')
                except queue.Empty:
                    pass
                try:
                    frame = source.messages.get_nowait()
                except queue.Empty:
                    frame = None
                if isinstance(frame, Exception):
                    armed = False
                    failed = True
                    print(f'画面读取失败，已暂停：{frame}')
                    break
                if frame is not None:
                    latest_receipt = time.monotonic()
                    scene, floor, valid, scores, near_scores, anchor_error = analyze(frame.image, profile)
                    proposal, reason = propose(scores, near_scores, profile)
                    action, guard = gate.evaluate(frame, scene, proposal, anchor_error, profile, time.monotonic())
                    if frame.started < fresh_after or busy:
                        action, guard = 'wait', 'wait for post-action frame'
                        gate.reset_consensus()
                    latest_action = action
                    if action == 'stop':
                        armed = False
                    overlay = scene.copy()
                    overlay[valid > 0] = (0, 0, 230)
                    overlay[floor] = (0, 200, 0)
                    preview = cv2.addWeighted(scene, 0.6, overlay, 0.4, 0)
                    panel = np.zeros((preview.shape[0] + 110, max(700, preview.shape[1]), 3), dtype=np.uint8)
                    panel[110:, :preview.shape[1]] = preview
                    lines = [f'{"ARMED" if armed else "OBSERVE/PAUSED"} | {action} | {guard}',
                             f'L/C/R floor: {scores["left"]:.2f} / {scores["center"]:.2f} / {scores["right"]:.2f} | {reason}',
                             f'capture {frame.finished - frame.started:.2f}s | steps {steps}/{profile["max_steps"]} | G arm, SPACE pause, Q quit']
                    for index, text in enumerate(lines):
                        cv2.putText(panel, text, (8, 24 + index * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
                    cv2.imshow(window, panel)
                    record = {'time': time.time(), 'proposal': proposal, 'decision': action,
                              'guard': guard, 'floor_ratio': scores, 'near_ratio': near_scores,
                              'capture_s': frame.finished - frame.started, 'armed': armed, 'submitted': False}
                    if armed and not busy and action in profile['actions']:
                        next_turns = turns + 1 if action.startswith('turn_') else 0
                        if steps >= profile['max_steps'] or next_turns > profile['max_turns']:
                            armed = False
                            print('达到运行步数或连续转向上限，暂停。')
                        else:
                            button = profile['actions'][action]
                            target = pixel_point(points[button], frame.width, frame.height)
                            busy = True
                            steps += 1
                            turns = next_turns
                            record['submitted'] = True
                            threading.Thread(target=execute, args=(target,), daemon=True).start()
                    log.write(json.dumps(record, ensure_ascii=False) + '\n')
                    log.flush()
                if time.monotonic() - latest_receipt > profile['max_frame_age_s']:
                    armed = False
                    latest_action = 'stop'
                    if panel is not None:
                        stale_panel = panel.copy()
                        stale_panel[:110] = 0
                        cv2.putText(stale_panel, 'PAUSED: no fresh frame; check phone / USB',
                                    (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        cv2.imshow(window, stale_panel)
                key = cv2.waitKey(30) & 0xFF
                if key in (ord('q'), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    armed = False
                    break
                if key == 32:
                    armed = False
                    print('已暂停后续动作；正在执行的短按仍会完成并释放。')
                if key == ord('g'):
                    if drive and not busy and not failed and latest_action not in ('stop', 'wait'):
                        armed = True
                        steps = 0
                        turns = 0
                        print('开始有限步数局部导航；空格暂停。')
                    else:
                        print('未启用 --drive、判断未稳定或设备状态异常，保持观察。')
    finally:
        source.closed.set()
        if busy:
            print('等待当前短按释放；紧急情况请从手机或电源停止小车。')
            try:
                result = actions_done.get(timeout=30)
                if result:
                    print(f'释放状态无法确认：{result}')
            except queue.Empty:
                print('未确认释放，请人工停止小车。')
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='可校准的地面颜色分割与有限步数局部避障原型。')
    parser.add_argument('--adb')
    parser.add_argument('--serial')
    parser.add_argument('--profile', type=Path, default=PROFILE_PATH)
    parser.add_argument('--points', type=Path, default=CONFIG_PATH)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--calibrate', action='store_true')
    mode.add_argument('--check', action='store_true', help='只读检查连接、截图和耗时')
    mode.add_argument('--drive', action='store_true', help='启用实车模式，仍需窗口内 G 启动')
    parser.add_argument('--log', type=Path, default=Path('navigation_logs/decisions.jsonl'))
    args = parser.parse_args()
    try:
        controller = PhoneController(find_adb(args.adb), args.serial)
        controller.connect()
        if args.check:
            frame = capture(controller)
            print(f'{frame.width}x{frame.height}; capture={frame.finished - frame.started:.3f}s; no touch sent')
        elif args.calibrate:
            calibrate(controller, args.profile)
        else:
            if not args.profile.is_file():
                raise ValueError('请先运行 --calibrate 选择道路、屏蔽界面、采样地面并选择固定 App 图标。')
            profile = json.loads(args.profile.read_text(encoding='utf-8'))
            validate_profile(profile)
            monitor(controller, profile, args.points, args.drive, args.log)
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.TimeoutExpired, cv2.error) as error:
        parser.exit(1, f'错误：{error}\n')


if __name__ == '__main__':
    main()
