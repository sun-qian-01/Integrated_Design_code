"""Eight-direction Android touch control via ADB; standard library only."""

import argparse
import base64
import json
import os
from pathlib import Path
import queue
import shutil
import struct
import subprocess
import threading


CONFIG_PATH = Path(__file__).with_name('phone_direction_points.json')
DIRECTIONS = {
    'up_left': ('↖ 左上', 1112, 275),
    'up': ('↑ 上', 1176, 249),
    'up_right': ('↗ 右上', 1240, 275),
    'left': ('← 左', 1087, 339),
    'right': ('→ 右', 1267, 339),
    'down_left': ('↙ 左下', 1112, 403),
    'down': ('↓ 下', 1176, 429),
    'down_right': ('↘ 右下', 1240, 403),
}
REFERENCE_SIZE = (1420, 642)


def default_points():
    return {
        name: [values[1] / REFERENCE_SIZE[0], values[2] / REFERENCE_SIZE[1]]
        for name, values in DIRECTIONS.items()
    }


def load_points(path):
    if not path.exists():
        return default_points()
    points = json.loads(path.read_text(encoding='utf-8'))
    if set(points) != set(DIRECTIONS):
        raise ValueError('坐标文件必须包含全部八个方向。')
    for point in points.values():
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError('每个方向必须包含两个归一化坐标。')
        if any(type(value) not in (int, float) or not 0 <= value < 1 for value in point):
            raise ValueError('坐标必须是大于等于 0、小于 1 的数值。')
    return points


def pixel_point(point, width, height):
    if width <= height:
        raise ValueError('请先让手机进入截图中的横屏小车控制界面。')
    return min(round(point[0] * width), width - 1), min(round(point[1] * height), height - 1)


def png_size(data):
    if len(data) < 24 or data[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError('ADB 未返回有效 PNG 截图，请检查设备状态。')
    return struct.unpack('>II', data[16:24])


def find_adb(explicit=None):
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise ValueError(f'找不到 ADB：{path}')
        return str(path)
    package_dir = Path(os.environ.get('LOCALAPPDATA', '')) / 'Microsoft/WinGet/Packages'
    candidates = sorted(package_dir.glob('Genymobile.scrcpy_*/scrcpy-*/adb.exe'))
    if candidates:
        return str(candidates[-1])
    path = shutil.which('adb')
    if path:
        return path
    raise ValueError('未找到 adb.exe，请使用 --adb 指定完整路径。')


class PhoneController:
    def __init__(self, adb, serial=None):
        self.adb = adb
        self.serial = serial

    def run(self, *args, timeout=15):
        command = [self.adb]
        if self.serial:
            command.extend(['-s', self.serial])
        command.extend(str(value) for value in args)
        result = subprocess.run(
            command, capture_output=True, timeout=timeout,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
        )
        if result.returncode:
            raise RuntimeError(result.stderr.decode('utf-8', errors='replace').strip() or 'ADB 执行失败')
        return result.stdout

    def connect(self):
        output = self.run('devices').decode('utf-8', errors='replace')
        devices = [line.split()[0] for line in output.splitlines()
                   if len(line.split()) >= 2 and line.split()[1] == 'device']
        if self.serial:
            if self.serial not in devices:
                raise ValueError('指定手机未连接或尚未授权，请运行 adb devices 检查。')
        elif len(devices) == 1:
            self.serial = devices[0]
        else:
            raise ValueError('需要一台已授权手机；多台设备时请用 --serial 指定。')

    def screenshot(self):
        data = self.run('exec-out', 'screencap', '-p')
        width, height = png_size(data)
        if width <= height:
            raise ValueError('当前手机不是横屏，请先打开小车控制界面。')
        return data, width, height

    def press(self, point, duration_ms):
        if not 50 <= duration_ms <= 2000:
            raise ValueError('按压时长须在 50～2000 毫秒之间。')
        horizontal, vertical = point
        try:
            self.run('shell', 'input', 'touchscreen', 'swipe',
                     horizontal, vertical, horizontal, vertical, duration_ms,
                     timeout=duration_ms / 1000 + 10)
        finally:
            self.run('shell', 'input', 'touchscreen', 'motionevent', 'UP', horizontal, vertical)


def launch_gui(controller, points, config_path, initial_duration):
    import tkinter as tk
    from tkinter import messagebox, ttk

    root = tk.Tk()
    root.title('手机八方向控制 · ADB')
    root.geometry('1080x800')
    selected = tk.StringVar(value='up')
    duration = tk.StringVar(value=str(initial_duration))
    calibrating = tk.BooleanVar(value=True)
    status = tk.StringVar(value='先刷新截图，选择方向，再点击截图中的对应位置校准。')
    events = queue.Queue()
    state = {'busy': False, 'size': None, 'image': None, 'scale': 1}
    controls = []

    ttk.Label(root, text=f'手机：{controller.serial}    每次单击发送一次定时按压，默认不发送触控。').pack(pady=5)
    options = ttk.Frame(root)
    options.pack()
    ttk.Label(options, text='按压时长（毫秒，50～2000）：').pack(side='left')
    ttk.Entry(options, textvariable=duration, width=8).pack(side='left')
    toggle = ttk.Checkbutton(options, text='校准模式（不发送触控）', variable=calibrating)
    toggle.pack(side='left', padx=15)
    controls.append(toggle)
    direction_frame = ttk.Frame(root)
    direction_frame.pack(pady=6)
    canvas = tk.Canvas(root, width=1000, height=450, background='#20252b', highlightthickness=0)
    ttk.Label(root, textvariable=status, wraplength=1000).pack(pady=5)
    ttk.Label(root, text='斜向位置仅为预估；先架起车轮验证。释放触摸不等于保证小车停止。').pack()

    def set_busy(value):
        state['busy'] = value
        for control in controls:
            control.configure(state='disabled' if value else 'normal')

    def submit(kind, operation):
        if state['busy']:
            return
        set_busy(True)

        def worker():
            try:
                events.put((kind, operation(), None))
            except Exception as error:
                events.put((kind, None, str(error)))

        threading.Thread(target=worker, daemon=True).start()

    def redraw():
        canvas.delete('marker')
        if state['size'] is None:
            return
        width, height = state['size']
        for name, point in points.items():
            horizontal, vertical = pixel_point(point, width, height)
            horizontal /= state['scale']
            vertical /= state['scale']
            color = '#ffcc33' if name == selected.get() else '#00e5ff'
            canvas.create_oval(horizontal - 5, vertical - 5, horizontal + 5, vertical + 5,
                               outline=color, width=2, tags='marker')
            canvas.create_text(horizontal, vertical - 14, text=DIRECTIONS[name][0],
                               fill=color, tags='marker')

    def choose(name):
        selected.set(name)
        redraw()
        if calibrating.get():
            status.set(f'已选择 {DIRECTIONS[name][0]}：点击截图上的目标位置，然后保存坐标。')
            return
        try:
            milliseconds = int(duration.get())
            if not 50 <= milliseconds <= 2000:
                raise ValueError('时长必须在 50～2000 毫秒之间。')
        except ValueError as error:
            messagebox.showerror('时长无效', str(error))
            return
        point = list(points[name])

        def send():
            _, width, height = controller.screenshot()
            target = pixel_point(point, width, height)
            controller.press(target, milliseconds)
            return f'{DIRECTIONS[name][0]}：{target}，按压 {milliseconds} ms，已发送释放事件；请观察实车响应。'

        status.set('正在发送定时按压，请等待释放……')
        submit('press', send)

    layout = [('up_left', 0, 0), ('up', 0, 1), ('up_right', 0, 2),
              ('left', 1, 0), ('right', 1, 2),
              ('down_left', 2, 0), ('down', 2, 1), ('down_right', 2, 2)]
    for name, row, column in layout:
        button = ttk.Button(direction_frame, text=DIRECTIONS[name][0], width=14,
                            command=lambda direction=name: choose(direction))
        button.grid(row=row, column=column, padx=4, pady=3)
        controls.append(button)
    ttk.Label(direction_frame, text='八方向').grid(row=1, column=1)

    def calibrate(event):
        if not calibrating.get() or state['busy'] or state['size'] is None:
            return
        width, height = state['size']
        point = [event.x * state['scale'] / width, event.y * state['scale'] / height]
        if all(0 <= value < 1 for value in point):
            points[selected.get()] = point
            redraw()
            status.set('坐标已修改（尚未保存）。校准点击未发送到手机。')

    canvas.bind('<Button-1>', calibrate)

    def save():
        try:
            config_path.write_text(json.dumps(points, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            status.set(f'八方向坐标已保存：{config_path}')
        except OSError as error:
            messagebox.showerror('保存失败', str(error))

    actions = ttk.Frame(root)
    actions.pack(pady=5)
    refresh = ttk.Button(actions, text='刷新手机截图（只读）', command=lambda: submit('screenshot', controller.screenshot))
    refresh.pack(side='left', padx=5)
    save_button = ttk.Button(actions, text='保存坐标', command=save)
    save_button.pack(side='left', padx=5)
    controls.extend([refresh, save_button])
    canvas.pack(padx=10, pady=5, fill='both', expand=True)

    def poll():
        try:
            while True:
                kind, result, error = events.get_nowait()
                set_busy(False)
                if error:
                    status.set(f'失败：{error}。若小车在运动，请使用手机或物理开关停止。')
                elif kind == 'screenshot':
                    data, width, height = result
                    scale = max(1, (width + 999) // 1000, (height + 449) // 450)
                    photo = tk.PhotoImage(data=base64.b64encode(data).decode('ascii')).subsample(scale)
                    state.update(size=(width, height), image=photo, scale=scale)
                    canvas.delete('all')
                    canvas.create_image(0, 0, anchor='nw', image=photo)
                    redraw()
                    status.set(f'截图 {width}×{height}；这是一张静态校准截图，实时画面请看 scrcpy。')
                else:
                    status.set(result)
        except queue.Empty:
            pass
        root.after(100, poll)

    def close():
        if state['busy']:
            status.set('操作正在执行，请等待完成后关闭；紧急情况使用手机或物理开关停止。')
        else:
            root.destroy()

    root.protocol('WM_DELETE_WINDOW', close)
    root.after(100, poll)
    submit('screenshot', controller.screenshot)
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description='安卓手机八方向定时触控；默认打开校准 GUI。')
    parser.add_argument('--adb', help='adb.exe 完整路径；默认自动查找 WinGet/PATH')
    parser.add_argument('--serial', help='手机序列号；仅一台已授权设备时自动选择')
    parser.add_argument('--config', type=Path, default=CONFIG_PATH)
    parser.add_argument('--direction', choices=list(DIRECTIONS), help='命令行发送一次指定方向按压')
    parser.add_argument('--duration-ms', type=int, default=200)
    parser.add_argument('--dry-run', action='store_true', help='只读取截图、输出坐标，不发送触控')
    args = parser.parse_args()
    try:
        if not 50 <= args.duration_ms <= 2000:
            raise ValueError('时长必须在 50～2000 毫秒之间。')
        points = load_points(args.config)
        controller = PhoneController(find_adb(args.adb), args.serial)
        controller.connect()
        if args.direction or args.dry_run:
            _, width, height = controller.screenshot()
            names = [args.direction] if args.direction else list(DIRECTIONS)
            for name in names:
                target = pixel_point(points[name], width, height)
                print(f'{name}: {target}, display={width}x{height}, duration={args.duration_ms}ms')
                if not args.dry_run:
                    controller.press(target, args.duration_ms)
            print('只读检查完成，未发送触控。' if args.dry_run else '按压和释放命令已发送；请观察实际响应。')
        else:
            launch_gui(controller, points, args.config, args.duration_ms)
    except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f'错误：{error}\n')


if __name__ == '__main__':
    main()
