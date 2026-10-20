"""Persistent scrcpy H.264 connection with a bounded latest-frame buffer."""

import atexit
import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import secrets
import socket
import struct
import subprocess
import tempfile
import threading
import time
from fractions import Fraction


@dataclass
class VideoFrame:
    image: object
    width: int
    height: int
    received: float
    sequence: int


def receive_exact(connection, size):
    chunks = bytearray()
    while len(chunks) < size:
        try:
            block = connection.recv(size - len(chunks))
        except socket.timeout:
            if chunks:
                raise ConnectionError('视频包读取超时，连接不能继续复用。')
            raise
        if not block:
            raise ConnectionError('手机视频连接已断开。')
        chunks.extend(block)
    return bytes(chunks)


def display_size(controller):
    description = controller.run('shell', 'dumpsys', 'input').decode('utf-8', errors='replace')
    matches = re.findall(r'Viewport INTERNAL:[^\n]*displayId=0[^\n]*logicalFrame=\[0,\s*0,\s*(\d+),\s*(\d+)\]', description)
    if not matches:
        raise RuntimeError('无法读取手机当前逻辑显示尺寸；请确认手机已解锁。')
    return tuple(map(int, matches[-1]))


class PhoneVideoStream:
    def __init__(self, controller):
        self.controller = controller
        self.condition = threading.Condition()
        self.latest = None
        self.error = None
        self.closed = False
        self.connection = None
        self.process = None
        self.port = None
        self.sequence = 0
        self.thread = None
        self.diagnostics = tempfile.TemporaryFile()
        self.start_lock = threading.Lock()
        atexit.register(self.close)

    def start(self):
        with self.start_lock:
            if self.thread is not None:
                return
            if self.closed:
                raise RuntimeError('视频流已关闭。')
            try:
                import av
            except ImportError as error:
                raise RuntimeError('缺少视频解码依赖，请运行 python -m pip install -r requirements-vision.txt。') from error
            directory = Path(self.controller.adb).resolve().parent
            server = directory / 'scrcpy-server'
            executable = directory / 'scrcpy.exe'
            if not server.is_file() or not executable.is_file():
                raise RuntimeError('请使用 scrcpy 安装目录内的 adb.exe；缺少 scrcpy-server 或 scrcpy.exe。')
            version_output = subprocess.run([str(executable), '--version'], capture_output=True,
                                            check=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)).stdout.decode()
            match = re.search(r'scrcpy\s+(\d+\.\d+(?:\.\d+)?)', version_output)
            if not match:
                raise RuntimeError('无法确定 scrcpy 服务版本。')
            self.protocol_v4 = int(match[1].split('.')[0]) >= 4
            self.native_size = display_size(self.controller)
            if self.native_size[0] <= self.native_size[1]:
                raise ValueError('请先解锁手机并打开横屏小车 App，再启动视频取帧。')
            session = secrets.randbelow(0x7fffffff)
            socket_name = f'scrcpy_{session:08x}'
            self.controller.run('push', str(server), '/data/local/tmp/scrcpy-server.jar')
            self.port = int(self.controller.run('forward', 'tcp:0', f'localabstract:{socket_name}').strip())
            command = [self.controller.adb, '-s', self.controller.serial, 'shell',
                       'CLASSPATH=/data/local/tmp/scrcpy-server.jar', 'app_process', '/',
                       'com.genymobile.scrcpy.Server', match[1], f'scid={session:08x}',
                       'tunnel_forward=true', 'audio=false', 'control=false', 'video_codec=h264',
                       'max_size=1280', 'max_fps=30', 'video_bit_rate=4000000',
                       'send_device_meta=false', 'send_dummy_byte=true',
                       f'{"send_stream_meta" if self.protocol_v4 else "send_codec_meta"}=true',
                       'send_frame_meta=true', 'cleanup=true']
            self.process = subprocess.Popen(command, stdout=self.diagnostics, stderr=subprocess.STDOUT,
                                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            self.thread = threading.Thread(target=self.decode, args=(av,), daemon=True)
            self.thread.start()

    def decode(self, av):
        try:
            deadline = time.monotonic() + 12
            while True:
                if self.closed:
                    return
                try:
                    self.connection = socket.create_connection(('127.0.0.1', self.port), timeout=1)
                    if receive_exact(self.connection, 1) != b'\0':
                        raise RuntimeError('scrcpy 视频握手不匹配。')
                    break
                except OSError:
                    if self.connection is not None:
                        self.connection.close()
                        self.connection = None
                    if time.monotonic() >= deadline or self.process.poll() is not None:
                        raise RuntimeError('scrcpy 视频服务启动失败，请检查 USB 和手机授权。')
                    time.sleep(0.1)
            self.connection.settimeout(5)
            codec = struct.unpack('>I', receive_exact(self.connection, 4))[0]
            if codec != 0x68323634:
                raise RuntimeError('scrcpy 视频格式不是预期的 H.264。')
            modern = getattr(self, 'protocol_v4', False)
            encoded_size = None if modern else struct.unpack('>II', receive_exact(self.connection, 8))
            initial_landscape = self.native_size[0] > self.native_size[1]
            decoder = av.CodecContext.create('h264', 'r')
            origin = None
            while not self.closed:
                try:
                    header = receive_exact(self.connection, 12)
                except socket.timeout:
                    continue
                flags, size = struct.unpack('>QI', header)
                if modern and flags & (1 << 63):
                    _, width, height = struct.unpack('>III', header)
                    if not width or not height or (width > height) != initial_landscape:
                        raise RuntimeError('手机视频方向或尺寸异常，请保持横屏并重启程序。')
                    if encoded_size is not None and encoded_size != (width, height):
                        raise RuntimeError('手机视频尺寸已变化，请重启程序。')
                    encoded_size = (width, height)
                    continue
                if not 0 < size <= 32 * 1024 * 1024:
                    raise RuntimeError('视频数据包长度异常。')
                payload = receive_exact(self.connection, size)
                config_bit = 62 if modern else 63
                config = bool(flags & (1 << config_bit))
                timestamp = (flags & ((1 << (config_bit - 1)) - 1)) / 1_000_000
                if not config and origin is None:
                    origin = time.monotonic() - timestamp
                if config:
                    decoder.extradata = payload
                    continue
                packet = av.Packet(payload)
                packet.pts = round(timestamp * 1_000_000)
                packet.time_base = Fraction(1, 1_000_000)
                for decoded in decoder.decode(packet):
                    image = decoded.to_ndarray(format='bgr24')
                    if (image.shape[1] > image.shape[0]) != initial_landscape:
                        raise RuntimeError('手机旋转，视频与触控尺寸已变化；请重启程序。')
                    now = time.monotonic()
                    stamp = min(now, origin + float(decoded.pts * decoded.time_base)) if origin is not None and decoded.pts is not None and decoded.time_base is not None else now
                    with self.condition:
                        self.sequence += 1
                        self.latest = VideoFrame(image, *self.native_size, stamp, self.sequence)
                        self.condition.notify_all()
        except Exception as error:
            if not self.closed:
                with self.condition:
                    self.error = error
                    self.condition.notify_all()

    def read(self, after=None, timeout=10):
        try:
            self.start()
        except Exception:
            self.close()
            raise
        cutoff = time.monotonic() if after is None else after
        deadline = time.monotonic() + timeout
        with self.condition:
            while True:
                if self.error:
                    raise RuntimeError(f'视频流错误：{self.error}') from self.error
                if self.closed:
                    raise RuntimeError('视频流已关闭。')
                if self.latest is not None and self.latest.received >= cutoff:
                    frame = self.latest
                    return VideoFrame(frame.image.copy(), frame.width, frame.height, frame.received, frame.sequence)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('没有收到更新的视频帧，已停止后续控制；请检查手机是否息屏、锁屏或画面停止更新。')
                self.condition.wait(remaining)

    def close(self):
        with self.condition:
            if self.closed:
                return
            self.closed = True
            self.condition.notify_all()
        if self.connection:
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.connection.close()
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.port:
            try:
                self.controller.run('forward', '--remove', f'tcp:{self.port}', timeout=3)
            except Exception:
                pass
        self.diagnostics.close()


def main():
    from phone_direction_control import PhoneController, find_adb
    parser = argparse.ArgumentParser(description='持续视频流只读测速，不发送触控。')
    parser.add_argument('--adb')
    parser.add_argument('--serial')
    parser.add_argument('--frames', type=int, default=60)
    args = parser.parse_args()
    if not 2 <= args.frames <= 1000:
        parser.error('frames 须为 2～1000。')
    controller = PhoneController(find_adb(args.adb), args.serial)
    try:
        controller.connect()
        warmup = time.monotonic()
        frame = controller.video_frame()
        print(f'首帧启动耗时 {time.monotonic() - warmup:.3f}s；触控尺寸 {frame.width}x{frame.height}；'
              f'视频尺寸 {frame.image.shape[1]}x{frame.image.shape[0]}', flush=True)
        started = time.monotonic()
        sequences = []
        ages = []
        for _ in range(args.frames):
            frame = controller.video_frame()
            sequences.append(frame.sequence)
            ages.append(time.monotonic() - frame.received)
        elapsed = time.monotonic() - started
        print(f'取帧 {args.frames} 次，耗时 {elapsed:.3f}s，取帧率 {args.frames / elapsed:.1f}/s，'
              f'平均接收/解码年龄 {sum(ages) / len(ages):.3f}s；首末帧序号 {sequences[0]}..{sequences[-1]}；未发送触控。')
    finally:
        controller.close()


if __name__ == '__main__':
    main()
