from fractions import Fraction
import socket
import struct
import threading
import time
import unittest
from unittest.mock import Mock, patch

import av
import numpy as np

from phone_direction_control import PhoneController
from phone_video_stream import PhoneVideoStream, VideoFrame, display_size, receive_exact
from vision_navigation import capture


class StreamTests(unittest.TestCase):
    def test_fragmented_packet_and_disconnect(self):
        connection = Mock()
        connection.recv.side_effect = [b'a', b'bc', b'']
        self.assertEqual(receive_exact(connection, 3), b'abc')
        with self.assertRaises(ConnectionError):
            receive_exact(connection, 1)

    def test_partial_timeout_does_not_silently_corrupt_protocol(self):
        connection = Mock()
        connection.recv.side_effect = [b'abc', socket.timeout()]
        with self.assertRaises(ConnectionError):
            receive_exact(connection, 12)

    def test_original_touch_dimensions(self):
        controller = Mock()
        controller.run.return_value = b'Viewport INTERNAL: displayId=0, logicalFrame=[0, 0, 2800, 1260]'
        self.assertEqual(display_size(controller), (2800, 1260))

    def test_read_waits_for_frame_after_cutoff_and_returns_copy(self):
        stream = PhoneVideoStream(Mock())
        self.addCleanup(stream.close)
        stream.latest = VideoFrame(np.zeros((4, 8, 3), dtype=np.uint8), 2800, 1260, 1, 1)
        with patch.object(stream, 'start'):
            with self.assertRaises(TimeoutError):
                stream.read(after=2, timeout=0.01)
            stream.latest = VideoFrame(np.zeros((4, 8, 3), dtype=np.uint8), 2800, 1260, 3, 2)
            result = stream.read(after=2)
            result.image[:] = 255
            self.assertFalse(stream.latest.image.any())

    def test_decoder_error_prevents_old_frame_use(self):
        stream = PhoneVideoStream(Mock())
        self.addCleanup(stream.close)
        stream.latest = VideoFrame(np.zeros((4, 8, 3)), 2800, 1260, time.monotonic(), 1)
        stream.error = ConnectionError('unplugged')
        with patch.object(stream, 'start'), self.assertRaises(RuntimeError):
            stream.read(after=0)

    def test_navigation_capture_uses_stream_without_png_roundtrip(self):
        controller = Mock()
        controller.video_frame.return_value = VideoFrame(np.zeros((576, 1280, 3), dtype=np.uint8),
                                                        2800, 1260, time.monotonic(), 42)
        frame = capture(controller)
        self.assertEqual(frame.image.shape, (432, 960, 3))
        self.assertEqual((frame.width, frame.height), (2800, 1260))
        controller.screenshot.assert_not_called()

    def test_controller_reuses_one_video_stream(self):
        controller = PhoneController('adb.exe', 'device')
        with patch('phone_video_stream.PhoneVideoStream') as factory:
            controller.video_frame()
            controller.video_frame()
            factory.assert_called_once_with(controller)
            controller.close()
            factory.return_value.close.assert_called_once()

    def test_real_h264_decode_through_scrcpy_packet_protocol(self):
        self.decode_protocol(False)

    def test_real_h264_decode_through_scrcpy_v4_session_protocol(self):
        self.decode_protocol(True)

    def decode_protocol(self, modern):
        codec = av.CodecContext.create('libx264', 'w')
        codec.width, codec.height = 96, 48
        codec.pix_fmt = 'yuv420p'
        codec.time_base = Fraction(1, 30)
        codec.options = {'preset': 'ultrafast', 'tune': 'zerolatency'}
        codec.flags |= av.codec.context.Flags.global_header
        codec.open()
        image = np.full((48, 96, 3), (40, 80, 120), dtype=np.uint8)
        encoded = codec.encode(av.VideoFrame.from_ndarray(image, format='bgr24')) + codec.encode(None)
        self.assertTrue(encoded)
        client, server = socket.socketpair()
        self.addCleanup(server.close)
        server.sendall(b'\0' + struct.pack('>I', 0x68323634))
        if modern:
            server.sendall(struct.pack('>III', 1 << 31, 96, 48))
        else:
            server.sendall(struct.pack('>II', 96, 48))
        config = codec.extradata
        server.sendall(struct.pack('>QI', 1 << (62 if modern else 63), len(config)) + config)
        for index, packet in enumerate(encoded):
            data = bytes(packet)
            server.sendall(struct.pack('>QI', index * 33333, len(data)) + data)
        stream = PhoneVideoStream(Mock())
        self.addCleanup(stream.close)
        stream.native_size = (2800, 1260)
        stream.protocol_v4 = modern
        stream.port = 1234
        stream.process = Mock()
        stream.process.poll.return_value = None
        not_ready = Mock()
        not_ready.recv.return_value = b''
        with patch('phone_video_stream.socket.create_connection', side_effect=[not_ready, client]) as connect:
            stream.thread = threading.Thread(target=stream.decode, args=(av,), daemon=True)
            stream.thread.start()
            with patch.object(stream, 'start'):
                frame = stream.read(after=0, timeout=3)
            self.assertEqual(frame.image.shape, (48, 96, 3))
            self.assertEqual((frame.width, frame.height), (2800, 1260))
            self.assertLess(float(np.abs(frame.image.astype(float) - image).mean()), 5)
            self.assertEqual(connect.call_count, 2)
            not_ready.close.assert_called_once()
            stream.close()
            stream.thread.join(timeout=2)
            self.assertFalse(stream.thread.is_alive())


if __name__ == '__main__':
    unittest.main()
