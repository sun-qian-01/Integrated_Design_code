import struct
import unittest
from unittest.mock import patch

from phone_direction_control import PhoneController, default_points, pixel_point, png_size


class DirectionControlTests(unittest.TestCase):
    def test_eight_points_are_in_landscape_display(self):
        points = default_points()
        self.assertEqual(len(points), 8)
        for point in points.values():
            horizontal, vertical = pixel_point(point, 2800, 1260)
            self.assertTrue(0 <= horizontal < 2800)
            self.assertTrue(0 <= vertical < 1260)
        self.assertEqual(pixel_point(points['up'], 2800, 1260), (2319, 489))

    def test_portrait_is_rejected(self):
        with self.assertRaises(ValueError):
            pixel_point([0.8, 0.5], 1260, 2800)

    def test_png_dimensions(self):
        data = b'\x89PNG\r\n\x1a\n' + b'\x00\x00\x00\rIHDR' + struct.pack('>II', 2800, 1260)
        self.assertEqual(png_size(data), (2800, 1260))
        with self.assertRaises(ValueError):
            png_size(b'error')

    def test_timed_press_and_release(self):
        controller = PhoneController('adb.exe', 'test-device')
        with patch.object(controller, 'run') as run:
            controller.press((100, 200), 300)
            self.assertEqual(run.call_args_list[0].args,
                             ('shell', 'input', 'touchscreen', 'swipe', 100, 200, 100, 200, 300))
            self.assertEqual(run.call_args_list[1].args,
                             ('shell', 'input', 'touchscreen', 'motionevent', 'UP', 100, 200))

    def test_release_attempted_after_failure(self):
        controller = PhoneController('adb.exe')
        with patch.object(controller, 'run', side_effect=[RuntimeError('failure'), b'']) as run:
            with self.assertRaises(RuntimeError):
                controller.press((100, 200), 300)
            self.assertEqual(run.call_args_list[-1].args,
                             ('shell', 'input', 'touchscreen', 'motionevent', 'UP', 100, 200))

    def test_invalid_duration_sends_nothing(self):
        controller = PhoneController('adb.exe')
        with patch.object(controller, 'run') as run:
            with self.assertRaises(ValueError):
                controller.press((100, 200), 10000)
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
