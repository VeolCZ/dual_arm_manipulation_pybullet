import json
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from environment.episode_recorder import EpisodeRecorder, EpisodeRecordingError
from view_episodes import _wait_for_key, discover_episodes, load_episode, main, make_canvas, play_episode, read_frame, verify_episode


class FakeEnvironment:
    camera = object()
    front_camera = object()
    left_arm_camera = object()
    camera_renderer = 123

    def __init__(self):
        self.steps = 0

    def step_simulation(self):
        self.steps += 1

    def get_external_cam_img(self, **kwargs):
        rgb = np.zeros((40, 40, 3), dtype=np.uint8)
        rgb[:, :, 0] = self.steps + 20
        return rgb, None, None

    get_left_arm_cam_img = get_external_cam_img
    get_front_cam_img = get_external_cam_img


class EpisodeRecordingTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.env = FakeEnvironment()
        self.original_step = self.env.step_simulation
        self.recorder = EpisodeRecorder(self.env, self.folder.name, fps=5,
                                        which=("external", "left"), physics_hz=20)
        self.addCleanup(self.recorder.close)

    def record(self, steps=9):
        with self.recorder.episode(scenario="test", run=1):
            for _ in range(steps):
                self.env.step_simulation()
        return load_episode(self.recorder.path)

    def test_sampling_timestamps_lossless_images_and_final_frame(self):
        episode = self.record()
        np.testing.assert_allclose([frame["timestamp"] for frame in episode.frames], [0, .2, .4, .45])
        self.assertEqual(episode.metadata["n_frames"], 4)
        self.assertEqual(episode.metadata["status"], "completed")
        self.assertEqual(episode.metadata["scenario_metadata"]["run"], 1)
        self.assertEqual(episode.metadata["image_sizes"], {"external": [40, 40], "left": [40, 40]})
        self.assertEqual(verify_episode(episode), {"external": (40, 40, 3), "left": (40, 40, 3)})
        np.testing.assert_array_equal(read_frame(episode, 3)["external"][0, 0], [0, 0, 29])
        self.recorder.close()
        self.assertEqual(self.env.step_simulation, self.original_step)

    def test_episode_boundaries_reset_time_and_preserve_previous_files(self):
        first = self.record(4)
        second = self.record(4)
        self.assertNotEqual(first.path, second.path)
        self.assertEqual(len(discover_episodes(self.folder.name)), 2)
        self.assertEqual(second.frames[0]["timestamp"], 0)
        np.testing.assert_array_equal(read_frame(first, 0)["external"][0, 0], [0, 0, 20])
        np.testing.assert_array_equal(read_frame(second, 0)["external"][0, 0], [0, 0, 24])

    def test_front_camera_is_recorded_by_default_and_replayed(self):
        self.recorder.close()
        self.recorder = EpisodeRecorder(self.env, self.folder.name, fps=5, physics_hz=20)
        self.addCleanup(self.recorder.close)
        episode = self.record(4)
        self.assertIn("front", episode.metadata["cameras"])
        self.assertEqual(episode.metadata["image_sizes"]["front"], [40, 40])
        self.assertTrue((episode.path / "front" / "000000.png").is_file())
        np.testing.assert_array_equal(read_frame(episode, 1)["front"][0, 0], [0, 0, 24])
        self.assertEqual(make_canvas(episode, 0, size=40).shape, (112, 120, 3))

    def test_recording_keeps_front_at_640_with_smaller_other_cameras(self):
        front = np.zeros((640, 640, 3), dtype=np.uint8)
        front[:, :, 0] = 220
        self.env.get_front_cam_img = lambda **kwargs: (front, None, None)
        self.recorder.close()
        self.recorder = EpisodeRecorder(self.env, self.folder.name, fps=5, physics_hz=20)
        self.addCleanup(self.recorder.close)
        episode = self.record(4)
        sizes = verify_episode(episode)
        self.assertEqual(sizes["front"], (640, 640, 3))
        self.assertEqual(sizes["external"], (40, 40, 3))
        self.assertEqual(episode.metadata["image_sizes"]["front"], [640, 640])
        np.testing.assert_array_equal(read_frame(episode, 0)["front"][0, 0], [0, 0, 220])
        self.assertEqual(make_canvas(episode, 0, size=40).shape, (112, 120, 3))

    def test_keyboard_interrupt_finishes_a_playable_partial_episode(self):
        with self.assertRaises(KeyboardInterrupt):
            with self.recorder.episode():
                for _ in range(4):
                    self.env.step_simulation()
                raise KeyboardInterrupt()
        episode = load_episode(self.recorder.path)
        self.assertEqual(episode.metadata["status"], "interrupted")
        self.assertEqual(len(episode.frames), 2)
        verify_episode(episode)

    @patch("cv2.imwrite", return_value=False)
    def test_writer_errors_are_reported_and_recorded(self, _write):
        with self.assertRaises(EpisodeRecordingError):
            self.record(0)
        metadata = json.loads((self.recorder.path / "episode.json").read_text())
        self.assertEqual(metadata["status"], "error")
        self.assertEqual(metadata["n_frames"], 0)
        self.assertIn("Failed to write", metadata["error"])

    def test_slow_disk_keeps_queue_bounded_and_counts_drops(self):
        release = threading.Event()
        original_writer = EpisodeRecorder._write_frames

        def slow_writer(recorder):
            release.wait(timeout=3)
            original_writer(recorder)

        with patch.object(EpisodeRecorder, "_write_frames", slow_writer):
            self.recorder.start()
            try:
                for _ in range(48):
                    self.env.step_simulation()
                self.assertEqual(self.recorder._queue.qsize(), 8)
                self.assertEqual(self.recorder._dropped, 5)
            finally:
                release.set()
                self.recorder.finish()
        self.assertEqual(load_episode(self.recorder.path).metadata["n_frames"], 8)

    def test_loader_recovers_complete_frames_after_truncated_index(self):
        episode = self.record()
        with (episode.path / "frames.jsonl").open("a") as stream:
            stream.write('{"timestamp":')
        recovered = load_episode(episode.path)
        self.assertEqual(len(recovered.frames), 4)

    @patch("cv2.imshow")
    def test_verify_needs_no_gui_and_canvas_uses_only_recorded_images(self, imshow):
        episode = self.record()
        self.assertEqual(main([self.folder.name, "--verify"]), 0)
        imshow.assert_not_called()
        canvas = make_canvas(episode, 0, size=40)
        self.assertEqual(canvas.shape, (112, 80, 3))
        np.testing.assert_array_equal(canvas[111, 39], [0, 0, 20])

    @patch("cv2.getWindowProperty", return_value=1)
    @patch("cv2.waitKeyEx", side_effect=[32, ord("d"), ord("a"), 32, ord("r"), ord("]")])
    @patch("cv2.imshow")
    @patch("view_episodes.make_canvas", return_value=np.zeros((10, 10, 3), dtype=np.uint8))
    def test_playback_pause_step_and_episode_navigation(self, canvas, _show, _wait, _visible):
        episode = self.record()
        self.assertEqual(play_episode(episode), "next")
        self.assertEqual([call.args[1] for call in canvas.call_args_list], [0, 0, 1, 0, 0, 0])

    @patch("cv2.getWindowProperty", return_value=1)
    @patch("cv2.waitKeyEx", side_effect=[-1, -1, ord("q")])
    def test_paused_wait_polls_instead_of_blocking(self, wait_key, _visible):
        self.assertEqual(_wait_for_key("viewer"), ord("q"))
        self.assertEqual([call.args[0] for call in wait_key.call_args_list], [20, 20, 20])

    @patch("cv2.getWindowProperty", return_value=1)
    @patch("cv2.waitKeyEx", return_value=-1)
    @patch("view_episodes.time.monotonic", side_effect=[0, .02, .02, .04, .04, .05])
    def test_event_polling_waits_until_frame_deadline(self, _clock, wait_key, _visible):
        self.assertEqual(_wait_for_key("viewer", deadline=.05), -1)
        self.assertEqual(wait_key.call_count, 3)
        self.assertTrue(all(1 <= call.args[0] <= 20 for call in wait_key.call_args_list))

    @patch("cv2.getWindowProperty", return_value=0)
    @patch("cv2.waitKeyEx", return_value=-1)
    def test_window_close_exits_paused_wait(self, _wait, _visible):
        self.assertEqual(_wait_for_key("viewer"), 27)

    @patch("cv2.destroyAllWindows")
    @patch("cv2.getWindowProperty", return_value=1)
    @patch("cv2.waitKeyEx", side_effect=KeyboardInterrupt)
    @patch("cv2.imshow")
    @patch("cv2.getBuildInformation", return_value="GUI: QT5")
    def test_terminal_interrupt_closes_windows_without_traceback(self, _build, _show, _wait, _visible, destroy):
        episode = self.record()
        self.assertEqual(main([str(episode.path)]), 130)
        destroy.assert_called_once()

    @patch("cv2.getWindowProperty", return_value=1)
    @patch("cv2.waitKeyEx", side_effect=[-1, ord("q")])
    @patch("cv2.imshow")
    def test_last_frame_remains_interruptible(self, _show, wait_key, _visible):
        episode = self.record(0)
        self.assertEqual(play_episode(episode), "quit")
        self.assertTrue(all(call.args[0] > 0 for call in wait_key.call_args_list))

    def test_invalid_recording_configuration_is_rejected(self):
        for fps in (0, float("nan"), 241):
            with self.assertRaises(ValueError):
                EpisodeRecorder(self.env, fps=fps)
        with self.assertRaises(ValueError):
            EpisodeRecorder(self.env, which=("missing",))


if __name__ == "__main__":
    unittest.main()
