import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

import view_episodes as viewer


class EpisodeExportTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def episode(self, name="episode", timestamps=(0.0, 0.2, 0.7), cameras=("front", "external")):
        path = self.root / name
        path.mkdir()
        colors = ((20, 80, 220), (50, 190, 40), (210, 60, 30))
        metadata = {
            "format_version": 1,
            "status": "completed",
            "n_frames": len(timestamps),
            "fps": 5,
            "cameras": list(cameras),
        }
        (path / "episode.json").write_text(json.dumps(metadata))
        frames = []
        for index, stamp in enumerate(timestamps):
            images = {}
            for camera_index, camera in enumerate(cameras):
                relative = Path(camera) / f"{index:06d}.png"
                (path / camera).mkdir(exist_ok=True)
                image = np.full((48, 48, 3), colors[(index + camera_index) % len(colors)], dtype=np.uint8)
                self.assertTrue(cv2.imwrite(str(path / relative), image))
                images[camera] = str(relative)
            frames.append({"timestamp": stamp, "images": images})
        (path / "frames.jsonl").write_text("".join(json.dumps(frame) + "\n" for frame in frames))
        return viewer.load_episode(path)

    def read_mp4(self, path):
        capture = cv2.VideoCapture(str(path))
        self.assertTrue(capture.isOpened(), f"Cannot decode {path}")
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            frames = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
        finally:
            capture.release()
        return fps, frames

    def read_gif(self, path):
        frames, durations = [], []
        with Image.open(path) as gif:
            for index in range(gif.n_frames):
                gif.seek(index)
                frames.append(np.asarray(gif.convert("RGB"))[:, :, ::-1].copy())
                durations.append(gif.info["duration"])
        return frames, durations

    def assert_camera_color(self, canvas, episode, index, size=40, tolerance=0):
        for camera_index, source in enumerate(viewer.read_frame(episode, index).values()):
            actual = canvas[72 + size - 5, camera_index * size + size - 5].astype(float)
            expected = source[-5, -5].astype(float)
            np.testing.assert_allclose(actual, expected, atol=tolerance)

    def test_gif_preserves_camera_panels_overlays_and_irregular_timing(self):
        episode = self.episode()
        output = self.root / "recording.gif"
        with patch("cv2.imshow", side_effect=AssertionError("Export opened a GUI")), \
                patch("cv2.waitKeyEx", side_effect=AssertionError("Export waited for a key")):
            viewer.export_episode(episode, output, size=40)
        frames, durations = self.read_gif(output)
        self.assertEqual(durations, [200, 500, 200])
        self.assertEqual(len(frames), 3)
        for index, frame in enumerate(frames):
            expected = viewer.make_canvas(episode, index, size=40, controls=False)
            self.assertEqual(frame.shape, expected.shape)
            # The complete viewer canvas includes the timestamp and camera labels.
            self.assertLess(np.abs(frame.astype(float) - expected.astype(float)).mean(), 3)
            self.assert_camera_color(frame, episode, index)

    def test_gif_speed_changes_each_frame_delay(self):
        episode = self.episode()
        output = self.root / "faster.gif"
        viewer.export_episode(episode, output, speed=2, size=40)
        _, durations = self.read_gif(output)
        self.assertEqual(durations, [100, 250, 100])

    def test_nonzero_timestamp_origin_and_gif_centisecond_quantization(self):
        episode = self.episode(timestamps=(10.0, 10.207, 10.714))
        gif_path = self.root / "offset.gif"
        viewer.export_episode(episode, gif_path, size=40)
        _, durations = self.read_gif(gif_path)
        self.assertEqual(len(durations), 3)
        for actual, expected in zip(durations, (207, 507, 200)):
            self.assertEqual(actual % 10, 0)
            self.assertAlmostEqual(actual, expected, delta=10)
        mp4_path = self.root / "offset.mp4"
        viewer.export_episode(episode, mp4_path, size=40)
        fps, frames = self.read_mp4(mp4_path)
        self.assertAlmostEqual(len(frames) / fps, .914, delta=1 / fps)
        self.assert_camera_color(frames[0], episode, 0, tolerance=12)

    def test_mp4_resamples_timestamps_and_speed_without_losing_colors(self):
        episode = self.episode()
        for speed in (1, 2):
            with self.subTest(speed=speed):
                output = self.root / f"recording_{speed}.mp4"
                viewer.export_episode(episode, output, speed=speed, size=40)
                fps, frames = self.read_mp4(output)
                self.assertAlmostEqual(fps, 30, places=3)
                self.assertAlmostEqual(len(frames) / fps, 0.9 / speed, delta=1 / fps)
                self.assertEqual(frames[0].shape, (112, 80, 3))
                for source_index, stamp in enumerate((0.05, 0.35, 0.75)):
                    output_index = round(stamp / speed * fps)
                    self.assert_camera_color(frames[output_index], episode, source_index, tolerance=12)
                expected = viewer.make_canvas(episode, 0, size=40, controls=False)
                self.assertLess(np.abs(frames[0].astype(float) - expected.astype(float)).mean(), 15)

    def test_single_frame_preserves_gif_size_and_pads_mp4_to_even_dimensions(self):
        episode = self.episode(timestamps=(0.0,), cameras=("front",))
        for suffix in ("gif", "mp4"):
            with self.subTest(suffix=suffix):
                output = self.root / f"single.{suffix}"
                viewer.export_episode(episode, output, size=41)
                if suffix == "gif":
                    frames, durations = self.read_gif(output)
                    self.assertEqual(durations, [200])
                    expected_shape = (113, 41, 3)
                else:
                    fps, frames = self.read_mp4(output)
                    self.assertAlmostEqual(len(frames) / fps, .2, delta=1 / fps)
                    expected_shape = (114, 42, 3)
                self.assertTrue(frames)
                self.assertEqual(frames[0].shape, expected_shape)
                self.assert_camera_color(frames[0], episode, 0, size=41, tolerance=12)

    def test_empty_episode_is_rejected_without_creating_output(self):
        episode = self.episode(timestamps=())
        for suffix in ("gif", "mp4"):
            with self.subTest(suffix=suffix):
                output = self.root / f"empty.{suffix}"
                with self.assertRaises(ValueError):
                    viewer.export_episode(episode, output)
                self.assertFalse(output.exists())

    def test_invalid_recording_fps_is_rejected_before_creating_output(self):
        episode = self.episode()
        for fps in (0, -1, float("nan"), float("inf")):
            for suffix in ("gif", "mp4"):
                with self.subTest(fps=fps, suffix=suffix):
                    episode.metadata["fps"] = fps
                    output = self.root / f"invalid-fps.{suffix}"
                    with self.assertRaises(ValueError):
                        viewer.export_episode(episode, output)
                    self.assertFalse(output.exists())

    def test_cli_exports_headlessly_without_starting_playback(self):
        episode = self.episode()
        output = self.root / "cli.gif"
        with patch("cv2.getBuildInformation", return_value="GUI: NONE"), \
                patch.object(viewer, "play_episode", side_effect=AssertionError("Started playback")), \
                patch("cv2.imshow", side_effect=AssertionError("Opened a GUI")), \
                patch("cv2.destroyAllWindows", side_effect=AssertionError("Touched a GUI")):
            self.assertEqual(viewer.main([str(episode.path), "--save-video", str(output),
                                          "--speed=2", "--size=40"]), 0)
        _, durations = self.read_gif(output)
        self.assertEqual(durations, [100, 250, 100])

    def test_relative_cli_outputs_are_saved_inside_episode_not_working_directory(self):
        episode = self.episode()
        working_directory = self.root / "working"
        working_directory.mkdir()
        previous_directory = Path.cwd()
        try:
            os.chdir(working_directory)
            for filename in ("episode.gif", "episode.mp4", "clips/episode.gif"):
                with self.subTest(filename=filename):
                    self.assertEqual(viewer.main([
                        str(episode.path), "--save-video", filename, "--size=40",
                    ]), 0)
                    output = episode.path / filename
                    self.assertTrue(output.is_file())
                    self.assertFalse((working_directory / filename).exists())
                    if output.suffix == ".gif":
                        frames, _ = self.read_gif(output)
                    else:
                        _, frames = self.read_mp4(output)
                    self.assertTrue(frames)
        finally:
            os.chdir(previous_directory)

    def test_cli_rejects_unknown_format_multiple_episodes_and_verify_conflict(self):
        episode = self.episode()
        self.episode(name="another")
        invalid_arguments = (
            [str(episode.path), "--save-video", str(self.root / "output.avi")],
            [str(self.root), "--save-video", str(self.root / "output.mp4")],
            [str(episode.path), "--verify", "--save-video", str(self.root / "output.gif")],
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    viewer.main(arguments)
                self.assertEqual(error.exception.code, 2)
        self.assertFalse(any(self.root.glob("output.*")))

    def test_failed_or_interrupted_export_keeps_destination_and_cleans_temporary_files(self):
        episode = self.episode()
        original_canvas = viewer.make_canvas
        for suffix in ("gif", "mp4"):
            for error in (ValueError("Missing frame"), KeyboardInterrupt()):
                with self.subTest(suffix=suffix, error=type(error).__name__):
                    output = self.root / f"preserve.{suffix}"
                    output.write_bytes(b"existing export")
                    before = set(self.root.iterdir())

                    def fail_after_first(current_episode, index, *args, **kwargs):
                        if index >= 1:
                            raise error
                        return original_canvas(current_episode, index, *args, **kwargs)

                    with patch.object(viewer, "make_canvas", side_effect=fail_after_first):
                        with self.assertRaises(type(error)):
                            viewer.export_episode(episode, output, size=40)
                    self.assertEqual(output.read_bytes(), b"existing export")
                    self.assertEqual(set(self.root.iterdir()), before)


if __name__ == "__main__":
    unittest.main()
