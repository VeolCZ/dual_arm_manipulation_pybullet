"""Sample simulation cameras and write lossless episodes for offline review."""

import atexit
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import uuid

import numpy as np


CAMERA_METHODS = {
    "external": "get_external_cam_img",
    "front": "get_front_cam_img",
    "head": "get_head_cam_img",
    "left": "get_left_arm_cam_img",
    "right": "get_right_arm_cam_img",
}
CAMERAS = tuple(CAMERA_METHODS)


class EpisodeRecordingError(RuntimeError):
    pass


class EpisodeRecorder:
    """Hook environment steps; only camera capture runs on the physics thread.

    A bounded queue sends frames to a PNG writer thread. If the disk falls behind,
    frames are dropped and counted rather than blocking the simulation.
    """

    def __init__(self, env, output_dir="episodes", fps=5.0,
                 which=("external", "front", "left", "right"), physics_hz=240):
        if not math.isfinite(fps) or not 0 < fps <= physics_hz:
            raise ValueError(f"Recording FPS must be between 0 and {physics_hz}")
        which = tuple(dict.fromkeys(which))
        if not which or any(name not in CAMERAS for name in which):
            raise ValueError(f"Recording cameras must be selected from {CAMERAS}")
        self.env = env
        self.output_dir = Path(output_dir)
        self.requested_fps = float(fps)
        self.step_seconds = 1.0 / physics_hz
        self.record_every = max(1, round(physics_hz / fps))
        attributes = {"external": "camera", "front": "front_camera", "head": "head_camera",
                      "left": "left_arm_camera", "right": "right_arm_camera"}
        self.cameras = tuple(name for name in which if getattr(env, attributes[name], None) is not None)
        if not self.cameras:
            raise ValueError("None of the requested recording cameras is available")
        missing = set(which) - set(self.cameras)
        if missing:
            print(f"[record] Unavailable cameras skipped: {', '.join(sorted(missing))}")
        self.active = False
        self._original_step = env.step_simulation
        self._closed = False
        env.step_simulation = self._step
        atexit.register(self.close)

    def _step(self):
        result = self._original_step()
        if self.active:
            self._steps += 1
            self._check_writer()
            if self._steps % self.record_every == 0:
                self._capture()
        return result

    def _check_writer(self):
        if self._error is not None:
            raise EpisodeRecordingError(f"Episode writer failed: {self._error}") from self._error

    def _capture(self):
        frames = {}
        for name in self.cameras:
            try:
                rgb, _, _ = getattr(self.env, CAMERA_METHODS[name])(
                    return_depth=False, return_seg=False, renderer=self.env.camera_renderer
                )
                frame = np.asarray(rgb)
                if frame.ndim != 3 or frame.shape[2] < 3:
                    raise ValueError(f"Invalid RGB image shape: {frame.shape}")
                # The writer owns its copy while physics and future captures continue.
                frames[name] = np.array(frame[:, :, :3], dtype=np.uint8, copy=True)
            except Exception as exc:
                raise EpisodeRecordingError(f"Cannot record {name} camera: {exc}") from exc
        if "image_sizes" not in self.metadata:
            self.metadata["image_sizes"] = {name: [frame.shape[1], frame.shape[0]] for name, frame in frames.items()}
        try:
            self._queue.put_nowait((self._steps * self.step_seconds, frames))
        except Full:
            self._dropped += 1
        self._last_capture_step = self._steps

    def _write_manifest(self):
        temporary = self.path / "episode.json.tmp"
        temporary.write_text(json.dumps(self.metadata, indent=2) + "\n")
        temporary.replace(self.path / "episode.json")

    def start(self, **metadata):
        if self._closed or self.active:
            raise RuntimeError("Recorder is closed or an episode is already active")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.path = self.output_dir / f"episode_{stamp}_{uuid.uuid4().hex[:6]}"
        self.path.mkdir()
        for name in self.cameras:
            (self.path / name).mkdir()
        self.metadata = {
            "format_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "cameras": list(self.cameras),
            "fps": 1.0 / (self.record_every * self.step_seconds),
            "requested_fps": self.requested_fps,
            "time_base": "simulation",
            "status": "recording",
            "n_frames": 0,
            "dropped_frames": 0,
            "scenario_metadata": metadata,
        }
        self._write_manifest()
        self._queue = Queue(maxsize=8)
        self._stop = threading.Event()
        self._error = None
        self._steps = self._written = self._dropped = 0
        self._last_capture_step = -1
        self.active = True
        self._thread = threading.Thread(target=self._write_frames, name="episode-writer", daemon=True)
        self._thread.start()
        print(f"[record] Recording {self.path} ({self.metadata['fps']:g} FPS in simulation time)")
        try:
            self._capture()
        except BaseException:
            self.finish(status="error")
            raise
        return self.path

    def _write_frames(self):
        try:
            import cv2

            with (self.path / "frames.jsonl").open("w", buffering=1) as index:
                while not self._stop.is_set() or not self._queue.empty():
                    try:
                        timestamp, frames = self._queue.get(timeout=0.05)
                    except Empty:
                        continue
                    images = {}
                    for name, rgb in frames.items():
                        relative = f"{name}/{self._written:06d}.png"
                        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        if not cv2.imwrite(str(self.path / relative), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                            raise OSError(f"Failed to write {relative}")
                        images[name] = relative
                    # Publish a frame only after every camera image is on disk.
                    index.write(json.dumps({"timestamp": timestamp, "images": images}) + "\n")
                    self._written += 1
        except BaseException as exc:
            self._error = exc

    def finish(self, status="completed"):
        if not self.active:
            return
        capture_error = None
        try:
            if status == "completed" and self._error is None and self._steps != self._last_capture_step:
                self._capture()
        except BaseException as exc:
            capture_error = exc
        finally:
            self.active = False
            self._stop.set()
            self._thread.join()
            self.metadata.update(
                status="error" if self._error is not None or capture_error is not None else status,
                n_frames=self._written,
                dropped_frames=self._dropped,
                duration_sim_seconds=self._steps * self.step_seconds,
            )
            if self._error is not None:
                self.metadata["error"] = str(self._error)
            self._write_manifest()
            print(f"[record] Saved {self._written} frames to {self.path} ({self._dropped} dropped)")
        if capture_error is not None:
            raise capture_error
        self._check_writer()

    @contextmanager
    def episode(self, **metadata):
        self.start(**metadata)
        try:
            yield self
        except BaseException as exc:
            self.finish(status="interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "error")
            raise
        else:
            self.finish()

    def close(self):
        if self._closed:
            return
        try:
            self.finish(status="interrupted")
        finally:
            self.env.step_simulation = self._original_step
            self._closed = True
            atexit.unregister(self.close)
