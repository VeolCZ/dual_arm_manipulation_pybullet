"""OpenCV display worker; this module never calls PyBullet."""

import atexit
import multiprocessing as mp
import os
from queue import Empty, Full

import numpy as np


def camera_label(name):
    return "topdown" if name == "external" else (name or "")


def camera_panel_order(cameras):
    """Keep the four-tile layout; add a third column when all five cameras show."""
    if "front" not in cameras:
        return ("external", "head", "left", "right")
    if "head" in cameras:
        return ("external", "front", "head", "left", "right", None)
    return ("external", "front", "left", "right")


def make_camera_mosaics(rgb_frames, depth_frames=None):
    """Build display mosaics using the smallest selected camera's tile size."""
    import cv2

    smallest = min(rgb_frames.values(), key=lambda frame: frame.shape[0] * frame.shape[1])
    height, width = smallest.shape[:2]
    order = camera_panel_order(rgb_frames)

    def rgb_tile(name):
        rgb = rgb_frames.get(name)
        if rgb is not None and rgb.shape[:2] != (height, width):
            rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
        tile = np.zeros((height, width, 3), dtype=np.uint8) if rgb is None else cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if name is not None:
            label = camera_label(name)
            cv2.putText(tile, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(tile, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        return tile

    def mosaic(tiles):
        columns = len(order) // 2
        return np.vstack((np.hstack(tiles[:columns]), np.hstack(tiles[columns:])))

    rgb_mosaic = mosaic([rgb_tile(name) for name in order])
    depth_mosaic = None
    if depth_frames is not None:
        def depth_tile(name):
            depth = depth_frames.get(name, np.zeros((height, width), dtype=np.float32))
            if depth.shape[:2] != (height, width):
                depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)

        depth_mosaic = mosaic([depth_tile(name) for name in order])
    return rgb_mosaic, depth_mosaic


def _run_opencv_viewer(frames, stop, window_name):
    """Own all GUI calls on this process's main thread, even between captures."""
    import cv2

    opened = False
    try:
        while not stop.is_set():
            parent = mp.parent_process()
            if parent is not None and not parent.is_alive():
                break
            try:
                packet = frames.get(timeout=0.02)
                # Display the most recent packet if the producer got ahead.
                while True:
                    try:
                        packet = frames.get_nowait()
                    except Empty:
                        break
            except Empty:
                packet = None

            if packet is not None:
                rgb, depth = make_camera_mosaics(*packet)
                cv2.imshow(window_name, rgb)
                if depth is not None:
                    cv2.imshow(f"{window_name}_depth", depth)
                if not opened:
                    height, width = rgb.shape[:2]
                    print(f"[viewer] OpenCV display process {os.getpid()}: '{window_name}' ({width}x{height} mosaic)", flush=True)
                    opened = True

            if opened:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                try:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) == 0:
                        break
                except cv2.error:
                    pass  # Not all window systems support visibility queries.
    except Exception as exc:
        print(f"[viewer] OpenCV display process stopped: {exc}", flush=True)
    finally:
        stop.set()
        cv2.destroyAllWindows()


class OpenCVCameraViewer:
    """Publish without waiting for display; at most one frame waits in the queue."""

    def __init__(self, window_name="cameras"):
        # Never fork a process that already has PyBullet/OpenGL/Qt threads.
        context = mp.get_context("spawn")
        self._frames = context.Queue(maxsize=1)
        self._stop = context.Event()
        self._process = context.Process(
            target=_run_opencv_viewer,
            args=(self._frames, self._stop, window_name),
            name="camera-viewer",
            daemon=True,
        )
        self._closed = False
        try:
            self._process.start()
        except Exception:
            self._frames.close()
            raise
        atexit.register(self.close)

    def is_alive(self):
        return not self._closed and not self._stop.is_set() and self._process.is_alive()

    def submit(self, rgb_frames, depth_frames=None):
        if not self.is_alive():
            return False
        packet = (rgb_frames, depth_frames)
        try:
            self._frames.put_nowait(packet)
        except Full:
            try:
                self._frames.get_nowait()
            except Empty:
                pass
            try:
                self._frames.put_nowait(packet)
            except Full:
                pass  # Feeder still owns the slot; skip rather than wait.
        return True

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._process.join(timeout=0.5)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=0.5)
        # A slow or closed consumer must never hold up interpreter shutdown.
        self._frames.cancel_join_thread()
        self._frames.close()
        atexit.unregister(self.close)
