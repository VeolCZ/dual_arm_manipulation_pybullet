"""Play recorded camera episodes without importing PyBullet or the grasp network."""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
from environment.camera_viewer import camera_label


@dataclass
class Episode:
    path: Path
    metadata: dict
    frames: list


def discover_episodes(path):
    path = Path(path)
    if path.is_file() and path.name == "episode.json":
        return [path.parent]
    if (path / "episode.json").is_file():
        return [path]
    return sorted(manifest.parent for manifest in path.glob("**/episode.json"))


def load_episode(path):
    path = Path(path)
    metadata = json.loads((path / "episode.json").read_text())
    if metadata.get("format_version") != 1:
        raise ValueError(f"Unsupported episode format: {path}")
    frames = []
    index = path / "frames.jsonl"
    if index.exists():
        with index.open() as stream:
            for line in stream:
                try:
                    frame = json.loads(line)
                except json.JSONDecodeError:
                    if not line.endswith("\n"):
                        break  # Recover complete frames after an interrupted write.
                    raise
                stamp = float(frame["timestamp"])
                if not math.isfinite(stamp) or stamp < 0 or (frames and stamp <= frames[-1]["timestamp"]):
                    raise ValueError(f"Invalid frame timestamp in {path}")
                if set(frame["images"]) != set(metadata["cameras"]):
                    raise ValueError(f"Frame cameras do not match metadata in {path}")
                frame["timestamp"] = stamp
                frames.append(frame)
    return Episode(path, metadata, frames)


def read_frame(episode, index):
    result = {}
    for camera in episode.metadata["cameras"]:
        path = episode.path / episode.frames[index]["images"][camera]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read recorded image: {path}")
        result[camera] = image
    return result


def verify_episode(episode):
    sizes = {}
    for index in range(len(episode.frames)):
        for camera, image in read_frame(episode, index).items():
            if camera in sizes and image.shape != sizes[camera]:
                raise ValueError(f"Camera {camera} changes size in {episode.path}")
            sizes[camera] = image.shape
    if episode.metadata["status"] == "completed" and episode.metadata["n_frames"] != len(episode.frames):
        raise ValueError(f"Frame count does not match metadata in {episode.path}")
    return sizes


def make_canvas(episode, index, size=320, paused=False):
    tiles = []
    for camera, image in read_frame(episode, index).items():
        tile = cv2.resize(image, (size, size))
        label = camera_label(camera)
        cv2.putText(tile, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(tile, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    panels = np.hstack(tiles)
    header = np.full((72, panels.shape[1], 3), 24, dtype=np.uint8)
    stamp = episode.frames[index]["timestamp"]
    state = "PAUSED" if paused else "END" if index == len(episode.frames) - 1 else "PLAY"
    info = f"{index + 1}/{len(episode.frames)}  t={stamp:.3f}s (simulation)  {state}  {episode.metadata['status']}"
    cv2.putText(header, info, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(header, "Space: pause  A/D: step  R: restart  [/]: episode  Q/Esc/Ctrl+C: quit", (8, 51),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 220, 255), 1, cv2.LINE_AA)
    return np.vstack((header, panels))


def _wait_for_key(window, deadline=None):
    """Pump GUI events while regularly returning to Python for signal handling."""
    while True:
        delay_ms = 20
        if deadline is not None:
            delay_ms = max(1, min(delay_ms, math.ceil((deadline - time.monotonic()) * 1000)))
        # A zero delay stays inside native GUI code indefinitely and can defer
        # Python's SIGINT handler. Poll even when paused or at the final frame.
        key = cv2.waitKeyEx(delay_ms)
        try:
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) == 0:
                return 27
        except cv2.error:
            pass
        if key != -1:
            return key
        if deadline is not None and time.monotonic() >= deadline:
            return -1


def play_episode(episode, speed=1.0, size=320):
    """Return 'next', 'previous', or 'quit' after keyboard navigation."""
    if not episode.frames:
        print(f"[viewer] Skipping empty episode: {episode.path}")
        return "next"
    print(f"[viewer] {episode.path} ({len(episode.frames)} frames)")
    window = "Recorded Episodes"
    frame = 0
    paused = False
    deadline = time.monotonic()
    left = {ord("a"), 81, 2424832, 65361, 63234}
    right = {ord("d"), 83, 2555904, 65363, 63235}
    while True:
        cv2.imshow(window, make_canvas(episode, frame, size, paused))
        at_end = frame == len(episode.frames) - 1
        if paused or at_end:
            next_deadline = None
        else:
            next_deadline = deadline + (episode.frames[frame + 1]["timestamp"] - episode.frames[frame]["timestamp"]) / speed
        key = _wait_for_key(window, next_deadline)
        if key in (ord("q"), 27, 3):
            return "quit"
        if key in (ord("]"), ord("n")):
            return "next"
        if key in (ord("["), ord("p")):
            return "previous"
        if key == 32:
            if at_end:
                frame, paused = 0, False
            else:
                paused = not paused
            deadline = time.monotonic()
        elif key in left | right:
            frame = max(0, min(len(episode.frames) - 1, frame + (-1 if key in left else 1)))
            paused = True
            deadline = time.monotonic()
        elif key == ord("r"):
            frame, paused = 0, False
            deadline = time.monotonic()
        elif key == -1 and not paused and not at_end:
            frame += 1
            deadline = next_deadline


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Episode folder, episode.json, or root directory containing episodes")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed relative to recorded simulation time")
    parser.add_argument("--size", type=int, default=320, help="Display size in pixels per camera panel")
    parser.add_argument("--verify", action="store_true", help="Check metadata and every image without opening a GUI")
    args = parser.parse_args(argv)
    if not math.isfinite(args.speed) or args.speed <= 0 or args.size <= 0:
        parser.error("--speed and --size must be positive")
    paths = discover_episodes(args.path)
    if not paths:
        parser.error(f"No recorded episodes found in {args.path}")
    if args.verify:
        for path in paths:
            episode = load_episode(path)
            sizes = verify_episode(episode)
            print(f"[verify] {path}: {len(episode.frames)} frames, {sizes}, status={episode.metadata['status']}")
        return 0
    gui = next((line for line in cv2.getBuildInformation().splitlines() if "GUI:" in line), "")
    if "NONE" in gui:
        parser.error("Playback needs a GUI-enabled OpenCV build; --verify also works with headless OpenCV")
    index = 0
    try:
        while 0 <= index < len(paths):
            action = play_episode(load_episode(paths[index]), args.speed, args.size)
            if action == "quit":
                break
            index = max(0, index - 1) if action == "previous" else index + 1
    except KeyboardInterrupt:
        print("\n[viewer] Stopped.")
        return 130
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
