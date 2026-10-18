from grasp_generator import GraspGenerator
from environment.utilities import Camera
from environment.env import Environment
from behaviors import Behaviors
from utils import YcbObjects, PackPileData, IsolatedObjData, summarize
from contextlib import nullcontext
from copy import deepcopy
from environment.episode_recorder import CAMERAS, EpisodeRecorder, EpisodeRecordingError
from environment.domain_randomization import DomainRandomizer

import numpy as np
import pybullet as p
import argparse
import os
import sys
import cv2
import math
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import time
import traceback


class GrasppingScenarios:
    """A Behaviors-based version of simulation.py.

    The core scenario logic (network inference, object spawning, metrics) is kept
    the same, but robot actions are routed through the `Behaviors` layer.
    """

    def __init__(self, network_model: str = "GGCNN"):
        self.network_model = network_model

        if network_model == "GR_ConvNet":
            self.IMG_SIZE = 224
            self.network_path = (
                "trained_models/GR_ConvNet/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98"
            )
            sys.path.append("trained_models/GR_ConvNet")
        
        # elif (network_model == "GGCNN"):
        
        
        else:
            print("The selected network has not been implemented yet!")
            raise SystemExit(1)

        self.CAM_Z = 1.9
        self.depth_radius = 1
        self.ATTEMPTS = 4

        # Camera streaming defaults (used by all scenarios).
        self.stream_show_depth = False
        self.stream_max_fps = 5.0
        self.stream_pause_s = 0.0
        self.stream_use_blit = True
        self.stream_backend = "opencv"
        self.stream_async = True
        self.stream_which = None
        self.record_episodes = False
        self.record_dir = "episodes"
        self.record_fps = 3.0
        self.record_which = ("front", "external", "left", "right")
        self.domain_randomization = False
        self.domain_seed = 3
        self._env = None

        # Offscreen figure for optional network visualisations (avoids opening a GUI window).
        self.fig = Figure(figsize=(10, 10))
        FigureCanvasAgg(self.fig)

    def _make_env(
        self,
        camera: Camera,
        *,
        vis: bool,
        debug: bool,
        finger_length: float = 0.06,
        hide_gui_panels: bool = True,
        stream_cameras: bool = False,
        stream_which=None,
    ) -> Environment:
        env = Environment(
            camera,
            vis=vis,
            debug=debug,
            finger_length=finger_length,
            hide_gui_panels=hide_gui_panels,
        )
        self._env = env

        if self.domain_randomization:
            env._domain_randomizer = DomainRandomizer(env, seed=self.domain_seed)

        if stream_cameras and vis:
            kwargs = dict(
                enabled=True,
                show_depth=bool(self.stream_show_depth),
                max_fps=float(self.stream_max_fps),
                pause_s=float(self.stream_pause_s),
                use_blit=bool(self.stream_use_blit),
                backend=self.stream_backend,
                async_display=self.stream_async,
            )
            which = self.stream_which if self.stream_which is not None else stream_which
            if which is not None:
                kwargs["which"] = which
            env.enable_camera_streaming(**kwargs)

            # Disable expensive debug drawing (axes + rays) for speed.
            try:
                env.enable_camera_frame_debug(enabled=False)
            except Exception:
                pass

        if self.record_episodes:
            env._episode_recorder = EpisodeRecorder(
                env, self.record_dir, fps=self.record_fps, which=self.record_which
            )
        return env

    def _record_episode(self, env, **metadata):
        recorder = getattr(env, "_episode_recorder", None)
        randomizer = getattr(env, "_domain_randomizer", None)
        if randomizer is not None:
            metadata["domain_randomization"] = deepcopy(randomizer.metadata)
        return recorder.episode(network=self.network_model, **metadata) if recorder else nullcontext()

    @staticmethod
    def _begin_domain_episode(env):
        randomizer = getattr(env, "_domain_randomizer", None)
        if randomizer is not None:
            settings = randomizer.begin_episode()
            print(f"[domain] Episode seed: {settings['episode_seed']}")

    def close(self):
        if self._env is not None:
            try:
                self._env.close()
            finally:
                self._env = None

    def draw_predicted_top_down_grasp(self, grasps, color=(0, 0, 1), lineIDs=None):
        x, y, z, yaw, opening_len, obj_height = grasps

        z += 0.1 # Lift the visualization slightly above the table for better visibility.

        if lineIDs is None:
            lineIDs = []

        # Draw the gripper's feasible opening, with 2 cm total visual padding.
        # Network widths are full gaps; the endpoints need half that span.
        # Keep the raw prediction unchanged for inference and execution.
        displayed_opening = float(np.clip(opening_len, 0.0, GraspGenerator.MAX_GRASP))
        half_span = (displayed_opening + 0.02) / 2.0
        finger_size = 0.12
        lineIDs.append(p.addUserDebugLine([x, y, z], [x, y, z + 0.15], color, lineWidth=6))

        lineIDs.append(
            p.addUserDebugLine(
                [x - half_span * math.sin(yaw), y - half_span * math.cos(yaw), z],
                [x + half_span * math.sin(yaw), y + half_span * math.cos(yaw), z],
                color,
                lineWidth=6,
            )
        )

        lineIDs.append(
            p.addUserDebugLine(
                [x - half_span * math.sin(yaw), y - half_span * math.cos(yaw), z],
                [x - half_span * math.sin(yaw), y - half_span * math.cos(yaw), z - finger_size],
                color,
                lineWidth=6,
            )
        )
        lineIDs.append(
            p.addUserDebugLine(
                [x + half_span * math.sin(yaw), y + half_span * math.cos(yaw), z],
                [x + half_span * math.sin(yaw), y + half_span * math.cos(yaw), z - finger_size],
                color,
                lineWidth=6,
            )
        )

        return lineIDs

    def _cameras_viewer(self, env: Environment, bgr, vis: bool, viewer_warned: bool, show_cameras: bool = True) -> bool:
        if (not show_cameras) or (not vis) or getattr(env, "_camera_stream_configured", False):
            return viewer_warned

        ok = False
        try:
            if self.stream_backend == "opencv":
                ok = env.visualize_cameras(max_fps=self.stream_max_fps, async_display=self.stream_async)
            else:
                ok = env.visualize_cameras_matplotlib(
                    pause_s=self.stream_pause_s,
                    max_fps=self.stream_max_fps,
                )
        except Exception as exc:
            if not viewer_warned:
                viewer_warned = True
                print(f"[viewer] {self.stream_backend} viewer failed: {exc}")
        if not ok and not viewer_warned:
            viewer_warned = True
            print(f"[viewer] {self.stream_backend} camera viewer unavailable")

        return viewer_warned

    def remove_drawing(self, lineIDs):
        for line in lineIDs:
            p.removeUserDebugItem(line)

    def dummy_simulation_steps(self, n, env: Environment | None = None):
        n_i = int(n)
        if n_i <= 0:
            return

        if env is None:
            step = p.stepSimulation
            for _ in range(n_i):
                step()
            return

        step = env.step_simulation
        for _ in range(n_i):
            step()

    @staticmethod
    def _env_has_any_objects(env: Environment | None) -> bool:
        """Best-effort check whether the Environment still has spawned objects."""
        if env is None:
            return False
        try:
            obj_ids = getattr(env, "obj_ids", None)
            if obj_ids is None:
                return False
            return len(obj_ids) > 0
        except Exception:
            return False

    @staticmethod
    def _segmentation_visible_body_ids(seg: np.ndarray | None) -> set[int]:
        """Extract visible body ids from a PyBullet segmentation mask buffer.

        PyBullet commonly encodes: seg = (objectUniqueId << 24) | linkIndex.
        Some renderers / flags may return raw object ids already. We support both.
        """
        if seg is None:
            return set()
        try:
            s = np.asarray(seg, dtype=np.int64)
            if s.size == 0:
                return set()

            raw_ids = set(int(v) for v in np.unique(s).tolist() if int(v) >= 0)
            shifted_ids = set(int(v) for v in np.unique(np.right_shift(s, 24)).tolist() if int(v) >= 0)

            # If the buffer is encoded, the raw values will be huge (~1<<24).
            # Prefer decoded ids, but keep raw as a fallback.
            if int(np.max(s)) >= (1 << 24):
                return shifted_ids
            return raw_ids | shifted_ids
        except Exception:
            return set()

    @staticmethod
    def _camera_sees_any_env_object(env: Environment | None, seg: np.ndarray | None) -> bool:
        """Return True if the external camera seg-mask includes any env.obj_ids."""
        if env is None:
            return False
        try:
            obj_ids = getattr(env, "obj_ids", None)
            if not obj_ids:
                return False
            env_ids = set(int(i) for i in obj_ids)
            visible_ids = GrasppingScenarios._segmentation_visible_body_ids(seg)
            return len(env_ids.intersection(visible_ids)) > 0
        except Exception:
            return False

    @staticmethod
    def _scene_has_objects_from_top_down_depth_camera(depth: np.ndarray, threshold: float = 0.0025) -> bool:
        if depth is None:
            return False
        try:
            return bool(float(depth.max() - depth.min()) >= float(threshold))
        except Exception:
            return False

    def is_there_any_object(self, camera, env: Environment | None = None):
        self.dummy_simulation_steps(10, env=env)
        if self._env_has_any_objects(env):
            return True
        if env is not None:
            _, depth, _ = env.get_external_cam_img(return_depth=True, return_seg=False)
        else:
            _, depth, _ = camera.get_cam_img(return_depth=True, return_seg=False)
        return self._scene_has_objects_from_top_down_depth_camera(depth)

    def isolated_obj_scenario(self, runs, device, vis, stream_cameras, output, debug):
        print("visualization = ", vis)
        print("stream cameras = ", stream_cameras)

        objects = YcbObjects(
            "objects/ycb_objects",
            mod_orn=["ChipsCan", "MustardBottle", "TomatoSoupCan"],
            mod_stiffness=["Strawberry"],
        )

        data = IsolatedObjData(objects.obj_names, runs, "results")

        center_x, center_y, center_z = 0.0, -0.54, self.CAM_Z
        camera = Camera(
            (center_x, center_y, center_z),
            (center_x, center_y, 0.785),
            0.2,
            2.0,
            (self.IMG_SIZE, self.IMG_SIZE),
            40,
        )
        env = self._make_env(
            camera,
            vis=vis,
            debug=debug,
            finger_length=0.06,
            hide_gui_panels=True,
            stream_cameras=bool(stream_cameras),
            stream_which=("front", "left", "right"),
        )
        behaviors = Behaviors(env)

        generator = GraspGenerator(
            self.network_path,
            camera,
            self.depth_radius,
            self.fig,
            self.IMG_SIZE,
            self.network_model,
            device,
        )

        viewer_warned = False
        bgr2rgb = cv2.cvtColor
        bgr2rgb_code = cv2.COLOR_BGR2RGB

        objects.shuffle_objects()
        for i in range(runs):
            print("----------- run ", i + 1, " -----------")
            print("network model = ", self.network_model)
            print("size of input image (W, H) = (", self.IMG_SIZE, " ,", self.IMG_SIZE, ")")

            for obj_name in objects.obj_names:
                print(obj_name)

                behaviors.reset_robot("dual", steps=1000)

                path, mod_orn, mod_stiffness = objects.get_obj_info(obj_name)
                self._begin_domain_episode(env)
                env.load_isolated_obj(path, mod_orn, mod_stiffness)
                self.dummy_simulation_steps(20, env=env)

                with self._record_episode(env, scenario="isolated", run=i + 1, object=obj_name):
                    number_of_attempts = self.ATTEMPTS
                    number_of_failures = 0
                    idx = 0
                    failed_grasp_counter = 0
                    while number_of_failures < number_of_attempts:
                        self.dummy_simulation_steps(10, env=env)

                        bgr, depth, _ = env.get_external_cam_img(return_depth=True, return_seg=False)
                        if not self._scene_has_objects_from_top_down_depth_camera(depth):
                            break

                        viewer_warned = self._cameras_viewer(
                            env,
                            bgr,
                            vis=vis,
                            viewer_warned=viewer_warned,
                            show_cameras=bool(stream_cameras),
                        )

                        rgb = bgr2rgb(bgr, bgr2rgb_code)

                        grasps, save_name = generator.predict_grasp(
                            rgb,
                            depth,
                            n_grasps=number_of_attempts,
                            show_output=output,
                        )
                        if not grasps:
                            self.dummy_simulation_steps(50, env=env)
                            if failed_grasp_counter > 3:
                                print("Failed to find a grasp points > 3 times. Skipping.")
                                break

                            failed_grasp_counter += 1
                            continue

                        if idx > len(grasps) - 1:
                            print("idx = ", idx)
                            if len(grasps) > 0:
                                idx = len(grasps) - 1
                            else:
                                number_of_failures += 1
                                continue

                        # Keep cleanup safe for headless (DIRECT) runs, where no
                        # PyBullet debug lines are created.
                        LID = []
                        if vis:
                            for g in grasps:
                                LID = self.draw_predicted_top_down_grasp(g, color=[1, 0, 1], lineIDs=LID)
                            time.sleep(0.5)
                            self.remove_drawing(LID)
                            self.dummy_simulation_steps(10, env=env)

                        # Visualize the selected grasp pose
                        if vis:
                            LID = self.draw_predicted_top_down_grasp(grasps[idx])


                        x, y, z, yaw, opening_len, obj_height = grasps[idx]
                        succes_grasp, succes_target = behaviors.grasp(
                            position_xyz=(x, y, z),
                            yaw=yaw,
                            opening_length=opening_len,
                            obj_height=obj_height,
                            debug=False,
                        )

                        behaviors.reset_robot("dual", steps=1000)
                        self.remove_drawing(LID)
                        self.dummy_simulation_steps(20, env=env)

  
                        data.add_try(obj_name)

                        if succes_grasp:
                            data.add_succes_grasp(obj_name)
                        if succes_target:
                            data.add_succes_target(obj_name)

                        behaviors.reset_robot("dual", steps=1000)

                        if succes_target:
                            number_of_failures = 0
                            if vis:
                                debugID = p.addUserDebugText("success", [-0.0, -0.9, 0.8], [0, 0.50, 0], textSize=2)
                                time.sleep(0.25)
                                p.removeUserDebugItem(debugID)

                            if save_name is not None:
                                os.rename(save_name + ".png", save_name + f"_SUCCESS_grasp{i}.png")

                        else:
                            number_of_failures += 1
                            idx += 1

                            if vis:
                                debugID = p.addUserDebugText("failed", [-0.0, -0.9, 0.8], [0.5, 0, 0], textSize=2)
                                time.sleep(0.25)
                                p.removeUserDebugItem(debugID)

        data.write_json(self.network_model)
        summarize(data.save_dir, runs, self.network_model)

    def packed_or_pile_scenario(self, runs, scenario, device, vis, stream_cameras, output, debug):
        number_of_objects = 5
        if scenario == "packed":
            objects = YcbObjects(
                "objects/ycb_objects",
                mod_orn=["ChipsCan", "MustardBottle", "TomatoSoupCan"],
                mod_stiffness=["Strawberry"],
                exclude=["CrackerBox"],
            )

            data = PackPileData(number_of_objects, runs, "results", "packed")

        elif scenario == "pile":
            objects = YcbObjects(
                "objects/ycb_objects",
                mod_orn=["ChipsCan", "MustardBottle", "TomatoSoupCan"],
                mod_stiffness=["Strawberry"],
                exclude=["CrackerBox"],
            )

            data = PackPileData(number_of_objects, runs, "results", "pile")
        else:
            raise ValueError("scenario must be 'packed' or 'pile'")

        center_x, center_y, center_z = 0.05, -0.52, self.CAM_Z
        camera = Camera(
            (center_x, center_y, center_z),
            (center_x, center_y, 0.785),
            0.2,
            2.0,
            (self.IMG_SIZE, self.IMG_SIZE),
            40,
        )
        env = self._make_env(
            camera,
            vis=vis,
            debug=debug,
            finger_length=0.06,
            hide_gui_panels=True,
            stream_cameras=bool(stream_cameras),
            stream_which=None,
        )
        behaviors = Behaviors(env)

        generator = GraspGenerator(
            self.network_path,
            camera,
            self.depth_radius,
            self.fig,
            self.IMG_SIZE,
            self.network_model,
            device,
        )

        viewer_warned = False
        bgr2rgb = cv2.cvtColor
        bgr2rgb_code = cv2.COLOR_BGR2RGB

        for i in range(runs):
            env.remove_all_obj()
            behaviors.reset_robot("dual", steps=1000)
            print("----------- run ", i + 1, " -----------")
            print("network model = ", self.network_model)
            print("size of input image (W, H) = (", self.IMG_SIZE, " ,", self.IMG_SIZE, ")")

            if vis:
                debugID = p.addUserDebugText(f"Experiment {i+1}", [-0.0, -0.9, 0.8], [0, 0, 255], textSize=2)
                time.sleep(0.5)
                p.removeUserDebugItem(debugID)

            number_of_failures = 0
            objects.shuffle_objects()

            info = objects.get_n_first_obj_info(number_of_objects)

            self._begin_domain_episode(env)
            if scenario == "packed":
                env.create_packed(info)
            elif scenario == "pile":
                env.create_pile(info)

            # If object spawning failed (or objects were cleared), end this run early.
            if not self._env_has_any_objects(env):
                print("No objects in the scene after spawning; skipping this run.")
                continue

            with self._record_episode(env, scenario=scenario, run=i + 1, objects=[item[0] for item in info]):
                number_of_attempts = self.ATTEMPTS
                failed_grasp_counter = 0
                flag_failed_grasp_counter = False

                while number_of_failures < number_of_attempts:
                    self.dummy_simulation_steps(10, env=env)

                    # Robust early-exit: once all objects are removed/moved, stop trying to grasp.
                    if not self._env_has_any_objects(env):
                        number_of_failures = number_of_attempts
                        break

                    try:
                        idx = 0
                        for attempt_idx in range(number_of_attempts):
                            bgr, depth, seg = env.get_external_cam_img(return_depth=True, return_seg=True)
                            if not self._scene_has_objects_from_top_down_depth_camera(depth):
                                number_of_failures = number_of_attempts
                                break

                            # If the objects exist but are not in the camera view, don't spam network inference.
                            if (seg is not None) and self._env_has_any_objects(env) and (not self._camera_sees_any_env_object(env, seg)):
                                if debug:
                                    print("Objects exist in env but are not visible in the external camera; ending run.")
                                number_of_failures = number_of_attempts
                                break

                            if not self._env_has_any_objects(env):
                                number_of_failures = number_of_attempts
                                break

                            data.add_try()
                            viewer_warned = self._cameras_viewer(
                                env,
                                bgr,
                                vis=vis,
                                viewer_warned=viewer_warned,
                                show_cameras=bool(stream_cameras),
                            )

                            rgb = bgr2rgb(bgr, bgr2rgb_code)

                            grasps, save_name = generator.predict_grasp(
                                rgb,
                                depth,
                                n_grasps=number_of_attempts,
                                show_output=output,
                            )
                            if not grasps:
                                self.dummy_simulation_steps(30, env=env)
                                if not self._env_has_any_objects(env):
                                    number_of_failures = number_of_attempts
                                    break

                                # Depth can be flat/invalid even when objects exist; print minimal stats.
                                try:
                                    d = np.asarray(depth, dtype=float)
                                    d_min = float(np.nanmin(d))
                                    d_max = float(np.nanmax(d))
                                    d_rng = float(d_max - d_min)
                                    n_nan = int(np.isnan(d).sum())
                                    # If this looks like a PyBullet depth buffer (0..1), also show metric depth.
                                    if 0.0 <= d_min <= 1.0 and 0.0 <= d_max <= 1.0:
                                        near = float(getattr(camera, "near", 0.2))
                                        far = float(getattr(camera, "far", 2.0))
                                        dd = np.clip(d, 1e-6, 1.0 - 1e-6)
                                        dm = (far * near) / (far - (far - near) * dd)
                                        dm_min = float(np.nanmin(dm))
                                        dm_max = float(np.nanmax(dm))
                                        dm_rng = float(dm_max - dm_min)
                                        print(
                                            f"could not find a grasp point! (depthbuf min={d_min:.4f} max={d_max:.4f} range={d_rng:.4f} nan={n_nan}; "
                                            f"depth_m min={dm_min:.3f} max={dm_max:.3f} range={dm_rng:.3f})"
                                        )
                                    else:
                                        print(f"could not find a grasp point! (depth min={d_min:.4f} max={d_max:.4f} range={d_rng:.4f} nan={n_nan})")
                                except Exception:
                                    print("could not find a grasp point!")
                                if failed_grasp_counter > 5:
                                    print("Failed to find grasp points > 5 times. Skipping.")
                                    flag_failed_grasp_counter = True
                                    break

                                failed_grasp_counter += 1
                                continue

                            # Keep cleanup safe for headless (DIRECT) runs, where
                            # no PyBullet debug lines are created.
                            LID = []
                            if vis:
                                for g in grasps:
                                    LID = self.draw_predicted_top_down_grasp(g, color=[1, 0, 1], lineIDs=LID)
                                self.dummy_simulation_steps(20, env=env)
                                # time.sleep(0.5)
                                self.remove_drawing(LID)
                                self.dummy_simulation_steps(20, env=env)

                            if idx > len(grasps) - 1:
                                print("idx = ", idx)
                                if len(grasps) > 0:
                                    idx = len(grasps) - 1
                                else:
                                    number_of_failures += 1
                                    continue

                            # Visualize the selected grasp pose
                            if vis:
                                LID = self.draw_predicted_top_down_grasp(grasps[idx])


                            x, y, z, yaw, opening_len, obj_height = grasps[idx]
                            succes_grasp, succes_target = behaviors.grasp(
                                position_xyz=(x, y, z),
                                yaw=yaw,
                                opening_length=opening_len,
                                obj_height=obj_height,
                                debug=False,
                            )

                            behaviors.reset_robot("dual", steps=1000)
                            self.remove_drawing(LID)
                            self.dummy_simulation_steps(20, env=env)

                            if succes_grasp:
                                data.add_succes_grasp()

                            if succes_target:
                                data.add_succes_target()
                                number_of_failures = 0

                                if vis:
                                    debugID = p.addUserDebugText("success", [-0.0, -0.9, 0.8], [0, 0.50, 0], textSize=2)
                                    time.sleep(0.25)
                                    p.removeUserDebugItem(debugID)

                                    if save_name is not None:
                                        os.rename(save_name + ".png", save_name + f"_SUCCESS_grasp{attempt_idx}.png")

                            else:
                                number_of_failures += 1
                                idx += 1

                                if vis:
                                    debugID = p.addUserDebugText("failed", [-0.0, -0.9, 0.8], [0.5, 0, 0], textSize=2)
                                    time.sleep(0.25)
                                    p.removeUserDebugItem(debugID)

                                if number_of_failures == number_of_attempts:
                                    if vis:
                                        debugID = p.addUserDebugText(
                                            "breaking point", [-0.0, -0.9, 0.8], [0.5, 0, 0], textSize=2
                                        )
                                        time.sleep(0.25)
                                        p.removeUserDebugItem(debugID)

                                    break

                    except Exception as exc:
                        if isinstance(exc, EpisodeRecordingError):
                            raise
                        print(f"An exception occurred during the experiment: {exc}")
                        if debug:
                            traceback.print_exc()
                        number_of_failures += 1
                        try:
                            behaviors.reset_robot("dual", steps=1000)
                        except Exception:
                            pass

                    if flag_failed_grasp_counter:
                        flag_failed_grasp_counter = False
                        break

        data.summarize()


def parse_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v is None:
            return False
        s = str(v).strip().lower()
        if s in ("yes", "y", "true", "t", "1", "on"):
            return True
        if s in ("no", "n", "false", "f", "0", "off"):
            return False
        raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v!r}")

    parser = argparse.ArgumentParser(description="Grasping demo (Behaviors-based)")

    parser.add_argument("--scenario", type=str, default="isolated", help="Grasping scenario (isolated/packed/pile)")
    parser.add_argument("--network", choices=("GR_ConvNet", "GGCNN"), default="GR_ConvNet", help="Grasp network: GR_ConvNet (224x224 RGB-D) or GGCNN (300x300 depth)")

    parser.add_argument("--runs", type=int, default=1, help="Number of runs the scenario is executed")
    parser.add_argument("--attempts", type=int, default=4, help="Consecutive placement failures before ending a round (default: 4)")

    parser.add_argument(
        "--save-network-output",
        dest="output",
        type=str2bool,
        default=True,
        help="Save network output (default: true)",
    )

    parser.add_argument("--device", type=str, default="GPU", help="device (CPU/GPU; default: GPU)")
    parser.add_argument("--gui", type=str2bool, default=True, help="GUI (True/False)")
    parser.add_argument("--domain_randomization", type=str2bool, default=False, help="Randomize only table/floor colors, object textures, lighting and RGB brightness per episode")
    parser.add_argument("--domain_seed", type=int, default=3, help="Seed for episode appearance (default: 3; does not seed object placement or physics)")
    parser.add_argument("--stream_cameras", type=str2bool, default=False, help="Show camera windows (default: false)")
    parser.add_argument("--stream_backend", choices=("opencv", "matplotlib"), default="opencv", help="Camera viewer backend (default: opencv)")
    parser.add_argument("--stream_async", type=str2bool, default=True, help="Display OpenCV cameras in a separate process (default: true; capture still uses the simulation thread)")
    parser.add_argument("--stream_fps", type=float, default=5.0, help="Maximum camera preview FPS (lower values leave more time for physics)")
    parser.add_argument("--stream_which", type=str, default=None, help="Comma-separated preview cameras: external,front,head,left,right (default: scenario cameras)")
    parser.add_argument("--record_episodes", type=str2bool, default=False, help="Save camera episodes for offline playback")
    parser.add_argument("--record_dir", default="episodes", help="Root folder for recorded episodes")
    parser.add_argument("--record_fps", type=float, default=3.0, help="Recording frames per simulated second (default: 3; camera capture adds cost)")
    parser.add_argument("--record_which", default="front,external,left,right", help="Comma-separated recording cameras (default: front,external,left,right; head is also available)")
    # parser.add_argument("--report", type=str2bool, default=True, help="report (True/False)")

    args = parser.parse_args()
    if not math.isfinite(args.record_fps) or not 0 < args.record_fps <= 240:
        parser.error("--record_fps must be positive and no greater than 240")
    args.record_which = tuple(name.strip().lower() for name in args.record_which.split(","))
    if any(name not in CAMERAS for name in args.record_which):
        parser.error("--record_which must contain camera names: external,front,head,left,right")
    if not math.isfinite(args.stream_fps) or args.stream_fps <= 0:
        parser.error("--stream_fps must be a positive finite number")
    if args.stream_which is not None:
        args.stream_which = tuple(name.strip().lower() for name in args.stream_which.split(","))
        if not args.stream_which or any(name not in CAMERAS for name in args.stream_which):
            parser.error("--stream_which must contain camera names: external,front,head,left,right")
    return args


if __name__ == "__main__":
    args = parse_args()
    output = args.output
    runs = args.runs
    device = args.device
    vis = args.gui

    grasp = GrasppingScenarios(args.network)
    grasp.ATTEMPTS = int(args.attempts)
    grasp.stream_backend = args.stream_backend
    grasp.stream_async = args.stream_async
    grasp.stream_max_fps = args.stream_fps
    grasp.stream_which = args.stream_which
    grasp.record_episodes = args.record_episodes
    grasp.record_dir = args.record_dir
    grasp.record_fps = args.record_fps
    grasp.record_which = args.record_which
    grasp.domain_randomization = args.domain_randomization
    grasp.domain_seed = args.domain_seed

    try:
        if args.scenario == "isolated":
            grasp.isolated_obj_scenario(
                runs,
                device,
                vis,
                stream_cameras=args.stream_cameras,
                output=output,
                debug=False,
            )
        elif args.scenario == "packed":
            grasp.packed_or_pile_scenario(
                runs,
                args.scenario,
                device,
                vis,
                stream_cameras=args.stream_cameras,
                output=output,
                debug=False,
            )
        elif args.scenario == "pile":
            grasp.packed_or_pile_scenario(
                runs,
                args.scenario,
                device,
                vis,
                stream_cameras=args.stream_cameras,
                output=output,
                debug=False,
            )
    finally:
        grasp.close()
