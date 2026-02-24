from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from typing import Any, Iterable, Literal, Optional, Sequence

import numpy as np
import pybullet as p
from PIL import Image

from environment.env import Environment

RobotSelector = Literal["left", "right", "dual", "any"]


@dataclass(frozen=True)
class BehaviorInfo:
    name: str
    robot: RobotSelector
    description: str


class Behaviors:
    """Robot behaviors wrapper for this PyBullet-based repo.

    This is intentionally lightweight (no MoveIt/ROS): it provides a single place
    to call common arm/gripper/object operations and keeps a registry of
    behaviors (left/right/dual/any) similar to your C++ service list.
    """

    def __init__(self, env: Environment):
        self.env = env

        # Debug visualization state (PyBullet user debug items).
        self._ee_target_debug_ids: dict[str, list[int]] = {"left": [], "right": []}
        self._ee_target_mesh_uid: dict[str, int] = {"left": -1, "right": -1}
        self._ee_traj_points: dict[str, list[tuple[float, float, float]]] = {"left": [], "right": []}
        self._ee_traj_debug_ids: dict[str, list[int]] = {"left": [], "right": []}

    # ---------------------------------------------------------------------
    # Registry
    # ---------------------------------------------------------------------

    @staticmethod
    def available_behaviors() -> list[BehaviorInfo]:
        return [
            # ================= COMMON =================
            BehaviorInfo("list_behaviors", "any", "Print list of available behaviors"),
            BehaviorInfo("get_object_pose_from_pybullet", "any", "Read object pose from PyBullet"),
            BehaviorInfo("print_object_pose_from_pybullet", "any", "Print object pose from PyBullet"),
            BehaviorInfo("set_object_pose_in_pybullet", "any", "Set object pose in PyBullet"),
            BehaviorInfo("text_to_speech", "any", "Publish text to speech (stub)"),

            # Cameras (wrappers around Environment.get_*_cam_img)
            BehaviorInfo("get_external_cam_img", "any", "Capture external RGB-D camera image"),
            BehaviorInfo("get_head_cam_img", "any", "Capture head camera image"),
            BehaviorInfo("get_left_arm_cam_img", "any", "Capture left wrist camera image"),
            BehaviorInfo("get_right_arm_cam_img", "any", "Capture right wrist camera image"),

            # Cameras (save to disk)
            BehaviorInfo("save_external_cam_img", "any", "Capture + save external RGB-D image to disk"),
            BehaviorInfo("save_head_cam_img", "any", "Capture + save head camera image to disk"),
            BehaviorInfo("save_left_arm_cam_img", "any", "Capture + save left wrist camera image to disk"),
            BehaviorInfo("save_right_arm_cam_img", "any", "Capture + save right wrist camera image to disk"),

            # ================= LEFT =================
            BehaviorInfo("go_to_initial_pose", "left", "Move left arm to initial pose"),
            BehaviorInfo("go_to_initial_top_pose", "left", "Move left arm to top pose"),
            BehaviorInfo("go_to_initial_top_close_pose", "left", "Move left arm to close top pose"),
            BehaviorInfo("IK_move_to_pose", "left", "Move left arm using IK"),
            BehaviorInfo("IK_move_to_pose_real_time", "left", "IK realtime control (best-effort)"),
            BehaviorInfo("tossing_object", "left", "Throw object with arm motion (stub)"),
            BehaviorInfo("read_all_joints", "left", "Read left arm joints"),
            BehaviorInfo("get_specific_joint_value", "left", "Read specific left joint"),
            BehaviorInfo("set_all_joints", "left", "Set all left joints"),
            BehaviorInfo("set_specific_joint_value", "left", "Set specific left joint"),
            BehaviorInfo("directly_set_all_joints", "left", "Send direct joint trajectory"),
            BehaviorInfo("open_gripper", "left", "Open left gripper"),
            BehaviorInfo("close_gripper", "left", "Close left gripper"),
            BehaviorInfo("partially_open_gripper", "left", "Partially open left gripper"),
            BehaviorInfo("read_current_ee_pose", "left", "Read left tool0 pose"),
            BehaviorInfo("enable_eef_mesh_debug", "left", "Attach tool0 mesh marker to left EE"),
            BehaviorInfo("visualize_target_ee_pose", "left", "Visualize left EE target"),
            BehaviorInfo("visualize_ee_trajectory", "left", "Visualize left EE trajectory"),
            BehaviorInfo("clear_ee_trajectory", "left", "Clear left EE trajectory"),

            # ================= RIGHT =================
            BehaviorInfo("go_to_initial_pose", "right", "Move right arm to initial pose"),
            BehaviorInfo("go_to_initial_top_pose", "right", "Move right arm to top pose"),
            BehaviorInfo("go_to_initial_top_close_pose", "right", "Move right arm to close top pose"),
            BehaviorInfo("IK_move_to_pose", "right", "Move right arm using IK"),
            BehaviorInfo("IK_move_to_pose_real_time", "right", "IK realtime control (best-effort)"),
            BehaviorInfo("tossing_object", "right", "Throw object with arm motion (stub)"),
            BehaviorInfo("read_all_joints", "right", "Read right arm joints"),
            BehaviorInfo("get_specific_joint_value", "right", "Read specific right joint"),
            BehaviorInfo("set_all_joints", "right", "Set all right joints"),
            BehaviorInfo("set_specific_joint_value", "right", "Set specific right joint"),
            BehaviorInfo("directly_set_all_joints", "right", "Send direct joint trajectory"),
            BehaviorInfo("open_gripper", "right", "Open right gripper"),
            BehaviorInfo("close_gripper", "right", "Close right gripper"),
            BehaviorInfo("partially_open_gripper", "right", "Partially open right gripper"),
            BehaviorInfo("read_current_ee_pose", "right", "Read right tool0 pose"),
            BehaviorInfo("enable_eef_mesh_debug", "right", "Attach tool0 mesh marker to right EE"),
            BehaviorInfo("visualize_target_ee_pose", "right", "Visualize right EE target"),
            BehaviorInfo("visualize_ee_trajectory", "right", "Visualize right EE trajectory"),
            BehaviorInfo("clear_ee_trajectory", "right", "Clear right EE trajectory"),

            # ================= DUAL =================
            BehaviorInfo("go_to_initial_pose", "dual", "Move both arms to initial pose"),
            BehaviorInfo("go_to_initial_top_pose", "dual", "Move both arms to top pose"),
            BehaviorInfo("go_to_initial_top_close_pose", "dual", "Move both arms to close top pose"),
            BehaviorInfo("IK_move_to_pose", "dual", "IK move both arms"),
            BehaviorInfo("IK_move_to_pose_real_time", "dual", "Realtime IK both arms (best-effort)"),
            BehaviorInfo("tossing_object", "dual", "Throw object with arm motion (stub)"),
            BehaviorInfo("read_all_joints", "dual", "Read all joints"),
            BehaviorInfo("get_specific_joint_value", "dual", "Read joint by index"),
            BehaviorInfo("set_all_joints", "dual", "Set all joints"),
            BehaviorInfo("set_specific_joint_value", "dual", "Set joint by index"),
            BehaviorInfo("directly_set_all_joints", "dual", "Send direct joint trajectory"),
            BehaviorInfo("open_gripper", "dual", "Open both grippers"),
            BehaviorInfo("close_gripper", "dual", "Close both grippers"),
            BehaviorInfo("partially_open_gripper", "dual", "Partially open both grippers"),
            BehaviorInfo("read_current_ee_pose", "dual", "Read left+right tool0 pose"),
            BehaviorInfo("enable_eef_mesh_debug", "dual", "Attach tool0 mesh marker to both EEs"),
            BehaviorInfo("visualize_target_ee_pose", "dual", "Visualize both EE targets"),
            BehaviorInfo("visualize_ee_trajectory", "dual", "Visualize both EE trajectories"),
            BehaviorInfo("clear_ee_trajectory", "dual", "Clear both EE trajectories"),
        ]

    def list_behaviors(self) -> list[BehaviorInfo]:
        behaviors = self.available_behaviors()
        for b in behaviors:
            print(f"{b.name:28s} | robot={b.robot:5s} | {b.description}")
        return behaviors

    def run(self, behavior: str, robot: RobotSelector = "any", **kwargs):
        """Dispatch a behavior by name.

        Intended to mirror the service-style usage in your C++ reference:
        you can pass `behavior` as a string and `robot` as left/right/dual/any.
        """

        fn = getattr(self, str(behavior), None)
        if fn is None or not callable(fn):
            raise ValueError(f"Unknown behavior: {behavior!r}")

        # Common behaviors ignore `robot`.
        if behavior in {
            "list_behaviors",
            "get_object_pose_from_pybullet",
            "print_object_pose_from_pybullet",
            "set_object_pose_in_pybullet",
            "text_to_speech",
        }:
            return fn(**kwargs)

        return fn(robot, **kwargs)

    # ---------------------------------------------------------------------
    # Object pose (PyBullet)
    # ---------------------------------------------------------------------

    def get_object_pose_from_pybullet(self, body_id: int):
        pos, orn = p.getBasePositionAndOrientation(int(body_id))
        return {"position": pos, "orientation_xyzw": orn}

    def print_object_pose_from_pybullet(self, body_id: int) -> None:
        pose = self.get_object_pose_from_pybullet(body_id)
        pos = pose["position"]
        orn = pose["orientation_xyzw"]
        print(
            f"object[{int(body_id)}] pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}) "
            f"quat=({orn[0]:.4f}, {orn[1]:.4f}, {orn[2]:.4f}, {orn[3]:.4f})"
        )

    def set_object_pose_in_pybullet(
        self,
        body_id: int,
        *,
        position: Sequence[float],
        orientation_xyzw: Optional[Sequence[float]] = None,
        orientation_rpy: Optional[Sequence[float]] = None,
    ) -> None:
        if orientation_xyzw is None:
            if orientation_rpy is None:
                orientation_xyzw = (0.0, 0.0, 0.0, 1.0)
            else:
                orientation_xyzw = p.getQuaternionFromEuler(list(orientation_rpy))
        p.resetBasePositionAndOrientation(int(body_id), list(position), list(orientation_xyzw))

    # ---------------------------------------------------------------------
    # Misc
    # ---------------------------------------------------------------------

    def text_to_speech(self, text: str) -> None:
        # Stub: this repo has no TTS backend; keep as a single place to hook one in.
        print(f"[tts] {text}")

    # ---------------------------------------------------------------------
    # Cameras
    # ---------------------------------------------------------------------

    @staticmethod
    def _save_cam_outputs(
        *,
        rgb: Any,
        depth: Any,
        seg: Any,
        out_dir: str,
        prefix: str,
    ) -> dict[str, str]:
        os.makedirs(out_dir, exist_ok=True)

        paths: dict[str, str] = {}

        if rgb is not None:
            rgb_arr = np.asarray(rgb)
            # Common cases: uint8 [H,W,3] or float [0..1].
            if rgb_arr.dtype != np.uint8:
                mx = float(np.max(rgb_arr)) if rgb_arr.size else 0.0
                if mx <= 1.0:
                    rgb_arr = rgb_arr * 255.0
                rgb_arr = np.clip(rgb_arr, 0.0, 255.0).astype(np.uint8)

            # If an alpha channel exists, keep it (PIL supports RGBA).
            rgb_path = os.path.join(out_dir, f"{prefix}_rgb.png")
            Image.fromarray(rgb_arr).save(rgb_path)
            paths["rgb_png"] = rgb_path

        if depth is not None:
            depth_path = os.path.join(out_dir, f"{prefix}_depth.npy")
            np.save(depth_path, np.asarray(depth))
            paths["depth_npy"] = depth_path

        if seg is not None:
            seg_path = os.path.join(out_dir, f"{prefix}_seg.npy")
            np.save(seg_path, np.asarray(seg))
            paths["seg_npy"] = seg_path

        return paths

    @staticmethod
    def _timestamp_suffix() -> str:
        # Include microseconds to avoid collisions when saving multiple frames quickly.
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    @classmethod
    def _default_cam_prefix(cls, camera_name: str) -> str:
        return f"{camera_name}_{cls._timestamp_suffix()}"

    @classmethod
    def _prefix_with_timestamp(cls, prefix: str) -> str:
        # Always suffix timestamp to prevent replacement when called repeatedly.
        return f"{str(prefix)}_{cls._timestamp_suffix()}"

    def get_external_cam_img(
        self,
        robot: RobotSelector = "any",
        *,
        return_depth: bool = True,
        return_seg: bool = False,
        renderer=None,
    ):
        """Wrapper for Environment.get_external_cam_img().

        Note: `robot` is accepted for uniform REPL calling but is ignored.
        """
        _ = robot
        return self.env.get_external_cam_img(
            return_depth=bool(return_depth),
            return_seg=bool(return_seg),
            renderer=renderer,
        )

    def save_external_cam_img(
        self,
        robot: RobotSelector = "any",
        *,
        out_dir: str = "results/camera",
        prefix: Optional[str] = None,
        return_depth: bool = True,
        return_seg: bool = False,
        renderer=None,
    ) -> dict[str, str]:
        """Capture external RGB-D camera and save to disk.

        Saves RGB as PNG and depth/seg as NPY (lossless, preserves units).
        Returns a dict of written file paths.

        Note: `robot` is accepted for uniform REPL calling but is ignored.
        """
        _ = robot
        rgb, depth, seg = self.get_external_cam_img(
            return_depth=bool(return_depth),
            return_seg=bool(return_seg),
            renderer=renderer,
        )
        if prefix is None:
            prefix = self._default_cam_prefix("external")
        else:
            prefix = self._prefix_with_timestamp(prefix)
        return self._save_cam_outputs(rgb=rgb, depth=depth, seg=seg, out_dir=str(out_dir), prefix=str(prefix))

    def get_head_cam_img(
        self,
        robot: RobotSelector = "any",
        *,
        return_depth: bool = True,
        return_seg: bool = True,
        renderer=None,
    ):
        """Wrapper for Environment.get_head_cam_img().

        Note: `robot` is accepted for uniform REPL calling but is ignored.
        """
        _ = robot
        return self.env.get_head_cam_img(
            return_depth=bool(return_depth),
            return_seg=bool(return_seg),
            renderer=renderer,
        )

    def save_head_cam_img(
        self,
        robot: RobotSelector = "any",
        *,
        out_dir: str = "results/camera",
        prefix: Optional[str] = None,
        return_depth: bool = True,
        return_seg: bool = True,
        renderer=None,
    ) -> dict[str, str]:
        """Capture head camera and save to disk (RGB PNG, depth/seg NPY)."""
        _ = robot
        rgb, depth, seg = self.get_head_cam_img(
            return_depth=bool(return_depth),
            return_seg=bool(return_seg),
            renderer=renderer,
        )
        if prefix is None:
            prefix = self._default_cam_prefix("head")
        else:
            prefix = self._prefix_with_timestamp(prefix)
        return self._save_cam_outputs(rgb=rgb, depth=depth, seg=seg, out_dir=str(out_dir), prefix=str(prefix))

    def get_left_arm_cam_img(
        self,
        robot: RobotSelector = "any",
        *,
        return_depth: bool = True,
        return_seg: bool = True,
        renderer=None,
    ):
        """Wrapper for Environment.get_left_arm_cam_img().

        Note: `robot` is accepted for uniform REPL calling but is ignored.
        """
        _ = robot
        return self.env.get_left_arm_cam_img(
            return_depth=bool(return_depth),
            return_seg=bool(return_seg),
            renderer=renderer,
        )

    def save_left_arm_cam_img(
        self,
        robot: RobotSelector = "any",
        *,
        out_dir: str = "results/camera",
        prefix: Optional[str] = None,
        return_depth: bool = True,
        return_seg: bool = True,
        renderer=None,
    ) -> dict[str, str]:
        """Capture left wrist camera and save to disk (RGB PNG, depth/seg NPY)."""
        _ = robot
        rgb, depth, seg = self.get_left_arm_cam_img(
            return_depth=bool(return_depth),
            return_seg=bool(return_seg),
            renderer=renderer,
        )
        if prefix is None:
            prefix = self._default_cam_prefix("left_wrist")
        else:
            prefix = self._prefix_with_timestamp(prefix)
        return self._save_cam_outputs(rgb=rgb, depth=depth, seg=seg, out_dir=str(out_dir), prefix=str(prefix))

    def get_right_arm_cam_img(
        self,
        robot: RobotSelector = "any",
        *,
        return_depth: bool = True,
        return_seg: bool = True,
        renderer=None,
    ):
        """Wrapper for Environment.get_right_arm_cam_img().

        Note: `robot` is accepted for uniform REPL calling but is ignored.
        """
        _ = robot
        return self.env.get_right_arm_cam_img(
            return_depth=bool(return_depth),
            return_seg=bool(return_seg),
            renderer=renderer,
        )

    def save_right_arm_cam_img(
        self,
        robot: RobotSelector = "any",
        *,
        out_dir: str = "results/camera",
        prefix: Optional[str] = None,
        return_depth: bool = True,
        return_seg: bool = True,
        renderer=None,
    ) -> dict[str, str]:
        """Capture right wrist camera and save to disk (RGB PNG, depth/seg NPY)."""
        _ = robot
        rgb, depth, seg = self.get_right_arm_cam_img(
            return_depth=bool(return_depth),
            return_seg=bool(return_seg),
            renderer=renderer,
        )
        if prefix is None:
            prefix = self._default_cam_prefix("right_wrist")
        else:
            prefix = self._prefix_with_timestamp(prefix)
        return self._save_cam_outputs(rgb=rgb, depth=depth, seg=seg, out_dir=str(out_dir), prefix=str(prefix))

    # ---------------------------------------------------------------------
    # Low-level helpers (arms/joints)
    # ---------------------------------------------------------------------

    def _arm_joint_names(self, robot: Literal["left", "right"]) -> list[str]:
        if robot == "left":
            return list(self.env.left_controlJoints)
        return list(self.env.right_controlJoints)

    def _eef_link_id(self, robot: Literal["left", "right"]) -> int:
        return int(self.env.left_eef_id if robot == "left" else self.env.right_eef_id)

    def _set_arm_joints_position_control(
        self,
        robot: Literal["left", "right"],
        joints: Sequence[float],
        *,
        steps: int = 240,
    ) -> bool:
        names = self._arm_joint_names(robot)
        if len(joints) != len(names):
            raise ValueError(f"Expected {len(names)} joint values, got {len(joints)}")

        for _ in range(int(steps)):
            for i, name in enumerate(names):
                j = self.env.joints[name]
                p.setJointMotorControl2(
                    self.env.robot_id,
                    j.id,
                    p.POSITION_CONTROL,
                    targetPosition=float(joints[i]),
                    force=float(j.maxForce) * float(getattr(self.env, "ARM_FORCE_SCALE", 1.0)),
                    maxVelocity=float(j.maxVelocity),
                    positionGain=float(getattr(self.env, "ARM_POSITION_GAIN", 0.04)),
                    velocityGain=float(getattr(self.env, "ARM_VELOCITY_GAIN", 1.0)),
                )
            self.env.step_simulation()
        return True

    def _reset_ee_debug(self, robot: Literal["left", "right"]) -> None:
        for uid in list(self._ee_target_debug_ids.get(robot, [])):
            try:
                p.removeUserDebugItem(int(uid))
            except Exception:
                pass
        self._ee_target_debug_ids[robot] = []

        uid_mesh = int(self._ee_target_mesh_uid.get(robot, -1))
        if uid_mesh >= 0:
            try:
                p.removeBody(uid_mesh)
            except Exception:
                pass
        self._ee_target_mesh_uid[robot] = -1

    # ---------------------------------------------------------------------
    # Preset poses
    # ---------------------------------------------------------------------

    # Joint presets copied from your reference (C++ example) and aligned to the
    # env's control joint ordering (6 joints per arm).
    _PRESET_JOINTS: dict[str, dict[str, list[float]]] = {
        "left": {
            "initial_pose": [-0.035442654286519826, -1.5423272413066407, -2.494051933288574, 4.104316874141357, -0.09125262895692998, 3.072849750518799],
            "initial_top_pose": [-0.059717957173482716, -1.5630534452250977, -1.5702199935913086, -1.6049581966795863, 1.5958514213562012, 3.0467772483825684],
            "initial_top_close_pose": [-0.061277214680806935, -1.5659197133830567, -1.552475929260254, 1.5286590295979003, -1.5769899527179163, 4.651482582092285],
        },
        "right": {
            "initial_pose": [0.14553141593933105, -1.5720273456969203, 2.3909576574908655, -0.7894459527781983, 0.13949155807495117, -3.1556242148028772],
            "initial_top_pose": [0.08292031288146973, -1.5741168461241664, 1.5776737372027796, 4.726433026581564, -1.496913258229391, -3.1232035795794886],
            "initial_top_close_pose": [0.056728482246398926, -1.5901724300780238, 1.5467003027545374, 1.5719284254261474, 1.563293218612671, -4.658060614262716],
        },
    }

    def go_to_initial_pose(self, robot: RobotSelector = "dual", *, steps: int = 1000) -> bool:
        if robot == "dual" or robot == "any":
            ok_l = self.go_to_initial_pose("left", steps=steps)
            ok_r = self.go_to_initial_pose("right", steps=steps)
            return bool(ok_l and ok_r)
        if robot not in ("left", "right"):
            raise ValueError("robot must be 'left'|'right'|'dual'")
        return self._set_arm_joints_position_control(robot, self._PRESET_JOINTS[robot]["initial_pose"], steps=steps)

    def go_to_initial_top_pose(self, robot: RobotSelector = "dual", *, steps: int = 500) -> bool:
        if robot == "dual" or robot == "any":
            ok_l = self.go_to_initial_top_pose("left", steps=steps)
            ok_r = self.go_to_initial_top_pose("right", steps=steps)
            return bool(ok_l and ok_r)
        if robot not in ("left", "right"):
            raise ValueError("robot must be 'left'|'right'|'dual'")
        return self._set_arm_joints_position_control(robot, self._PRESET_JOINTS[robot]["initial_top_pose"], steps=steps)

    def go_to_initial_top_close_pose(self, robot: RobotSelector = "dual", *, steps: int = 500) -> bool:
        if robot == "dual" or robot == "any":
            ok_l = self.go_to_initial_top_close_pose("left", steps=steps)
            ok_r = self.go_to_initial_top_close_pose("right", steps=steps)
            return bool(ok_l and ok_r)
        if robot not in ("left", "right"):
            raise ValueError("robot must be 'left'|'right'|'dual'")
        return self._set_arm_joints_position_control(robot, self._PRESET_JOINTS[robot]["initial_top_close_pose"], steps=steps)

    # ---------------------------------------------------------------------
    # IK-based motions (using Environment.move_ee)
    # ---------------------------------------------------------------------

    def IK_move_to_pose(
        self,
        robot: RobotSelector,
        *,
        position_xyz: Sequence[float],
        left_position_xyz: Optional[Sequence[float]] = None,
        right_position_xyz: Optional[Sequence[float]] = None,
        yaw: float = 0.0,
        orientation_xyzw: Optional[Sequence[float]] = None,
        left_yaw: Optional[float] = None,
        right_yaw: Optional[float] = None,
        left_orientation_xyzw: Optional[Sequence[float]] = None,
        right_orientation_xyzw: Optional[Sequence[float]] = None,
        max_step: int | None = 600,
        custom_velocity: Optional[float] = None,
        try_close_gripper: bool = False,
        verbose: bool = False,
    ):
        """Move tool0 to a target pose using IK.

        Args:
            position_xyz: Target EE position (x,y,z) in world.
            orientation_xyzw: Optional target quaternion (x,y,z,w). If provided,
                it overrides `yaw`.
            yaw: Backward-compatible shorthand to generate a top-down grasp-like
                orientation (used only when orientation_xyzw is not provided).
            max_step: Safety cap on iterations. The controller always breaks early
                if the target is reached. Set to None to let the environment pick
                an automatic cap based on initial distance.

        For `robot='dual'`, provide `left_position_xyz` and `right_position_xyz`.
        You may also provide `left_orientation_xyzw`/`right_orientation_xyzw`.
        If those are omitted but `orientation_xyzw` is provided, it is used for
        both arms.
        """

        if robot == "any":
            raise ValueError("robot must be 'left'|'right'|'dual' (not 'any')")

        if robot == "dual":
            if left_position_xyz is None or right_position_xyz is None:
                raise ValueError("dual IK_move_to_pose requires left_position_xyz and right_position_xyz")

            l_orn = orientation_xyzw if left_orientation_xyzw is None else left_orientation_xyzw
            r_orn = orientation_xyzw if right_orientation_xyzw is None else right_orientation_xyzw

            ok_l, _ = self.IK_move_to_pose(
                "left",
                position_xyz=left_position_xyz,
                yaw=float(yaw if left_yaw is None else left_yaw),
                orientation_xyzw=l_orn,
                max_step=max_step,
                custom_velocity=custom_velocity,
                try_close_gripper=try_close_gripper,
                verbose=verbose,
            )
            ok_r, _ = self.IK_move_to_pose(
                "right",
                position_xyz=right_position_xyz,
                yaw=float(yaw if right_yaw is None else right_yaw),
                orientation_xyzw=r_orn,
                max_step=max_step,
                custom_velocity=custom_velocity,
                try_close_gripper=try_close_gripper,
                verbose=verbose,
            )
            return bool(ok_l and ok_r), None

        if robot not in ("left", "right"):
            raise ValueError("robot must be 'left'|'right'")

        self.env.set_active_arm(robot)
        if orientation_xyzw is not None:
            if len(list(orientation_xyzw)) != 4:
                raise ValueError("orientation_xyzw must be length-4 (x,y,z,w)")
            orn = tuple(float(v) for v in orientation_xyzw)
        else:
            # Uses the same top-down convention as grasping.
            orn = self.env._tool0_quat_from_grasp_yaw(float(yaw))
        action = (float(position_xyz[0]), float(position_xyz[1]), float(position_xyz[2]), orn)
        return self.env.move_ee(
            action,
            max_step=(None if max_step is None else int(max_step)),
            custom_velocity=custom_velocity,
            try_close_gripper=bool(try_close_gripper),
            verbose=bool(verbose),
        )

    def IK_move_to_pose_real_time(self, robot: RobotSelector, **kwargs):
        """Best-effort realtime-ish IK control.

        In this PyBullet repo, we don’t have a separate realtime controller loop.
        This calls `IK_move_to_pose()` with the same backend.
        """

        return self.IK_move_to_pose(robot, **kwargs)

    def tossing_object(self, robot: RobotSelector, *args, **kwargs):
        raise NotImplementedError("tossing_object is not implemented for this PyBullet controller")

    # ---------------------------------------------------------------------
    # Joint IO
    # ---------------------------------------------------------------------

    def read_all_joints(self, robot: RobotSelector) -> list[float]:
        if robot == "dual":
            return self.read_all_joints("left") + self.read_all_joints("right")
        if robot not in ("left", "right"):
            raise ValueError("robot must be 'left'|'right'|'dual'")

        out: list[float] = []
        for name in self._arm_joint_names(robot):
            jid = self.env.joints[name].id
            out.append(float(p.getJointState(self.env.robot_id, jid)[0]))
        return out

    def get_specific_joint_value(self, robot: RobotSelector, joint_number: int) -> float:
        joints = self.read_all_joints("dual" if robot == "dual" else robot)
        if joint_number < 0 or joint_number >= len(joints):
            raise IndexError(f"joint_number out of range: 0..{len(joints)-1}")
        return float(joints[int(joint_number)])

    def set_all_joints(self, robot: RobotSelector, joints: Sequence[float], *, steps: int = 240) -> bool:
        if robot == "dual":
            if len(joints) != 12:
                raise ValueError("dual expects 12 joints (6 left + 6 right)")
            ok_l = self.set_all_joints("left", joints[:6], steps=steps)
            ok_r = self.set_all_joints("right", joints[6:], steps=steps)
            return bool(ok_l and ok_r)
        if robot not in ("left", "right"):
            raise ValueError("robot must be 'left'|'right'|'dual'")
        return self._set_arm_joints_position_control(robot, joints, steps=steps)

    def set_specific_joint_value(
        self,
        robot: RobotSelector,
        joint_number: int,
        joint_value: float,
        *,
        steps: int = 240,
    ) -> bool:
        if robot == "dual":
            joints = self.read_all_joints("dual")
            joints[int(joint_number)] = float(joint_value)
            return self.set_all_joints("dual", joints, steps=steps)
        if robot not in ("left", "right"):
            raise ValueError("robot must be 'left'|'right'|'dual'")

        joints = self.read_all_joints(robot)
        if joint_number < 0 or joint_number >= len(joints):
            raise IndexError(f"joint_number out of range: 0..{len(joints)-1}")
        joints[int(joint_number)] = float(joint_value)
        return self.set_all_joints(robot, joints, steps=steps)

    def directly_set_all_joints(self, robot: RobotSelector, joints: Sequence[float]) -> bool:
        if robot == "dual":
            if len(joints) != 12:
                raise ValueError("dual expects 12 joints (6 left + 6 right)")
            ok_l = self.directly_set_all_joints("left", joints[:6])
            ok_r = self.directly_set_all_joints("right", joints[6:])
            return bool(ok_l and ok_r)

        if robot not in ("left", "right"):
            raise ValueError("robot must be 'left'|'right'|'dual'")

        names = self._arm_joint_names(robot)
        if len(joints) != len(names):
            raise ValueError(f"Expected {len(names)} joint values, got {len(joints)}")

        for i, name in enumerate(names):
            jid = self.env.joints[name].id
            p.resetJointState(self.env.robot_id, jid, float(joints[i]))

        # One step to propagate state.
        self.env.step_simulation()
        return True

    # ---------------------------------------------------------------------
    # Gripper
    # ---------------------------------------------------------------------

    def open_gripper(self, robot: RobotSelector, *, opening_length: float = 0.085):
        if robot == "dual" or robot == "any":
            self.open_gripper("left", opening_length=opening_length)
            self.open_gripper("right", opening_length=opening_length)
            return
        if robot not in ("left", "right"):
            raise ValueError("robot must be 'left'|'right'|'dual'")

        self.env.set_active_arm(robot)
        self.env.move_gripper(float(opening_length), step=120)

    def close_gripper(self, robot: RobotSelector, *, check_contact: bool = False) -> bool:
        if robot == "dual" or robot == "any":
            ok_l = self.close_gripper("left", check_contact=check_contact)
            ok_r = self.close_gripper("right", check_contact=check_contact)
            return bool(ok_l and ok_r)
        if robot not in ("left", "right"):
            raise ValueError("robot must be 'left'|'right'|'dual'")

        self.env.set_active_arm(robot)
        return bool(self.env.auto_close_gripper(check_contact=bool(check_contact)))

    def partially_open_gripper(self, robot: RobotSelector, opening_length: float):
        return self.open_gripper(robot, opening_length=float(opening_length))

    # ---------------------------------------------------------------------
    # High-level convenience wrappers (used by simulation_v2)
    # ---------------------------------------------------------------------

    def reset_robot(self, robot: RobotSelector = "dual", *, steps: int = 1000) -> bool:
        """Reset robot to a safe initial pose (both arms) and open grippers.

        Matches the intent of Environment.reset_robot() but routes through Behaviors.
        """
        if robot != "dual" and robot != "any":
            raise ValueError("reset_robot currently supports only robot='dual'|'any'")
        ok = self.go_to_initial_top_pose("dual", steps=int(steps))
        # Keep legacy semantics used by Environment.reset_robot(): 0.0 is fully open.
        self.open_gripper("dual", opening_length=0.0)
        return bool(ok)

    def grasp(
        self,
        *,
        position_xyz: Sequence[float],
        yaw: float,
        opening_length: float,
        obj_height: float,
        debug: bool = False,
    ):
        """Perform the repo's grasp-and-place routine.

        This delegates to Environment.grasp() (which auto-selects left/right arm
        based on X) but gives callers a Behaviors entrypoint.
        """
        return self.env.grasp(
            (float(position_xyz[0]), float(position_xyz[1]), float(position_xyz[2])),
            float(yaw),
            float(opening_length),
            float(obj_height),
            debug=bool(debug),
        )

    # ---------------------------------------------------------------------
    # EE pose + visualization
    # ---------------------------------------------------------------------

    def read_current_ee_pose(self, robot: RobotSelector):
        if robot == "dual":
            return {
                "left": self.read_current_ee_pose("left"),
                "right": self.read_current_ee_pose("right"),
            }
        if robot not in ("left", "right"):
            raise ValueError("robot must be 'left'|'right'|'dual'")

        li = self._eef_link_id(robot)
        pos, orn = p.getLinkState(self.env.robot_id, li, computeForwardKinematics=True)[0:2]
        return {"position": pos, "orientation_xyzw": orn}

    def enable_eef_mesh_debug(
        self,
        robot: RobotSelector,
        *,
        enabled: bool = True,
        stl_path: Optional[str] = None,
        mesh_scale: float = 1.0,
        rgba: Sequence[float] = (1.0, 1.0, 1.0, 0.9),
        parent_to_mesh_xyz: Sequence[float] = (0.0, 0.0, 0.0),
        parent_to_mesh_quat_xyzw: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
    ):
        """Toggle a visual mesh marker attached to the current tool0 pose.

        This is the PyBullet equivalent of the ROS mesh marker, useful for
        visualizing the *current* EE pose (position + orientation) as the robot moves.
        """

        if robot == "any":
            raise ValueError("robot must be 'left'|'right'|'dual' (not 'any')")

        arms = ("left", "right") if robot == "dual" else (str(robot),)
        self.env.enable_eef_mesh_debug(
            enabled=bool(enabled),
            arms=arms,
            stl_path=stl_path,
            mesh_scale=float(mesh_scale),
            rgba=rgba,
            parent_to_mesh_xyz=parent_to_mesh_xyz,
            parent_to_mesh_quat_xyzw=parent_to_mesh_quat_xyzw,
        )

    def visualize_target_ee_pose(
        self,
        robot: RobotSelector,
        *,
        position_xyz: Sequence[float],
        orientation_xyzw: Optional[Sequence[float]] = None,
        left_position_xyz: Optional[Sequence[float]] = None,
        right_position_xyz: Optional[Sequence[float]] = None,
        left_orientation_xyzw: Optional[Sequence[float]] = None,
        right_orientation_xyzw: Optional[Sequence[float]] = None,
        duration_s: float = 5.0,
        axis_len: float = 0.12,
        stl_path: Optional[str] = None,
        mesh_scale: float = 1.0,
        # Color control (preferred): r/g/b (+ alpha).
        r: Optional[float] = None,
        g: Optional[float] = None,
        b: Optional[float] = None,
        alpha: float = 0.6,
        # Backwards compatible: explicit RGBA tuple/list.
        rgba: Optional[Sequence[float]] = None,
    ):
        if robot == "any":
            raise ValueError("robot must be 'left'|'right'|'dual' (not 'any')")
        if robot == "dual":
            if left_position_xyz is None or right_position_xyz is None:
                raise ValueError("dual visualize_target_ee_pose requires left_position_xyz and right_position_xyz")
            self.visualize_target_ee_pose(
                "left",
                position_xyz=left_position_xyz,
                orientation_xyzw=left_orientation_xyzw,
                duration_s=duration_s,
                axis_len=axis_len,
                stl_path=stl_path,
                mesh_scale=mesh_scale,
                r=r,
                g=g,
                b=b,
                alpha=alpha,
                rgba=rgba,
            )
            self.visualize_target_ee_pose(
                "right",
                position_xyz=right_position_xyz,
                orientation_xyzw=right_orientation_xyzw,
                duration_s=duration_s,
                axis_len=axis_len,
                stl_path=stl_path,
                mesh_scale=mesh_scale,
                r=r,
                g=g,
                b=b,
                alpha=alpha,
                rgba=rgba,
            )
            return
        if robot not in ("left", "right"):
            raise ValueError("robot must be 'left'|'right'")

        self._reset_ee_debug(robot)

        # Spawn an STL mesh marker at the target pose (position + orientation).
        # This better communicates the full target pose than 3 debug lines.
        if stl_path is None:
            stl_path = os.path.join(os.path.dirname(__file__), "full_hand_hamid_corrected_axes_tool0.stl")

        orn = orientation_xyzw
        if orn is None:
            orn = (0.0, 0.0, 0.0, 1.0)

        if (r is None) != (g is None) or (r is None) != (b is None):
            raise ValueError("If specifying color, pass all of r, g, b")

        if rgba is not None and len(rgba) >= 3 and r is None:
            rr, gg, bb = float(rgba[0]), float(rgba[1]), float(rgba[2])
            aa = float(rgba[3]) if len(rgba) >= 4 else float(alpha)
        elif r is not None:
            rr, gg, bb = float(r), float(g), float(b)
            aa = float(alpha)
        else:
            # Per-arm defaults requested: left=yellow, right=green.
            if robot == "left":
                rr, gg, bb = 1.0, 1.0, 0.0
            else:
                rr, gg, bb = 0.0, 0.8, 0.0
            aa = float(alpha)

        vshape = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=str(stl_path),
            meshScale=[float(mesh_scale), float(mesh_scale), float(mesh_scale)],
            rgbaColor=[float(rr), float(gg), float(bb), float(aa)],
        )
        uid = p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=int(vshape),
            baseCollisionShapeIndex=-1,
            basePosition=[float(position_xyz[0]), float(position_xyz[1]), float(position_xyz[2])],
            baseOrientation=[float(orn[0]), float(orn[1]), float(orn[2]), float(orn[3])],
        )
        self._ee_target_mesh_uid[robot] = int(uid)

        # Best-effort: expire the body after duration_s (only processed in GUI mode).
        try:
            self.env.register_debug_body_expiry(int(uid), float(duration_s))
        except Exception:
            pass

        # Keep axis_len arg for backwards compatibility (no-op now).
        _ = float(axis_len)

    def visualize_ee_trajectory(
        self,
        robot: RobotSelector,
        *,
        position_xyz: Sequence[float],
        left_position_xyz: Optional[Sequence[float]] = None,
        right_position_xyz: Optional[Sequence[float]] = None,
        duration_s: float = 60.0,
        max_points: int = 2000,
        line_width: float = 2.0,
    ):
        if robot == "any":
            raise ValueError("robot must be 'left'|'right'|'dual' (not 'any')")
        if robot == "dual":
            if left_position_xyz is None or right_position_xyz is None:
                raise ValueError("dual visualize_ee_trajectory requires left_position_xyz and right_position_xyz")
            self.visualize_ee_trajectory(
                "left",
                position_xyz=left_position_xyz,
                duration_s=duration_s,
                max_points=max_points,
                line_width=line_width,
            )
            self.visualize_ee_trajectory(
                "right",
                position_xyz=right_position_xyz,
                duration_s=duration_s,
                max_points=max_points,
                line_width=line_width,
            )
            return
        if robot not in ("left", "right"):
            raise ValueError("robot must be 'left'|'right'")

        pts = self._ee_traj_points[robot]
        pts.append((float(position_xyz[0]), float(position_xyz[1]), float(position_xyz[2])))
        if len(pts) > int(max_points):
            del pts[: len(pts) - int(max_points)]

        # Remove previous lines and redraw (simple + robust; good enough for debug).
        for uid in list(self._ee_traj_debug_ids.get(robot, [])):
            try:
                p.removeUserDebugItem(int(uid))
            except Exception:
                pass
        self._ee_traj_debug_ids[robot] = []

        color = [1, 1, 0] if robot == "left" else [0, 0.8, 0]
        life = float(duration_s)

        ids: list[int] = []
        for a, b in zip(pts[:-1], pts[1:]):
            uid = p.addUserDebugLine(a, b, color, lifeTime=life, lineWidth=float(line_width))
            ids.append(int(uid))
        self._ee_traj_debug_ids[robot] = ids

    def clear_ee_trajectory(self, robot: RobotSelector):
        if robot == "dual" or robot == "any":
            self.clear_ee_trajectory("left")
            self.clear_ee_trajectory("right")
            return
        if robot not in ("left", "right"):
            raise ValueError("robot must be 'left'|'right'|'dual'")

        self._ee_traj_points[robot] = []
        for uid in list(self._ee_traj_debug_ids.get(robot, [])):
            try:
                p.removeUserDebugItem(int(uid))
            except Exception:
                pass
        self._ee_traj_debug_ids[robot] = []
