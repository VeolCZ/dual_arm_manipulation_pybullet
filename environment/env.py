from environment.utilities import setup_sisbot, setup_dual_arm_sisbot
from environment.camera import Camera, LinkCamera
import math
import time
import numpy as np
import pybullet as p
import pybullet_data
import random
import os
from typing import Iterable, Optional, Sequence
import time
import traceback

class FailToReachTargetError(RuntimeError):
    pass


class Environment:
    OBJECT_INIT_HEIGHT = 1.05
    GRIPPER_MOVING_HEIGHT = 1.25
    GRIPPER_GRASPED_LIFT_HEIGHT = 1.4
    # Drop zones (baskets). Left arm uses the original target; right arm uses the new basket
    # placed to the right of the robot.
    TARGET_ZONE_POS = [0.7, 0.0, 0.685]  # left basket (backwards compatible name)
    TARGET_ZONE_POS_RIGHT = [-0.7, 0.0, 0.685]
    SIMULATION_STEP_DELAY = 0.0005 #speed of simulator - the lower the fatser ## should be a param
    FINGER_LENGTH = 0.06
    Z_TABLE_TOP = 0.785
    GRIP_REDUCTION = 0.60
    # Tuned to match this project's table/workspace coordinates.
    # The dual-arm URDF is modeled facing +Y; our table/workspace is around Y≈-0.5.
    # So we rotate 180° around Z to face the table.
    ROBOT_BASE_POS = [0.0, 0.075, 0.78]
    ROBOT_BASE_RPY = [0.0, 0.0, np.pi]

    # Motor/physics tuning to reduce jitter in POSITION_CONTROL.
    ARM_FORCE_SCALE = 1.0
    ARM_POSITION_GAIN = 0.04
    ARM_VELOCITY_GAIN = 1.0
    ROBOT_LINK_LINEAR_DAMPING = 0.01
    ROBOT_LINK_ANGULAR_DAMPING = 0.01

    # Optional per-arm yaw offsets (radians) for matching tool frame conventions.
    GRASP_YAW_OFFSET_LEFT = 0.0
    # The right tool0 frame is effectively rotated 180deg around Z vs the grasp yaw convention,
    # so compensate here.
    GRASP_YAW_OFFSET_RIGHT = -np.pi

    @staticmethod
    def _quat_from_rotm(R: np.ndarray):
        """Convert 3x3 rotation matrix to (x,y,z,w) quaternion."""
        t = float(np.trace(R))
        if t > 0.0:
            S = math.sqrt(t + 1.0) * 2.0
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = math.sqrt(1.0 + float(R[0, 0]) - float(R[1, 1]) - float(R[2, 2])) * 2.0
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = math.sqrt(1.0 + float(R[1, 1]) - float(R[0, 0]) - float(R[2, 2])) * 2.0
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = math.sqrt(1.0 + float(R[2, 2]) - float(R[0, 0]) - float(R[1, 1])) * 2.0
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
        return [float(qx), float(qy), float(qz), float(qw)]

    def _grasp_yaw_offset(self) -> float:
        if getattr(self, "active_arm", "right") == "left":
            return float(self.GRASP_YAW_OFFSET_LEFT)
        return float(self.GRASP_YAW_OFFSET_RIGHT)

    def _tool0_quat_from_grasp_yaw(self, yaw: float):
        """Build a top-down tool0 orientation from grasp yaw.

        Convention used here:
        - tool0 Z axis points down (-world Z) for top-down grasps
        - tool0 X axis aligns with the grasp's opening direction in the XY plane
        """
        yaw = float(yaw) + self._grasp_yaw_offset()
        x_axis = np.array([math.sin(yaw), math.cos(yaw), 0.0], dtype=float)
        x_axis /= (np.linalg.norm(x_axis) + 1e-12)
        z_axis = np.array([0.0, 0.0, -1.0], dtype=float)
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= (np.linalg.norm(y_axis) + 1e-12)

        # Re-orthogonalize x to avoid drift.
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= (np.linalg.norm(x_axis) + 1e-12)

        R = np.stack([x_axis, y_axis, z_axis], axis=1)
        return self._quat_from_rotm(R)

    def _tool0_quat_topdown(self):
        return self._tool0_quat_from_grasp_yaw(0.0)

    def __init__(
        self,
        camera: Camera,
        vis: bool = False,
        debug: bool = False,
        gripper_type: str = '140',
        finger_length: float = 0.06,
        enable_shadows: bool = False,
        hide_gui_panels: bool = True,
    ) -> None:
        self.vis = vis
        self.debug = debug
        self.camera = camera

        # EE pose visualization using a mesh marker (optional GUI helper).
        # When enabled, we attach a visual-only mesh to left_tool0/right_tool0 via
        # a fixed constraint so it follows the current EE pose continuously.
        self._eef_mesh_debug_enabled = False
        self._eef_mesh_debug_stl_path: Optional[str] = None
        self._eef_mesh_debug_mesh_scale = 1.0
        self._eef_mesh_debug_rgba = (1.0, 1.0, 1.0, 1.0)
        self._eef_mesh_debug_parent_to_mesh_xyz = (0.0, 0.0, 0.0)
        self._eef_mesh_debug_parent_to_mesh_quat_xyzw = (0.0, 0.0, 0.0, 1.0)
        # Per-arm created bodies/constraints.
        self._eef_mesh_debug_body_uid: dict[str, int] = {}
        self._eef_mesh_debug_cid: dict[str, int] = {}

        # Debug bodies that should be removed after a wall-clock timeout.
        # Items are (bodyUniqueId, expiry_time_s).
        self._debug_bodies_expiry: list[tuple[int, float]] = []

        # Optional: auto-update camera mosaic viewer on each simulation step.
        # Disabled by default to avoid adding GUI overhead to training runs.
        self.auto_stream_cameras = False
        self.auto_stream_cameras_which = ("external", "head", "left", "right")
        self.auto_stream_cameras_show_depth = False
        self.auto_stream_cameras_max_fps = 5.0
        # Important: avoid sleeping inside the GUI viewer when called from step_simulation().
        self.auto_stream_cameras_pause_s = 0.0
        self.auto_stream_cameras_use_blit = True

        # Debug drawing (axes + rays) for wrist cameras.
        # This is fairly expensive; default to enabled only when debug=True.
        self.draw_camera_frame_debug = bool(debug)

        # Cache the external camera look-at target so link-mounted cameras can
        # be configured to match it.
        self._external_cam_target = (float(camera.x_t), float(camera.y_t), float(camera.z_t))

        self.obj_init_pos = (camera.x, camera.y)
        self.obj_ids = []
        self.obj_positions = []
        self.obj_orientations = []

        # Cache of previously loaded isolated objects to avoid repeated remove/load cycles.
        # Key: (urdf_path, mod_orn, mod_stiffness) -> bodyUniqueId
        self._isolated_obj_cache: dict[tuple[str, bool, bool], int] = {}
        self._hidden_object_pose = ([10.0, 10.0, -10.0], p.getQuaternionFromEuler([0.0, 0.0, 0.0]))

        if gripper_type not in ('85', '140'):
            raise NotImplementedError(
                'Gripper %s not implemented.' % gripper_type)
        self.gripper_type = gripper_type
        self.finger_length = finger_length

        # define environment
        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)

        # Optional debug: trace unexpected body removals/hides.
        # Enable with env var: PB_TRACE_REMOVEBODY=1 (prints uid + short stack).
        self._trace_removebody = str(os.getenv("PB_TRACE_REMOVEBODY", "")).strip().lower() in ("1", "true", "yes", "on")
        if self._trace_removebody:
            try:
                if not hasattr(p, "_copilot_orig_removeBody"):
                    p._copilot_orig_removeBody = p.removeBody

                def _removeBody_traced(uid):
                    try:
                        uid_i = int(uid)
                    except Exception:
                        uid_i = uid
                    try:
                        stack = "".join(traceback.format_stack(limit=8))
                    except Exception:
                        stack = "<no stack>"
                    print(f"[PB_TRACE_REMOVEBODY] removeBody({uid_i})\n{stack}")
                    return p._copilot_orig_removeBody(uid)

                p.removeBody = _removeBody_traced
            except Exception:
                pass

        # Optionally start PyBullet GUI without the left-side debug panels.
        # This matches the effect of pressing 'G' in the GUI.
        if self.vis and bool(hide_gui_panels):
            self.set_gui_panels_enabled(False)

        # GUI renderer visuals (shadows are enabled by default in PyBullet).
        if self.vis:
            try:
                p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1 if bool(enable_shadows) else 0)
            except Exception:
                pass

        # Renderer used for getCameraImage(). If this ends up slow (hundreds of ms),
        # you're likely running without hardware-accelerated OpenGL.
        if self.vis:
            self.camera_renderer = getattr(p, "ER_BULLET_HARDWARE_OPENGL", None)
        else:
            self.camera_renderer = getattr(p, "ER_TINY_RENDERER", None)

        # Default renderer for the external camera wrapper (used heavily by simulation.py).
        # Falls back silently if the passed camera doesn't support this attribute.
        try:
            self.camera.default_renderer = self.camera_renderer
        except Exception:
            pass
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(1.0 / 240.0)
        p.setPhysicsEngineParameter(numSolverIterations=200)
        # Improve contact stability for high-friction grasps.
        try:
            p.setPhysicsEngineParameter(enableConeFriction=1)
        except Exception:
            pass
        try:
            p.setPhysicsEngineParameter(useSplitImpulse=1)
        except Exception:
            pass
        p.setGravity(0, 0, -10)
        self.planeID = p.loadURDF('plane.urdf')
        self.tableID = p.loadURDF('environment/urdf/objects/table.urdf',
                                  [0.0, -0.65, 0.76],
                                  p.getQuaternionFromEuler([0, 0, 0]),
                                  useFixedBase=True)
        # Left basket (existing)
        self.target_table_id = p.loadURDF(
            'environment/urdf/objects/target_table.urdf',
            [self.TARGET_ZONE_POS[0], self.TARGET_ZONE_POS[1], 0.66],
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
        )
        self.target_id = p.loadURDF(
            'environment/urdf/objects/traybox.urdf',
            self.TARGET_ZONE_POS,
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
            globalScaling=0.7,
        )

        # Right basket (new)
        self.target_table_id_right = p.loadURDF(
            'environment/urdf/objects/target_table.urdf',
            [self.TARGET_ZONE_POS_RIGHT[0], self.TARGET_ZONE_POS_RIGHT[1], 0.66],
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
        )
        self.target_id_right = p.loadURDF(
            'environment/urdf/objects/traybox.urdf',
            self.TARGET_ZONE_POS_RIGHT,
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
            globalScaling=0.7,
        )

        self._target_zone_pos_by_arm = {
            "left": self.TARGET_ZONE_POS,
            "right": self.TARGET_ZONE_POS_RIGHT,
        }
        self._target_id_by_arm = {
            "left": self.target_id,
            "right": self.target_id_right,
        }
        self.UR5Stand_id = p.loadURDF('environment/urdf/objects/ur5_stand.urdf',
                                    [-0.7, -0.36, 0.0],
                                    p.getQuaternionFromEuler([0, 0, 0]),
                                    useFixedBase=True)

        # Debug visualization: camera link reference frames (axis gizmos).
        # Stored as: key -> {'x': id, 'y': id, 'z': id}
        self._camera_frame_debug_ids: dict[str, dict[str, int]] = {}
        # Debug visualization: rendered camera frame (thicker) + target point.
        self._camera_render_frame_debug_ids: dict[str, dict[str, int]] = {}
        self._camera_target_debug_ids: dict[str, dict[str, int]] = {}
        self.camera_frame_axis_len = 0.08  # meters
        self.camera_render_frame_axis_len = 0.12  # meters

        # -----------------------------
        # Load dual-arm robot
        # -----------------------------
        # Load from an absolute path to make relative mesh paths reliable.
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        # dual_urdf_path = os.path.join(repo_root, "environment", "urdf", "dual_arm_robot_without_wrist_cameras.urdf.urdf")
        dual_urdf_path = os.path.join(repo_root, "environment", "urdf", "dual_arm_robot_with_wrist_cameras.urdf.urdf")

        self.robot_id = p.loadURDF(
            dual_urdf_path,
            self.ROBOT_BASE_POS,
            p.getQuaternionFromEuler(self.ROBOT_BASE_RPY),
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
        )

        # Add a bit of damping on robot links to reduce steady-state oscillation/jitter.
        for link_idx in range(-1, p.getNumJoints(self.robot_id)):
            p.changeDynamics(
                self.robot_id,
                link_idx,
                linearDamping=self.ROBOT_LINK_LINEAR_DAMPING,
                angularDamping=self.ROBOT_LINK_ANGULAR_DAMPING,
            )

        (
            self.joints,
            self.left_controlJoints,
            self.right_controlJoints,
            self._control_lgripper,
            self._control_rgripper,
            self.joint_id_to_movable_index,
            self.link_name_to_id,
        ) = setup_dual_arm_sisbot(p, self.robot_id, gripper_type)

        # Gripper self-collisions can cause severe contact instability (links appear to
        # "disconnect"/explode during grasps). Disable collisions within each gripper.
        disable_gripper_self_collision = str(os.getenv("PB_GRIPPER_DISABLE_SELF_COLLISION", "1")).strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        if disable_gripper_self_collision:
            try:
                for prefix in ("lgripper_", "rgripper_"):
                    link_ids = [int(ji.id) for name, ji in getattr(self, "joints", {}).items() if str(name).startswith(prefix)]
                    # Pairwise disable collisions within the gripper.
                    for i in range(len(link_ids)):
                        for j in range(i + 1, len(link_ids)):
                            try:
                                p.setCollisionFilterPair(self.robot_id, self.robot_id, link_ids[i], link_ids[j], enableCollision=0)
                            except Exception:
                                pass
            except Exception:
                pass

        # -----------------------------
        # Link-mounted cameras
        # -----------------------------
        # Head-mounted Kinect RGB frame (requested).
        if "head_mount_kinect_rgb_frame" in self.link_name_to_id:
            self._head_camera_link_name = "head_mount_kinect_rgb_frame"
        elif "head_mount_kinect_rgb_optical_frame" in self.link_name_to_id:
            self._head_camera_link_name = "head_mount_kinect_rgb_optical_frame"
        elif "head_mount_kinect_link" in self.link_name_to_id:
            self._head_camera_link_name = "head_mount_kinect_link"
        else:
            self._head_camera_link_name = None

        # Wrist cameras.
        # Prefer the physical camera link if present; its axes are usually aligned with
        # typical camera conventions (see user example), and avoids surprises from the
        # intermediate *_rgb_frame / *_rgb_optical_frame transforms.
        if "left_arm_camera_link" in self.link_name_to_id:
            self._left_arm_camera_link_name = "left_arm_camera_link"
        elif "left_arm_camera_rgb_optical_frame" in self.link_name_to_id:
            self._left_arm_camera_link_name = "left_arm_camera_rgb_optical_frame"
        elif "left_arm_camera_rgb_frame" in self.link_name_to_id:
            self._left_arm_camera_link_name = "left_arm_camera_rgb_frame"
        else:
            self._left_arm_camera_link_name = None

        if "right_arm_camera_link" in self.link_name_to_id:
            self._right_arm_camera_link_name = "right_arm_camera_link"
        elif "right_arm_camera_rgb_optical_frame" in self.link_name_to_id:
            self._right_arm_camera_link_name = "right_arm_camera_rgb_optical_frame"
        elif "right_arm_camera_rgb_frame" in self.link_name_to_id:
            self._right_arm_camera_link_name = "right_arm_camera_rgb_frame"
        else:
            self._right_arm_camera_link_name = None

        self.head_camera = None
        if self._head_camera_link_name is not None:
            self.head_camera = LinkCamera(
                near=float(camera.near),
                far=float(camera.far),
                size=(int(camera.width), int(camera.height)),
                fov=float(camera.fov),
                # If we are using an optical frame, use ROS optical axes.
                forward_axis=(0.0, 0.0, 1.0)
                if self._head_camera_link_name.endswith("_optical_frame")
                else (1.0, 0.0, 0.0),
                up_axis=(0.0, -1.0, 0.0)
                if self._head_camera_link_name.endswith("_optical_frame")
                else (0.0, 0.0, 1.0),
            )
            try:
                self.head_camera.default_renderer = self.camera_renderer
            except Exception:
                pass

        self.left_arm_camera = None
        if self._left_arm_camera_link_name is not None:
            if self._left_arm_camera_link_name.endswith("_rgb_optical_frame"):
                l_fwd, l_up = (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)
            elif self._left_arm_camera_link_name.endswith("_camera_link"):
                # Match common camera_link convention (and the provided example):
                # forward = -Z, up = +X.
                l_fwd, l_up = (0.0, 0.0, -1.0), (-1.0, 0.0, 0.0)
            else:
                l_fwd, l_up = (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)

            self.left_arm_camera = LinkCamera(
                near=0.02,
                far=float(camera.far),
                size=(int(camera.width), int(camera.height)),
                fov=float(camera.fov),
                forward_axis=l_fwd,
                up_axis=l_up,
                eye_offset=0.05,
            )
            try:
                self.left_arm_camera.default_renderer = self.camera_renderer
            except Exception:
                pass

        self.right_arm_camera = None
        if self._right_arm_camera_link_name is not None:
            if self._right_arm_camera_link_name.endswith("_rgb_optical_frame"):
                r_fwd, r_up = (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)
            elif self._right_arm_camera_link_name.endswith("_camera_link"):
                r_fwd, r_up = (0.0, 0.0, -1.0), (-1.0, 0.0, 0.0)
            else:
                r_fwd, r_up = (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)

            self.right_arm_camera = LinkCamera(
                near=0.02,
                far=float(camera.far),
                size=(int(camera.width), int(camera.height)),
                fov=float(camera.fov),
                forward_axis=r_fwd,
                up_axis=r_up,
                eye_offset=0.05,
            )
            try:
                self.right_arm_camera.default_renderer = self.camera_renderer
            except Exception:
                pass

            # Wrist camera aiming parameters (used by get_left/right_arm_cam_img).
            # These are in meters in world space.
            self.wrist_cam_look_ahead = 0.20
            # Tilt-down angle (degrees). Positive = tilt down.
            # Internally converted to a camera-local Y offset using:
            #   y = -tan(deg) * z
            self.wrist_cam_tilt_down = 0.0

            # Optional per-axis aiming offsets in the camera's local frame (meters).
            # Camera local axes (for these parameters):
            # - X: right
            # - Y: up
            # - Z: forward
            # If left as None, we fall back to (look_ahead, tilt_down):
            #   X = 0, Y = -tilt_down, Z = look_ahead
            self.wrist_cam_look_ahead_x = None
            self.wrist_cam_look_ahead_y = None
            self.wrist_cam_look_ahead_z = None

        if "left_tool0" not in self.link_name_to_id or "right_tool0" not in self.link_name_to_id:
            raise RuntimeError("Could not find left_tool0/right_tool0 in loaded URDF.")

        self.left_eef_id = self.link_name_to_id["left_tool0"]
        self.right_eef_id = self.link_name_to_id["right_tool0"]

        # Default to right arm to keep existing single-arm API stable.
        self.active_arm = "right"
        self.controlJoints = self.right_controlJoints
        self.eef_id = self.right_eef_id

        # Map gripper driver joint names (used for reading current state)
        self.mimicParentName_by_arm = {
            "left": "lgripper_finger_joint",
            "right": "rgripper_finger_joint",
        }

        # Joint ids for the inner finger pads (used for contact/force checks)
        self._pad_joint_by_arm = {
            "left": (
                "lgripper_left_inner_finger_pad_joint",
                "lgripper_right_inner_finger_pad_joint",
            ),
            "right": (
                "rgripper_left_inner_finger_pad_joint",
                "rgripper_right_inner_finger_pad_joint",
            ),
        }

        # Add force sensors + friction for both grippers if available
        for arm in ("left", "right"):
            left_pad, right_pad = self._pad_joint_by_arm[arm]
            if left_pad in self.joints:
                p.enableJointForceTorqueSensor(self.robot_id, self.joints[left_pad].id)
                p.changeDynamics(
                    self.robot_id,
                    self.joints[left_pad].id,
                    lateralFriction=2.5,
                    spinningFriction=0.1,
                    rollingFriction=0.1,
                )
            if right_pad in self.joints:
                p.enableJointForceTorqueSensor(self.robot_id, self.joints[right_pad].id)
                p.changeDynamics(
                    self.robot_id,
                    self.joints[right_pad].id,
                    lateralFriction=2.5,
                    spinningFriction=0.1,
                    rollingFriction=0.1,
                )

        # custom sliders to tune parameters (name of the parameter,range,initial value)
        # Task space (Cartesian space)
        if debug:
            self.xin = p.addUserDebugParameter('x', -0.4, 0.4, 0.11)
            self.yin = p.addUserDebugParameter('y', -0.8, 0, -0.49)
            self.zin = p.addUserDebugParameter('z', 0.9, 1.3, 1.1)
            self.rollId = p.addUserDebugParameter(
                'roll', -3.14, 3.14, 0)  # -1.57 yaw
            self.pitchId = p.addUserDebugParameter(
                'pitch', -3.14, 3.14, np.pi/2)
            self.yawId = p.addUserDebugParameter(
                'yaw', -np.pi/2, np.pi/2, 0)  # -3.14 pitch
            self.gripper_opening_length_control = p.addUserDebugParameter(
                'gripper_opening_length', 0, 0.1, 0.085)

        # Add debug lines for end effector and camera
        if vis:
            self.eef_debug_lineID = None
            # p.addUserDebugLine([camera.x, camera.y, camera.z - 0.15], [camera.x, camera.y, camera.z - 0.4], [0, 1, 0], lineWidth=5)
            dist = 1.8
            yaw = 0.0
            pitch = -28.40
            target = [0.04, -0.03, 0.9]
            p.resetDebugVisualizerCamera(dist, yaw, pitch, target)

            vis_camera_z =  camera.z + 0.1 #m - adjust camera height to be more in line with the table/workspace and avoid looking at the ceiling.
            alpha = 0.025 #m 
            z_cam_l1 = vis_camera_z #m 

            color = [0.3, 0, 0]
            p.addUserDebugText("External RGB-D camera", [camera.x-0.2, camera.y, vis_camera_z+0.02], color, textSize=1)
             
            p.addUserDebugLine([camera.x+alpha, camera.y+alpha, z_cam_l1], [camera.x+alpha, camera.y-alpha, z_cam_l1], color, lineWidth=4)
            p.addUserDebugLine([camera.x+alpha, camera.y-alpha, z_cam_l1], [camera.x-alpha, camera.y-alpha, z_cam_l1], color, lineWidth=4)
            p.addUserDebugLine([camera.x-alpha, camera.y-alpha, z_cam_l1], [camera.x-alpha, camera.y+alpha, z_cam_l1], color, lineWidth=4)
            p.addUserDebugLine([camera.x-alpha, camera.y+alpha, z_cam_l1], [camera.x+alpha, camera.y+alpha, z_cam_l1], color, lineWidth=4)
        
            z_cam_l2 = vis_camera_z - 0.05 #m 
               
            p.addUserDebugLine([camera.x+alpha, camera.y+alpha, z_cam_l2], [camera.x+alpha, camera.y-alpha, z_cam_l2], color, lineWidth=4)
            p.addUserDebugLine([camera.x+alpha, camera.y-alpha, z_cam_l2], [camera.x-alpha, camera.y-alpha, z_cam_l2], color, lineWidth=4)
            p.addUserDebugLine([camera.x-alpha, camera.y-alpha, z_cam_l2], [camera.x-alpha, camera.y+alpha, z_cam_l2], color, lineWidth=4)
            p.addUserDebugLine([camera.x-alpha, camera.y+alpha, z_cam_l2], [camera.x+alpha, camera.y+alpha, z_cam_l2], color, lineWidth=4)

            ### body
            p.addUserDebugLine([camera.x+alpha, camera.y+alpha, z_cam_l1], [camera.x+alpha, camera.y+alpha, z_cam_l2], color, lineWidth=4)
            p.addUserDebugLine([camera.x+alpha, camera.y-alpha, z_cam_l1], [camera.x+alpha, camera.y-alpha, z_cam_l2], color, lineWidth=4)
            p.addUserDebugLine([camera.x-alpha, camera.y-alpha, z_cam_l1], [camera.x-alpha, camera.y-alpha, z_cam_l2], color, lineWidth=4)
            p.addUserDebugLine([camera.x-alpha, camera.y+alpha, z_cam_l1], [camera.x-alpha, camera.y+alpha, z_cam_l2], color, lineWidth=4)

            ### Third rectangle             
            z_cam_l3 = vis_camera_z - 0.105 #m 
            beta = alpha * 2 
            p.addUserDebugLine([camera.x+beta, camera.y+beta, z_cam_l3], [camera.x+beta, camera.y-beta, z_cam_l3], color, lineWidth=4)
            p.addUserDebugLine([camera.x+beta, camera.y-beta, z_cam_l3], [camera.x-beta, camera.y-beta, z_cam_l3], color, lineWidth=4)
            p.addUserDebugLine([camera.x-beta, camera.y-beta, z_cam_l3], [camera.x-beta, camera.y+beta, z_cam_l3], color, lineWidth=4)
            p.addUserDebugLine([camera.x-beta, camera.y+beta, z_cam_l3], [camera.x+beta, camera.y+beta, z_cam_l3], color, lineWidth=4)

            ### body
            p.addUserDebugLine([camera.x+alpha, camera.y+alpha, z_cam_l2], [camera.x+beta, camera.y+beta, z_cam_l3], color, lineWidth=4)
            p.addUserDebugLine([camera.x+alpha, camera.y-alpha, z_cam_l2], [camera.x+beta, camera.y-beta, z_cam_l3], color, lineWidth=4)
            p.addUserDebugLine([camera.x-alpha, camera.y-alpha, z_cam_l2], [camera.x-beta, camera.y-beta, z_cam_l3], color, lineWidth=4)
            p.addUserDebugLine([camera.x-alpha, camera.y+alpha, z_cam_l2], [camera.x-beta, camera.y+beta, z_cam_l3], color, lineWidth=4)


            # External camera model (D435) as a visual mesh instead of debug lines.
            # This is GUI-only; for DIRECT runs it does nothing.
            # self._external_camera_model_id = None
            # try:
            #     repo_root_vis = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            #     d435_mesh_path = os.path.join(repo_root_vis, "environment", "meshes", "d435.dae")

            #     # Compute a reasonable pose: put the model at (camera.x,y,z) and point it
            #     # toward the camera look-at target.
            #     eye = np.array([float(camera.x), float(camera.y), float(camera.z)], dtype=float)
            #     tgt = np.array([float(camera.x_t), float(camera.y_t), float(camera.z_t)], dtype=float)
            #     fwd = tgt - eye
            #     fn = float(np.linalg.norm(fwd))
            #     if fn < 1e-9:
            #         fwd = np.array([1.0, 0.0, 0.0], dtype=float)
            #     else:
            #         fwd = fwd / fn
            #     up = np.array([0.0, 0.0, 1.0], dtype=float)
            #     right = np.cross(fwd, up)
            #     rn = float(np.linalg.norm(right))
            #     if rn < 1e-9:
            #         # If fwd is parallel to world-up, pick an arbitrary right axis.
            #         right = np.array([1.0, 0.0, 0.0], dtype=float)
            #     else:
            #         right = right / rn
            #     up_ortho = np.cross(right, fwd)
            #     un = float(np.linalg.norm(up_ortho))
            #     if un < 1e-9:
            #         up_ortho = up
            #     else:
            #         up_ortho = up_ortho / un

            #     # Local convention for the mesh: X=forward, Z=up.
            #     # (If the mesh appears rotated, tweak the fixed RPY below.)
            #     x_axis = fwd
            #     z_axis = up_ortho
            #     # Right-handed Y axis.
            #     y_axis = np.cross(z_axis, x_axis)
            #     yn = float(np.linalg.norm(y_axis))
            #     if yn < 1e-9:
            #         y_axis = right
            #     else:
            #         y_axis = y_axis / yn
            #     # Re-orthogonalize Z.
            #     z_axis = np.cross(x_axis, y_axis)
            #     zn = float(np.linalg.norm(z_axis))
            #     if zn < 1e-9:
            #         z_axis = up_ortho
            #     else:
            #         z_axis = z_axis / zn

            #     R_cam = np.stack([x_axis, y_axis, z_axis], axis=1)
            #     q_xyzw = self._quat_from_rotm(R_cam)

            #     # Optional fixed correction if the mesh's local axes don't match.
            #     # Keeping this as an attribute makes it easy to tune from the REPL.
            #     self.external_camera_model_rpy_offset = getattr(
            #         self,
            #         "external_camera_model_rpy_offset",
            #         (0.0, 0.0, 0.0),
            #     )
            #     q_off = p.getQuaternionFromEuler(
            #         [
            #             float(self.external_camera_model_rpy_offset[0]),
            #             float(self.external_camera_model_rpy_offset[1]),
            #             float(self.external_camera_model_rpy_offset[2]),
            #         ]
            #     )
            #     q_xyzw = p.multiplyTransforms([0, 0, 0], q_xyzw, [0, 0, 0], q_off)[1]

            #     # D435 mesh scale: default to 1.0 (meters). If it looks wrong, adjust.
            #     self.external_camera_model_scale = float(getattr(self, "external_camera_model_scale", 1.0))

            #     vis_shape = p.createVisualShape(
            #         shapeType=p.GEOM_MESH,
            #         fileName=d435_mesh_path,
            #         meshScale=[self.external_camera_model_scale] * 3,
            #         rgbaColor=[0.2, 0.2, 0.2, 1.0],
            #         specularColor=[0.2, 0.2, 0.2],
            #     )
            #     self._external_camera_model_id = p.createMultiBody(
            #         baseMass=0.0,
            #         baseVisualShapeIndex=vis_shape,
            #         baseCollisionShapeIndex=-1,
            #         basePosition=eye.tolist(),
            #         baseOrientation=q_xyzw,
            #     )
            #     # Make sure it doesn't affect dynamics.
            #     p.changeDynamics(self._external_camera_model_id, -1, lateralFriction=0.0)
            # except Exception as _e:
            #     # Fall back to a simple label if the mesh cannot be loaded.
            #     try:
            #         color = [0.3, 0, 0]
            #         p.addUserDebugText(
            #             "External RGB-D camera",
            #             [camera.x - 0.1, camera.y, camera.z + 0.02],
            #             color,
            #             textSize=2,
            #         )
            #     except Exception:
            #         pass

            ### working area             
            working_area = 0.79 #m 
            beta = 0.415 
            p.addUserDebugLine([camera.x+beta, camera.y+beta, working_area], [camera.x+beta, camera.y-beta, working_area], [0, 1, 0], lineWidth=5)
            p.addUserDebugLine([camera.x+beta, camera.y-beta, working_area], [camera.x-beta, camera.y-beta, working_area], [0, 1, 0], lineWidth=5)
            p.addUserDebugLine([camera.x-beta, camera.y-beta, working_area], [camera.x-beta, camera.y+beta, working_area], [0, 1, 0], lineWidth=5)
            p.addUserDebugLine([camera.x-beta, camera.y+beta, working_area], [camera.x+beta, camera.y+beta, working_area], [0, 1, 0], lineWidth=5)



        # Setup some Limit
        self.gripper_open_limit = (0.0, 0.1)
        self.ee_position_limit = ((-0.8, 0.8),
                                  (-0.8, 0.8),
                                  (0.785, 1.4))
        self.reset_robot()

    def set_gui_panels_enabled(self, enabled: bool = True):
        """Show/hide the PyBullet GUI side panels.

        When hidden, this behaves like pressing 'G' in the GUI. We also hide the
        RGB/depth/segmentation preview widgets so the left panel stays clean.
        """
        if not getattr(self, "vis", False):
            return

        enabled_i = 1 if bool(enabled) else 0
        for flag_name in (
            "COV_ENABLE_GUI",
            "COV_ENABLE_RGB_BUFFER_PREVIEW",
            "COV_ENABLE_DEPTH_BUFFER_PREVIEW",
            "COV_ENABLE_SEGMENTATION_MARK_PREVIEW",
        ):
            flag = getattr(p, flag_name, None)
            if flag is None:
                continue
            try:
                p.configureDebugVisualizer(flag, enabled_i)
            except Exception:
                pass

    def get_external_cam_img(
        self,
        return_depth: bool = True,
        return_seg: bool = False,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    ):
        """Return (rgb, depth, seg) from the external fixed RGB-D camera."""
        if getattr(self, "camera", None) is None:
            raise RuntimeError("External camera is not configured")
        effective_renderer = self.camera_renderer if renderer is None else renderer
        return self.camera.get_cam_img(
            return_depth=bool(return_depth),
            return_seg=bool(return_seg),
            renderer=effective_renderer,
        )

    def get_head_cam_img(self, return_depth: bool = True, return_seg: bool = True, renderer=p.ER_BULLET_HARDWARE_OPENGL):
        """Return (rgb, depth, seg) from the head-mounted Kinect camera."""
        if self.head_camera is None or self._head_camera_link_name is None:
            raise RuntimeError("Head camera link not present in the loaded URDF")
        link_index = self.link_name_to_id[self._head_camera_link_name]
        # Match the external camera's look-at target.
        return self.head_camera.get_cam_img(
            self.robot_id,
            link_index,
            target_world=self._external_cam_target,
            return_depth=bool(return_depth),
            return_seg=bool(return_seg),
            renderer=self.camera_renderer if renderer is None else renderer,
        )

    def get_left_arm_cam_img(self, return_depth: bool = True, return_seg: bool = True, renderer=p.ER_BULLET_HARDWARE_OPENGL):
        """Return (rgb, depth, seg) from the left wrist RGB camera."""
        if self.left_arm_camera is None or self._left_arm_camera_link_name is None:
            raise RuntimeError("Left arm camera link not present in the loaded URDF")
        link_index = self.link_name_to_id[self._left_arm_camera_link_name]
        ls = p.getLinkState(self.robot_id, link_index, computeForwardKinematics=True)
        cam_link_pos, cam_link_orn = (ls[4], ls[5]) if len(ls) >= 6 else (ls[0], ls[1])
        tool_ls = p.getLinkState(self.robot_id, self.left_eef_id, computeForwardKinematics=True)
        tool_pos = tool_ls[4] if len(tool_ls) >= 6 else tool_ls[0]

        # Apply parent->camera transform (same as LinkCamera) so target is computed
        # from the same pose used for rendering.
        cam_pos, cam_orn = p.multiplyTransforms(
            cam_link_pos,
            cam_link_orn,
            self.left_arm_camera.parent_to_camera_xyz,
            self.left_arm_camera.parent_to_camera_quat,
        )
        # Target: look ahead and slightly down to see the workspace.
        m = p.getMatrixFromQuaternion(cam_orn)
        fx, fy, fz = self.left_arm_camera.forward_axis
        fwd_world_for_offset = (
            float(m[0]) * float(fx) + float(m[1]) * float(fy) + float(m[2]) * float(fz),
            float(m[3]) * float(fx) + float(m[4]) * float(fy) + float(m[5]) * float(fz),
            float(m[6]) * float(fx) + float(m[7]) * float(fy) + float(m[8]) * float(fz),
        )
        fwd_world = fwd_world_for_offset
        ux, uy, uz = self.left_arm_camera.up_axis
        up_world = (
            float(m[0]) * float(ux) + float(m[1]) * float(uy) + float(m[2]) * float(uz),
            float(m[3]) * float(ux) + float(m[4]) * float(uy) + float(m[5]) * float(uz),
            float(m[6]) * float(ux) + float(m[7]) * float(uy) + float(m[8]) * float(uz),
        )

        # If the camera is pointing back toward tool0, flip its forward axis.
        # This avoids the common failure mode where the wrist camera mostly sees the gripper base.
        to_tool = (
            float(tool_pos[0]) - float(cam_pos[0]),
            float(tool_pos[1]) - float(cam_pos[1]),
            float(tool_pos[2]) - float(cam_pos[2]),
        )
        if (fwd_world[0] * to_tool[0] + fwd_world[1] * to_tool[1] + fwd_world[2] * to_tool[2]) > 0.0:
            fx, fy, fz = (-float(fx), -float(fy), -float(fz))
            fwd_world = (
                float(m[0]) * float(fx) + float(m[1]) * float(fy) + float(m[2]) * float(fz),
                float(m[3]) * float(fx) + float(m[4]) * float(fy) + float(m[5]) * float(fz),
                float(m[6]) * float(fx) + float(m[7]) * float(fy) + float(m[8]) * float(fz),
            )

        # Use the exact same eye offset convention as LinkCamera:
        # eye_offset is always applied along the camera's configured forward axis.
        cam_eye = (
            float(cam_pos[0]) + float(self.left_arm_camera.eye_offset) * fwd_world_for_offset[0],
            float(cam_pos[1]) + float(self.left_arm_camera.eye_offset) * fwd_world_for_offset[1],
            float(cam_pos[2]) + float(self.left_arm_camera.eye_offset) * fwd_world_for_offset[2],
        )

        # Build a camera-local right/up basis for per-axis offsets.
        right_world = np.cross(np.array(fwd_world, dtype=float), np.array(up_world, dtype=float))
        rn = float(np.linalg.norm(right_world) + 1e-12)
        right_world = (right_world / rn).tolist()
        up_ortho_world = np.cross(np.array(right_world, dtype=float), np.array(fwd_world, dtype=float))
        un = float(np.linalg.norm(up_ortho_world) + 1e-12)
        up_ortho_world = (up_ortho_world / un).tolist()

        ox = 0.0 if self.wrist_cam_look_ahead_x is None else float(self.wrist_cam_look_ahead_x)
        oz = float(self.wrist_cam_look_ahead) if self.wrist_cam_look_ahead_z is None else float(self.wrist_cam_look_ahead_z)
        if self.wrist_cam_look_ahead_y is None:
            tilt_deg = float(self.wrist_cam_tilt_down)
            oy = -math.tan(math.radians(tilt_deg)) * oz
        else:
            oy = float(self.wrist_cam_look_ahead_y)

        target = (
            float(cam_eye[0]) + oz * float(fwd_world[0]) + oy * float(up_ortho_world[0]) + ox * float(right_world[0]),
            float(cam_eye[1]) + oz * float(fwd_world[1]) + oy * float(up_ortho_world[1]) + ox * float(right_world[1]),
            float(cam_eye[2]) + oz * float(fwd_world[2]) + oy * float(up_ortho_world[2]) + ox * float(right_world[2]),
        )

        # Prefer looking through the fingertip if we can estimate it.
        fingertip = self._get_fingertip_marker("left", tool_pos)
        if fingertip is not None:
            dv = np.array(
                [
                    float(fingertip[0]) - float(cam_eye[0]),
                    float(fingertip[1]) - float(cam_eye[1]),
                    float(fingertip[2]) - float(cam_eye[2]),
                ],
                dtype=float,
            )
            if float(np.linalg.norm(dv)) > 1e-6:
                target = fingertip
        return self.left_arm_camera.get_cam_img(
            self.robot_id,
            link_index,
            target_world=target,
            return_depth=bool(return_depth),
            return_seg=bool(return_seg),
            renderer=self.camera_renderer if renderer is None else renderer,
        )

    def get_right_arm_cam_img(self, return_depth: bool = True, return_seg: bool = True, renderer=p.ER_BULLET_HARDWARE_OPENGL):
        """Return (rgb, depth, seg) from the right wrist RGB camera."""
        if self.right_arm_camera is None or self._right_arm_camera_link_name is None:
            raise RuntimeError("Right arm camera link not present in the loaded URDF")
        link_index = self.link_name_to_id[self._right_arm_camera_link_name]
        ls = p.getLinkState(self.robot_id, link_index, computeForwardKinematics=True)
        cam_link_pos, cam_link_orn = (ls[4], ls[5]) if len(ls) >= 6 else (ls[0], ls[1])
        tool_ls = p.getLinkState(self.robot_id, self.right_eef_id, computeForwardKinematics=True)
        tool_pos = tool_ls[4] if len(tool_ls) >= 6 else tool_ls[0]

        cam_pos, cam_orn = p.multiplyTransforms(
            cam_link_pos,
            cam_link_orn,
            self.right_arm_camera.parent_to_camera_xyz,
            self.right_arm_camera.parent_to_camera_quat,
        )
        m = p.getMatrixFromQuaternion(cam_orn)
        fx, fy, fz = self.right_arm_camera.forward_axis
        fwd_world_for_offset = (
            float(m[0]) * float(fx) + float(m[1]) * float(fy) + float(m[2]) * float(fz),
            float(m[3]) * float(fx) + float(m[4]) * float(fy) + float(m[5]) * float(fz),
            float(m[6]) * float(fx) + float(m[7]) * float(fy) + float(m[8]) * float(fz),
        )
        fwd_world = fwd_world_for_offset
        ux, uy, uz = self.right_arm_camera.up_axis
        up_world = (
            float(m[0]) * float(ux) + float(m[1]) * float(uy) + float(m[2]) * float(uz),
            float(m[3]) * float(ux) + float(m[4]) * float(uy) + float(m[5]) * float(uz),
            float(m[6]) * float(ux) + float(m[7]) * float(uy) + float(m[8]) * float(uz),
        )

        to_tool = (
            float(tool_pos[0]) - float(cam_pos[0]),
            float(tool_pos[1]) - float(cam_pos[1]),
            float(tool_pos[2]) - float(cam_pos[2]),
        )
        if (fwd_world[0] * to_tool[0] + fwd_world[1] * to_tool[1] + fwd_world[2] * to_tool[2]) > 0.0:
            fx, fy, fz = (-float(fx), -float(fy), -float(fz))
            fwd_world = (
                float(m[0]) * float(fx) + float(m[1]) * float(fy) + float(m[2]) * float(fz),
                float(m[3]) * float(fx) + float(m[4]) * float(fy) + float(m[5]) * float(fz),
                float(m[6]) * float(fx) + float(m[7]) * float(fy) + float(m[8]) * float(fz),
            )
        cam_eye = (
            float(cam_pos[0]) + float(self.right_arm_camera.eye_offset) * fwd_world_for_offset[0],
            float(cam_pos[1]) + float(self.right_arm_camera.eye_offset) * fwd_world_for_offset[1],
            float(cam_pos[2]) + float(self.right_arm_camera.eye_offset) * fwd_world_for_offset[2],
        )

        right_world = np.cross(np.array(fwd_world, dtype=float), np.array(up_world, dtype=float))
        rn = float(np.linalg.norm(right_world) + 1e-12)
        right_world = (right_world / rn).tolist()
        up_ortho_world = np.cross(np.array(right_world, dtype=float), np.array(fwd_world, dtype=float))
        un = float(np.linalg.norm(up_ortho_world) + 1e-12)
        up_ortho_world = (up_ortho_world / un).tolist()

        ox = 0.0 if self.wrist_cam_look_ahead_x is None else float(self.wrist_cam_look_ahead_x)
        oz = float(self.wrist_cam_look_ahead) if self.wrist_cam_look_ahead_z is None else float(self.wrist_cam_look_ahead_z)
        if self.wrist_cam_look_ahead_y is None:
            tilt_deg = float(self.wrist_cam_tilt_down)
            # print (f"Right wrist cam tilt (deg): {tilt_deg}, offset (m): {oz}, computed Y offset (m): {-math.tan(math.radians(tilt_deg)) * oz}")
            oy = -math.tan(math.radians(tilt_deg)) * oz
        else:
            oy = float(self.wrist_cam_look_ahead_y)

        target = (
            float(cam_eye[0]) + oz * float(fwd_world[0]) + oy * float(up_ortho_world[0]) + ox * float(right_world[0]),
            float(cam_eye[1]) + oz * float(fwd_world[1]) + oy * float(up_ortho_world[1]) + ox * float(right_world[1]),
            float(cam_eye[2]) + oz * float(fwd_world[2]) + oy * float(up_ortho_world[2]) + ox * float(right_world[2]),
        )

        fingertip = self._get_fingertip_marker("right", tool_pos)
        # fingertip[2] -= self.wrist_cam_look_ahead # Add a bit of height to the fingertip target to reduce collisions with the table.
        if fingertip is not None:
            dv = np.array(
                [
                    float(fingertip[0]) - float(cam_eye[0]),
                    float(fingertip[1]) - float(cam_eye[1]),
                    float(fingertip[2]) - float(cam_eye[2]),
                ],
                dtype=float,
            )
            if float(np.linalg.norm(dv)) > 1e-6:
                target = fingertip
        return self.right_arm_camera.get_cam_img(
            self.robot_id,
            link_index,
            target_world=target,
            return_depth=bool(return_depth),
            return_seg=bool(return_seg),
            renderer=self.camera_renderer if renderer is None else renderer,
        )

    def visualize_cameras(
        self,
        which: Iterable[str] = ("external", "head", "left", "right"),
        show_depth: bool = False,
        window_name: str = "cameras",
        wait_ms: int = 1,
        renderer=None,
    ) -> bool:
        """Visualize camera streams live in an OpenCV window.

        Call this once per control/sim step to stream frames while the robot moves.

        Returns:
            True to continue, False if user requested quit (q/esc) or visualization is unavailable.
        """
        try:
            import cv2
        except Exception:
            # Likely opencv not installed (or headless build missing GUI).
            return False

        which_set = set(str(s).lower() for s in which)
        effective_renderer = self.camera_renderer if renderer is None else renderer

        def _as_uint8_rgb(img):
            arr = np.asarray(img)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return arr

        def _rgb_to_bgr(rgb_u8):
            return cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)

        def _depth_to_vis(depth):
            d = np.asarray(depth)
            d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
            d_norm = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX)
            d_u8 = d_norm.astype(np.uint8)
            return cv2.applyColorMap(d_u8, cv2.COLORMAP_TURBO)

        rgb_frames: dict[str, np.ndarray] = {}
        depth_frames: dict[str, np.ndarray] = {}

        try:
            if "external" in which_set:
                rgb, depth, _seg = self.get_external_cam_img(
                    return_depth=bool(show_depth),
                    return_seg=False,
                    renderer=effective_renderer,
                )
                rgb_frames["external"] = _as_uint8_rgb(rgb)
                if depth is not None:
                    depth_frames["external"] = np.asarray(depth)
        except Exception:
            pass

        try:
            if "head" in which_set and self.head_camera is not None:
                rgb, depth, _seg = self.get_head_cam_img(
                    return_depth=bool(show_depth),
                    return_seg=False,
                    renderer=effective_renderer,
                )
                rgb_frames["head"] = _as_uint8_rgb(rgb)
                if depth is not None:
                    depth_frames["head"] = np.asarray(depth)
        except Exception:
            pass

        try:
            if "left" in which_set and self.left_arm_camera is not None:
                rgb, depth, _seg = self.get_left_arm_cam_img(
                    return_depth=bool(show_depth),
                    return_seg=False,
                    renderer=effective_renderer,
                )
                rgb_frames["left"] = _as_uint8_rgb(rgb)
                if depth is not None:
                    depth_frames["left"] = np.asarray(depth)
        except Exception:
            pass

        try:
            if "right" in which_set and self.right_arm_camera is not None:
                rgb, depth, _seg = self.get_right_arm_cam_img(
                    return_depth=bool(show_depth),
                    return_seg=False,
                    renderer=effective_renderer,
                )
                rgb_frames["right"] = _as_uint8_rgb(rgb)
                if depth is not None:
                    depth_frames["right"] = np.asarray(depth)
        except Exception:
            pass

        if not rgb_frames:
            return False

        any_frame = next(iter(rgb_frames.values()))
        h, w = int(any_frame.shape[0]), int(any_frame.shape[1])
        blank = np.zeros((h, w, 3), dtype=np.uint8)

        def _labeled(name: str):
            rgb_u8 = rgb_frames.get(name)
            bgr = blank.copy() if rgb_u8 is None else _rgb_to_bgr(rgb_u8)
            cv2.putText(
                bgr,
                name,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            return bgr

        mosaic_rgb = np.vstack(
            [
                np.hstack([_labeled("external"), _labeled("head")]),
                np.hstack([_labeled("left"), _labeled("right")]),
            ]
        )

        try:
            cv2.imshow(window_name, mosaic_rgb)
            if show_depth:
                def _d(name: str):
                    return _depth_to_vis(depth_frames.get(name, np.zeros((h, w), dtype=np.float32)))

                mosaic_depth = np.vstack(
                    [
                        np.hstack([_d("external"), _d("head")]),
                        np.hstack([_d("left"), _d("right")]),
                    ]
                )
                cv2.imshow(f"{window_name}_depth", mosaic_depth)

            key = int(cv2.waitKey(int(wait_ms))) & 0xFF
            if key in (27, ord('q')):
                return False
            return True
        except Exception:
            return False

    def visualize_cameras_matplotlib(
        self,
        which: Iterable[str] = ("external", "head", "left", "right"),
        show_depth: bool = False,
        window_title: str = "cameras",
        pause_s: float = 0.001,
        max_fps: float | None = None,
        use_blit: bool = True,
        visible: bool = True,
        renderer=None,
    ) -> bool:
        """Visualize camera streams live using Matplotlib.

        This avoids OpenCV GUI backends (which can be problematic when
        `opencv-python-headless` is installed). Call once per sim/control step.

        Returns:
            True to continue, False if visualization is unavailable.
        """

        if not bool(visible):
            return False

        try:
            import matplotlib
            import matplotlib.pyplot as plt
        except Exception:
            return False

        # Optional FPS throttle (useful to reduce GUI overhead).
        viewer_existing = getattr(self, "_cam_viewer", None)
        if viewer_existing is not None and max_fps is not None:
            try:
                max_fps_f = float(max_fps)
            except Exception:
                max_fps_f = 0.0
            if max_fps_f > 0.0:
                now = float(time.time())
                last = float(getattr(self, "_cam_last_draw_ts", 0.0))
                min_dt = 1.0 / max_fps_f
                if (now - last) < min_dt:
                    # Called from the physics loop; do NOT sleep here.
                    # Just let the GUI backend process pending events (non-blocking).
                    try:
                        fig = viewer_existing.get("fig")
                        if fig is not None:
                            fig.canvas.flush_events()
                    except Exception:
                        pass
                    return True
                self._cam_last_draw_ts = now

        which_set = set(str(s).lower() for s in which)
        effective_renderer = self.camera_renderer if renderer is None else renderer

        def _as_u8_rgb(img) -> np.ndarray:
            arr = np.asarray(img)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            if arr.ndim == 2:
                arr = np.repeat(arr[:, :, None], 3, axis=2)
            return arr

        rgb_frames: dict[str, np.ndarray] = {}
        depth_frames: dict[str, np.ndarray] = {}

        try:
            if "external" in which_set:
                rgb, depth, _seg = self.get_external_cam_img(
                    return_depth=bool(show_depth),
                    return_seg=False,
                    renderer=effective_renderer,
                )
                rgb_frames["external"] = _as_u8_rgb(rgb)
                if depth is not None:
                    depth_frames["external"] = np.asarray(depth)
        except Exception:
            pass

        try:
            if "head" in which_set and self.head_camera is not None:
                rgb, depth, _seg = self.get_head_cam_img(
                    return_depth=bool(show_depth),
                    return_seg=False,
                    renderer=effective_renderer,
                )
                rgb_frames["head"] = _as_u8_rgb(rgb)
                if depth is not None:
                    depth_frames["head"] = np.asarray(depth)
        except Exception:
            pass

        try:
            if "left" in which_set and self.left_arm_camera is not None:
                rgb, depth, _seg = self.get_left_arm_cam_img(
                    return_depth=bool(show_depth),
                    return_seg=False,
                    renderer=effective_renderer,
                )
                rgb_frames["left"] = _as_u8_rgb(rgb)
                if depth is not None:
                    depth_frames["left"] = np.asarray(depth)
        except Exception:
            pass

        try:
            if "right" in which_set and self.right_arm_camera is not None:
                rgb, depth, _seg = self.get_right_arm_cam_img(
                    return_depth=bool(show_depth),
                    return_seg=False,
                    renderer=effective_renderer,
                )
                rgb_frames["right"] = _as_u8_rgb(rgb)
                if depth is not None:
                    depth_frames["right"] = np.asarray(depth)
        except Exception:
            pass

        if not rgb_frames:
            return False

        # Determine a consistent tile size.
        any_rgb = next(iter(rgb_frames.values()))
        h, w = int(any_rgb.shape[0]), int(any_rgb.shape[1])
        blank_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        blank_depth = np.zeros((h, w), dtype=np.float32)

        def _get_rgb(name: str) -> np.ndarray:
            img = rgb_frames.get(name)
            return blank_rgb if img is None else img

        def _get_depth_norm(name: str) -> np.ndarray:
            d = depth_frames.get(name)
            if d is None:
                return blank_depth
            d = np.asarray(d, dtype=np.float32)
            d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
            mn, mx = float(np.min(d)), float(np.max(d))
            if mx - mn < 1e-6:
                return blank_depth
            return (d - mn) / (mx - mn)

        order = ("external", "head", "left", "right")

        viewer = getattr(self, "_cam_viewer", None)
        if viewer is None or "fig" not in viewer:
            plt.ion()
            fig, axes = plt.subplots(2, 2, num=window_title)
            try:
                fig.canvas.manager.set_window_title(window_title)
            except Exception:
                pass

            ims = {}
            for ax, name in zip(axes.ravel(), order):
                ax.set_title(name)
                ax.axis('off')
                ims[name] = ax.imshow(_get_rgb(name), animated=bool(use_blit))

            depth_fig = None
            depth_axes = None
            depth_ims = None
            if bool(show_depth):
                depth_fig, depth_axes = plt.subplots(2, 2, num=f"{window_title}_depth")
                try:
                    depth_fig.canvas.manager.set_window_title(f"{window_title}_depth")
                except Exception:
                    pass
                depth_ims = {}
                for ax, name in zip(depth_axes.ravel(), order):
                    ax.set_title(name)
                    ax.axis('off')
                    depth_ims[name] = ax.imshow(
                        _get_depth_norm(name),
                        cmap='turbo',
                        vmin=0.0,
                        vmax=1.0,
                        animated=bool(use_blit),
                    )

            fig.tight_layout(pad=0.5)
            if depth_fig is not None:
                depth_fig.tight_layout(pad=0.5)

            self._cam_viewer = {
                "fig": fig,
                "axes": axes,
                "ims": ims,
                "depth_fig": depth_fig,
                "depth_axes": depth_axes,
                "depth_ims": depth_ims,
                "use_blit": bool(use_blit),
                "blit_bg": None,
                "depth_blit_bg": None,
            }
            viewer = self._cam_viewer

            if not getattr(self, "_cam_viewer_opened", False):
                self._cam_viewer_opened = True
                try:
                    backend = matplotlib.get_backend()
                except Exception:
                    backend = "<unknown>"
                print(f"[viewer] Opened Matplotlib camera window '{window_title}' (backend={backend}, DISPLAY={os.environ.get('DISPLAY')})")

            try:
                plt.show(block=False)
                # Prime the canvases so blitting has a valid background.
                fig.canvas.draw()
                fig.canvas.flush_events()
                if depth_fig is not None:
                    depth_fig.canvas.draw()
                    depth_fig.canvas.flush_events()

                if bool(viewer.get("use_blit", False)):
                    try:
                        viewer["blit_bg"] = fig.canvas.copy_from_bbox(fig.bbox)
                    except Exception:
                        viewer["blit_bg"] = None
                    if depth_fig is not None:
                        try:
                            viewer["depth_blit_bg"] = depth_fig.canvas.copy_from_bbox(depth_fig.bbox)
                        except Exception:
                            viewer["depth_blit_bg"] = None

                    def _on_resize(_evt):
                        try:
                            viewer["blit_bg"] = None
                            viewer["depth_blit_bg"] = None
                        except Exception:
                            pass

                    try:
                        fig.canvas.mpl_connect("resize_event", _on_resize)
                    except Exception:
                        pass
                    if depth_fig is not None:
                        try:
                            depth_fig.canvas.mpl_connect("resize_event", _on_resize)
                        except Exception:
                            pass
            except Exception:
                self._cam_viewer = None
                return False

        # If user closed the window, stop.
        try:
            if not plt.fignum_exists(viewer["fig"].number):
                self._cam_viewer = None
                return False
        except Exception:
            pass

        # Update RGB
        try:
            if bool(viewer.get("use_blit", False)):
                fig = viewer["fig"]
                bg = viewer.get("blit_bg")
                if bg is None:
                    fig.canvas.draw()
                    try:
                        bg = fig.canvas.copy_from_bbox(fig.bbox)
                        viewer["blit_bg"] = bg
                    except Exception:
                        bg = None

                if bg is not None:
                    fig.canvas.restore_region(bg)

                for name in order:
                    im = viewer.get("ims", {}).get(name)
                    if im is None:
                        continue
                    im.set_data(_get_rgb(name))
                    try:
                        im.axes.draw_artist(im)
                    except Exception:
                        pass

                try:
                    fig.canvas.blit(fig.bbox)
                except Exception:
                    fig.canvas.draw_idle()
                fig.canvas.flush_events()
            else:
                for name in order:
                    if name in viewer.get("ims", {}):
                        viewer["ims"][name].set_data(_get_rgb(name))
                viewer["fig"].canvas.draw_idle()
                viewer["fig"].canvas.flush_events()
        except Exception:
            return False

        # Update depth (optional)
        if bool(show_depth) and viewer.get("depth_fig") is not None and viewer.get("depth_ims") is not None:
            try:
                depth_fig = viewer["depth_fig"]
                if bool(viewer.get("use_blit", False)):
                    bg = viewer.get("depth_blit_bg")
                    if bg is None:
                        depth_fig.canvas.draw()
                        try:
                            bg = depth_fig.canvas.copy_from_bbox(depth_fig.bbox)
                            viewer["depth_blit_bg"] = bg
                        except Exception:
                            bg = None

                    if bg is not None:
                        depth_fig.canvas.restore_region(bg)

                    for name in order:
                        im = viewer.get("depth_ims", {}).get(name)
                        if im is None:
                            continue
                        im.set_data(_get_depth_norm(name))
                        try:
                            im.axes.draw_artist(im)
                        except Exception:
                            pass

                    try:
                        depth_fig.canvas.blit(depth_fig.bbox)
                    except Exception:
                        depth_fig.canvas.draw_idle()
                    depth_fig.canvas.flush_events()
                else:
                    for name in order:
                        if name in viewer["depth_ims"]:
                            viewer["depth_ims"][name].set_data(_get_depth_norm(name))
                    depth_fig.canvas.draw_idle()
                    depth_fig.canvas.flush_events()
            except Exception:
                pass

        try:
            ps = float(pause_s)
            # pause_s==0 can freeze some backends; use an epsilon.
            plt.pause(ps if ps > 0.0 else 1e-6)
        except Exception:
            pass
        return True

    def profile_camera_capture(
        self,
        which: Iterable[str] = ("external", "head", "left", "right"),
        iters: int = 50,
        renderer=None,
    ):
        """Profile camera capture time (no plotting).

        Prints average milliseconds per frame for each enabled camera.
        This helps distinguish capture cost (PyBullet getCameraImage) from plotting cost.
        """
        which_set = set(str(s).lower() for s in which)
        iters_i = max(1, int(iters))

        timers = {"external": 0.0, "head": 0.0, "left": 0.0, "right": 0.0}
        counts = {k: 0 for k in timers}

        def _t(fn, key: str):
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
            timers[key] += (t1 - t0)
            counts[key] += 1

        for _ in range(iters_i):
            if "external" in which_set:
                try:
                    _t(
                        lambda: self.get_external_cam_img(
                            return_depth=True,
                            return_seg=False,
                            renderer=renderer,
                        ),
                        "external",
                    )
                except Exception:
                    pass
            if "head" in which_set and getattr(self, "head_camera", None) is not None:
                try:
                    _t(lambda: self.get_head_cam_img(return_depth=True, return_seg=False, renderer=renderer), "head")
                except Exception:
                    pass
            if "left" in which_set and getattr(self, "left_arm_camera", None) is not None:
                try:
                    _t(lambda: self.get_left_arm_cam_img(return_depth=True, return_seg=False, renderer=renderer), "left")
                except Exception:
                    pass
            if "right" in which_set and getattr(self, "right_arm_camera", None) is not None:
                try:
                    _t(lambda: self.get_right_arm_cam_img(return_depth=True, return_seg=False, renderer=renderer), "right")
                except Exception:
                    pass

        print("[profile] camera capture avg ms/frame:")
        for k in ("external", "head", "left", "right"):
            if counts[k] == 0:
                continue
            ms = 1000.0 * timers[k] / float(counts[k])
            print(f"  - {k}: {ms:.3f} ms")

    def profile_matplotlib_update(self, which: Iterable[str] = ("external", "head", "left", "right"), iters: int = 50):
        """Profile the full visualize_cameras_matplotlib() call (capture + draw)."""
        iters_i = max(1, int(iters))
        t_total = 0.0
        ok_count = 0
        for _ in range(iters_i):
            t0 = time.perf_counter()
            ok = False
            try:
                ok = bool(
                    self.visualize_cameras_matplotlib(
                        which=which,
                        show_depth=False,
                        window_title="cameras",
                        pause_s=0.0,
                        max_fps=None,
                        use_blit=True,
                        visible=True,
                    )
                )
            except Exception:
                ok = False
            t1 = time.perf_counter()
            if ok:
                t_total += (t1 - t0)
                ok_count += 1

        if ok_count <= 0:
            print("[profile] matplotlib viewer update failed")
            return

        ms = 1000.0 * t_total / float(ok_count)
        print(f"[profile] visualize_cameras_matplotlib avg ms/call: {ms:.3f} ms (n={ok_count})")

    def _get_arm(self, arm: str | None):
        arm = self.active_arm if arm is None else arm
        if arm not in ("left", "right"):
            raise ValueError("arm must be 'left' or 'right'")
        if arm == "left":
            return self.left_eef_id, self.left_controlJoints, self._control_lgripper
        return self.right_eef_id, self.right_controlJoints, self._control_rgripper

    def set_active_arm(self, arm: str):
        """Switch which arm the single-arm-style APIs act on."""
        if arm not in ("left", "right"):
            raise ValueError("arm must be 'left' or 'right'")
        self.active_arm = arm
        if arm == "left":
            self.controlJoints = self.left_controlJoints
            self.eef_id = self.left_eef_id
        else:
            self.controlJoints = self.right_controlJoints
            self.eef_id = self.right_eef_id

    def _get_pad_joint_ids(self, arm: str | None):
        arm = self.active_arm if arm is None else arm
        left_pad_name, right_pad_name = self._pad_joint_by_arm[arm]
        return self.joints[left_pad_name].id, self.joints[right_pad_name].id

    def controlGripper(self, controlMode, targetPosition: float, arm: str | None = None):
        """Compatibility wrapper for the old single-arm gripper controller.

        For the dual-arm robot, this drives the active arm's gripper.
        """
        if controlMode != p.POSITION_CONTROL:
            raise NotImplementedError("Only POSITION_CONTROL is supported")

        _eef_id, _arm_joint_names, ctrl = self._get_arm(arm)
        if ctrl is None:
            return
        ctrl(float(targetPosition))

    def step_simulation(self):
        """
        Hook p.stepSimulation()
        """
        # try:
        #     self._refresh_rendering()
        # except Exception:
        #     pass

        p.stepSimulation()
        if self.vis:
            # Clean up any expired debug bodies.
            if getattr(self, "_debug_bodies_expiry", None):
                try:
                    now = time.time()
                    keep: list[tuple[int, float]] = []
                    for uid, t_exp in list(self._debug_bodies_expiry):
                        uid_i = int(uid)
                        if float(t_exp) <= now:
                            if getattr(self, "_trace_removebody", False):
                                try:
                                    active = list(int(x) for x in getattr(self, "obj_ids", []) if x is not None)
                                except Exception:
                                    active = []
                                try:
                                    cached = list(set(int(v) for v in getattr(self, "_isolated_obj_cache", {}).values() if v is not None))
                                except Exception:
                                    cached = []
                                print(
                                    f"[PB_TRACE_EXPIRY] firing uid={uid_i} active_obj_ids={active} cached_obj_ids_sample={cached[:5]}"
                                )

                            # Safety: never let debug cleanup remove real scene bodies.
                            protected: set[int] = set(int(x) for x in getattr(self, "obj_ids", []) if x is not None)
                            for attr in (
                                "robot_id",
                                "planeID",
                                "tableID",
                                "target_table_id",
                                "target_id",
                                "target_table_id_right",
                                "target_id_right",
                            ):
                                try:
                                    v = getattr(self, attr, None)
                                    if v is not None:
                                        protected.add(int(v))
                                except Exception:
                                    pass

                            if uid_i in protected:
                                # Drop it from the expiry list; it's not a debug body.
                                continue

                            # Additional safety: only remove visual-only marker bodies.
                            # Our marker bodies are created with baseMass=0 and baseCollisionShapeIndex=-1.
                            try:
                                dyn = p.getDynamicsInfo(uid_i, -1)
                                mass = float(dyn[0]) if dyn is not None and len(dyn) > 0 else 0.0
                            except Exception:
                                mass = 0.0
                            if abs(mass) > 1e-9:
                                # Not a massless debug marker -> don't delete it.
                                continue
                            try:
                                cdata = p.getCollisionShapeData(uid_i, -1)
                                has_collision = bool(cdata)
                            except Exception:
                                has_collision = True
                            if has_collision:
                                # Has collision geometry -> not our visual-only marker.
                                continue

                            try:
                                p.removeBody(uid_i)
                            except Exception:
                                pass
                        else:
                            keep.append((uid_i, float(t_exp)))
                    self._debug_bodies_expiry = keep
                except Exception:
                    pass

            # Ensure EE mesh debug is created if enabled.
            if getattr(self, "_eef_mesh_debug_enabled", False):
                try:
                    self._ensure_eef_mesh_debug()
                except Exception:
                    # Debug viz should never break the simulation.
                    pass

            # Update camera frame axis visualization (wrist-mounted cameras).
            if getattr(self, "draw_camera_frame_debug", False):
                try:
                    self._update_camera_frame_debug()
                except Exception:
                    # Debug viz should never break the simulation.
                    pass
            if self.debug:
                if self.eef_debug_lineID is not None:
                    p.removeUserDebugItem(self.eef_debug_lineID)
                eef_xyz = p.getLinkState(self.robot_id, self.eef_id)[0:1]
                end = np.array(eef_xyz[0])
                end[2] -= 0.5
                self.eef_debug_lineID = p.addUserDebugLine(np.array(eef_xyz[0]), end, [1, 0, 0])

            # Optional camera stream driven by stepping (useful when the caller
            # does not explicitly call visualize_cameras_matplotlib() every loop).
            # if getattr(self, "auto_stream_cameras", True):
            if (self.auto_stream_cameras):
                try:
                    # print("[auto_stream_cameras] Streaming camera views...")
                    self.visualize_cameras_matplotlib(
                        which=getattr(self, "auto_stream_cameras_which", ("external", "head", "left", "right")),
                        show_depth=bool(getattr(self, "auto_stream_cameras_show_depth", False)),
                        window_title="cameras",
                        pause_s=float(getattr(self, "auto_stream_cameras_pause_s", 0.001)),
                        max_fps=float(getattr(self, "auto_stream_cameras_max_fps", 10.0)),
                        use_blit=bool(getattr(self, "auto_stream_cameras_use_blit", True)),
                    )
                except Exception:
                    pass
            time.sleep(self.SIMULATION_STEP_DELAY)

    def register_debug_body_expiry(self, body_uid: int, lifetime_s: float) -> None:
        """Request that a visual-only debug body be removed after lifetime_s.

        Note: removal is processed from step_simulation() and only when running
        with GUI enabled (self.vis=True).
        """
        try:
            uid = int(body_uid)
            life = float(lifetime_s)
        except Exception:
            return
        if uid < 0 or life <= 0.0:
            return
        try:
            # De-dup: keep only the most recent expiry for a given uid.
            self.unregister_debug_body_expiry(uid)
            self._debug_bodies_expiry.append((uid, time.time() + life))
            if getattr(self, "_trace_removebody", False):
                try:
                    print(f"[PB_TRACE_EXPIRY] register uid={int(uid)} lifetime_s={float(life)}")
                except Exception:
                    pass
        except Exception:
            pass

    def unregister_debug_body_expiry(self, body_uid: int) -> None:
        """Remove a body from the debug expiry queue.

        Needed because debug bodies can be removed early (e.g., when a new
        marker replaces an old one). If the uid is later re-used by PyBullet,
        a stale expiry entry could delete a real object.
        """
        try:
            uid = int(body_uid)
        except Exception:
            return
        if uid < 0:
            return
        try:
            self._debug_bodies_expiry = [(int(u), float(t)) for (u, t) in self._debug_bodies_expiry if int(u) != uid]
            if getattr(self, "_trace_removebody", False):
                try:
                    print(f"[PB_TRACE_EXPIRY] unregister uid={int(uid)}")
                except Exception:
                    pass
        except Exception:
            pass

    def enable_eef_mesh_debug(
        self,
        enabled: bool = True,
        *,
        arms: Sequence[str] = ("left", "right"),
        stl_path: Optional[str] = None,
        mesh_scale: float = 1.0,
        rgba: Sequence[float] = (1.0, 1.0, 1.0, 0.9),
        parent_to_mesh_xyz: Sequence[float] = (0.0, 0.0, 0.0),
        parent_to_mesh_quat_xyzw: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
        clear_existing: bool = True,
    ):
        """Enable/disable mesh visualization for the current EE pose.

        This attaches a visual-only mesh to each tool0 link via a fixed constraint.

        Args:
            enabled: Toggle mesh visualization.
            arms: Which arms to attach to (any of: "left", "right").
            stl_path: STL mesh to use. Defaults to repo root
                `full_hand_hamid_corrected_axes_tool0.stl`.
            mesh_scale: Uniform mesh scale.
            rgba: RGBA color (alpha < 1 is helpful for seeing through).
            parent_to_mesh_xyz: Offset applied in the parent (tool0) frame.
            parent_to_mesh_quat_xyzw: Orientation offset applied in the parent frame.
            clear_existing: Remove any previously created debug mesh bodies.
        """

        enabled_b = bool(enabled)
        if not enabled_b and clear_existing:
            self._clear_eef_mesh_debug()

        self._eef_mesh_debug_enabled = enabled_b
        if not enabled_b:
            return

        if stl_path is None:
            # Repo root is one level above the environment/ folder.
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            stl_path = os.path.join(repo_root, "full_hand_hamid_corrected_axes_tool0.stl")

        self._eef_mesh_debug_stl_path = str(stl_path)
        self._eef_mesh_debug_mesh_scale = float(mesh_scale)
        self._eef_mesh_debug_rgba = (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]))
        self._eef_mesh_debug_parent_to_mesh_xyz = (
            float(parent_to_mesh_xyz[0]),
            float(parent_to_mesh_xyz[1]),
            float(parent_to_mesh_xyz[2]),
        )
        self._eef_mesh_debug_parent_to_mesh_quat_xyzw = (
            float(parent_to_mesh_quat_xyzw[0]),
            float(parent_to_mesh_quat_xyzw[1]),
            float(parent_to_mesh_quat_xyzw[2]),
            float(parent_to_mesh_quat_xyzw[3]),
        )
        self._eef_mesh_debug_arms = tuple(str(a).lower() for a in arms)

        if clear_existing:
            self._clear_eef_mesh_debug()

        # Create immediately if possible.
        self._ensure_eef_mesh_debug()

    def _clear_eef_mesh_debug(self):
        # Remove constraints first.
        for _arm, cid in list(getattr(self, "_eef_mesh_debug_cid", {}).items()):
            try:
                if int(cid) >= 0:
                    p.removeConstraint(int(cid))
            except Exception:
                pass
        try:
            self._eef_mesh_debug_cid.clear()
        except Exception:
            self._eef_mesh_debug_cid = {}

        for _arm, uid in list(getattr(self, "_eef_mesh_debug_body_uid", {}).items()):
            try:
                if int(uid) >= 0:
                    p.removeBody(int(uid))
            except Exception:
                pass
        try:
            self._eef_mesh_debug_body_uid.clear()
        except Exception:
            self._eef_mesh_debug_body_uid = {}

    def _ensure_eef_mesh_debug(self):
        if not getattr(self, "_eef_mesh_debug_enabled", False):
            return

        if not hasattr(self, "robot_id") or self.robot_id is None:
            return
        if not hasattr(self, "link_name_to_id"):
            return

        stl_path = getattr(self, "_eef_mesh_debug_stl_path", None)
        if not stl_path:
            return

        # Resolve tool0 link indices.
        left_li = None
        right_li = None
        if "left_tool0" in self.link_name_to_id:
            left_li = int(self.link_name_to_id["left_tool0"])
        if "right_tool0" in self.link_name_to_id:
            right_li = int(self.link_name_to_id["right_tool0"])

        arm_to_li = {"left": left_li, "right": right_li}
        arms = getattr(self, "_eef_mesh_debug_arms", ("left", "right"))

        for arm in arms:
            if arm not in ("left", "right"):
                continue
            if arm in self._eef_mesh_debug_body_uid:
                continue
            li = arm_to_li.get(arm)
            if li is None:
                continue

            rgba = getattr(self, "_eef_mesh_debug_rgba", (1.0, 1.0, 1.0, 1.0))
            mesh_scale = float(getattr(self, "_eef_mesh_debug_mesh_scale", 1.0))

            vshape = p.createVisualShape(
                shapeType=p.GEOM_MESH,
                fileName=str(stl_path),
                meshScale=[mesh_scale, mesh_scale, mesh_scale],
                rgbaColor=[float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3])],
            )
            # Spawn it at the current tool pose (it will be constrained after).
            ls = p.getLinkState(self.robot_id, int(li), computeForwardKinematics=True)
            tool_pos, tool_orn = (ls[4], ls[5]) if len(ls) >= 6 else (ls[0], ls[1])
            uid = p.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=int(vshape),
                baseCollisionShapeIndex=-1,
                basePosition=[float(tool_pos[0]), float(tool_pos[1]), float(tool_pos[2])],
                baseOrientation=[float(tool_orn[0]), float(tool_orn[1]), float(tool_orn[2]), float(tool_orn[3])],
            )

            parent_to_xyz = getattr(self, "_eef_mesh_debug_parent_to_mesh_xyz", (0.0, 0.0, 0.0))
            parent_to_quat = getattr(self, "_eef_mesh_debug_parent_to_mesh_quat_xyzw", (0.0, 0.0, 0.0, 1.0))

            cid = p.createConstraint(
                parentBodyUniqueId=int(self.robot_id),
                parentLinkIndex=int(li),
                childBodyUniqueId=int(uid),
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[float(parent_to_xyz[0]), float(parent_to_xyz[1]), float(parent_to_xyz[2])],
                childFramePosition=[0.0, 0.0, 0.0],
                parentFrameOrientation=[
                    float(parent_to_quat[0]),
                    float(parent_to_quat[1]),
                    float(parent_to_quat[2]),
                    float(parent_to_quat[3]),
                ],
                childFrameOrientation=[0.0, 0.0, 0.0, 1.0],
            )

            self._eef_mesh_debug_body_uid[arm] = int(uid)
            self._eef_mesh_debug_cid[arm] = int(cid)

    def enable_camera_frame_debug(self, enabled: bool = True, clear_existing: bool = True):
        """Enable/disable the wrist-camera debug axes + target rays.

        Disabling this can significantly speed up GUI simulations.
        """
        enabled_b = bool(enabled)
        if not enabled_b and clear_existing:
            # Remove existing debug items so they disappear immediately.
            for store in (
                getattr(self, "_camera_frame_debug_ids", {}),
                getattr(self, "_camera_render_frame_debug_ids", {}),
                getattr(self, "_camera_target_debug_ids", {}),
            ):
                try:
                    for ids in list(store.values()):
                        for uid in list(ids.values()):
                            try:
                                if int(uid) >= 0:
                                    p.removeUserDebugItem(int(uid))
                            except Exception:
                                pass
                except Exception:
                    pass
                try:
                    store.clear()
                except Exception:
                    pass

        self.draw_camera_frame_debug = enabled_b

    def enable_camera_streaming(
        self,
        enabled: bool = True,
        which: Iterable[str] = ("external", "head", "left", "right"),
        show_depth: bool = False,
        max_fps: float = 5.0,
        pause_s: float = 0.0,
        use_blit: bool = True,
    ):
        """Enable/disable auto camera streaming driven by step_simulation()."""
        self.auto_stream_cameras = bool(enabled)
        self.auto_stream_cameras_which = tuple(str(s).lower() for s in which)
        self.auto_stream_cameras_show_depth = bool(show_depth)
        self.auto_stream_cameras_max_fps = float(max_fps)
        self.auto_stream_cameras_pause_s = float(pause_s)
        self.auto_stream_cameras_use_blit = bool(use_blit)

    def _update_link_frame_debug(self, key: str, link_index: int, axis_len: float | None = None):
        axis_len = float(self.camera_frame_axis_len if axis_len is None else axis_len)
        ls = p.getLinkState(self.robot_id, link_index, computeForwardKinematics=True)
        link_pos, link_orn = (ls[4], ls[5]) if len(ls) >= 6 else (ls[0], ls[1])
        m = p.getMatrixFromQuaternion(link_orn)

        x_axis = (float(m[0]), float(m[3]), float(m[6]))
        y_axis = (float(m[1]), float(m[4]), float(m[7]))
        z_axis = (float(m[2]), float(m[5]), float(m[8]))

        origin = (float(link_pos[0]), float(link_pos[1]), float(link_pos[2]))
        x_end = (origin[0] + axis_len * x_axis[0], origin[1] + axis_len * x_axis[1], origin[2] + axis_len * x_axis[2])
        y_end = (origin[0] + axis_len * y_axis[0], origin[1] + axis_len * y_axis[1], origin[2] + axis_len * y_axis[2])
        z_end = (origin[0] + axis_len * z_axis[0], origin[1] + axis_len * z_axis[1], origin[2] + axis_len * z_axis[2])

        ids = self._camera_frame_debug_ids.get(key)
        if ids is None:
            ids = {"x": -1, "y": -1, "z": -1}
            self._camera_frame_debug_ids[key] = ids

        ids["x"] = p.addUserDebugLine(origin, x_end, [1, 0, 0], lineWidth=3, replaceItemUniqueId=int(ids["x"]))
        ids["y"] = p.addUserDebugLine(origin, y_end, [0, 1, 0], lineWidth=3, replaceItemUniqueId=int(ids["y"]))
        ids["z"] = p.addUserDebugLine(origin, z_end, [0, 0, 1], lineWidth=3, replaceItemUniqueId=int(ids["z"]))

    @staticmethod
    def _rot_vec_world(quat_xyzw, v_local_xyz):
        m = p.getMatrixFromQuaternion(quat_xyzw)
        return (
            float(m[0]) * float(v_local_xyz[0]) + float(m[1]) * float(v_local_xyz[1]) + float(m[2]) * float(v_local_xyz[2]),
            float(m[3]) * float(v_local_xyz[0]) + float(m[4]) * float(v_local_xyz[1]) + float(m[5]) * float(v_local_xyz[2]),
            float(m[6]) * float(v_local_xyz[0]) + float(m[7]) * float(v_local_xyz[1]) + float(m[8]) * float(v_local_xyz[2]),
        )

    def _get_fingertip_marker(self, arm: str, tool_pos):
        """Estimate a point near the fingertip for the given arm.

        Uses the midpoint between inner finger pad links and extrapolates outward
        away from tool0 by roughly FINGER_LENGTH.

        Returns:
            (x, y, z) tuple in world coordinates, or None if unavailable.
        """
        try:
            left_pad_id, right_pad_id = self._get_pad_joint_ids(arm)
            ls_l = p.getLinkState(self.robot_id, int(left_pad_id), computeForwardKinematics=True)
            ls_r = p.getLinkState(self.robot_id, int(right_pad_id), computeForwardKinematics=True)
            pos_l = ls_l[4] if len(ls_l) >= 6 else ls_l[0]
            pos_r = ls_r[4] if len(ls_r) >= 6 else ls_r[0]
            mid = np.array(
                [
                    0.5 * (float(pos_l[0]) + float(pos_r[0])),
                    0.5 * (float(pos_l[1]) + float(pos_r[1])),
                    0.5 * (float(pos_l[2]) + float(pos_r[2])),
                ],
                dtype=float,
            )
            tp = np.array([float(tool_pos[0]), float(tool_pos[1]), float(tool_pos[2])], dtype=float)
            d = mid - tp
            n = float(np.linalg.norm(d) + 1e-12)
            d = d / n
            tip = mid + float(getattr(self, "FINGER_LENGTH", 0.06)) * d
            return (float(tip[0]), float(tip[1]), float(tip[2]))
        except Exception:
            return None

    def _wrist_camera_eye_axes_target(self, arm: str):
        """Compute wrist camera debug geometry.

        Camera frame definition here is graphics-style:
        - X (red): right
        - Y (green): up
        - Z (blue): forward (view direction)
        """
        if arm not in ("left", "right"):
            raise ValueError("arm must be 'left' or 'right'")

        if arm == "left":
            cam_obj = self.left_arm_camera
            link_name = self._left_arm_camera_link_name
            tool_link = self.left_eef_id
        else:
            cam_obj = self.right_arm_camera
            link_name = self._right_arm_camera_link_name
            tool_link = self.right_eef_id

        if cam_obj is None or link_name is None:
            raise RuntimeError("Wrist camera not available")

        link_index = self.link_name_to_id[link_name]
        ls = p.getLinkState(self.robot_id, link_index, computeForwardKinematics=True)
        link_pos, link_orn = (ls[4], ls[5]) if len(ls) >= 6 else (ls[0], ls[1])

        # Apply parent->camera transform (same as LinkCamera).
        cam_pos, cam_orn = p.multiplyTransforms(
            link_pos,
            link_orn,
            cam_obj.parent_to_camera_xyz,
            cam_obj.parent_to_camera_quat,
        )

        fwd_axis = cam_obj.forward_axis
        # Match LinkCamera: eye_offset always uses the configured forward axis
        # (even if we later flip fwd for aiming).
        fwd_world_for_offset = self._rot_vec_world(cam_orn, fwd_axis)
        fwd_world = fwd_world_for_offset
        up_world = self._rot_vec_world(cam_orn, cam_obj.up_axis)

        # Ensure forward points away from tool0 (avoid looking into gripper base).
        tool_ls = p.getLinkState(self.robot_id, tool_link, computeForwardKinematics=True)
        tool_pos = tool_ls[4] if len(tool_ls) >= 6 else tool_ls[0]
        to_tool = (
            float(tool_pos[0]) - float(cam_pos[0]),
            float(tool_pos[1]) - float(cam_pos[1]),
            float(tool_pos[2]) - float(cam_pos[2]),
        )
        if (fwd_world[0] * to_tool[0] + fwd_world[1] * to_tool[1] + fwd_world[2] * to_tool[2]) > 0.0:
            fwd_axis = (-float(fwd_axis[0]), -float(fwd_axis[1]), -float(fwd_axis[2]))
            fwd_world = self._rot_vec_world(cam_orn, fwd_axis)

        # Eye offset (push the render origin out of camera housing).
        # `cam_pos` is the camera link/optical origin (after parent_to_camera_*).
        # `eye` must match the exact render eye used by LinkCamera.
        eye = (
            float(cam_pos[0]) + float(cam_obj.eye_offset) * float(fwd_world_for_offset[0]),
            float(cam_pos[1]) + float(cam_obj.eye_offset) * float(fwd_world_for_offset[1]),
            float(cam_pos[2]) + float(cam_obj.eye_offset) * float(fwd_world_for_offset[2]),
        )

        # Build an orthonormal camera-local basis in world coordinates.
        # We'll use this basis to apply (x,y,z) offsets in the camera frame:
        # X=right, Y=up, Z=forward.
        right_basis = np.cross(np.array(fwd_world, dtype=float), np.array(up_world, dtype=float))
        rn = float(np.linalg.norm(right_basis) + 1e-12)
        right_basis = (right_basis / rn).tolist()
        up_basis = np.cross(np.array(right_basis, dtype=float), np.array(fwd_world, dtype=float))
        un = float(np.linalg.norm(up_basis) + 1e-12)
        up_basis = (up_basis / un).tolist()

        ox = 0.0 if getattr(self, "wrist_cam_look_ahead_x", None) is None else float(self.wrist_cam_look_ahead_x)
        oz = float(self.wrist_cam_look_ahead) if getattr(self, "wrist_cam_look_ahead_z", None) is None else float(self.wrist_cam_look_ahead_z)
        if getattr(self, "wrist_cam_look_ahead_y", None) is None:
            tilt_deg = float(getattr(self, "wrist_cam_tilt_down", 0.0))
            oy = -math.tan(math.radians(tilt_deg)) * oz
        else:
            oy = float(self.wrist_cam_look_ahead_y)

        target = (
            float(eye[0]) + oz * float(fwd_world[0]) + oy * float(up_basis[0]) + ox * float(right_basis[0]),
            float(eye[1]) + oz * float(fwd_world[1]) + oy * float(up_basis[1]) + ox * float(right_basis[1]),
            float(eye[2]) + oz * float(fwd_world[2]) + oy * float(up_basis[2]) + ox * float(right_basis[2]),
        )

        # Match rendering: if fingertip estimate is available, look through it.
        fingertip = self._get_fingertip_marker(arm, tool_pos)
        if fingertip is not None:
            dv = np.array(
                [
                    float(fingertip[0]) - float(eye[0]),
                    float(fingertip[1]) - float(eye[1]),
                    float(fingertip[2]) - float(eye[2]),
                ],
                dtype=float,
            )
            if float(np.linalg.norm(dv)) > 1e-6:
                target = fingertip

        # Rendered view direction is the look-at vector (target - eye).
        aim = np.array(
            [float(target[0]) - float(eye[0]), float(target[1]) - float(eye[1]), float(target[2]) - float(eye[2])],
            dtype=float,
        )
        aim_n = float(np.linalg.norm(aim) + 1e-12)
        aim = (aim / aim_n).tolist()

        # Build a rendered camera frame from (aim, up_world).
        # This matches how `computeViewMatrix(eye, target, up)` defines the view.
        up_w = np.array(up_world, dtype=float)
        up_proj = up_w - float(np.dot(up_w, aim)) * np.array(aim, dtype=float)
        up_n = float(np.linalg.norm(up_proj) + 1e-12)
        up = (up_proj / up_n).tolist()
        right = np.cross(np.array(up, dtype=float), np.array(aim, dtype=float))
        rn = float(np.linalg.norm(right) + 1e-12)
        right = (right / rn).tolist()

        # Return axes as tuples.
        x_axis = (float(right[0]), float(right[1]), float(right[2]))
        y_axis = (float(up[0]), float(up[1]), float(up[2]))
        z_axis = (float(aim[0]), float(aim[1]), float(aim[2]))
        origin = (float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]))
        tool_pos_t = (float(tool_pos[0]), float(tool_pos[1]), float(tool_pos[2]))
        return origin, eye, (x_axis, y_axis, z_axis), target, tool_pos_t

    def _update_camera_render_frame_debug(self, key: str, eye, axes, axis_len: float | None = None):
        axis_len = float(self.camera_render_frame_axis_len if axis_len is None else axis_len)
        x_axis, y_axis, z_axis = axes
        origin = (float(eye[0]), float(eye[1]), float(eye[2]))

        x_end = (origin[0] + axis_len * x_axis[0], origin[1] + axis_len * x_axis[1], origin[2] + axis_len * x_axis[2])
        y_end = (origin[0] + axis_len * y_axis[0], origin[1] + axis_len * y_axis[1], origin[2] + axis_len * y_axis[2])
        z_end = (origin[0] + axis_len * z_axis[0], origin[1] + axis_len * z_axis[1], origin[2] + axis_len * z_axis[2])

        ids = self._camera_render_frame_debug_ids.get(key)
        if ids is None:
            ids = {"x": -1, "y": -1, "z": -1}
            self._camera_render_frame_debug_ids[key] = ids

        # Thicker than link-frame axes.
        ids["x"] = p.addUserDebugLine(origin, x_end, [1, 0, 0], lineWidth=7, replaceItemUniqueId=int(ids["x"]))
        ids["y"] = p.addUserDebugLine(origin, y_end, [0, 1, 0], lineWidth=7, replaceItemUniqueId=int(ids["y"]))
        ids["z"] = p.addUserDebugLine(origin, z_end, [0, 0, 1], lineWidth=7, replaceItemUniqueId=int(ids["z"]))

    def _update_camera_target_debug(self, key: str, eye, target, marker=None):
        eye = (float(eye[0]), float(eye[1]), float(eye[2]))
        target = (float(target[0]), float(target[1]), float(target[2]))

        if marker is None:
            marker = target
        marker = (float(marker[0]), float(marker[1]), float(marker[2]))

        ids = self._camera_target_debug_ids.get(key)
        if ids is None:
            ids = {"ray": -1, "cx": -1, "cy": -1, "cz": -1}
            self._camera_target_debug_ids[key] = ids

        # Eye-to-target ray (yellow)
        ids["ray"] = p.addUserDebugLine(eye, target, [1, 1, 0], lineWidth=3, replaceItemUniqueId=int(ids["ray"]))

        # Small cross at marker point (yellow)
        s = 0.03
        cx0 = (marker[0] - s, marker[1], marker[2])
        cx1 = (marker[0] + s, marker[1], marker[2])
        cy0 = (marker[0], marker[1] - s, marker[2])
        cy1 = (marker[0], marker[1] + s, marker[2])
        cz0 = (marker[0], marker[1], marker[2] - s)
        cz1 = (marker[0], marker[1], marker[2] + s)
        ids["cx"] = p.addUserDebugLine(cx0, cx1, [1, 1, 0], lineWidth=3, replaceItemUniqueId=int(ids["cx"]))
        ids["cy"] = p.addUserDebugLine(cy0, cy1, [1, 1, 0], lineWidth=3, replaceItemUniqueId=int(ids["cy"]))
        ids["cz"] = p.addUserDebugLine(cz0, cz1, [1, 1, 0], lineWidth=3, replaceItemUniqueId=int(ids["cz"]))

    def _update_camera_frame_debug(self):
        if not getattr(self, "draw_camera_frame_debug", False):
            return
        # Wrist-mounted camera links
        if getattr(self, "_left_arm_camera_link_name", None) in getattr(self, "link_name_to_id", {}):
            li = self.link_name_to_id[self._left_arm_camera_link_name]
            self._update_link_frame_debug("left_cam", li)
            origin, eye, axes, target, tool_pos = self._wrist_camera_eye_axes_target("left")
            # Draw the thick frame at the physical camera origin so it aligns with the mesh.
            self._update_camera_render_frame_debug("left_cam", origin, axes)
            # Draw the ray so it passes through the finger tips (preferred),
            # falling back to tool0 if fingertip links are unavailable.
            marker = self._get_fingertip_marker("left", tool_pos) or tool_pos
            v = np.array([marker[0] - origin[0], marker[1] - origin[1], marker[2] - origin[2]], dtype=float)
            n = float(np.linalg.norm(v) + 1e-12)
            d = v / n
            ray_len = 2.0
            ray_target = (origin[0] + ray_len * float(d[0]), origin[1] + ray_len * float(d[1]), origin[2] + ray_len * float(d[2]))
            self._update_camera_target_debug("left_cam", origin, ray_target, marker=marker)
        if getattr(self, "_right_arm_camera_link_name", None) in getattr(self, "link_name_to_id", {}):
            ri = self.link_name_to_id[self._right_arm_camera_link_name]
            self._update_link_frame_debug("right_cam", ri)
            origin, eye, axes, target, tool_pos = self._wrist_camera_eye_axes_target("right")
            self._update_camera_render_frame_debug("right_cam", origin, axes)
            marker = self._get_fingertip_marker("right", tool_pos) or tool_pos
            v = np.array([marker[0] - origin[0], marker[1] - origin[1], marker[2] - origin[2]], dtype=float)
            n = float(np.linalg.norm(v) + 1e-12)
            d = v / n
            ray_len = 2.0
            ray_target = (origin[0] + ray_len * float(d[0]), origin[1] + ray_len * float(d[1]), origin[2] + ray_len * float(d[2]))
            self._update_camera_target_debug("right_cam", origin, ray_target, marker=marker)

    @staticmethod
    def is_still(handle):
        still_eps = 1e-3
        lin_vel, ang_vel = p.getBaseVelocity(handle)
        # print(np.abs(lin_vel).sum() + np.abs(ang_vel).sum())
        return np.abs(lin_vel).sum() + np.abs(ang_vel).sum() < still_eps

    def wait_until_still(self, objID, max_wait_epochs=500):
        for _ in range(max_wait_epochs):
            self.step_simulation()
            if self.is_still(objID):
                return
        if self.debug:
            print('Warning: Not still after MAX_WAIT_EPOCHS = %d.' %
                  max_wait_epochs)

    def wait_until_all_still(self, max_wait_epochs=1000):
        for _ in range(max_wait_epochs):
            self.step_simulation()
            if np.all(list(self.is_still(obj_id) for obj_id in self.obj_ids)):
                return
        if self.debug:
            print('Warning: Not still after MAX_WAIT_EPOCHS = %d.' %
                  max_wait_epochs)

    def read_debug_parameter(self):
        # read the value of task parameter
        x = p.readUserDebugParameter(self.xin)
        y = p.readUserDebugParameter(self.yin)
        z = p.readUserDebugParameter(self.zin)
        roll = p.readUserDebugParameter(self.rollId)
        pitch = p.readUserDebugParameter(self.pitchId)
        yaw = p.readUserDebugParameter(self.yawId)
        gripper_opening_length = p.readUserDebugParameter(
            self.gripper_opening_length_control)

        return x, y, z, roll, pitch, yaw, gripper_opening_length

    def reset_robot(self):
        # Initialize both arms to safe defaults (from your dual-arm IK server).
        # right_init = [0.0, -np.pi / 2, 1.832017, -0.31, 0.0, np.pi]
        # left_init = [0.0, -np.pi / 2, -1.832017, -2.8, 0.0, np.pi]
        left_init = [-0.059717957173482716, -1.5630534452250977,
                    -1.5702199935913086, -1.6049581966795863,
                    1.5958514213562012, 3.0467772483825684]

        right_init = [0.08292031288146973, -1.5741168461241664,
                    1.5776737372027796, 4.726433026581564,
                    -1.496913258229391, -3.1232035795794886]

        for _ in range(1000):
            for i, name in enumerate(self.right_controlJoints):
                joint = self.joints[name]
                p.setJointMotorControl2(
                    self.robot_id,
                    joint.id,
                    p.POSITION_CONTROL,
                    targetPosition=float(right_init[i]),
                    force=float(joint.maxForce) * self.ARM_FORCE_SCALE,
                    maxVelocity=float(joint.maxVelocity),
                    positionGain=self.ARM_POSITION_GAIN,
                    velocityGain=self.ARM_VELOCITY_GAIN,
                )
            for i, name in enumerate(self.left_controlJoints):
                joint = self.joints[name]
                p.setJointMotorControl2(
                    self.robot_id,
                    joint.id,
                    p.POSITION_CONTROL,
                    targetPosition=float(left_init[i]),
                    force=float(joint.maxForce) * self.ARM_FORCE_SCALE,
                    maxVelocity=float(joint.maxVelocity),
                    positionGain=self.ARM_POSITION_GAIN,
                    velocityGain=self.ARM_VELOCITY_GAIN,
                )

            # default: open both grippers
            self.controlGripper(controlMode=p.POSITION_CONTROL, targetPosition=0.0, arm="left")
            self.controlGripper(controlMode=p.POSITION_CONTROL, targetPosition=0.0, arm="right")
            self.step_simulation()



    def move_arm_away(self):
        # Keep API but move the active arm
        _, arm_joint_names, _ = self._get_arm(None)
        joint = self.joints[arm_joint_names[0]]
        for _ in range(300):
            p.setJointMotorControl2(self.robot_id, joint.id, p.POSITION_CONTROL,
                                    targetPosition=0.,
                                    force=float(joint.maxForce) * self.ARM_FORCE_SCALE,
                                    maxVelocity=float(joint.maxVelocity),
                                    positionGain=self.ARM_POSITION_GAIN,
                                    velocityGain=self.ARM_VELOCITY_GAIN)

                                    
            self.step_simulation()

    def check_grasped(self):
        left_index, right_index = self._get_pad_joint_ids(None)

        contact_left = p.getContactPoints(
            bodyA=self.robot_id, linkIndexA=left_index)
        contact_right = p.getContactPoints(
            bodyA=self.robot_id, linkIndexA=right_index)
        contact_ids = set(item[2] for item in contact_left +
                          contact_right if item[2] in [self.obj_id])
        if len(contact_ids) == 1:
            return True
        return False

    def check_grasped_id(self):
        left_index, right_index = self._get_pad_joint_ids(None)

        contact_left = p.getContactPoints(
            bodyA=self.robot_id, linkIndexA=left_index)
        contact_right = p.getContactPoints(
            bodyA=self.robot_id, linkIndexA=right_index)
        contact_ids = set(item[2] for item in contact_left +
                          contact_right if item[2] in self.obj_ids)
        if len(contact_ids) > 1:
            if self.debug:
                print('Warning: Multiple items in hand!')
        return list(item_id for item_id in contact_ids if item_id in self.obj_ids)

    def check_contact(self, id_a, id_b):
        contact_a = p.getContactPoints(bodyA=id_a)
        contact_ids = set(item[2] for item in contact_a if item[2] in [id_b])
        if len(contact_ids) == 1:
            return True
        return False

    def check_target_reached(self, obj_id, target_id=None):
        target_id = self.target_id if target_id is None else target_id
        aabb = p.getAABB(target_id, -1)
        x_min, x_max = aabb[0][0], aabb[1][0]
        y_min, y_max = aabb[0][1], aabb[1][1]
        pos = p.getBasePositionAndOrientation(obj_id)
        x, y = pos[0][0], pos[0][1]
        if x > x_min and x < x_max and y > y_min and y < y_max:
            return True
        return False

    def gripper_contact(self, bool_operator='and', force=250):
        left_index, right_index = self._get_pad_joint_ids(None)

        contact_left = p.getContactPoints(bodyA=self.robot_id, linkIndexA=left_index)
        contact_right = p.getContactPoints(bodyA=self.robot_id, linkIndexA=right_index)

        # Filter to contacts against currently spawned objects when possible.
        try:
            obj_ids = set(getattr(self, "obj_ids", []) or [])
        except Exception:
            obj_ids = set()
        if obj_ids:
            contact_left = [cp for cp in contact_left if cp[2] in obj_ids]
            contact_right = [cp for cp in contact_right if cp[2] in obj_ids]

        if bool_operator == 'and' and not (contact_right and contact_left):
            return False

        # Prefer contact normal forces from getContactPoints (more directly tied to
        # actual contacts than joint reaction forces).
        def _sum_normal_forces(cps):
            total = 0.0
            for cp in cps:
                try:
                    # Normal force is index 9 in PyBullet's contact point tuple.
                    total += float(cp[9])
                except Exception:
                    pass
            return float(total)

        left_nf = _sum_normal_forces(contact_left)
        right_nf = _sum_normal_forces(contact_right)

        # If force is <= 0, treat any qualifying contact as contact.
        try:
            force_th = float(force)
        except Exception:
            force_th = 0.0

        if force_th <= 0.0:
            if bool_operator == 'and':
                return bool(contact_left) and bool(contact_right)
            return bool(contact_left) or bool(contact_right)

        if bool_operator == 'and':
            return left_nf > force_th and right_nf > force_th
        else:
            return left_nf > force_th or right_nf > force_th

    def _close_contact_detected(self) -> bool:
        """Contact predicate used for 'stop closing' behavior.

        Uses a more permissive default than `gripper_contact()`'s legacy defaults,
        and is configurable via env vars.
        """
        bool_op = str(os.getenv("PB_GRIPPER_CLOSE_CONTACT_BOOL", "or")).strip().lower()
        if bool_op not in ("and", "or"):
            bool_op = "or"
        try:
            force_th = float(os.getenv("PB_GRIPPER_CLOSE_CONTACT_FORCE", "5"))
        except Exception:
            force_th = 5.0
        try:
            debug = str(os.getenv("PB_GRIPPER_CLOSE_CONTACT_DEBUG", "0")).strip().lower() not in (
                "0",
                "false",
                "no",
                "off",
                "",
            )
        except Exception:
            debug = False

        hit = bool(self.gripper_contact(bool_operator=bool_op, force=force_th))
        if debug:
            print(f"[gripper] close_contact: op={bool_op} force_th={force_th} -> {int(hit)}")
        return hit

    def move_gripper(self, gripper_opening_length: float, step: int = 120):
        gripper_opening_length = np.clip( gripper_opening_length, *self.gripper_open_limit)
        gripper_opening_angle = 0.715 - math.asin((gripper_opening_length - 0.010) / 0.1143)  # angle calculation
            
        for _ in range(step):
            # Clamp to the parent joint limits if known
            parent_name = self.mimicParentName_by_arm[self.active_arm]
            if parent_name in self.joints:
                parent = self.joints[parent_name]
                gripper_opening_angle = float(np.clip(gripper_opening_angle, parent.lowerLimit, parent.upperLimit))
            self.controlGripper(controlMode=p.POSITION_CONTROL, targetPosition=gripper_opening_angle)

            self.step_simulation()

    def auto_close_gripper(self, step: int = 170, check_contact: bool = False) -> bool:
        # Get initial gripper open position
        parent_name = self.mimicParentName_by_arm[self.active_arm]
        initial_position = p.getJointState(self.robot_id, self.joints[parent_name].id)[0]
        initial_position = math.sin(0.715 - initial_position) * 0.1143 + 0.010
        for step_idx in range(1, step):
            current_target_open_length = initial_position - step_idx / step * initial_position

            self.move_gripper(current_target_open_length, 1)
            if current_target_open_length < 1e-5:
                return False

            # time.sleep(1 / 120)
            if check_contact and self._close_contact_detected():
                # Stop squeezing and hold briefly to let contacts settle.
                try:
                    hold_steps = int(os.getenv("PB_GRIPPER_CLOSE_HOLD_STEPS", "10"))
                except Exception:
                    hold_steps = 10
                if hold_steps > 0:
                    self.move_gripper(current_target_open_length, hold_steps)
                return True
        return False

    def close_gripper(self, check_contact: bool = False) -> bool:
        # Backwards-compat hook (used by move_ee when try_close_gripper=True)
        return self.auto_close_gripper(check_contact=check_contact)

    def calc_z_offset(self, gripper_opening_length: float):
        gripper_opening_length = np.clip(
            gripper_opening_length, *self.gripper_open_limit)
        gripper_opening_angle = 0.715 - \
            math.asin((gripper_opening_length - 0.010) / 0.1143)
        if self.gripper_type == '140':
            gripper_length = 10.3613 * \
                np.sin(1.64534-0.24074 * (gripper_opening_angle / np.pi)) - 10.1219
        else:
            gripper_length = 1.231 - 1.1
        return gripper_length

    def _refresh_rendering(self):
        """Force a small renderer refresh (mainly for GUI/OpenGL)."""
        try:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        except Exception:
            pass

    def _set_rendering_enabled(self, enabled: bool):
        """Enable/disable rendering in GUI to avoid OpenGL glitches during load/remove."""
        # Only meaningful in GUI; harmless in DIRECT.
        try:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1 if enabled else 0)
        except Exception:
            pass

    def _set_body_collision_enabled(self, body_id: int, enabled: bool):
        try:
            num_joints = p.getNumJoints(body_id)
        except Exception:
            return
        group = 1 if enabled else 0
        mask = 1 if enabled else 0
        for link_idx in range(-1, num_joints):
            try:
                p.setCollisionFilterGroupMask(body_id, link_idx, group, mask)
            except Exception:
                pass

    def _hide_body(self, body_id: int):
        if getattr(self, "_trace_removebody", False):
            try:
                print(f"[PB_TRACE_HIDE] hide_body({int(body_id)})")
            except Exception:
                pass
        self._set_body_collision_enabled(body_id, False)
        try:
            p.resetBasePositionAndOrientation(body_id, *self._hidden_object_pose)
            p.resetBaseVelocity(body_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        except Exception:
            pass

    def _show_body(self, body_id: int):
        self._set_body_collision_enabled(body_id, True)

    def _ensure_body_opaque(self, body_id: int):
        try:
            vs = p.getVisualShapeData(body_id)
        except Exception:
            vs = None

        if not vs:
            # Fallback
            try:
                p.changeVisualShape(body_id, -1, rgbaColor=[1, 1, 1, 1])
            except Exception:
                pass
            return

        for item in vs:
            # item fields: (bodyUniqueId, linkIndex, visualGeometryType, dimensions, filename,
            #                localVisualFramePosition, localVisualFrameOrientation, rgbaColor, ...)
            link_index = int(item[1])
            rgba = item[7]
            try:
                r, g, b, _a = [float(v) for v in rgba]
                p.changeVisualShape(body_id, link_index, rgbaColor=[r, g, b, 1.0])
            except Exception:
                pass

    def _body_exists(self, body_id: int) -> bool:
        try:
            p.getBasePositionAndOrientation(body_id)
            return True
        except Exception:
            return False

    def _apply_object_dynamics(self, obj_id: int, mod_stiffness: bool):
        try:
            # Objects must have sufficient friction too; otherwise they will still
            # slip/roll out even if the finger pads are high-friction.
            obj_lat_fric = float(os.getenv("PB_OBJECT_LATERAL_FRICTION", "2.5"))
            obj_roll_fric = float(os.getenv("PB_OBJECT_ROLLING_FRICTION", "0.01"))
            obj_spin_fric = float(os.getenv("PB_OBJECT_SPINNING_FRICTION", "0.01"))
            if mod_stiffness:
                p.changeDynamics(
                    obj_id,
                    -1,
                    lateralFriction=obj_lat_fric,
                    rollingFriction=obj_roll_fric,
                    spinningFriction=obj_spin_fric,
                    restitution=0.0,
                    frictionAnchor=1,
                    contactStiffness=100000,
                    contactDamping=50.0,
                )
            else:
                p.changeDynamics(
                    obj_id,
                    -1,
                    lateralFriction=obj_lat_fric,
                    rollingFriction=obj_roll_fric,
                    spinningFriction=obj_spin_fric,
                    restitution=0.0,
                    frictionAnchor=1,
                )
        except Exception:
            pass

    def _load_obj_body(self, path, pos, yaw, mod_orn=False, mod_stiffness=False):
        # NOTE: This method does not register the object in self.obj_ids/positions/orientations.
        pos = list(pos)
        orn = p.getQuaternionFromEuler([0, 0, yaw])
        self._set_rendering_enabled(False)
        obj_id = p.loadURDF(path, pos, orn, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        # adjust position according to height
        aabb = p.getAABB(obj_id, -1)
        if mod_orn:
            minm, maxm = aabb[0][1], aabb[1][1]
            orn = p.getQuaternionFromEuler([0, np.pi * 0.5, yaw])
        else:
            minm, maxm = aabb[0][2], aabb[1][2]

        pos[2] += (maxm - minm) / 2
        p.resetBasePositionAndOrientation(obj_id, pos, orn)
        self._ensure_body_opaque(obj_id)
        self._apply_object_dynamics(obj_id, bool(mod_stiffness))
        self._set_rendering_enabled(True)
        self._refresh_rendering()
        return obj_id, pos, orn

    def remove_obj(self, obj_id):
        # Get index of obj in id list, then remove object from simulation
        self._set_rendering_enabled(False)
        idx = self.obj_ids.index(obj_id)
        self.obj_orientations.pop(idx)
        self.obj_positions.pop(idx)
        self.obj_ids.pop(idx)
        p.removeBody(obj_id)
        self._set_rendering_enabled(True)
        self._refresh_rendering()

    def remove_all_obj(self):
        self._set_rendering_enabled(False)
        self.obj_positions.clear()
        self.obj_orientations.clear()
        for obj_id in self.obj_ids:
            p.removeBody(obj_id)
        self.obj_ids.clear()
        self._set_rendering_enabled(True)
        self._refresh_rendering()

    def reset_all_obj(self):
        for i, obj_id in enumerate(self.obj_ids):
            p.resetBasePositionAndOrientation(
                obj_id, self.obj_positions[i], self.obj_orientations[i])
        self.wait_until_all_still()

    def update_obj_states(self):
        for i, obj_id in enumerate(self.obj_ids):
            pos, orn = p.getBasePositionAndOrientation(obj_id)
            self.obj_positions[i] = pos
            self.obj_orientations[i] = orn

    def load_obj(self, path, pos, yaw, mod_orn=False, mod_stiffness=False):
        obj_id, pos_out, orn_out = self._load_obj_body(path, pos, yaw, mod_orn, mod_stiffness)
        self.obj_ids.append(obj_id)
        self.obj_positions.append(pos_out)
        self.obj_orientations.append(orn_out)
        return obj_id, pos_out, orn_out

    def cycle_isolated_obj(self, path, mod_orn=False, mod_stiffness=False):
        """Spawn an isolated object using a cache (avoids removeBody/loadURDF loops).

        This is a GUI-stability workaround: repeatedly removing/loading URDFs with complex
        visuals can cause the main viewport to stop drawing solid geometry.
        """
        # Hide currently active objects.
        for obj_id in list(self.obj_ids):
            self._hide_body(obj_id)

        self.obj_ids.clear()
        self.obj_positions.clear()
        self.obj_orientations.clear()

        r_x = random.uniform(self.obj_init_pos[0] - 0.1, self.obj_init_pos[0] + 0.1)
        r_y = random.uniform(self.obj_init_pos[1] - 0.1, self.obj_init_pos[1] + 0.1)
        yaw = random.uniform(0, np.pi)
        pos = [r_x, r_y, self.Z_TABLE_TOP]

        key = (str(path), bool(mod_orn), bool(mod_stiffness))
        obj_id = self._isolated_obj_cache.get(key)

        if obj_id is None or not self._body_exists(obj_id):
            obj_id, pos_out, orn_out = self._load_obj_body(path, pos, yaw, mod_orn, mod_stiffness)
            self._isolated_obj_cache[key] = obj_id
            if getattr(self, "_trace_removebody", False):
                try:
                    print(f"[PB_TRACE_OBJ] load new obj_id={int(obj_id)} path={str(path)}")
                except Exception:
                    pass
        else:
            # Reuse cached body.
            self._show_body(obj_id)
            if getattr(self, "_trace_removebody", False):
                try:
                    print(f"[PB_TRACE_OBJ] reuse obj_id={int(obj_id)} path={str(path)}")
                except Exception:
                    pass
            orn_out = p.getQuaternionFromEuler([0, np.pi * 0.5, yaw]) if mod_orn else p.getQuaternionFromEuler([0, 0, yaw])
            p.resetBasePositionAndOrientation(obj_id, pos, orn_out)
            # Recompute height offset for current orientation.
            aabb = p.getAABB(obj_id, -1)
            minm, maxm = (aabb[0][1], aabb[1][1]) if mod_orn else (aabb[0][2], aabb[1][2])
            pos_out = list(pos)
            pos_out[2] += (maxm - minm) / 2
            p.resetBasePositionAndOrientation(obj_id, pos_out, orn_out)
            p.resetBaseVelocity(obj_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
            self._ensure_body_opaque(obj_id)
            self._apply_object_dynamics(obj_id, bool(mod_stiffness))
            self._refresh_rendering()

        self.obj_ids.append(obj_id)
        self.obj_positions.append(pos_out)
        self.obj_orientations.append(orn_out)

        for _ in range(100):
            self.step_simulation()
        self.wait_until_still(obj_id)
        self.update_obj_states()

    def load_isolated_obj(self, path, mod_orn=False, mod_stiffness=False):
        # Backwards compatible name; prefer cycle_isolated_obj() for GUI stability.
        return self.cycle_isolated_obj(path, mod_orn=mod_orn, mod_stiffness=mod_stiffness)

    def create_temp_box(self, width, num):
        box_width = width
        box_height = 0.2
        box_z = self.Z_TABLE_TOP + (box_height/2)
        id1 = p.loadURDF(f'environment/urdf/objects/slab{num}.urdf',
                         [self.obj_init_pos[0] - box_width /
                             2, self.obj_init_pos[1], box_z],
                         p.getQuaternionFromEuler([0, 0, 0]),
                         useFixedBase=True)
        id2 = p.loadURDF(f'environment/urdf/objects/slab{num}.urdf',
                         [self.obj_init_pos[0] + box_width /
                             2, self.obj_init_pos[1], box_z],
                         p.getQuaternionFromEuler([0, 0, 0]),
                         useFixedBase=True)
        id3 = p.loadURDF(f'environment/urdf/objects/slab{num}.urdf',
                         [self.obj_init_pos[0], self.obj_init_pos[1] +
                             box_width/2, box_z],
                         p.getQuaternionFromEuler([0, 0, np.pi*0.5]),
                         useFixedBase=True)
        id4 = p.loadURDF(f'environment/urdf/objects/slab{num}.urdf',
                         [self.obj_init_pos[0], self.obj_init_pos[1] -
                             box_width/2, box_z],
                         p.getQuaternionFromEuler([0, 0, np.pi*0.5]),
                         useFixedBase=True)
        return [id1, id2, id3, id4]

    def create_pile(self, obj_info):
        box_ids = self.create_temp_box(0.36, 1)
        for path, mod_orn, mod_stiffness in obj_info:
            margin = 0.025
            r_x = random.uniform(
                self.obj_init_pos[0] - margin, self.obj_init_pos[0] + margin)
            r_y = random.uniform(
                self.obj_init_pos[1] - margin, self.obj_init_pos[1] + margin)
            yaw = random.uniform(0, np.pi)
            pos = [r_x, r_y, 1.0]

            obj_id, _, _ = self.load_obj(
                path, pos, yaw, mod_orn, mod_stiffness)
            for _ in range(10):
                self.step_simulation()
            self.wait_until_still(obj_id, 30)

        self.wait_until_all_still()
        for handle in box_ids:
            p.removeBody(handle)
        box_ids = self.create_temp_box(0.47, 2)
        self.wait_until_all_still(300)
        for handle in box_ids:
            p.removeBody(handle)
        self.wait_until_all_still(300)
        self.update_obj_states()

    def move_obj_along_axis(self, obj_id, axis, operator, step, stop):
        collison = False
        while not collison:
            pos, orn = p.getBasePositionAndOrientation(obj_id)
            new_pos = list(pos)
            if operator == '+':
                new_pos[axis] += step
                if new_pos[axis] > stop:
                    break
            else:
                new_pos[axis] -= step
                if new_pos[axis] < stop:
                    break
            # Move object towards center
            p.resetBasePositionAndOrientation(obj_id, new_pos, orn)
            p.stepSimulation()
            contact_a = p.getContactPoints(obj_id)
            # If object collides with any other object, stop
            contact_ids = set(item[2]
                              for item in contact_a if item[2] in self.obj_ids)
            if len(contact_ids) != 0:
                collison = True
        # Move one step back
        pos, orn = p.getBasePositionAndOrientation(obj_id)
        new_pos = list(pos)
        if operator == '+':
            new_pos[axis] -= step
        else:
            new_pos[axis] += step
        p.resetBasePositionAndOrientation(obj_id, new_pos, orn)

    def create_packed(self, obj_info):
        init_x, init_y, init_z = self.obj_init_pos[0], self.obj_init_pos[1], self.Z_TABLE_TOP
        yaw = random.uniform(0, np.pi)
        path, mod_orn, mod_stiffness = obj_info[0]
        center_obj, _, _ = self.load_obj(
            path, [init_x, init_y, init_z], yaw, mod_orn, mod_stiffness)

        margin = 0.3
        yaw = random.uniform(0, np.pi)
        path, mod_orn, mod_stiffness = obj_info[1]
        left_obj_id, _, _ = self.load_obj(
            path, [init_x-margin, init_y, init_z], yaw, mod_orn, mod_stiffness)
        yaw = random.uniform(0, np.pi)
        path, mod_orn, mod_stiffness = obj_info[2]
        top_obj_id, _, _ = self.load_obj(
            path, [init_x, init_y+margin, init_z], yaw, mod_orn, mod_stiffness)
        yaw = random.uniform(0, np.pi)
        path, mod_orn, mod_stiffness = obj_info[3]
        right_obj_id, _, _ = self.load_obj(
            path, [init_x+margin, init_y, init_z], yaw, mod_orn, mod_stiffness)
        yaw = random.uniform(0, np.pi)
        path, mod_orn, mod_stiffness = obj_info[4]
        down_obj_id, _, _ = self.load_obj(
            path, [init_x, init_y-margin, init_z], yaw, mod_orn, mod_stiffness)

        self.wait_until_all_still()
        step = 0.01
        self.move_obj_along_axis(left_obj_id, 0, '+', step, init_x)
        self.move_obj_along_axis(top_obj_id, 1, '-', step, init_y)
        self.move_obj_along_axis(right_obj_id, 0, '-', step, init_x)
        self.move_obj_along_axis(down_obj_id, 1, '+', step, init_y)
        self.update_obj_states()


    def move_ee(
        self,
        action,
        max_step=600,
        check_collision_config=None,
        custom_velocity=None,
        try_close_gripper=False,
        verbose=False,
    ):
        x, y, z, orn = action
        x = np.clip(x, *self.ee_position_limit[0])
        y = np.clip(y, *self.ee_position_limit[1])
        z = np.clip(z, *self.ee_position_limit[2])
        # set damping for robot arm and gripper
        jd = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
              0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        jd = jd * 0
        still_open_flag_ = True  # Hot fix

        eef_id, arm_joint_names, _ = self._get_arm(None)
        t0 = float(time.time())
        real_xyz, real_xyzw = p.getLinkState(self.robot_id, eef_id)[0:2]

        # Precompute target orientation in Euler once (orn is constant for this call).
        target_roll, target_pitch, target_yaw = p.getEulerFromQuaternion(orn)

        # Dynamic step budget: if max_step is None (or <=0), estimate a reasonable
        # cap based on initial distance, and still break early on convergence.
        pos_tol = 0.005
        # Orientation tolerance in radians (via quaternion angular distance).
        orn_tol = 0.02
        try:
            d0 = float(np.linalg.norm(np.array((x, y, z)) - np.array(real_xyz)))
        except Exception:
            d0 = 0.0

        if max_step is None:
            max_step_i = 0
        else:
            try:
                max_step_i = int(max_step)
            except Exception:
                max_step_i = 0

        if max_step_i <= 0:
            # Rough bound using a fixed alpha derived from initial distance.
            # Remaining distance after n steps: ~ (1-alpha)^n * d0
            alpha0 = float(np.clip(d0 * 0.5, 0.05, 0.2))
            if d0 <= pos_tol:
                est = 1
            else:
                try:
                    est = int(np.ceil(np.log(pos_tol / max(d0, 1e-9)) / np.log(max(1.0 - alpha0, 1e-6))))
                except Exception:
                    est = 600
            # Keep within a sane range. Minimum 200 helps avoid premature failure
            # when dynamics/gains require extra settling.
            max_step_i = int(np.clip(est, 200, 2000))

        reached = False
        
        for _ in range(max_step_i):

            # Smoothing: blend target toward current EE position.
            ## Option1: A fixed alpha to a small value (e.g., 0.1-0.3) for consistent timing and smoother trajectories, but may require tuning of max_step_i to ensure convergence within the step budget. Too small alpha can cause slow convergence, while too large can cause overshooting and instability, especially with complex dynamics or low control gains.
            alpha = 0.2  # Fixed alpha for more consistent timing; may require tuning of max_step_i.

            # Option2: Adaptable alpha based on distance: use larger steps when far, and slow down when near the target to avoid overshooting and allow settling. The specific scaling factor (0.5) and thresholds (0.05, 0.02) may require tuning based on the robot's dynamics and control gains.
            # Larger distance -> larger alpha (faster), near target -> smaller alpha (smoother).
            try:
                distance = float(np.linalg.norm(np.array((x, y, z)) - np.array(real_xyz)))
            except Exception:
                distance = 0.0
            # Use a larger alpha when far (faster approach), and snap to alpha=1.0
            # when very close to avoid stalling with tiny incremental targets.
            alpha = float(np.clip(distance * 0.5, 0.05, 0.5))
            if distance < 0.02:
                alpha = 1.0

            

            # apply IK
            x_tmp = alpha * x + (1-alpha)*real_xyz[0]
            y_tmp = alpha * y + (1-alpha)*real_xyz[1]
            z_tmp = alpha * z + (1-alpha)*real_xyz[2]
            
            joint_poses = p.calculateInverseKinematics(bodyUniqueId=self.robot_id, endEffectorLinkIndex=eef_id, 
                                                       targetPosition=[x_tmp, y_tmp, z_tmp], targetOrientation=orn, 
                                                       maxNumIterations=50)

            # PyBullet IK returns values for all movable joints in the model.
            # Extract only the 6 arm joints for the active arm.
            arm_positions = []
            for name in arm_joint_names:
                jid = self.joints[name].id
                idx = self.joint_id_to_movable_index[jid]
                arm_positions.append(joint_poses[idx])

            # Ramp up position gain near target to reduce steady-state error.
            pos_gain = float(self.ARM_POSITION_GAIN)
            if distance < 0.05:
                pos_gain = min(0.2, pos_gain * 4.0)
            if distance < 0.02:
                pos_gain = min(0.3, pos_gain * 6.0)

            for i, name in enumerate(arm_joint_names):
                joint = self.joints[name]
                pose = float(arm_positions[i])
                p.setJointMotorControl2(
                    self.robot_id,
                    joint.id,
                    p.POSITION_CONTROL,
                    targetPosition=pose,
                    force=float(joint.maxForce) * self.ARM_FORCE_SCALE,
                    maxVelocity=float(joint.maxVelocity)
                    if custom_velocity is None
                    else float(custom_velocity) * (i + 1),
                    positionGain=pos_gain,
                    #velocityGain=self.ARM_VELOCITY_GAIN,
                )

               
            # Let the arm settle toward the current IK target for a few physics steps.
            # Near the target alpha is small -> use more substeps.
            try:
                substeps = int(np.clip(round(0.2 / max(alpha, 1e-6)), 1, 5))
            except Exception:
                substeps = 1
            for _sub in range(substeps):
                self.step_simulation()

            if try_close_gripper and still_open_flag_ and not self._close_contact_detected():
                still_open_flag_ = self.close_gripper(check_contact=True)

            # Check if contact with objects
            # if check_collision_config and self.gripper_contact(**check_collision_config):
            #     if self.debug:
            #         print('Collision detected!', self.check_grasped_id())
            #     return False, p.getLinkState(self.robot_id, eef_id)[0:2]

            # Check xyz and rpy error
            real_xyz, real_xyzw = p.getLinkState(self.robot_id, eef_id)[0:2]
            pos_err = float(np.linalg.norm(np.array((x, y, z)) - np.array(real_xyz)))

            # Quaternion angular distance (robust vs Euler wrap-around).
            try:
                q_real = np.array(real_xyzw, dtype=np.float64)
                q_tgt = np.array(orn, dtype=np.float64)
                dot = float(np.dot(q_real, q_tgt))
                dot = float(np.clip(abs(dot), 0.0, 1.0))
                orn_err = float(2.0 * math.acos(dot))
            except Exception:
                # Fallback to Euler diffs (less robust).
                real_roll, real_pitch, real_yaw = p.getEulerFromQuaternion(real_xyzw)
                orn_err = float(
                    np.abs(
                        np.array(
                            [
                                target_roll - real_roll,
                                target_pitch - real_pitch,
                                target_yaw - real_yaw,
                            ]
                        )
                    ).sum()
                )

            if pos_err < pos_tol and orn_err < orn_tol:
                reached = True
                if verbose:
                    print('Reach target with', _, 'steps')
                break

        if verbose:
            print(f"move_ee duration: {time.time()-t0:.3f}s")

        if (not reached) and self.debug:
            print('Failed to reach the target')

        # Return the latest EE state regardless; caller can inspect error.
        return bool(reached), (real_xyz, real_xyzw)

    def grasp(self, pos: tuple, yaw: float, gripper_opening_length: float, obj_height: float, debug: bool = False):
        """
        Method to perform grasp
        pos [x y z]: The axis in real-world coordinate
        yaw: float,   for grasp, it should be in [-pi/2, pi/2)
        """
        succes_grasp, succes_target = False, False
        grasped_obj_id = None

        x, y, z = pos

        # Auto-select arm: negative x -> right arm, otherwise -> left arm.
        self.set_active_arm("right" if float(x) < 0.0 else "left")
        # Substracht gripper finger length from z
        z -= self.finger_length
        z = np.clip(z, *self.ee_position_limit[2])

        # Move above target
        #self.reset_robot()
        self.move_gripper(0.1)
        orn = self._tool0_quat_from_grasp_yaw(yaw)
        self.move_ee([x, y, self.GRIPPER_MOVING_HEIGHT, orn])

        # Reduce grip to get a tighter grip
        gripper_opening_length *= self.GRIP_REDUCTION

        # Grasp and lift object
        z_offset = self.calc_z_offset(gripper_opening_length)
        self.move_ee([x, y, z + z_offset, orn])
        # self.move_gripper(gripper_opening_length)
        self.auto_close_gripper(check_contact=True)
        for _ in range(150):
            self.step_simulation()
        # time.sleep(0.15)

        self.move_ee([x, y, self.GRIPPER_MOVING_HEIGHT, orn])

        # # If the object has been grasped and lifted off the table
        # grasped_id = self.check_grasped_id()
        # if len(grasped_id) == 1:
        #     succes_grasp = True
        #     grasped_obj_id = grasped_id[0]
        # else:
        #     return succes_target, succes_grasp

        # Move object to target zone
        target_pos = self._target_zone_pos_by_arm[self.active_arm]
        target_id = self._target_id_by_arm[self.active_arm]

        y_drop = target_pos[2] + z_offset + obj_height + 0.15
        # Keep the tool vertical during place.
        y_orn = self._tool0_quat_topdown()

        #self.move_arm_away()
        self.move_ee([target_pos[0], target_pos[1], 1.25, y_orn])
        self.move_ee([target_pos[0], target_pos[1], y_drop, y_orn])
        self.move_gripper(0.085)
        self.move_ee([target_pos[0], target_pos[1], self.GRIPPER_MOVING_HEIGHT, y_orn])

        # Wait then check if object is in target zone
        for _ in range(20):
            self.step_simulation()

        if self.check_target_reached(grasped_obj_id, target_id=target_id):
            succes_target = True
            #self.remove_obj(grasped_obj_id)

        

        return succes_grasp, succes_target

    def close(self):
        p.disconnect(self.physicsClient)

    

