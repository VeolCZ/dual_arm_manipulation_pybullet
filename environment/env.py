from environment.utilities import setup_sisbot, setup_dual_arm_sisbot, Camera, LinkCamera
import math
import time
import numpy as np
import pybullet as p
import pybullet_data
import random
import os


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
    ROBOT_LINK_LINEAR_DAMPING = 0.04
    ROBOT_LINK_ANGULAR_DAMPING = 0.04

    # Optional per-arm yaw offsets (radians) for matching tool frame conventions.
    GRASP_YAW_OFFSET_LEFT = 0.0
    GRASP_YAW_OFFSET_RIGHT = 0.0

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

    def __init__(self, camera: Camera, vis=False, debug=False, gripper_type='140', finger_length=0.06) -> None:
        self.vis = vis
        self.debug = debug
        self.camera = camera

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
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(1.0 / 240.0)
        p.setPhysicsEngineParameter(numSolverIterations=200)
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

        # -----------------------------
        # Load dual-arm robot
        # -----------------------------
        # Load from an absolute path to make relative mesh paths reliable.
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        # dual_urdf_path = os.path.join(repo_root, "environment", "urdf", "dual_arm_robot.urdf")
        dual_urdf_path = os.path.join(repo_root, "environment", "urdf", "full_stationary_robot.urdf")

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

        # -----------------------------
        # Head-mounted camera (Kinect)
        # -----------------------------
        # Prefer an optical frame if present (ROS convention: Z forward, Y down).
        if "head_mount_kinect_rgb_optical_frame" in self.link_name_to_id:
            self._head_camera_link_name = "head_mount_kinect_rgb_optical_frame"
        elif "head_mount_kinect_link" in self.link_name_to_id:
            self._head_camera_link_name = "head_mount_kinect_link"
        else:
            self._head_camera_link_name = None

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
                p.changeDynamics(self.robot_id, self.joints[left_pad].id, lateralFriction=1)
            if right_pad in self.joints:
                p.enableJointForceTorqueSensor(self.robot_id, self.joints[right_pad].id)
                p.changeDynamics(self.robot_id, self.joints[right_pad].id, lateralFriction=1)

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
            dist = 1.5
            yaw = 30
            pitch = -50
            target = [0.1, -0.30, 0.95]
            p.resetDebugVisualizerCamera(dist, yaw, pitch, target)

            # camera model
            alpha = 0.025 #m 
            z_cam_l1 = camera.z #m 

            color = [0.3, 0, 0]
            p.addUserDebugText("RGB-D camera", [camera.x-0.1, camera.y, camera.z+0.02], color, textSize=2)
             
            p.addUserDebugLine([camera.x+alpha, camera.y+alpha, z_cam_l1], [camera.x+alpha, camera.y-alpha, z_cam_l1], color, lineWidth=4)
            p.addUserDebugLine([camera.x+alpha, camera.y-alpha, z_cam_l1], [camera.x-alpha, camera.y-alpha, z_cam_l1], color, lineWidth=4)
            p.addUserDebugLine([camera.x-alpha, camera.y-alpha, z_cam_l1], [camera.x-alpha, camera.y+alpha, z_cam_l1], color, lineWidth=4)
            p.addUserDebugLine([camera.x-alpha, camera.y+alpha, z_cam_l1], [camera.x+alpha, camera.y+alpha, z_cam_l1], color, lineWidth=4)
        
            z_cam_l2 = camera.z - 0.07 #m 
               
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
            z_cam_l3 = camera.z - 0.125 #m 
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

            ### working area             
            working_area = 0.79 #m 
            beta = 0.4 
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

    def get_head_cam_img(self):
        """Return (rgb, depth, seg) from the head-mounted Kinect camera."""
        if self.head_camera is None or self._head_camera_link_name is None:
            raise RuntimeError("Head camera link not present in the loaded URDF")
        link_index = self.link_name_to_id[self._head_camera_link_name]
        return self.head_camera.get_cam_img(self.robot_id, link_index)

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
        p.stepSimulation()
        if self.vis:
            if self.debug:
                if self.eef_debug_lineID is not None:
                    p.removeUserDebugItem(self.eef_debug_lineID)
                eef_xyz = p.getLinkState(self.robot_id, self.eef_id)[0:1]
                end = np.array(eef_xyz[0])
                end[2] -= 0.5
                self.eef_debug_lineID = p.addUserDebugLine(np.array(eef_xyz[0]), end, [1, 0, 0])
            time.sleep(self.SIMULATION_STEP_DELAY)

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

        for _ in range(500):
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

        contact_left = p.getContactPoints( bodyA=self.robot_id, linkIndexA=left_index)
        contact_right = p.getContactPoints( bodyA=self.robot_id, linkIndexA=right_index)

        if bool_operator == 'and' and not (contact_right and contact_left):
            return False

        # Check the force
        left_force = p.getJointState(self.robot_id, left_index)[2][:3]  # 6DOF, Torque is ignored
        right_force = p.getJointState(self.robot_id, right_index)[2][:3]
        left_norm, right_norm = np.linalg.norm(left_force), np.linalg.norm(right_force)
        
        # print(left_norm, right_norm)
        if bool_operator == 'and':
            return left_norm > force and right_norm > force
        else:
            return left_norm > force or right_norm > force

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
            if check_contact and self.gripper_contact():
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
            if mod_stiffness:
                p.changeDynamics(
                    obj_id,
                    -1,
                    lateralFriction=1,
                    rollingFriction=0.001,
                    spinningFriction=0.002,
                    restitution=0.01,
                    contactStiffness=100000,
                    contactDamping=0.0,
                )
            else:
                p.changeDynamics(
                    obj_id,
                    -1,
                    lateralFriction=1,
                    rollingFriction=0.002,
                    spinningFriction=0.001,
                    restitution=0.01,
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
        else:
            # Reuse cached body.
            self._show_body(obj_id)
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


    def move_ee(self, action, max_step=600, check_collision_config=None, custom_velocity=None,
                try_close_gripper=False, verbose=False):
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
        real_xyz, real_xyzw = p.getLinkState(self.robot_id, eef_id)[0:2]
        alpha = 0.2 # this parameter can be tuned to make the movement  smoother
        
        for _ in range(max_step):

            # apply IK
            x_tmp = alpha * x + (1-alpha)*real_xyz[0]
            y_tmp = alpha * y + (1-alpha)*real_xyz[1]
            z_tmp = alpha * z + (1-alpha)*real_xyz[2]
            
            joint_poses = p.calculateInverseKinematics(bodyUniqueId=self.robot_id, endEffectorLinkIndex=eef_id, 
                                                       targetPosition=[x_tmp, y_tmp, z_tmp], targetOrientation=orn, 
                                                       maxNumIterations=200)

            # PyBullet IK returns values for all movable joints in the model.
            # Extract only the 6 arm joints for the active arm.
            arm_positions = []
            for name in arm_joint_names:
                jid = self.joints[name].id
                idx = self.joint_id_to_movable_index[jid]
                arm_positions.append(joint_poses[idx])

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
                    positionGain=self.ARM_POSITION_GAIN,
                    velocityGain=self.ARM_VELOCITY_GAIN,
                )

            self.step_simulation()
            if try_close_gripper and still_open_flag_ and not self.gripper_contact():
                still_open_flag_ = self.close_gripper(check_contact=True)

            # Check if contact with objects
            if check_collision_config and self.gripper_contact(**check_collision_config):
                if self.debug:
                    print('Collision detected!', self.check_grasped_id())
                return False, p.getLinkState(self.robot_id, eef_id)[0:2]

            # Check xyz and rpy error
            real_xyz, real_xyzw = p.getLinkState(self.robot_id, eef_id)[0:2]
            roll, pitch, yaw = p.getEulerFromQuaternion(orn)
            real_roll, real_pitch, real_yaw = p.getEulerFromQuaternion(real_xyzw)
            if np.linalg.norm(np.array((x, y, z)) - real_xyz) < 0.001 \
                    and np.abs((roll - real_roll, pitch - real_pitch, yaw - real_yaw)).sum() < 0.001:
                if verbose:
                    print('Reach target with', _, 'steps')
                return True, (real_xyz, real_xyzw)

        # raise FailToReachTargetError
        if self.debug:
            print('Failed to reach the target')
        return False, p.getLinkState(self.robot_id, eef_id)[0:2]

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
        for _ in range(100):
            self.step_simulation()
        self.move_ee([x, y, self.GRIPPER_MOVING_HEIGHT, orn])

        # If the object has been grasped and lifted off the table
        grasped_id = self.check_grasped_id()
        if len(grasped_id) == 1:
            succes_grasp = True
            grasped_obj_id = grasped_id[0]
        else:
            return succes_target, succes_grasp

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

    

