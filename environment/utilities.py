import pybullet as p
from collections import namedtuple
from attrdict import AttrDict
import functools
import os
from datetime import datetime


def setup_sisbot(p, robotID, gripper_type):
    controlJoints = ["shoulder_pan_joint", "shoulder_lift_joint",
                     "elbow_joint", "wrist_1_joint",
                     "wrist_2_joint", "wrist_3_joint",
                     "finger_joint"]
    jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
    numJoints = p.getNumJoints(robotID)
    jointInfo = namedtuple("jointInfo",
                           ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity",
                            "controllable"])
    joints = AttrDict()
    for i in range(numJoints):
        info = p.getJointInfo(robotID, i)
        jointID = info[0]
        jointName = info[1].decode("utf-8")
        jointType = jointTypeList[info[2]]
        jointLowerLimit = info[8]
        jointUpperLimit = info[9]
        jointMaxForce = info[10]
        jointMaxVelocity = info[11]
        controllable = True if jointName in controlJoints else False
        info = jointInfo(jointID, jointName, jointType, jointLowerLimit,
                         jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
        if info.type == "REVOLUTE":  # set revolute joint to static
            p.setJointMotorControl2(
                robotID, info.id, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        joints[info.name] = info

    # explicitly deal with mimic joints
    def controlGripper(robotID, parent, children, mul, **kwargs):
        controlMode = kwargs.pop("controlMode")
        if controlMode == p.POSITION_CONTROL:
            pose = kwargs.pop("targetPosition")
            # move parent joint
            p.setJointMotorControl2(robotID, parent.id, controlMode, targetPosition=pose,
                                    force=parent.maxForce, maxVelocity=parent.maxVelocity)
            # move child joints
            for name in children:
                child = children[name]
                childPose = pose * mul[child.name]
                p.setJointMotorControl2(robotID, child.id, controlMode, targetPosition=childPose,
                                        force=child.maxForce, maxVelocity=child.maxVelocity)
        else:
            raise NotImplementedError(
                "controlGripper does not support \"{}\" control mode".format(controlMode))
        # check if there
        if len(kwargs) != 0:
            raise KeyError("No keys {} in controlGripper".format(
                ", ".join(kwargs.keys())))

    assert gripper_type in ['85', '140']
    mimicParentName = "finger_joint"
    if gripper_type == '85':
        mimicChildren = {"right_outer_knuckle_joint": 1,
                         "left_inner_knuckle_joint": 1,
                         "right_inner_knuckle_joint": 1,
                         "left_inner_finger_joint": -1,
                         "right_inner_finger_joint": -1}
    else:
        mimicChildren = {
            "right_outer_knuckle_joint": -1,
            "left_inner_knuckle_joint": -1,
            "right_inner_knuckle_joint": -1,
            "left_inner_finger_joint": 1,
            "right_inner_finger_joint": 1}
    parent = joints[mimicParentName]
    children = AttrDict((j, joints[j])
                        for j in joints if j in mimicChildren.keys())
    controlRobotiqC2 = functools.partial(
        controlGripper, robotID, parent, children, mimicChildren)

    return joints, controlRobotiqC2, controlJoints, mimicParentName


def setup_dual_arm_sisbot(p, robotID, gripper_type: str = "140"):
    """Setup helper for the dual-arm URDF.

    Returns:
        joints: AttrDict joint_name -> jointInfo
        left_arm_joints: list[str] (6 UR joints)
        right_arm_joints: list[str] (6 UR joints)
        control_lgripper: callable(target_position: float)
        control_rgripper: callable(target_position: float)
        joint_id_to_movable_index: dict[int,int] mapping PyBullet joint id -> IK solution index
        link_name_to_id: dict[str,int] mapping child link name -> PyBullet joint id (link index)
    """

    if gripper_type not in ("85", "140"):
        raise ValueError("gripper_type must be '85' or '140'")

    jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
    numJoints = p.getNumJoints(robotID)
    jointInfo = namedtuple(
        "jointInfo",
        [
            "id",
            "name",
            "type",
            "lowerLimit",
            "upperLimit",
            "maxForce",
            "maxVelocity",
        ],
    )

    joints = AttrDict()
    joint_id_to_movable_index = {}
    movable_joint_ids = []
    link_name_to_id = {}

    for i in range(numJoints):
        info = p.getJointInfo(robotID, i)
        jointID = info[0]
        jointName = info[1].decode("utf-8")
        jointType = jointTypeList[info[2]]
        ll = float(info[8])
        ul = float(info[9])
        maxF = float(info[10])
        maxV = float(info[11])

        child_link_name = info[12].decode("utf-8")
        link_name_to_id[child_link_name] = jointID

        if info[2] != p.JOINT_FIXED:
            joint_id_to_movable_index[jointID] = len(movable_joint_ids)
            movable_joint_ids.append(jointID)

        ji = jointInfo(jointID, jointName, jointType, ll, ul, maxF, maxV)

        if ji.type == "REVOLUTE":
            p.setJointMotorControl2(robotID, ji.id, p.VELOCITY_CONTROL, targetVelocity=0.0, force=0.0)

        joints[ji.name] = ji

    arm_base = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    left_arm_joints = [f"left_{n}" for n in arm_base]
    right_arm_joints = [f"right_{n}" for n in arm_base]

    def _make_robotiq_mimic_controller(prefix: str):
        parent_name = prefix + "finger_joint"
        if parent_name not in joints:
            return None

        if gripper_type == "85":
            mimic_children = {
                prefix + "right_outer_knuckle_joint": 1,
                prefix + "left_inner_knuckle_joint": 1,
                prefix + "right_inner_knuckle_joint": 1,
                prefix + "left_inner_finger_joint": -1,
                prefix + "right_inner_finger_joint": -1,
            }
        else:
            mimic_children = {
                prefix + "right_outer_knuckle_joint": -1,
                prefix + "left_inner_knuckle_joint": -1,
                prefix + "right_inner_knuckle_joint": -1,
                prefix + "left_inner_finger_joint": 1,
                prefix + "right_inner_finger_joint": 1,
            }

        parent = joints[parent_name]
        children = []
        for cname, mul in mimic_children.items():
            if cname in joints:
                children.append((joints[cname], mul))

        def control(target_position: float):
            target_position = float(target_position)
            p.setJointMotorControl2(
                robotID,
                parent.id,
                p.POSITION_CONTROL,
                targetPosition=target_position,
                force=float(parent.maxForce),
                maxVelocity=float(parent.maxVelocity),
            )
            for child, mul in children:
                p.setJointMotorControl2(
                    robotID,
                    child.id,
                    p.POSITION_CONTROL,
                    targetPosition=float(target_position * mul),
                    force=float(child.maxForce),
                    maxVelocity=float(child.maxVelocity),
                )

        return control

    control_lgripper = _make_robotiq_mimic_controller("lgripper_")
    control_rgripper = _make_robotiq_mimic_controller("rgripper_")

    return (
        joints,
        left_arm_joints,
        right_arm_joints,
        control_lgripper,
        control_rgripper,
        joint_id_to_movable_index,
        link_name_to_id,
    )


def setup_sisbot_force(p, robotID, gripper_type):
    controlJoints = ["shoulder_pan_joint", "shoulder_lift_joint",
                     "elbow_joint", "wrist_1_joint",
                     "wrist_2_joint", "wrist_3_joint",
                     "finger_joint"]
    jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
    numJoints = p.getNumJoints(robotID)
    jointInfo = namedtuple("jointInfo",
                           ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity",
                            "controllable", "jointAxis", "parentFramePos", "parentFrameOrn"])
    joints = AttrDict()
    for i in range(numJoints):
        info = p.getJointInfo(robotID, i)
        jointID = info[0]
        jointName = info[1].decode("utf-8")
        jointType = jointTypeList[info[2]]
        jointLowerLimit = info[8]
        jointUpperLimit = info[9]
        jointMaxForce = info[10]
        jointMaxVelocity = info[11]
        jointAxis = info[13]
        parentFramePos = info[14]
        parentFrameOrn = info[15]
        controllable = True if jointName in controlJoints else False
        info = jointInfo(jointID, jointName, jointType, jointLowerLimit,
                         jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable,
                         jointAxis, parentFramePos, parentFrameOrn)
        if info.type == "REVOLUTE":  # set revolute joint to static
            p.setJointMotorControl2(
                robotID, info.id, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        joints[info.name] = info
    for j in joints:
        print(joints[j])
    # explicitly deal with mimic joints

    def controlGripper(robotID, parent, children, mul, **kwargs):
        controlMode = kwargs.pop("controlMode")
        if controlMode == p.POSITION_CONTROL:
            pose = kwargs.pop("targetPosition")
            # move parent joint
            p.setJointMotorControl2(robotID, parent.id, controlMode, targetPosition=pose,
                                    force=parent.maxForce, maxVelocity=parent.maxVelocity)
            # p.setJointMotorControl2(robotID, parent.id, p.TORQUE_CONTROL,
            #                         force=10, maxVelocity=parent.maxVelocity)
            return
            # move child joints
            for name in children:
                child = children[name]
                childPose = pose * mul[child.name]
                p.setJointMotorControl2(robotID, child.id, controlMode, targetPosition=childPose,
                                        force=child.maxForce, maxVelocity=child.maxVelocity)
        else:
            raise NotImplementedError(
                "controlGripper does not support \"{}\" control mode".format(controlMode))
        # check if there
        if len(kwargs) != 0:
            raise KeyError("No keys {} in controlGripper".format(
                ", ".join(kwargs.keys())))

    assert gripper_type in ['85', '140']
    mimicParentName = "finger_joint"
    if gripper_type == '85':
        mimicChildren = {"right_outer_knuckle_joint": 1,
                         "left_inner_knuckle_joint": 1,
                         "right_inner_knuckle_joint": 1,
                         "left_inner_finger_joint": -1,
                         "right_inner_finger_joint": -1}
    else:
        mimicChildren = {
            "right_outer_knuckle_joint": -1,
            "left_inner_knuckle_joint": -1,
            "right_inner_knuckle_joint": -1,
            "left_inner_finger_joint": 1,
            "right_inner_finger_joint": 1}
    parent = joints[mimicParentName]
    children = AttrDict((j, joints[j])
                        for j in joints if j in mimicChildren.keys())
    # Create all the gear constraint
    for name in children:
        child = children[name]
        c = p.createConstraint(robotID, parent.id, robotID, child.id, p.JOINT_GEAR, child.jointAxis,
                               # child.parentFramePos, (0, 0, 0), child.parentFrameOrn, (0, 0, 0))
                               (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))
        p.changeConstraint(c, gearRatio=-mimicChildren[name], maxForce=10000)
    controlRobotiqC2 = functools.partial(
        controlGripper, robotID, parent, children, mimicChildren)

    return joints, controlRobotiqC2, controlJoints, mimicParentName


class Camera:
    def __init__(self, cam_pos, cam_target, near, far, size, fov):
        self.x, self.y, self.z = cam_pos
        self.x_t, self.y_t, self.z_t = cam_target
        self.width, self.height = size
        self.near, self.far = near, far
        self.fov = fov

        aspect = self.width / self.height
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov, aspect, near, far)
        self.view_matrix = p.computeViewMatrix(cam_pos, cam_target, [0, 1, 0])

        self.rec_id = None
        # Optional default renderer for getCameraImage (set by Environment).
        self.default_renderer = None

    def get_cam_img(self, return_depth: bool = True, return_seg: bool = True, renderer=None):
        """
        Method to get images from camera
        return:
        rgb
        depth
        segmentation mask
        """
        flags = None
        if not bool(return_seg):
            flags = getattr(p, "ER_NO_SEGMENTATION_MASK", None)

        if renderer is None:
            renderer = getattr(self, "default_renderer", None)
        renderer_kw = {}
        if renderer is not None:
            renderer_kw["renderer"] = renderer

        if flags is None:
            _w, _h, rgb, depth, seg = p.getCameraImage(
                self.width,
                self.height,
                self.view_matrix,
                self.projection_matrix,
                **renderer_kw,
            )
        else:
            _w, _h, rgb, depth, seg = p.getCameraImage(
                self.width,
                self.height,
                self.view_matrix,
                self.projection_matrix,
                flags=flags,
                **renderer_kw,
            )

        rgb_out = rgb[:, :, 0:3]
        depth_out = depth if bool(return_depth) else None
        seg_out = seg if bool(return_seg) else None
        return rgb_out, depth_out, seg_out

    def start_recording(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        file = f'{save_dir}/{now}.mp4'

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self.rec_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, file)

    def stop_recording(self):
        p.stopStateLogging(self.rec_id)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)


class LinkCamera:
    """A camera whose pose is driven by a link in the simulation.

    This is useful for head-mounted sensors. Each capture call computes the
    view matrix from the current link transform.
    """

    def __init__(
        self,
        near: float,
        far: float,
        size: tuple[int, int],
        fov: float,
        parent_to_camera_xyz=(0.0, 0.0, 0.0),
        parent_to_camera_rpy=(0.0, 0.0, 0.0),
        forward_axis=(1.0, 0.0, 0.0),
        up_axis=(0.0, 0.0, 1.0),
        eye_offset: float = 0.10,
    ):
        self.width, self.height = size
        self.near, self.far = float(near), float(far)
        self.fov = float(fov)

        aspect = self.width / self.height
        self.projection_matrix = p.computeProjectionMatrixFOV(self.fov, aspect, self.near, self.far)

        self.parent_to_camera_xyz = tuple(float(v) for v in parent_to_camera_xyz)
        self.parent_to_camera_rpy = tuple(float(v) for v in parent_to_camera_rpy)
        self.parent_to_camera_quat = p.getQuaternionFromEuler(self.parent_to_camera_rpy)

        self.forward_axis = tuple(float(v) for v in forward_axis)
        self.up_axis = tuple(float(v) for v in up_axis)
        self.eye_offset = float(eye_offset)
        # Optional default renderer for getCameraImage (set by Environment).
        self.default_renderer = None

    @staticmethod
    def _rotate_vec(quat_xyzw, v):
        """Rotate a 3D vector by quaternion (x,y,z,w)."""
        m = p.getMatrixFromQuaternion(quat_xyzw)
        # m is row-major 3x3
        x = m[0] * float(v[0]) + m[1] * float(v[1]) + m[2] * float(v[2])
        y = m[3] * float(v[0]) + m[4] * float(v[1]) + m[5] * float(v[2])
        z = m[6] * float(v[0]) + m[7] * float(v[1]) + m[8] * float(v[2])
        return (x, y, z)

    def get_cam_img(
        self,
        robot_id: int,
        link_index: int,
        target_world=None,
        return_depth: bool = True,
        return_seg: bool = True,
        renderer=None,
    ):
        """Render RGB, depth, seg from a link.

        Args:
            robot_id: pybullet bodyUniqueId
            link_index: pybullet link index (joint index for the child link)
            target_world: Optional (x,y,z) in world frame. If provided, the camera
                will look at this point. Otherwise it will look along its
                configured forward axis.
        """
        # IMPORTANT: Use the link *frame* pose, not the COM pose.
        # PyBullet's getLinkState returns:
        #   [0:2] = COM position/orientation
        #   [4:6] = link frame position/orientation (URDF link frame)
        # Using COM here often causes a visible offset between the rendered camera
        # pose and the expected URDF frame pose.
        ls = p.getLinkState(robot_id, link_index, computeForwardKinematics=True)
        if len(ls) >= 6:
            link_pos, link_orn = ls[4], ls[5]
        else:
            link_pos, link_orn = ls[0], ls[1]

        cam_pos, cam_orn = p.multiplyTransforms(
            link_pos,
            link_orn,
            self.parent_to_camera_xyz,
            self.parent_to_camera_quat,
        )

        up_world = self._rotate_vec(cam_orn, self.up_axis)

        # Optional eye offset to push the camera out of its own housing/geometry.
        # This is especially useful for wrist-mounted cameras where the link frame
        # is at the center of the CAD mesh.
        forward_world_for_offset = self._rotate_vec(cam_orn, self.forward_axis)
        if self.eye_offset != 0.0:
            cam_pos = (
                cam_pos[0] + self.eye_offset * forward_world_for_offset[0],
                cam_pos[1] + self.eye_offset * forward_world_for_offset[1],
                cam_pos[2] + self.eye_offset * forward_world_for_offset[2],
            )

        if target_world is None:
            forward_world = forward_world_for_offset
            target = [
                cam_pos[0] + forward_world[0],
                cam_pos[1] + forward_world[1],
                cam_pos[2] + forward_world[2],
            ]
        else:
            target = [float(target_world[0]), float(target_world[1]), float(target_world[2])]
        view_matrix = p.computeViewMatrix(cam_pos, target, up_world)

        if renderer is None:
            renderer = getattr(self, "default_renderer", None)

        flags = None
        if not bool(return_seg):
            flags = getattr(p, "ER_NO_SEGMENTATION_MASK", None)

        renderer_kw = {}
        if renderer is not None:
            renderer_kw["renderer"] = renderer

        if flags is None:
            _w, _h, rgb, depth, seg = p.getCameraImage(
                self.width,
                self.height,
                view_matrix,
                self.projection_matrix,
                **renderer_kw,
            )
        else:
            _w, _h, rgb, depth, seg = p.getCameraImage(
                self.width,
                self.height,
                view_matrix,
                self.projection_matrix,
                flags=flags,
                **renderer_kw,
            )
        rgb_out = rgb[:, :, 0:3]
        depth_out = depth if bool(return_depth) else None
        seg_out = seg if bool(return_seg) else None
        return rgb_out, depth_out, seg_out
