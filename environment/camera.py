import os
from datetime import datetime

import numpy as np
import pybullet as p


def _rgb_with_brightness(rgba, brightness):
    """Scale rendered RGB without modifying the source alpha, depth, or labels."""
    rgb = rgba[:, :, 0:3]
    if brightness == 1.0:
        return rgb
    return np.clip(rgb.astype(np.float32) * brightness, 0, 255).astype(np.uint8)


class Camera:
    def __init__(self, cam_pos, cam_target, near, far, size, fov, up_vector=(0, 1, 0)):
        self.x, self.y, self.z = cam_pos
        self.x_t, self.y_t, self.z_t = cam_target
        self.width, self.height = size
        self.near, self.far = near, far
        self.fov = fov
        self.up_vector = tuple(up_vector)

        aspect = self.width / self.height
        self.projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
        self.view_matrix = p.computeViewMatrix(cam_pos, cam_target, self.up_vector)

        self.rec_id = None
        # Optional default renderer for getCameraImage (set by Environment).
        self.default_renderer = None
        self.render_options = {}
        self.rgb_brightness = 1.0

    def get_cam_img(
        self,
        return_depth: bool = True,
        return_seg: bool = True,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    ):
        """Render RGB, depth, seg from a fixed look-at camera."""
        flags = None
        if not bool(return_seg):
            flags = getattr(p, "ER_NO_SEGMENTATION_MASK", None)

        if renderer is None:
            renderer = getattr(self, "default_renderer", None)
        renderer_kw = dict(self.render_options)
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

        rgb_out = _rgb_with_brightness(rgb, self.rgb_brightness)
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

    Useful for head/wrist-mounted sensors. Each capture call computes the
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
        self.render_options = {}
        self.rgb_brightness = 1.0

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
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    ):
        """Render RGB, depth, seg from a link."""
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

        renderer_kw = dict(self.render_options)
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

        rgb_out = _rgb_with_brightness(rgb, self.rgb_brightness)
        depth_out = depth if bool(return_depth) else None
        seg_out = seg if bool(return_seg) else None
        return rgb_out, depth_out, seg_out
