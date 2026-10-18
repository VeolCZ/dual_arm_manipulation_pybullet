import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from datetime import datetime
import sys
import pickle

from network.hardware.device import get_device
#from network.inference.post_process import post_process_output
from network.utils.data.camera_data import CameraData
from network.utils.visualisation.plot import plot_results
from network.utils.dataset_processing.grasp import detect_grasps
from skimage.filters import gaussian
import os
import cv2

class GraspGenerator:
    IMG_WIDTH = 224
    IMG_ROTATION = -np.pi * 0.5
    CAM_ROTATION = 0
    PIX_CONVERSION = 277
    DIST_BACKGROUND = 1.115
    MAX_GRASP = 0.085

    def __init__(self, net_path, camera, depth_radius, fig, IMG_WIDTH=None, network='GR_ConvNet', device='cpu'):
        if network not in ('GR_ConvNet', 'GGCNN'):
            raise ValueError(f'Unsupported grasp network: {network!r}')
        if IMG_WIDTH is None:
            IMG_WIDTH = 300 if network == 'GGCNN' else 224

        # Full-model checkpoints reference inference.* (GR-ConvNet) or
        # models.* (GGCNN), so their module root must be importable.
        try:
            net_dir = os.path.dirname(os.path.abspath(str(net_path)))
            model_root = os.path.abspath(os.path.join(net_dir, os.pardir))
            if any(os.path.isdir(os.path.join(model_root, name)) for name in ("inference", "models")) and model_root not in sys.path:
                sys.path.append(model_root)
        except Exception:
            pass

        device_s = str(device).strip().lower() if device is not None else "cpu"
        want_cuda = device_s not in ("cpu", "none", "false", "0")
        have_cuda = bool(torch.cuda.is_available())

        if want_cuda and not have_cuda:
            print("CUDA requested but torch.cuda.is_available() is False; falling back to CPU.")
        force_cpu = (not want_cuda) or (not have_cuda)
        self.device = get_device(force_cpu=force_cpu)

        # PyTorch 2.6+ defaults `weights_only=True`, which breaks loading full model objects.
        # These checkpoints were produced via torch.save(model), so we load with weights_only=False.
        try:
            self.net = torch.load(net_path, map_location=self.device, weights_only=False)
        except TypeError:
            # Older torch without weights_only arg.
            self.net = torch.load(net_path, map_location=self.device)
        except pickle.UnpicklingError:
            # As a last resort, retry without weights_only if caller is on a torch version that changed defaults.
            self.net = torch.load(net_path, map_location=self.device, weights_only=False)

        try:
            self.net.to(self.device)
        except Exception:
            pass
        try:
            self.net.eval()
        except Exception:
            pass

        # print (self.net)

        
        self.near = camera.near
        self.far = camera.far
        self.depth_r = depth_radius
        
        self.fig = fig
        self.network = network

        self.PIX_CONVERSION = 277 * IMG_WIDTH/224

        self.IMG_WIDTH = IMG_WIDTH
        print (self.IMG_WIDTH)

        # Get rotation matrix
        img_center = self.IMG_WIDTH / 2 - 0.5
        self.img_to_cam = self.get_transform_matrix(-img_center/self.PIX_CONVERSION,
                                                    img_center/self.PIX_CONVERSION,
                                                    0,
                                                    self.IMG_ROTATION)
        self.cam_to_robot_base = self.get_transform_matrix(
            camera.x, camera.y, camera.z, self.CAM_ROTATION)

    def get_transform_matrix(self, x, y, z, rot):
        return np.array([
                        [np.cos(rot),   -np.sin(rot),   0,  x],
                        [np.sin(rot),   np.cos(rot),    0,  y],
                        [0,             0,              1,  z],
                        [0,             0,              0,  1]
                        ])

    def grasp_to_robot_frame(self, grasp, depth_img):
        """
        return: x, y, z, yaw, opening length gripper, object height
        """
        # Get x, y, z of center pixel
        x_p, y_p = grasp.center[0], grasp.center[1]

        # Get area of depth values around center pixel
        x_min = np.clip(x_p-self.depth_r, 0, self.IMG_WIDTH)
        x_max = np.clip(x_p+self.depth_r, 0, self.IMG_WIDTH)
        y_min = np.clip(y_p-self.depth_r, 0, self.IMG_WIDTH)
        y_max = np.clip(y_p+self.depth_r, 0, self.IMG_WIDTH)
        depth_values = depth_img[x_min:x_max, y_min:y_max]

        # Get minimum depth value from selected area
        z_p = np.amin(depth_values)

        # Convert pixels to meters
        x_p /= self.PIX_CONVERSION
        y_p /= self.PIX_CONVERSION
        z_p = self.far * self.near / (self.far - (self.far - self.near) * z_p)

        # Convert image space to camera's 3D space
        img_xyz = np.array([x_p, y_p, -z_p, 1])
        cam_space = np.matmul(self.img_to_cam, img_xyz)

        # Convert camera's 3D space to robot frame of reference
        robot_frame_ref = np.matmul(self.cam_to_robot_base, cam_space)

        # Change direction of the angle and rotate by alpha rad
        yaw = grasp.angle * -1 + (self.IMG_ROTATION)
        if yaw < -np.pi / 2:
            yaw += np.pi

        # Covert pixel width to gripper width
        opening_length = (grasp.length / int(self.MAX_GRASP *
                          self.PIX_CONVERSION)) * self.MAX_GRASP

        obj_height = self.DIST_BACKGROUND - z_p

        # return x, y, z, yaw, opening length gripper
        return robot_frame_ref[0], robot_frame_ref[1], robot_frame_ref[2], yaw, opening_length, obj_height

    def post_process_output(self, q_img, cos_img, sin_img, width_img, pixels_max_grasp,
                            *, quality_sigma=1.0, angle_sigma=1.0):
        """
        Post-process the raw output of the network, convert to numpy arrays, apply filtering.
        :param q_img: Q output of network (as torch Tensors)
        :param cos_img: cos output of network
        :param sin_img: sin output of network
        :param width_img: Width output of network
        :return: Filtered Q output, Filtered Angle output, Filtered Width output
        """
        q_img = q_img.cpu().numpy().squeeze()
        ang_img = (torch.atan2(sin_img, cos_img) / 2.0).cpu().numpy().squeeze()
        width_img = width_img.cpu().numpy().squeeze() * pixels_max_grasp

        q_img = gaussian(q_img, quality_sigma, preserve_range=True)
        ang_img = gaussian(ang_img, angle_sigma, preserve_range=True)
        width_img = gaussian(width_img, 1.0, preserve_range=True)

        return q_img, ang_img, width_img

    def predict(self, rgb, depth, n_grasps=1, show_output=False):
        depth = np.asarray(depth, dtype=np.float32)
        if depth.size == 0 or not np.isfinite(depth).all():
            return [], None
        expected_shape = (self.IMG_WIDTH, self.IMG_WIDTH)
        if depth.shape != expected_shape:
            raise ValueError(f'Expected depth image {expected_shape}, got {depth.shape}')

        # Both networks need metric depth rather than PyBullet's nonlinear buffer.
        depth_for_net = depth
        if 0.0 <= float(depth.min()) and float(depth.max()) <= 1.0:
            d = np.clip(depth, 1e-6, 1.0 - 1e-6)
            depth_for_net = (self.far * self.near) / (self.far - (self.far - self.near) * d)

        max_val = float(np.max(depth_for_net))
        if not np.isfinite(max_val) or max_val <= 1e-9:
            return [], None

        if self.network == 'GR_ConvNet':
            # Preserve the existing RGB-D preprocessing for this checkpoint.
            depth_for_net = depth_for_net * (255.0 / max_val)
            depth_for_net = np.clip((depth_for_net - float(np.mean(depth_for_net))) / 175.0, -1.0, 1.0)
            depth_for_plot = np.expand_dims(depth_for_net, axis=2)
            img_data = CameraData(width=self.IMG_WIDTH, height=self.IMG_WIDTH)
            x, _, _ = img_data.get_data(rgb=rgb, depth=depth_for_plot)
            
        # elif self.network == 'GGCNN':

          
        else:
            raise ValueError(f'Unsupported grasp network: {self.network!r}')

        with torch.no_grad():
            xc = x.to(self.device)
            if self.network == 'GR_ConvNet':
                pred = self.net.predict(xc)
                pixels_max_grasp = int(self.MAX_GRASP * self.PIX_CONVERSION)
                q_img, ang_img, width_img = self.post_process_output(
                    pred['pos'], pred['cos'], pred['sin'], pred['width'], pixels_max_grasp
                )

            # elif self.network == 'GGCNN':
               

            else:
                raise ValueError(f'Unsupported grasp network: {self.network!r}')

        if any(img.shape != expected_shape for img in (q_img, ang_img, width_img)):
            raise ValueError(
                f'{self.network} output maps must match depth image {expected_shape}; '
                f'got {[img.shape for img in (q_img, ang_img, width_img)]}. '
                'Use a 300x300 camera/input for the bundled GGCNN checkpoint.'
            )

        save_name = None
        if show_output:
            im_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            plot = plot_results(self.fig,
                                rgb_img=im_bgr,
                                grasp_q_img=q_img,
                                grasp_angle_img=ang_img,
                                depth_img=depth_for_plot,
                                no_grasps=n_grasps,
                                grasp_width_img=width_img)
            os.makedirs('network_output', exist_ok=True)
            stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
            save_name = f'network_output/{stamp}'
            plot.savefig(save_name + '.png')
            plot.clf()

        grasps = detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=n_grasps)
        return grasps, save_name

    def predict_grasp(self, rgb, depth, n_grasps=1, show_output=False):
        predictions, save_name = self.predict(rgb, depth, n_grasps=n_grasps, show_output=show_output)
        grasps = []
        for grasp in predictions:
            x, y, z, roll, opening_len, obj_height = self.grasp_to_robot_frame(grasp, depth)
            grasps.append((x, y, z, roll, opening_len, obj_height))

        return grasps, save_name
