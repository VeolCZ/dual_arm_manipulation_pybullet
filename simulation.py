from grasp_generator import GraspGenerator
from environment.utilities import Camera
from environment.env import Environment
from utils import YcbObjects, PackPileData, IsolatedObjData, summarize
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

# env.profile_camera_capture(iters=20, renderer=p.ER_BULLET_HARDWARE_OPENGL)
# env.profile_camera_capture(iters=20, renderer=p.ER_TINY_RENDERER)

class GrasppingScenarios():

    def __init__(self,network_model="GGCNN"):
        
        self.network_model = network_model

        if (network_model == "GR_ConvNet"):
            ##### GR-ConvNet #####
            self.IMG_SIZE = 224
            self.network_path = 'trained_models/GR_ConvNet/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98'
            sys.path.append('trained_models/GR_ConvNet')
        else:
            # you need to add your network here!
            print("The selected network has not been implemented yet!")
            exit() 
        
        
        self.CAM_Z = 1.9
        self.depth_radius = 1
        self.ATTEMPTS = 3

        # Camera streaming defaults (used by all scenarios).
        self.stream_show_depth = False
        self.stream_max_fps = 5.0
        self.stream_pause_s = 0.0
        self.stream_use_blit = True

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
        """Create an Environment and (optionally) enable camera streaming.

        Centralizes the common setup used by isolated/packed/pile scenarios.
        """
        env = Environment(
            camera,
            vis=vis,
            debug=debug,
            finger_length=finger_length,
            hide_gui_panels=hide_gui_panels,
        )

        if stream_cameras:
            try:
                kwargs = dict(
                    enabled=True,
                    show_depth=bool(self.stream_show_depth),
                    max_fps=float(self.stream_max_fps),
                    pause_s=float(self.stream_pause_s),
                    use_blit=bool(self.stream_use_blit),
                )
                if stream_which is not None:
                    kwargs["which"] = stream_which
                env.enable_camera_streaming(**kwargs)
            except Exception:
                pass

            # Disable expensive debug drawing (axes + rays) for speed.
            try:
                env.enable_camera_frame_debug(enabled=False)
            except Exception:
                pass

        return env
       
                
    def draw_predicted_top_down_grasp(self, grasps, color=(0, 0, 1), lineIDs=None):
        x, y, z, yaw, opening_len, obj_height = grasps

        if lineIDs is None:
            lineIDs = []

        gripper_size = opening_len + 0.02 
        finger_size = 0.075
        # lineIDs = []
        lineIDs.append(p.addUserDebugLine([x, y, z], [x, y, z+0.15],color, lineWidth=6))

        lineIDs.append(p.addUserDebugLine([x - gripper_size*math.sin(yaw), y - gripper_size*math.cos(yaw), z], 
                                    [x + gripper_size*math.sin(yaw), y + gripper_size*math.cos(yaw), z], 
                                    color, lineWidth=6))

        lineIDs.append(p.addUserDebugLine([x - gripper_size*math.sin(yaw), y - gripper_size*math.cos(yaw), z], 
                                    [x - gripper_size*math.sin(yaw), y - gripper_size*math.cos(yaw), z-finger_size], 
                                    color, lineWidth=6))
        lineIDs.append(p.addUserDebugLine([x + gripper_size*math.sin(yaw), y + gripper_size*math.cos(yaw), z], 
                                    [x + gripper_size*math.sin(yaw), y + gripper_size*math.cos(yaw), z-finger_size], 
                                    color, lineWidth=6))
        
        return lineIDs

    def _visualize_cameras(
        self,
        env: Environment,
        bgr,
        vis: bool,
        viewer_warned: bool,
        show_cameras: bool = True,
    ) -> bool:
        """Show camera mosaics when visualization is enabled.

        Uses Matplotlib first (more robust in headless-ish setups), then OpenCV.
        Returns the updated `viewer_warned` flag.
        """
        if (not show_cameras) or (not vis) or getattr(env, "auto_stream_cameras", False):
            return viewer_warned

        ok = False
        try:
            ok = env.visualize_cameras_matplotlib(
                show_depth=False,
                window_title="cameras",
                visible=True,
                pause_s=0.001,
                max_fps=10.0,
                renderer=getattr(env, "camera_renderer", None),
            )
        except Exception as exc:
            if not viewer_warned:
                viewer_warned = True
                print(f"[viewer] Matplotlib viewer failed: {exc}")

        if not ok:
            try:
                ok = env.visualize_cameras(
                    show_depth=False,
                    window_name="cameras",
                    wait_ms=1,
                    renderer=getattr(env, "camera_renderer", None),
                )
            except Exception as exc:
                if not viewer_warned:
                    viewer_warned = True
                    print(f"[viewer] OpenCV mosaic viewer failed: {exc}")

        if not ok:
            try:
                cv2.imshow('rgb', bgr)
                cv2.waitKey(1)
            except Exception as exc:
                if not viewer_warned:
                    viewer_warned = True
                    print(f"[viewer] OpenCV imshow failed (likely headless): {exc}")

        return viewer_warned
    
    def remove_drawing(self,lineIDs):
        for line in lineIDs:
            p.removeUserDebugItem(line)
    
    def dummy_simulation_steps(self, n, env: Environment | None = None):
        """Advance the physics simulation.

        If an `Environment` is provided, route stepping through `env.step_simulation()`
        so GUI debug visualisations (camera frames, target rays, etc.) stay in sync.
        """
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
    def _scene_has_objects_from_depth(depth: np.ndarray, threshold: float = 0.0025) -> bool:
        """Heuristic: if the depth image has non-trivial variation, assume objects exist."""
        if depth is None:
            return False
        try:
            return bool(float(depth.max() - depth.min()) >= float(threshold))
        except Exception:
            return False

    def is_there_any_object(self, camera, env: Environment | None = None):
        # Route stepping through Environment when available so GUI debug + camera
        # streaming hooks stay in sync.
        self.dummy_simulation_steps(10, env=env)
        if env is not None:
            _, depth, _ = env.get_external_cam_img(return_depth=True, return_seg=False)
        else:
            _, depth, _ = camera.get_cam_img(return_depth=True, return_seg=False)
        # print ("min RGB = ", rgb.min(), "max RGB = ", rgb.max(), "rgb.avg() = ", np.average(rgb))
        # print ("min depth = ", depth.min(), "max depth = ", depth.max())
        return self._scene_has_objects_from_depth(depth)
                    
                    
    def isolated_obj_scenario(self, runs, device, vis, stream_cameras, output, debug):

        print ("visualization = ", vis)
        print ("stream cameras = ", stream_cameras)

        objects = YcbObjects('objects/ycb_objects',
                            mod_orn=['ChipsCan', 'MustardBottle', 'TomatoSoupCan'],
                            mod_stiffness=['Strawberry'])

        
        ## reporting the results at the end of experiments in the results folder
        data = IsolatedObjData(objects.obj_names, runs, 'results')

        ## camera settings: cam_pos, cam_target, near, far, size, fov
        center_x, center_y, center_z = 0.0, -0.54, self.CAM_Z
        camera = Camera((center_x, center_y, center_z), (center_x, center_y, 0.785), 0.2, 2.0, (self.IMG_SIZE, self.IMG_SIZE), 40)
        env = self._make_env(
            camera,
            vis=vis,
            debug=debug,
            finger_length=0.06,
            hide_gui_panels=True,
            stream_cameras=bool(stream_cameras),
            stream_which=("left", "right"),
        )
        
        generator = GraspGenerator(self.network_path, camera, self.depth_radius, self.fig, self.IMG_SIZE, self.network_model, device)

        viewer_warned = False
        bgr2rgb = cv2.cvtColor
        bgr2rgb_code = cv2.COLOR_BGR2RGB
        
        objects.shuffle_objects()
        for i in range(runs):
            print("----------- run ", i+1, " -----------")
            print ("network model = ", self.network_model)
            print ("size of input image (W, H) = (", self.IMG_SIZE," ," ,self.IMG_SIZE, ")")

            for obj_name in objects.obj_names:
                print(obj_name)

                env.reset_robot()          
               
                path, mod_orn, mod_stiffness = objects.get_obj_info(obj_name)
                env.load_isolated_obj(path, mod_orn, mod_stiffness)
                self.dummy_simulation_steps(20, env=env)

                number_of_attempts = self.ATTEMPTS
                number_of_failures = 0
                idx = 0 ## select the best grasp configuration
                failed_grasp_counter = 0
                while number_of_failures < number_of_attempts:
                    # Keep the old behavior (let objects settle) but avoid an extra camera render
                    # in the loop condition.
                    self.dummy_simulation_steps(10, env=env)

                    bgr, depth, _ = env.get_external_cam_img(return_depth=True, return_seg=False)
                    if not self._scene_has_objects_from_depth(depth):
                        break

                    viewer_warned = self._visualize_cameras(
                        env,
                        bgr,
                        vis=vis,
                        viewer_warned=viewer_warned,
                        show_cameras=bool(stream_cameras),
                    )

                    ##convert BGR to RGB
                    rgb = bgr2rgb(bgr, bgr2rgb_code)
                              
                    grasps, save_name = generator.predict_grasp( rgb, depth, n_grasps=number_of_attempts, show_output=output)
                    if not grasps:
                        self.dummy_simulation_steps(50, env=env)
                        #print ("could not find a grasp point!")
                        if failed_grasp_counter > 3:
                            print("Failed to find a grasp points > 3 times. Skipping.")
                            break
                            
                        failed_grasp_counter += 1                 
                        continue

                    #print ("grasps.length = ", len(grasps))
                    if (idx > len(grasps)-1):  
                        print ("idx = ", idx)
                        if len(grasps) > 0 :
                           idx = len(grasps)-1
                        else:
                           number_of_failures += 1
                           continue    

                    if vis:
                        LID =[]
                        for g in grasps:
                            LID = self.draw_predicted_top_down_grasp(g,color=[1,0,1],lineIDs=LID)
                        time.sleep(0.5)
                        self.remove_drawing(LID)
                        self.dummy_simulation_steps(10, env=env)
                        

                    lineIDs = self.draw_predicted_top_down_grasp(grasps[idx])

                    x, y, z, yaw, opening_len, obj_height = grasps[idx]
                    succes_grasp, succes_target = env.grasp((x, y, z), yaw, opening_len, obj_height)

                    data.add_try(obj_name)
                   
                    if succes_grasp:
                        data.add_succes_grasp(obj_name)
                    if succes_target:
                        data.add_succes_target(obj_name)

                    ## remove visualized grasp configuration 
                    if vis:
                        self.remove_drawing(lineIDs)

                    env.reset_robot()
                    
                    if succes_target:
                        number_of_failures = 0
                        if vis:
                            debugID = p.addUserDebugText("success", [-0.0, -0.9, 0.8], [0,0.50,0], textSize=2)
                            time.sleep(0.25)
                            p.removeUserDebugItem(debugID)
                        
                        if save_name is not None:
                            os.rename(save_name + '.png', save_name + f'_SUCCESS_grasp{i}.png')
                        

                    else:
                        number_of_failures += 1
                        idx +=1        
                        #env.reset_robot() 
                        # env.remove_all_obj()                        
                
                        if vis:
                            debugID = p.addUserDebugText("failed", [-0.0, -0.9, 0.8], [0.5,0,0], textSize=2)
                            time.sleep(0.25)
                            p.removeUserDebugItem(debugID)

        data.write_json(self.network_model)
        summarize(data.save_dir, runs, self.network_model)


    def packed_or_pile_scenario(self, runs, scenario, device, vis, stream_cameras, output, debug):
        
        ## reporting the results at the end of experiments in the results folder
        number_of_objects = 5
        if scenario=='packed':
            objects = YcbObjects('objects/ycb_objects',
                            mod_orn=['ChipsCan', 'MustardBottle', 'TomatoSoupCan'],
                            mod_stiffness=['Strawberry'],
                            exclude=['CrackerBox'])
            
            data = PackPileData(number_of_objects, runs, 'results', 'packed')

        elif scenario=='pile':
            objects = YcbObjects('objects/ycb_objects',
                            mod_orn=['ChipsCan', 'MustardBottle', 'TomatoSoupCan'],
                            mod_stiffness=['Strawberry'],
                            exclude=['CrackerBox'])

            data = PackPileData(number_of_objects, runs, 'results', 'pile')


        center_x, center_y, center_z = 0.05, -0.52, self.CAM_Z
        camera = Camera((center_x, center_y, center_z), (center_x, center_y, 0.785), 0.2, 2.0, (self.IMG_SIZE, self.IMG_SIZE), 40)
        env = self._make_env(
            camera,
            vis=vis,
            debug=debug,
            finger_length=0.06,
            hide_gui_panels=True,
            stream_cameras=bool(stream_cameras),
            # Keep previous behavior: in packed/pile, let Environment choose default cameras.
            stream_which=None,
        )
        
        generator = GraspGenerator(self.network_path, camera, self.depth_radius, self.fig, self.IMG_SIZE, self.network_model, device)

        
        viewer_warned = False
        bgr2rgb = cv2.cvtColor
        bgr2rgb_code = cv2.COLOR_BGR2RGB

        for i in range(runs):
            env.remove_all_obj()
            env.reset_robot()              
            print("----------- run ", i+1, " -----------")
            print ("network model = ", self.network_model)
            print ("size of input image (W, H) = (", self.IMG_SIZE," ," ,self.IMG_SIZE, ")")

            
            if vis:
                debugID = p.addUserDebugText(f'Experiment {i+1}', [-0.0, -0.9, 0.8], [0,0,255], textSize=2)
                time.sleep(0.5)
                p.removeUserDebugItem(debugID)

            number_of_failures = 0
            objects.shuffle_objects()

            info = objects.get_n_first_obj_info(number_of_objects)

            if scenario=='packed':
                env.create_packed(info)
            elif scenario=='pile':
                env.create_pile(info)
                
            #self.dummy_simulation_steps(50)

            number_of_failures = 0
            number_of_attempts = self.ATTEMPTS
            failed_grasp_counter = 0
            flag_failed_grasp_counter= False

            while number_of_failures < number_of_attempts:
                # Keep the old behavior (let pile settle) but avoid an extra render in a loop condition.
                self.dummy_simulation_steps(10, env=env)

                try:
                    idx = 0 ## select the best grasp configuration
                    for attempt_idx in range(number_of_attempts):
                        bgr, depth, _ = env.get_external_cam_img(return_depth=True, return_seg=False)
                        if not self._scene_has_objects_from_depth(depth):
                            # No objects left on the table.
                            number_of_failures = number_of_attempts
                            break

                        data.add_try()
                        viewer_warned = self._visualize_cameras(
                            env,
                            bgr,
                            vis=vis,
                            viewer_warned=viewer_warned,
                            show_cameras=bool(stream_cameras),
                        )

                        rgb = bgr2rgb(bgr, bgr2rgb_code)

                        grasps, save_name = generator.predict_grasp( rgb, depth, n_grasps=number_of_attempts, show_output=output)
                        if not grasps:
                                self.dummy_simulation_steps(30, env=env)
                                print ("could not find a grasp point!")    
                                if failed_grasp_counter > 5:
                                    print("Failed to find a grasp points > 3 times. Skipping.")
                                    flag_failed_grasp_counter= True
                                    break

                                failed_grasp_counter += 1 
                                continue
                        
                        if vis:
                            LID =[]
                            for g in grasps:
                                LID = self.draw_predicted_top_down_grasp(g,color=[1,0,1],lineIDs=LID)
                            time.sleep(0.5)
                            self.remove_drawing(LID)
                            self.dummy_simulation_steps(20, env=env)
                            
                        #print ("grasps.length = ", len(grasps))
                        if (idx > len(grasps)-1):  
                            print ("idx = ", idx)
                            if len(grasps) > 0 :
                                idx = len(grasps)-1
                            else:
                                number_of_failures += 1
                                continue  

                        lineIDs = self.draw_predicted_top_down_grasp(grasps[idx])
                        
                        ## perform object grasping and manipulation : 
                        #### succes_grasp means if the grasp was successful, 
                        #### succes_target means if the target object placed in the target basket successfully
                        x, y, z, yaw, opening_len, obj_height = grasps[idx]
                        succes_grasp, succes_target = env.grasp((x, y, z), yaw, opening_len, obj_height)
                        
                        if succes_grasp:
                            data.add_succes_grasp()                     

                        ## remove visualized grasp configuration 
                        if vis:
                            self.remove_drawing(lineIDs)

                        env.reset_robot()
                            
                        if succes_target:
                            data.add_succes_target()
                            number_of_failures = 0

                            if vis:
                                debugID = p.addUserDebugText("success", [-0.0, -0.9, 0.8], [0,0.50,0], textSize=2)
                                time.sleep(0.25)
                                p.removeUserDebugItem(debugID)
                                
                                if save_name is not None:
                                    os.rename(save_name + '.png', save_name + f'_SUCCESS_grasp{attempt_idx}.png')
                                
                        else:
                            #env.reset_robot()   
                            number_of_failures += 1
                            idx +=1     
                                                    
                            if vis:
                                debugID = p.addUserDebugText("failed", [-0.0, -0.9, 0.8], [0.5,0,0], textSize=2)
                                time.sleep(0.25)
                                p.removeUserDebugItem(debugID)

                            if number_of_failures == number_of_attempts:
                                if vis:
                                    debugID = p.addUserDebugText("breaking point",  [-0.0, -0.9, 0.8], [0.5,0,0], textSize=2)
                                    time.sleep(0.25)
                                    p.removeUserDebugItem(debugID)
                                
                                break

                        #env.reset_all_obj()
        
                except Exception as exc:
                    print(f"An exception occurred during the experiment: {exc}")
                    if debug:
                        traceback.print_exc()
                    number_of_failures += 1
                    try:
                        env.reset_robot()
                    except Exception:
                        pass
                    #print ("#objects = ", len(env.obj_ids), "#failed = ", number_of_failures , "#attempts =", number_of_attempts)
        
                if flag_failed_grasp_counter:
                    flag_failed_grasp_counter= False
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

    parser = argparse.ArgumentParser(description='Grasping demo')

    parser.add_argument('--scenario', type=str, default='isolated', help='Grasping scenario (isolated/packed/pile)')
    parser.add_argument('--network', type=str, default='GR_ConvNet', help='Network model (GR_ConvNet/...)')

    parser.add_argument('--runs', type=int, default=1, help='Number of runs the scenario is executed')
    parser.add_argument('--attempts', type=int, default=3, help='Number of attempts in case grasping failed')

    parser.add_argument(
        '--save-network-output',
        dest='output',
        type=str2bool,
        default=False,
        help='Save network output (True/False)'
    )

    parser.add_argument('--device', type=str, default='cpu', help='device (cpu/gpu)')
    parser.add_argument('--vis', type=str2bool, default=True, help='vis (True/False)')
    parser.add_argument('--stream_cameras', type=str2bool, default=True, help='Show/stream camera windows (True/False)')
    parser.add_argument('--report', type=str2bool, default=True, help='report (True/False)')

                        
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    output = args.output
    runs = args.runs
    ATTEMPTS = args.attempts
    device=args.device
    vis=args.vis
    report=args.report
    
    print ('visualization = ', vis)
    
    grasp = GrasppingScenarios(args.network)
    # Make CLI attempts parameter actually affect the scenarios.
    grasp.ATTEMPTS = int(args.attempts)

    if args.scenario == 'isolated':
        grasp.isolated_obj_scenario(
            runs,
            device,
            vis,
            stream_cameras=args.stream_cameras,
            output=output,
            debug=False,
        )
    elif args.scenario == 'packed':
        grasp.packed_or_pile_scenario(
            runs,
            args.scenario,
            device,
            vis,
            stream_cameras=args.stream_cameras,
            output=output,
            debug=False,
        )
    elif args.scenario == 'pile':
        grasp.packed_or_pile_scenario(
            runs,
            args.scenario,
            device,
            vis,
            stream_cameras=args.stream_cameras,
            output=output,
            debug=False,
        )

