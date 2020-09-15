import math
import os, sys
import airsim
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import copy
from datetime import datetime

# add utils_airsim path
sys.path.insert(0, "../eye_tracking/")
from utils_eyetracking import Tobii4C_Cpp

# add joystick agent to record interventions
from utils_airsim import JoystickAgent

mtl_path = os.getcwd().split('airsim')[0]+'mtl/'
sys.path.insert(0, mtl_path)

from common_utils import *

class MultimodalAirSimEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, exp_name='test', n_steps=200, custom_command=None,
                ref_initial_loc=[], get_obj_segmentation=True, agent=None,
                dynamic_target=False, save_image_data=True):
        # env parameters
        self.exp_name = exp_name
        self.n_steps = n_steps
        self.ref_initial_loc = ref_initial_loc
        self.ref_initial_loc_copy = copy.deepcopy(ref_initial_loc)
        self.get_obj_segmentation = get_obj_segmentation
        self.save_image_data = save_image_data
        self.agent = agent
        self.step_cnt = 0  # counts time steps for all experiment
        self.epi_num = -1  # counts number of episodes
        self.default_start_altitude = -3 # meters (negative z-axis)
        self.enable_interventions = False
        self.human_intervening = False

        # observation:
        #   512 resnet features
        #   16 IMU features (not using X and Y from GPS)
        self.observation_space = spaces.Dict({
            "img_rgb": spaces.Box(low=-1, high=1, shape=(240, 352, 3), dtype=np.float32),
            "states_bc": spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)})

        # actions:
        #   4 joystick readings
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(4,), dtype=np.float32)

        # setup log files
        if custom_command is None:
            # default log address
            self.exp_addr = f'airsim_data/{exp_name}/'
        else:
            # create log based on command given
            self.exp_addr = os.path.join(f'airsim_data/{exp_name}/',f'{custom_command.replace(" ", "_")}/')
        os.makedirs(self.exp_addr, exist_ok=True)
        os.makedirs(self.exp_addr+'seg', exist_ok=True)  # for rgb images
        os.makedirs(self.exp_addr+'rgb', exist_ok=True)  # for segmentation images
        os.makedirs(self.exp_addr+'depth', exist_ok=True)  # for depth images
        self.filename = self.exp_addr + 'log.csv'
        self.logfile = open(self.filename, 'w')
        self._label_log()

        self.logger = get_logger('GYM_ENV_LOGGER')

        # setup eye tracking hardware
        self.eye_tracker = Tobii4C_Cpp()

        self.setup_airsim()

        # use a separate buffer to save image data
        if self.save_image_data:
            self.img_buffer = []

        # define list of possible commands or can overwrite default commands
        # custom with argument
        self.custom_command = custom_command
        self.command_list = [
            'INTERCEPT TREE',
            'INTERCEPT VEHICLE',
            'INTERCEPT BUILDING']

        # setup joystick to enable interventions when not recording human demonstrations
        if self.enable_interventions:
            self.joystick = JoystickAgent()

        # player start (drone) default location
        self.dynamic_target = dynamic_target
        self.player_start_loc = {
            'pos_x': -3.57840824127197,
            'pos_y': 0.510065853595734,
            'pos_z': -1.862957239151}

    def setup_airsim(self):
        
        # setup airsim
        self.client = airsim.MultirotorClient()        
        self.client.confirmConnection()
        self.client.enableApiControl(True)

    def step(self, actions):
        # check if actions also includes gaze predictions
        if type(actions) in [tuple, list]:
            # action contains gaze and actions
            actions_joystick = actions[1].cpu().detach().numpy()[0]
            actions_gaze = actions[0].cpu().detach().numpy()[0]
        else:
            # actions are only actions (no gaze)
            actions_joystick = actions
            actions_gaze = None

        # propagate joystick commands
        self._vel_cmd(actions_joystick)

        # log data with current actions and observation
        self._write_log(actions_joystick, actions_gaze)

        # read new observation
        obs = self._read_obs()

        # compute reward
        reward = 0.

        # check if the episode is complete
        done = False
        info = {}

        # update step counter
        self.step_cnt += 1
        self.step_num += 1

        return obs, reward, done, info

    def _read_obs(self):
        # read eye gaze (x, y coordinates, normalized by image height and width)
        self.gaze_data = self.eye_tracker.read_gaze()

        # read IMU data
        self.state = self.client.getMultirotorState()
        (pitch, roll, yaw) = self._toEulerianAngle(self.state.kinematics_estimated.orientation)

        # check collision status
        self.vehicle_collided = self.client.simGetCollisionInfo().has_collided
        
        # store some states variables to be used later to process the actions
        self.alt = self.state.kinematics_estimated.position.z_val
        self.pitch = pitch
        self.roll = roll
        self.yaw = yaw

        # read cameras and append to buffer
        if self.get_obj_segmentation:
            img_responses = self.client.simGetImages([
                airsim.ImageRequest(0, airsim.ImageType.Scene, False, False),  # RGB
                airsim.ImageRequest(0, airsim.ImageType.Segmentation, False, False),  # Object segmentation
                airsim.ImageRequest(0, airsim.ImageType.DepthVis, True)])  # Depth
        else:
            # no request object segmentation to speed airsim loop
            img_responses = self.client.simGetImages([
                airsim.ImageRequest(0, airsim.ImageType.Scene, False, False),  # RGB
                airsim.ImageRequest(0, airsim.ImageType.DepthVis, True)])  # Depth
        if self.save_image_data:
            self.img_buffer.append((self.step_cnt, img_responses))

        try:
            # postprocess rgb image to be returned to the agent
            response = img_responses[0]
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
            # reshape it
            img_rgb = img1d.reshape(response.height, response.width, 3)
            # convert from bgr to rgb
            img_rgb = np.fliplr(img_rgb.reshape(-1,3)).reshape(img_rgb.shape)
        except:
            # error processing rgb data, most likely got an empty frame
            # solution: use previous frame
            img_rgb = self.prev_obs['img_rgb']

        # postprocess depth image to be returned to the agent 
        # convert to numpy (float)
        if self.get_obj_segmentation:
            response = img_responses[2]
        else:
            response = img_responses[1]
        try:
            depth = np.array(response.image_data_float, dtype=np.float32)
            # reshape it
            depth = depth.reshape(response.height, response.width)
            # scale pixels
            img_depth = np.array(np.abs(1-depth) * 255, dtype=np.uint8)  # reverse black and white   
        except:
            # error processing depth data, most likely got an empty frame
            # solution: use previous frame
            img_depth = self.prev_obs['img_depth']

        # create observation vector to be processed in the agent
        obs = {
            'img_depth': img_depth,
            'img_rgb': img_rgb,
            'states_bc': [self.state, pitch, roll, yaw],
            'gaze_data': self.gaze_data}

        # create copy of the current obs to be used as "previous" before new 
        # observation is read
        self.prev_obs = obs

        return obs

    def reset(self):
        # make sure seed is random
        np.random.seed()

        # sample language command
        if self.custom_command is not None:
            self.command = self.custom_command
        else:
            # random sample from possible list of commands
            self.command = np.random.choice(self.command_list)
        self._print_both(f'>>> COMMAND: {self.command}', severity=3)

        # resets
        self.client.reset()

        # try to handle error during takeoff
        takeoff_ok = False

        while True:
            try:
                # wait for a given amount of seconds while takeoff
                # if takes too long, retry
                self.client.enableApiControl(True)
                self.client.armDisarm(True)
                self.client.takeoffAsync(timeout_sec=5).join()
                state = self.client.getMultirotorState()

                if state.kinematics_estimated.position.z_val < self.default_start_altitude:
                    break
            except:
                self.client.armDisarm(False)
                self.client.enableApiControl(False)



        ## move vehicle to a different start location
        # read current position
        current_state = self.client.getMultirotorState()
        current_x = current_state.kinematics_estimated.position.x_val
        current_y = current_state.kinematics_estimated.position.y_val
        current_z = self.default_start_altitude

        # new location is either the same as a reference experiment or random
        if not self.following_ref_initial_loc:
            self._print_both('[*] Going to a initial RANDOM location...', severity=2)
            # sample new position
            # other side on street?
            if np.random.rand() >= 0.5:
                self.client.rotateToYawAsync(180).join()
                extra_x = 80.
            else:
                extra_x = 0.
            new_x = current_x + 5*(np.random.rand()-0.5) + extra_x
            new_y = current_y + 8*(np.random.rand()-0.5)
            new_z = current_z

            # save starting location for next agent, if needed
            self.ref_initial_loc.append({'x':new_x,'y':new_y,'z':new_z})

        else:
            # follows list of reference initial locations
            self._print_both('[*] Going to a initial REFERENCE location...', severity=2)

            # read reference location
            ref_location = self.ref_initial_loc.pop(0)
            new_x = ref_location['x']
            new_y = ref_location['y']
            new_z = ref_location['z']

            # make sure vehicle is oriented the correct way (other side of street)
            if new_x > 40:
                self.client.rotateToYawAsync(180).join()

        # move vehicle to new position
        self._print_both(f'[*] x: {new_x}, y: {new_y}, z: {new_z}')
        self.client.moveToPositionAsync(new_x, new_y, new_z, 5).join()
        time.sleep(2) # stabilizes vehicle after going to new position

        # read initial observation
        obs = self._read_obs()

        # update time steps and episode counters
        self.epi_num += 1
        self.step_num = 0

        return obs

    def _print_both(self, message, severity=1):
        """Print message to both terminal and simulator."""
        self.logger.info(message)
        self.client.simPrintLogMessage(message, severity=severity)


    def close(self):
        """Closes all devices and threads."""
        self._print_both('** COMPLETED DATA COLLECTION **', severity=3)
        # close logfiles, devices, and connection to airsim
        self.logfile.close()
        self.client.armDisarm(False)
        self.client.reset()
        self.client.enableApiControl(False)
        self.eye_tracker.close()


    def save_images_to_disk(self):
        # move images from buffer to disk
        if self.save_image_data:
            self._print_both('[*] Moving images from buffer to disk...')
            for k in range(len(self.img_buffer)):
                self._save_airsim_images(
                    self.img_buffer[k][1],
                    self.img_buffer[k][0])

        # # clean buffer
        # self.img_buffer = []

    def _vel_cmd(self, actions):
        """Move vehicle by sending high-level velocity commands (vx, vy, vz).
        """
        # send the action (converting to correct format: np.float64)        
        sc = 10 # scales joystick inputs
        duration = 1e-0
        vx = sc/1.5*actions[1]
        vy = sc/1.5*actions[0]
        vz = 10*sc*actions[3]
        ref_alt = self.alt + sc/2*actions[2]

        # translate from inertial to body frame
        C = np.zeros((2,2))
        C[0,0] = np.cos(self.yaw)
        C[0,1] = -np.sin(self.yaw)
        C[1,0] = -C[0,1]
        C[1,1] = C[0,0]
        vb = C.dot(np.array([vx,vy]))

        # send commands
        self.client.moveByVelocityZAsync(vb[0], vb[1], ref_alt, duration,
                airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(True, vz))

    def _toEulerianAngle(self, q):
        z = q.z_val
        y = q.y_val
        x = q.x_val
        w = q.w_val
        ysqr = y * y

        # roll (x-axis rotation)
        t0 = +2.0 * (w*x + y*z)
        t1 = +1.0 - 2.0*(x*x + ysqr)
        roll = math.atan2(t0, t1)

        # pitch (y-axis rotation)
        t2 = +2.0 * (w*y - z*x)
        if (t2 > 1.0):
            t2 = 1
        if (t2 < -1.0):
            t2 = -1.0
        pitch = math.asin(t2)

        # yaw (z-axis rotation)
        t3 = +2.0 * (w*z + x*y)
        t4 = +1.0 - 2.0 * (ysqr + z*z)
        yaw = math.atan2(t3, t4)

        return (pitch, roll, yaw)

    def print_both(message, client, severity=1):
        """Print message to both terminal and simulator."""
        self.logger.info(message)
        client.simPrintLogMessage(message, severity=severity)

    def reset_vehicle(client):
        """Resets to original position and takeoff"""
        # resets    
        client.armDisarm(False)
        client.reset()
        client.enableApiControl(False)

        # connect
        print_both('[*] Connecting and preparing for takeoff...', client)
        client.confirmConnection()
        client.enableApiControl(True)
        client.armDisarm(True)
        client.takeoffAsync().join()
        print_both('[*] Takeoff complete.', client)

    def _label_log(self):
        """Write log labels."""
        # define labels
        log_labels = '{},{},{},{},{},{},{},{},{},{}, \
                     {},{},{},{},{},{},{},{},{},{},{}, \
                     {},{},{},{},{},{},{},{},{},{},{}\n'.format('timestamp',
                                                          'epi_num',
                                                          'step_num',
                                                          'rgb_addr',
                                                          'segment_addr',
                                                          'depth_addr',
                                                          'collision_status',
                                                          'pos_x',
                                                          'pos_y',
                                                          'pos_z',
                                                          'roll',
                                                          'pitch',
                                                          'yaw',
                                                          'vel_x',
                                                          'vel_y',
                                                          'vel_z',
                                                          'roll_vel',
                                                          'pitch_vel',
                                                          'yaw_vel',
                                                          'acc_x',
                                                          'acc_y',
                                                          'acc_z',
                                                          'roll_acc',
                                                          'pitch_acc',
                                                          'yaw_acc',
                                                          'act_roll',
                                                          'act_pitch',
                                                          'act_throttle',
                                                          'act_yaw',
                                                          'gaze_x',
                                                          'gaze_y',
                                                          'command')

        # write to files
        self.logfile.write(log_labels)


    def _write_log(self, actions, gaze_pred):
        """Write data to log file."""
        # setup addresses
        rgb_addr = self.exp_addr + f'rgb/rgb_{self.step_cnt}.png'
        seg_addr = self.exp_addr + f'seg/seg_{self.step_cnt}.png'
        depth_addr = self.exp_addr + f'depth/depth_{self.step_cnt}.png'  

        # check if logging predicted gaze from the agent or real gaze from user
        if gaze_pred is None:
            gaze_x = self.gaze_data[0]
            gaze_y = self.gaze_data[1]
        else:
            gaze_x = gaze_pred[0]
            gaze_y = gaze_pred[1]

        # setup log string
        log_labels = '{},{},{},{},{},{},{},{},{},{}, \
                     {},{},{},{},{},{},{},{},{},{},{}, \
                     {},{},{},{},{},{},{},{},{},{},{}\n'.format(
                        self.state.timestamp,
                        self.epi_num,
                        self.step_num,
                        rgb_addr,
                        seg_addr,
                        depth_addr,
                        self.vehicle_collided,
                        self.state.kinematics_estimated.position.x_val,
                        self.state.kinematics_estimated.position.y_val,
                        self.state.kinematics_estimated.position.z_val,
                        self.roll,
                        self.pitch,
                        self.yaw,
                        self.state.kinematics_estimated.linear_velocity.x_val,
                        self.state.kinematics_estimated.linear_velocity.y_val,
                        self.state.kinematics_estimated.linear_velocity.z_val,
                        self.state.kinematics_estimated.angular_velocity.x_val,
                        self.state.kinematics_estimated.angular_velocity.y_val,
                        self.state.kinematics_estimated.angular_velocity.z_val,
                        self.state.kinematics_estimated.linear_acceleration.x_val,
                        self.state.kinematics_estimated.linear_acceleration.y_val,
                        self.state.kinematics_estimated.linear_acceleration.z_val,
                        self.state.kinematics_estimated.angular_acceleration.x_val,
                        self.state.kinematics_estimated.angular_acceleration.y_val,
                        self.state.kinematics_estimated.angular_acceleration.z_val,
                        actions[0],
                        actions[1],
                        actions[2],
                        actions[3],
                        gaze_x,
                        gaze_y,
                        self.command)

        # write to disk
        self.logfile.write(log_labels)

    def _save_airsim_images(self, img_responses, step_cnt):
        """Threaded function that writes AirSim images to disk."""
        # convert images to numpy format 
        converted_imgs = []
        rgb_addr = self.exp_addr + f'rgb/rgb_{step_cnt}.png'
        seg_addr = self.exp_addr + f'seg/seg_{step_cnt}.png'
        depth_addr = self.exp_addr + f'depth/depth_{step_cnt}.png'

        for response in img_responses:
            if response.pixels_as_float==False:
                # convert to numpy
                img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
                # reshape it
                img_rgb = img1d.reshape(response.height, response.width, 3)
                # convert from bgr to rgb
                img_rgb = np.fliplr(img_rgb.reshape(-1,3)).reshape(img_rgb.shape)
            else:  # for the depth camera
                # convert to numpy (float)
                depth = np.array(response.image_data_float, dtype=np.float32)
                # reshape it
                depth = depth.reshape(response.height, response.width)
                # scale pixels
                img_rgb = np.array(np.abs(1-depth) * 255, dtype=np.uint8)  # reverse black and white

            # append to be saved to disk later
            converted_imgs.append(img_rgb)

        # write to disk
        if self.get_obj_segmentation: 
            plt.imsave(rgb_addr, converted_imgs[0])
            plt.imsave(seg_addr, converted_imgs[1])
            plt.imsave(depth_addr, converted_imgs[2], cmap='gray')
        else:
            # no object segmentation to save
            plt.imsave(rgb_addr, converted_imgs[0])
            plt.imsave(depth_addr, converted_imgs[1], cmap='gray')

    def render(self, mode='human', close=False):
        raise NotImplementedError


class MultimodalAirSimMountainsEnv(MultimodalAirSimEnv):
    """MultimodalAirSim environment in the LandscapeMountains map."""

    def reset(self, initial_loc=None, target_loc=None):
        # make sure airsim is still running
        assert self.client.ping()
        
        # sample language command
        if self.custom_command is not None:
            self.command = self.custom_command
        else:
            # random sample from possible list of commands
            self.command = np.random.choice(self.command_list)
        self._print_both(f'>>> Episode: {self.epi_num+1} | COMMAND: {self.command}', severity=3)

        # define goal object
        self.goal_obj = 'Truck_4'#'MovableAPC2_1226'
        self.logger.info(f'Target object: {self.goal_obj}')

        # resets
        self.client.reset()

        # try to handle error during takeoff
        takeoff_ok = False

        while True:
            try:
                # wait for a given amount of seconds while takeoff
                # if takes too long, retry
                self.client.enableApiControl(True)
                self.client.armDisarm(True)
                self.client.takeoffAsync().join()
                state = self.client.getMultirotorState()

                if state.kinematics_estimated.position.z_val < self.default_start_altitude:
                    break
            except:
                self.client.armDisarm(False)
                self.client.enableApiControl(False)


        # move target to goal position
        if self.dynamic_target:
            if target_loc is not None:
                curr_target_pose = self.client.simGetObjectPose(self.goal_obj)
                curr_target_pose.position.x_val = target_loc['target_pos_x'] - self.player_start_loc['pos_x']
                curr_target_pose.position.y_val = target_loc['target_pos_y'] - self.player_start_loc['pos_y']
                curr_target_pose.position.z_val = target_loc['target_pos_z'] - self.player_start_loc['pos_z'] -1
                self.client.simSetObjectPose(self.goal_obj, curr_target_pose, teleport = True)


        ## move vehicle to a different start location
        # read current position
        current_state = self.client.getMultirotorState()
        current_x = current_state.kinematics_estimated.position.x_val
        current_y = current_state.kinematics_estimated.position.y_val
        current_z = self.default_start_altitude

        # new location is either the same as a reference experiment or random
        if initial_loc is None:
            self._print_both('[*] Going to a initial RANDOM location...', severity=2)
            # sample new position
            new_x = current_x + 70*(np.random.rand()-0.5)
            new_y = current_y + 70*(np.random.rand()-0.5)
            new_z = current_z - 2*np.random.rand()

            # also sample new heading
            new_yaw = np.deg2rad(np.random.randint(low=0, high=360))

            # save starting location for next agent, if needed
            self.ref_initial_loc.append({'x':new_x,'y':new_y,'z':new_z,'yaw':new_yaw})

        else:
            # follows list of reference initial locations
            self._print_both('[*] Going to a initial REFERENCE location...', severity=2)

            # read reference location
            new_x = initial_loc['pos_x']
            new_y = initial_loc['pos_y']
            new_z = initial_loc['pos_z'] -3 # meters, safety distance
            new_yaw = initial_loc['yaw']

        # move vehicle to new position and orientation
        self._print_both(f'[*] x: {new_x}, y: {new_y}, z: {new_z}')

        # first, move to safe altitude, then x,y location, than desired altitude
        # (this prevents crashing to tree or any other obstacle that might be in
        #  between the vehicle and the desired location)
        safe_z = -15 # meters
        self.client.moveToPositionAsync(current_x, current_y, safe_z, 5).join()
        self.client.moveToPositionAsync(new_x, new_y, safe_z, 5).join()
        self.client.moveToPositionAsync(new_x, new_y, new_z, 5).join()
        self.client.rotateToYawAsync(np.rad2deg(new_yaw)).join()

        # read initial observation
        obs = self._read_obs()

        # check for collisions after reset, if that becomes a problem needs to
        # ensure that the first state is collision-free
        self.logger.info(f'Initial collision: {self.vehicle_collided}')
        if self.vehicle_collided:
            initial_loc['pos_x'] += 0.5
            initial_loc['pos_y'] += 0.5
            self.reset(initial_loc=initial_loc, target_loc=target_loc)

        # update time steps and episode counters
        self.epi_num += 1
        self.step_num = 0

        return obs

    def step(self, actions):
        # # check if recording demonstrations
        # if self.agent is not None:
        #     self.logger.info(f'A trained agent was passed to the environment: {self.agent}')

        # check if actions also includes gaze predictions
        if type(actions) in [tuple, list]:
            # action contains gaze and actions
            actions_joystick = actions[1].cpu().detach().numpy()[0]
            actions_gaze = actions[0].cpu().detach().numpy()[0]
        else:
            # actions are only actions (no gaze)
            actions_joystick = actions
            actions_gaze = None

        # check for interventions before propagating commanded actions
        if self.enable_interventions:
            self.human_intervening = self.joystick._check_intervention()
            if self.human_intervening:
                # read joystick intervention commands
                actions_joystick = self.joystick.act()

        # propagate joystick commands
        self._vel_cmd(actions_joystick)

        # read new observation and log previous one (the one used to compute the action)
        self.prev_obs_log = copy.deepcopy(self.prev_obs)
        obs = self._read_obs()

        # compute reward
        reward = 0.

        # check if the episode is complete
        done = False
        # stopping conditions:
        #   1. max number of steps reached
        if self.step_num >= self.n_steps:
            done = True
            self._print_both(f'[*] Max number of steps reached. (Epi. {self.epi_num})', severity=2)

        #   2. collision with any obstacle
        if self.vehicle_collided:
            done = True
            self._print_both(f'[*] Vehicle collided. (Epi. {self.epi_num})', severity=2)

        #   3. task completed
        self.goal_dist = self._distance_to_object(object_name=self.goal_obj)
        self.task_done = False
        if self.goal_dist < 5: # meters from the centroid of target
            self._print_both(f'[*] Task completed! (Epi. {self.epi_num})', severity=2)
            self.task_done = True
            done = True

        # #   4. check if landed (or stuck at the ground by mistake)
        # landed = self.state.landed_state
        # if self.step_num>10 and landed:
        #     done = True
        #     self._print_both(f'[*] Landed on the ground. (Epi. {self.epi_num})', severity=2)

        # log data with actions, previous observation, done status, etc
        self._write_log(actions_joystick, actions_gaze)
        
        # return any additional info
        info = {'task_done': self.task_done}

        # update step counter
        self.step_cnt += 1
        self.step_num += 1

        return obs, reward, done, info

    def _distance_to_object(self, object_name):
        """Compute current distance from vehicle to object in Unreal Engine.

        object_name of interest:
            'MovableAPC2_1226': vehicle from AAAI-19 paper

        """
        # grab current position
        curr_x = self.state.kinematics_estimated.position.x_val
        curr_y = self.state.kinematics_estimated.position.y_val
        curr_z = self.state.kinematics_estimated.position.z_val

        # get position of target object
        target_pos = self.client.simGetObjectPose(object_name)
        self.target_pos_x = target_pos.position.x_val
        self.target_pos_y = target_pos.position.y_val
        self.target_pos_z = target_pos.position.z_val

        # compute Euclidean distance
        distance_to_target = np.sqrt(
            (curr_x-self.target_pos_x)**2 +
            (curr_y-self.target_pos_y)**2 +
            (curr_z-self.target_pos_z)**2)

        return distance_to_target

    def _label_log(self):
        """Write log labels."""
        # define labels
        log_labels = '{},{},{},{},{},{},{},{},{},{},{},{},{},{}, \
                     {},{},{},{},{},{},{},{},{},{},{},{}, \
                     {},{},{},{},{},{},{},{},{},{},{},{}\n'.format('timestamp',
                                                          'epi_num',
                                                          'step_num',
                                                          'rgb_addr',
                                                          'segment_addr',
                                                          'depth_addr',
                                                          'human_intervening',
                                                          'collision_status',
                                                          'task_done',
                                                          'target_pos_x',
                                                          'target_pos_y',
                                                          'target_pos_z',
                                                          'dist_to_target',
                                                          'pos_x',
                                                          'pos_y',
                                                          'pos_z',
                                                          'roll',
                                                          'pitch',
                                                          'yaw',
                                                          'vel_x',
                                                          'vel_y',
                                                          'vel_z',
                                                          'roll_vel',
                                                          'pitch_vel',
                                                          'yaw_vel',
                                                          'acc_x',
                                                          'acc_y',
                                                          'acc_z',
                                                          'roll_acc',
                                                          'pitch_acc',
                                                          'yaw_acc',
                                                          'act_roll',
                                                          'act_pitch',
                                                          'act_throttle',
                                                          'act_yaw',
                                                          'gaze_x',
                                                          'gaze_y',
                                                          'command')

        # write to files
        self.logfile.write(log_labels)


    def _write_log(self, actions, gaze_pred):
        """Write data to log file."""
        # parse previous observation        
        state = self.prev_obs_log['states_bc'][0]
        gaze_data = self.prev_obs_log['gaze_data']

        # setup addresses
        rgb_addr = self.exp_addr + f'rgb/rgb_{self.step_cnt}.png'
        seg_addr = self.exp_addr + f'seg/seg_{self.step_cnt}.png'
        depth_addr = self.exp_addr + f'depth/depth_{self.step_cnt}.png'  

        # check if logging predicted gaze from the agent or real gaze from user
        if gaze_pred is None:
            gaze_x = gaze_data[0]
            gaze_y = gaze_data[1]
        else:
            gaze_x = gaze_pred[0]
            gaze_y = gaze_pred[1]

        # setup log string
        log_labels = '{},{},{},{},{},{},{},{},{},{},{},{},{},{}, \
                     {},{},{},{},{},{},{},{},{},{},{},{}, \
                     {},{},{},{},{},{},{},{},{},{},{},{}\n'.format(
                        state.timestamp,
                        self.epi_num,
                        self.step_num,
                        rgb_addr,
                        seg_addr,
                        depth_addr,
                        self.human_intervening,
                        self.vehicle_collided,
                        self.task_done,
                        self.target_pos_x,
                        self.target_pos_y,
                        self.target_pos_z,
                        self.goal_dist,
                        state.kinematics_estimated.position.x_val,
                        state.kinematics_estimated.position.y_val,
                        state.kinematics_estimated.position.z_val,
                        self.roll,
                        self.pitch,
                        self.yaw,
                        state.kinematics_estimated.linear_velocity.x_val,
                        state.kinematics_estimated.linear_velocity.y_val,
                        state.kinematics_estimated.linear_velocity.z_val,
                        state.kinematics_estimated.angular_velocity.x_val,
                        state.kinematics_estimated.angular_velocity.y_val,
                        state.kinematics_estimated.angular_velocity.z_val,
                        state.kinematics_estimated.linear_acceleration.x_val,
                        state.kinematics_estimated.linear_acceleration.y_val,
                        state.kinematics_estimated.linear_acceleration.z_val,
                        state.kinematics_estimated.angular_acceleration.x_val,
                        state.kinematics_estimated.angular_acceleration.y_val,
                        state.kinematics_estimated.angular_acceleration.z_val,
                        actions[0],
                        actions[1],
                        actions[2],
                        actions[3],
                        gaze_x,
                        gaze_y,
                        self.command)

        # write to disk
        self.logfile.write(log_labels)
