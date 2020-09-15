"""viz_airsim_log.py
Visualize data gathered in AirSim.

Usage:
    python viz_airsim_log.py --exp_name <exp_name>

Example:
    python viz_airsim_log.py --exp_name truck_mountains_v8/Fly_to_the_nearest_truck --plot_gaze --no_seg
    python viz_airsim_log.py --exp_name interv_model/Fly_to_the_nearest_truck --plot_gaze --no_seg


Example (VanillaBC):
    python viz_airsim_log.py --exp_name '2020-08-22 12:55:09.609361/Fly_to_the_nearest_truck' --no_seg
Example (GazeBC):
    python viz_airsim_log.py --exp_name '2020-08-21 17:30:08.436480/Fly_to_the_nearest_truck' --plot_gaze --no_seg

Example (predictions overlay with demonstrated). NOTE: need to generated predictions file first
    python viz_airsim_log.py --from_file '2020-08-31 20:11:03.468076' --exp_name truck_mountains1 --demo_data --plot_gaze
    python viz_airsim_log.py --from_file '2020-08-31 20:47:27.754218' --exp_name truck_mountains1 --demo_data --plot_gaze
    python viz_airsim_log.py --from_file '2020-08-31 21:23:13.571091' --exp_name truck_mountains1 --demo_data --plot_gaze

"""
import time
from datetime import datetime
import argparse
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import cv2

# parse arguments
my_parser = argparse.ArgumentParser(
    prog='viz_airsim_log.py', usage='%(prog)s [options]',
    description='Visualize data gathered in AirSim.')
my_parser.add_argument('--exp_name', type=str, default='test')
my_parser.add_argument('--plot_gaze', action='store_true')
my_parser.add_argument('--demo_data', action='store_true')
my_parser.add_argument('--from_file', type=str, default=None,
    help='Reads csv files with predictions and overlay over logged gaze data.')
my_parser.add_argument('--gaze_importance', action='store_true')
my_parser.add_argument('--no_seg', action='store_false')
args = my_parser.parse_args()


# load log data
if args.demo_data:
    log_addr = f'../mtl/data/{args.exp_name}/Fly_to_the_nearest_truck/'
else:
    log_addr = f'../mtl/airsim_data/logs/{args.exp_name}/'
log_data = pd.read_csv(log_addr+'log.csv')
n_samples = log_data.shape[0]
os.makedirs(log_addr+'viz', exist_ok=True)
fps_cnt = np.zeros(n_samples-1)
n_experiments = 1


# check if have predicted gaze data saved on file
if args.from_file is not None:
    print('[*] Overlaying predicted and demonstrated gaze.')
    gaze_on_demo_dir = f"../mtl/logs/{args.from_file}/gaze_on_demo"
    files_cnt = 0
    for file in os.listdir(gaze_on_demo_dir):
        if file.endswith(".csv"):
            # load file and append to dataframe
            if files_cnt == 0:
                gaze_df = pd.read_csv(os.path.join(gaze_on_demo_dir, file))
            else:
                gaze_df = gaze_df.append(pd.read_csv(os.path.join(gaze_on_demo_dir, file)), ignore_index=True)
            files_cnt += 1

    # identify episodes, exp_name, and frame number in the log file
    start = 'truck_mountains'
    start_exp = '/data//'
    start_frame = 'rgb/rgb_'
    end = '/Fly_to_the_nearest_truck'
    end_frame = '.png'
    log_episodes = []
    log_exp_names = []
    log_frame_numbers = []
    for s in gaze_df['frame_addr'].values:
        log_episodes.append(s[s.find(start)+len(start):s.rfind(end)])
        log_exp_names.append(s[s.find(start_exp)+len(start_exp):s.rfind(end)])
        log_frame_numbers.append(s[s.find(start_frame)+len(start_frame):s.rfind(end_frame)])
    
    # add to dataframe of logged values
    gaze_df['episode'] = log_episodes
    gaze_df['exp_name'] = log_exp_names
    gaze_df['frame_number'] = log_frame_numbers
    gaze_df['frame_number'] = gaze_df['frame_number'].astype(int)

    # sort by exp_name and save to disk
    gaze_df = gaze_df.sort_values(by=['exp_name'])
    gaze_df.to_csv(f"../mtl/logs/{args.from_file}/gaze_preds_compiled.csv")

    # check number of unique experiments
    exp_names = gaze_df['exp_name'].unique()
    n_experiments = len(exp_names)

# generates one video per experiment
for j in range(n_experiments):
    exp_name = args.exp_name

    # split experiment data in case we are loading from file
    if args.from_file is not None:
        # update exp_name
        exp_name = exp_names[j]

        # load log data
        if args.demo_data:
            log_addr = f'../mtl/data/{exp_name}/Fly_to_the_nearest_truck/'
        else:
            log_addr = f'../mtl/airsim_data/logs/{exp_name}/'
        log_data = pd.read_csv(log_addr+'log.csv')
        n_samples = log_data.shape[0]
        os.makedirs(log_addr+'viz', exist_ok=True)
        fps_cnt = np.zeros(n_samples-1)
        n_experiments = 1

        # split gaze_df based on the exp_name we are parsing
        gaze_exp_name_df = gaze_df[gaze_df['exp_name'] == exp_name]
        gaze_exp_name_df = gaze_exp_name_df.sort_values(by=['frame_number'])
        gaze_exp_name_df.to_csv(f"../mtl/logs/{args.from_file}/{exp_name}_gaze_preds_exp_name.csv")
        gaze_exp_cnt = 0

    
    # loop over all data, show images and overlay with eye gaze
    for i in range(n_samples):
        if args.from_file is not None:
            
            # check if this demonstration sampled was predicted
            try:
                # print(exp_name, n_samples, int(gaze_exp_name_df['frame_number'].iloc[gaze_exp_cnt]), i, gaze_exp_cnt, gaze_exp_name_df['frame_addr'].iloc[gaze_exp_cnt])
                if int(gaze_exp_name_df['frame_number'].iloc[gaze_exp_cnt] != i):
                    continue 
            except:
                # print('no predicted frames for', i)
                break

        # print(f'[*] Processing {i} out of {n_samples} images.')
        # load images    
        if args.demo_data:
            img_rgb = cv2.imread(log_data['rgb_addr'][i])
            if args.no_seg:
                img_seg = cv2.imread(log_data['segment_addr'][i])
            img_depth = cv2.imread(log_data['depth_addr'][i])
        else:
            img_rgb = cv2.imread('../mtl/'+log_data['rgb_addr'][i])
            if args.no_seg:
                img_seg = cv2.imread('../mtl/'+log_data['segment_addr'][i])
            img_depth = cv2.imread('../mtl/'+log_data['depth_addr'][i])
        img_depth_src = img_depth.copy() # preserve original to compute metrics
        img_height = img_rgb.shape[0]
        img_width = img_rgb.shape[1]

        # compute time between frames (defines video FPS)
        if i > 0:
            timestamp = log_data['timestamp'][i]/1e9
            prev_timestamp = log_data['timestamp'][i-1]/1e9
            dt_object = datetime.fromtimestamp(timestamp)
            prev_dt_object = datetime.fromtimestamp(prev_timestamp)
            fps_cnt[i-1] = 1/(dt_object-prev_dt_object).total_seconds()

        # add gaze mark to image
        mark_radius = 40
        if args.plot_gaze:
            GAZE_COLOR = (255,0,255)
            gaze_overlay = img_rgb.copy()
            gaze_x = int(log_data['gaze_x'][i]*img_width)
            gaze_y = int(log_data['gaze_y'][i]*img_height)
            cv2.circle(gaze_overlay,(gaze_x, gaze_y), mark_radius, GAZE_COLOR, -1)
            if args.no_seg:
                cv2.circle(img_seg,(gaze_x, gaze_y), mark_radius, GAZE_COLOR, -1)
            cv2.circle(img_depth,(gaze_x, gaze_y), mark_radius, GAZE_COLOR, -1)

            # save location of first gaze marks
            first_gaze_x = gaze_x
            first_gaze_y = gaze_y

            # combine gaze overlay with transparency
            alpha=0.3
            img_rgb = cv2.addWeighted(gaze_overlay, alpha, img_rgb, 1 - alpha, 0)

            # add additional gaze marks from previous frames if data is available
            n_marks = 3
            if i > (n_marks-1):
                # additional marks
                for k in range(1, n_marks):
                    gaze_overlay = img_rgb.copy()
                    gaze_x = int(log_data['gaze_x'][i-k]*img_width)
                    gaze_y = int(log_data['gaze_y'][i-k]*img_height)
                    cv2.circle(gaze_overlay,(gaze_x, gaze_y), mark_radius-k*5, GAZE_COLOR, -1)
                    if args.no_seg:
                        cv2.circle(img_seg,(gaze_x, gaze_y), mark_radius-k*5, GAZE_COLOR, -1)
                    cv2.circle(img_depth,(gaze_x, gaze_y), mark_radius-k*5, GAZE_COLOR, -1)

                    # combine gaze overlay with transparency
                    alpha=0.3-0.1*k
                    img_rgb = cv2.addWeighted(gaze_overlay, alpha, img_rgb, 1 - alpha, 0)

        # add gaze mark to image (from file)
        if args.from_file is not None:
            # make sure agent was using the same RGB file
            GAZE_PRED_COLOR = (255,255,0)
            gaze_overlay = img_rgb.copy()
            gaze_x = int(gaze_exp_name_df['gaze_outputs_x'].iloc[gaze_exp_cnt]*img_width)
            gaze_y = int(gaze_exp_name_df['gaze_outputs_y'].iloc[gaze_exp_cnt]*img_height)
            cv2.circle(gaze_overlay,(gaze_x, gaze_y), mark_radius, GAZE_PRED_COLOR, -1)
            if args.no_seg:
                cv2.circle(img_seg,(gaze_x, gaze_y), mark_radius, GAZE_PRED_COLOR, -1)
            cv2.circle(img_depth,(gaze_x, gaze_y), mark_radius, GAZE_PRED_COLOR, -1)

            # save location of first gaze marks
            first_gaze_x = gaze_x
            first_gaze_y = gaze_y

            # combine gaze overlay with transparency
            alpha=0.3
            img_rgb = cv2.addWeighted(gaze_overlay, alpha, img_rgb, 1 - alpha, 0)

            # add additional gaze marks from previous frames if data is available
            n_marks = 3
            if i > (n_marks-1):
                # additional marks
                for k in range(1, n_marks):
                    gaze_overlay = img_rgb.copy()
                    gaze_x = int(gaze_exp_name_df['gaze_outputs_x'].iloc[gaze_exp_cnt-k]*img_width)
                    gaze_y = int(gaze_exp_name_df['gaze_outputs_y'].iloc[gaze_exp_cnt-k]*img_height)
                    cv2.circle(gaze_overlay,(gaze_x, gaze_y), mark_radius-k*5, GAZE_PRED_COLOR, -1)
                    if args.no_seg:
                        cv2.circle(img_seg,(gaze_x, gaze_y), mark_radius-k*5, GAZE_PRED_COLOR, -1)
                    cv2.circle(img_depth,(gaze_x, gaze_y), mark_radius-k*5, GAZE_PRED_COLOR, -1)

                    # combine gaze overlay with transparency
                    alpha=0.3-0.1*k
                    img_rgb = cv2.addWeighted(gaze_overlay, alpha, img_rgb, 1 - alpha, 0)

            # update gaze counter
            gaze_exp_cnt += 1

        # add text
        if args.from_file is None:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_rgb, log_data['command'][i], (15,25), font, .65,(0,0,0),2,cv2.LINE_AA)
            cv2.putText(img_rgb, log_data['command'][i], (15,25), font, .65,(255,255,255),1,cv2.LINE_AA)

        # put images side-by-side
        if args.no_seg:
            img_combined = np.hstack((img_rgb, img_seg, img_depth))
        else:
            img_combined = np.hstack((img_rgb, img_depth))

        # save to disk
        cv2.imwrite(log_addr+f'viz/frame{i}.png', img_combined)

        ## SEPARATE RGB VIDEO
        aspect_ratio = 480/704

        # # add intervention indicator
        # if log_data['human_intervening'][i]:
        #     # indicator light
        #     cv2.circle(
        #     img_rgb, (int(.5*img_width), int(.95*img_width*aspect_ratio)),
        #     15, (255,50,50), -1)
        # # contour to indicator
        # cv2.circle(
        #     img_rgb, (int(.5*img_width), int(.95*img_width*aspect_ratio)),
        #     15, (255,255,255), 2)

        # put controls
        if args.from_file is None:
            # joystick areas (left and right)
            center_left = (int(.1*img_width), int(.85*img_width*aspect_ratio))
            center_right = (int(.9*img_width), int(.85*img_width*aspect_ratio))
            radius = 50
            cv2.circle(img_rgb, center_left, radius, (255,255,255), 1)
            cv2.circle(img_rgb, center_right, radius, (255,255,255), 1)
            # joystick commands (throttle+yaw and pitch+roll)
            thr_cmd = log_data['act_throttle'][i]
            yaw_cmd = log_data['act_yaw'][i]
            thr_y_pos = center_left[1]-thr_cmd*radius
            yaw_x_pos = center_left[0]+yaw_cmd*radius
            cv2.circle(img_rgb, (int(yaw_x_pos), int(thr_y_pos)), int(0.1*radius), (255,255,255), -1)

            pitch_cmd = log_data['act_pitch'][i]
            roll_cmd = log_data['act_roll'][i]
            pitch_y_pos = center_right[1]-pitch_cmd*radius
            roll_x_pos = center_right[0]+roll_cmd*radius
            cv2.circle(img_rgb, (int(roll_x_pos), int(pitch_y_pos)), int(0.1*radius), (255,255,255), -1)

        # add gaze importance features based on depth of image where gaze is fixating
        if args.gaze_importance:
            # define patch size
            patch_size = 30 # pixels, side of the square centered at the gaze location

            # compute average depth value around gaze location
            # draw ROI rectangle between vertices pt1 and pt2
            pt1 = (
                int(np.clip(first_gaze_x-patch_size/2, 0, img_width)),
                int(np.clip(first_gaze_y-patch_size/2, 0, img_height))
            )
            pt2 = (
                int(np.clip(first_gaze_x+patch_size/2, 0, img_width)),
                int(np.clip(first_gaze_y+patch_size/2, 0, img_height))
            )

            # compute average value inside ROI based on depth frame
            roi_depth = img_depth_src[pt1[1]:pt2[1], pt1[0]:pt2[0]]
            depth_metric = np.mean(roi_depth)/255

            # draw metric bar
            pt1_bar = (int(0.9*img_width), int((0.15-.1*depth_metric)*img_height))
            pt2_bar = (int(0.95*img_width), int(0.15*img_height))
            cv2.rectangle(img_rgb, pt1_bar,  pt2_bar, (50,int(200*depth_metric)+55,80), -1)

            # draw bar around metric (bounds)
            pt1_bar = (int(0.9*img_width), int(0.05*img_height))
            pt2_bar = (int(0.95*img_width), int(0.15*img_height))
            cv2.rectangle(img_rgb, pt1_bar,  pt2_bar, (200,200,200), 1)

        # write to disk
        cv2.imwrite(log_addr+f'viz/rgb_frame{i}.png', img_rgb)


    # convert frames to video
    if args.from_file is not None:
        start_frame = int(gaze_exp_name_df['frame_number'].iloc[0])
        os.system(f'ffmpeg -r 10 -f image2 -start_number {start_frame} -i "{log_addr}viz/rgb_frame%d.png" -q:v 1 "../mtl/logs/{args.from_file}/{exp_name}_Human_Comparison.avi" -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -y')
    else:
        mean_fps = np.mean(fps_cnt)
        print('Average framerate:', mean_fps)
        os.system(f'ffmpeg -r {mean_fps} -f image2 -i "{log_addr}viz/frame%d.png" -q:v 1 "{log_addr}viz/video.avi" -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -y')
        os.system(f'ffmpeg -r {mean_fps} -f image2 -i "{log_addr}viz/rgb_frame%d.png" -q:v 1 "{log_addr}viz/rgb_video.avi" -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -y')

    # remove all viz png and leave only the video
    os.system(f'rm -rf "{log_addr}"viz/*.png')