"""gym_log_airsim.py
Log vehicle, camera, and human data while solving tasks using the AirSim Gym
environment.

Usage:
    python gym_log_airsim.py \
        --exp_name <exp_name> \
        --start_epi <start_epi> \
        --n_steps <n_steps> \
        --command <command> \
        [optional] --model_name <model_name>

Example (collecting data on Mountains environment):
    python gym_log_airsim.py \
            --env_name MultimodalAirSimMountains-v0 \
            --exp_name truck_mountains1 \
            --n_episodes 10 \
            --n_steps 2000 \
            --command "Fly to the nearest truck" \
            --use_joystick

Example (collecting data on Mountains environment, dynamic truck):
    python gym_log_airsim.py \
            --env_name MultimodalAirSimMountains-v0 \
            --exp_name moving_truck_mountains1 \
            --start_epi 0 \
            --max_episodes 5 \
            --n_steps 2000 \
            --command "Fly to the nearest truck" \
            --use_joystick
"""
import argparse
import sys, os
import gym
import time
import pandas as pd 

metrics_path = os.getcwd().split('airsim')[0]+'mtl/'
sys.path.insert(0, metrics_path)

from metrics import *
from common_utils import *

def restart_airsim():
    """Finds the binary PIDs, kill them, and restart the AirSim binary."""
    # find PIDs
    pids = os.popen('pgrep ARL_Test_Small').read().split('\n')[:-1]

    # kill each PID
    for pid in pids:
        os.popen(f'kill -9 {pid}')

    # delay to wait binary to be fully closed
    time.sleep(3)

    # restart binary
    os.popen('airsim_static')

    # delay to wait binary to be fully opened
    time.sleep(15)

def run_rollouts(**params):
    use_joystick = params['use_joystick']
    exp_name = params['exp_name']
    command = params['command']
    n_steps = params['n_steps']
    env_name = params['env_name']
    start_epi = params['start_epi']
    max_episodes = params['max_episodes']
    dynamic_target = params['dynamic_target']
    n_episodes = 0

    logger = get_logger('TESTING_LOGGER')


    try:
        import airsim_env
        from utils_airsim import JoystickAgent, AirSimBCAgent

        # create joystick agent or load behavior cloning model
        if use_joystick:
            # use joystick
            agent = JoystickAgent()
            get_obj_segmentation = True

            params['df'] = pd.read_csv('locs_config.csv')
            params['test_epi_ids'] = list(params['df'].epi_id.unique())
            params['num_trials_per_episode'] = 1
        else:
            # use pretrained model
            agent = AirSimBCAgent(exp_name, task=command)
            get_obj_segmentation = False
            params['num_trials_per_episode'] = 5

        # create env to be tested
        env = gym.make(
            env_name,
            n_steps=n_steps,
            exp_name=exp_name,
            custom_command=command,
            get_obj_segmentation=get_obj_segmentation,
            agent=agent,
            dynamic_target=dynamic_target,
            save_image_data=False)

        completions = 0.0

        # loop and collect data
        episode_id_cnt = 1
        for episode_id in params['test_epi_ids']:
            # make sure episode_id is updated based on start episode
            len_test_epis = len(params['test_epi_ids'])
            logger.info(f'>>>>> Episode ID: {episode_id} | Episode {episode_id_cnt}/{len_test_epis}')
           
            if use_joystick:
                logger.info('Using joystick to collect demonstration data.')
                episode_id += start_epi
                if episode_id >= len_test_epis:
                    # finished episodes from list
                    break            

            episode = params['df'].groupby('epi_id').get_group(episode_id)
            drone_params = ['pos_x','pos_y','pos_z','yaw']
            truck_params = ['target_pos_x','target_pos_y','target_pos_z']

            # configurations passed as dictionaries to reset function
            initial_loc = episode[drone_params].to_dict('records')[0]
            target_loc = episode[truck_params].to_dict('records')[0]

            # multiple trials per drone config to account for a stochastic environment
            for i in range(params['num_trials_per_episode']):
                
                logger.info(f'>>>>> Episode ID: {episode_id} | Episode {episode_id_cnt}/{len_test_epis} | Trial {i+1}')

                # reset function takes in initial drone spawn loc and target loc as input
                try:
                    with time_limit(180):
                        # check if stuck during reset
                        observation = env.reset(initial_loc=initial_loc, target_loc=target_loc)
                except:
                    logger.error('Crashed/Froze during RESET!', exc_info=True)
                    restart_airsim()
                    env.setup_airsim()
                    observation = env.reset(initial_loc=initial_loc, target_loc=target_loc)

                # episode loop
                while True:
                    action = agent.act(observation)
                    
                    try:
                        with time_limit(5):
                            # check if stuck during step
                            observation, reward, done, info = env.step(action)
                    except:
                        logger.error('Crashed/Froze during STEP!', exc_info=True)
                        restart_airsim()
                        env.setup_airsim()
                        observation = env.reset(initial_loc=initial_loc, target_loc=target_loc)
                        done = False

                    if done:
                        completions +=  info['task_done'] == True
                        n_episodes += 1
                        env.save_images_to_disk()
                        break

            # move to the next configuration, once all trials are complete
            episode_id_cnt += 1

            # check if already collected data for the desired number of episodes
            if use_joystick:
                if n_episodes >= max_episodes:
                    break

        # close everything
        env.close()

        logfile = env.filename

        # return the list of starting points used for the rollouts, if a preset list was used then the original list would have been fully popped to empty.
        ref_initial_loc = env.ref_initial_loc if len(env.ref_initial_loc) != 0 else env.ref_initial_loc_copy

        task_completion_rate = completions/n_episodes
        n_collisions = num_collisions(logfile)
        avg_task_length, avg_spl = compute_avg_task_length(logfile)

    except ModuleNotFoundError:
        task_completion_rate = 100
        n_collisions = 0
        avg_task_length = 0
        avg_spl = 0

    finally:
        # return metrics
        metrics = {
            'task_completion_rate':task_completion_rate, 
            'num_collisions':n_collisions, 
            'avg_task_length':avg_task_length,
            'avg_spl':task_completion_rate*avg_spl
        }

        # save demonstration metrics
        if use_joystick:
            metrics_df = pd.DataFrame.from_dict(metrics, orient="index")
            metrics_df_addr = os.path.join(env.exp_addr,'metrics.csv')
            metrics_df.to_csv(metrics_df_addr)
            logger.info(f'> Saved demonstration metrics at {metrics_df_addr}.')

    return metrics


if __name__=='__main__':
    # parse arguments
    my_parser = argparse.ArgumentParser(
        prog='gym_log_airsim.py',
        description='Logs demonstrations or perform rollouts of behavior cloning models.')
    my_parser.add_argument('--env_name', type=str, default='MultimodalAirSim-v0')
    my_parser.add_argument('--exp_name', type=str, default='test')
    my_parser.add_argument('--start_epi', type=int, default=0)
    my_parser.add_argument('--max_episodes', type=int, default=100)
    my_parser.add_argument('--n_steps', type=int, default=200)
    my_parser.add_argument('--command', type=str, default='TEST COMMAND')
    my_parser.add_argument('--use_joystick', action='store_true')
    my_parser.add_argument('--dynamic_target', action='store_true')
    my_parser.add_argument('--ref_initial_loc', type=str, default=[],
        help='Reference experiment address to use the same initial locations.')
    args = my_parser.parse_args()

    run_rollouts(**vars(args))
