"""eval_exp.py
Computes performance metrics for an experiment per episode.

Usage:
    python eval_exp.py \
        --exp_names data/fly_tree data/bc_fly_tree data/bc_gaze_fly_tree \
        --exp_labels Human BC RealGazeBC
    python eval_exp.py \
        --exp_names data/fly_tree data/joint_rollout data/bc_rollout \
        --exp_labels Human JointBC BC
    python eval_exp.py \
        --exp_names data/test_fly_red_vehicle/Fly_to_red_vehicle data/prev_multitask3/Fly_to_red_vehicle data/bl/Fly_to_red_vehicle \
        --exp_labels Human JointBC BC \
        --fig_name test_fly_red_vehicle/Fly_to_red_vehicle \
        --task_name "Fly to red vehicle"
    python eval_exp.py \
        --exp_names data/test_fly_utility_pole/Fly_to_the_top_of_utility_pole data/prev_multitask3/Fly_to_the_top_of_utility_pole data/bl/Fly_to_the_top_of_utility_pole \
        --exp_labels Human JointBC BC \
        --fig_name test_fly_utility_pole/Fly_to_the_top_of_utility_pole \
        --task_name "Fly to the top of utility pole"
"""
import os, sys, argparse
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from scipy.spatial.distance import directed_hausdorff
from scipy import signal

def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


class TrajMetrics(object):
    """Holds trajectory evaluation metrics."""
    def __init__(self, exp_name, task_name, n_episodes=None):
        # load log
        self.exp_name = exp_name
        logfile_addr = os.path.join(exp_name, 'log.csv')
        self.data = pd.read_csv(logfile_addr)

        # find number of episodes if reference trajectory
        if n_episodes is None:
            n_episodes = self.data['epi_num'].max()+1
        self.n_episodes = n_episodes

        # trajectories from different starting points
        self.trajs_side1 = []
        self.trajs_side2 = []

        # metrics
        self.final_coords = np.zeros((self.n_episodes, 3))
        self.hausdorff_dists = np.zeros(2) # 2 sides of the map
        self.dist_to_goal = np.zeros(self.n_episodes)
        self.completed_goal = np.zeros(self.n_episodes)

        # report        
        print(f'Loaded {self.exp_name} for {self.n_episodes} episodes.')

        # define task goal locations and radius
        print(f'Working on task: "{task_name}"')
        # fly_red_vehicle
        if task_name == "Fly to red vehicle":
            self.goal_loc = np.array([51.30679, 2.917661, 1.00407])
            self.goal_radius = 2*np.sqrt(
                (55.41195297-self.goal_loc[0])**2 +
                (2.604412794-self.goal_loc[1])**2 +
                (0.194623098-self.goal_loc[2])**2)

        # fly_utility_pole
        elif task_name == "Fly to the top of utility pole":
            self.goal_loc = np.array([28.90716, 5.911726, 12.1986])
            self.goal_radius = 4*np.sqrt(
                (30.7957-self.goal_loc[0])**2 +
                (5.634659-self.goal_loc[1])**2 +
                (11.9692-self.goal_loc[2])**2)

        # fly_tree
        elif task_name == "Fly to the tree nearby gray vehicle":
            self.goal_loc = np.array([26.70623, -10.5306, 10.6685])
            self.goal_radius = 2*np.sqrt(
                (23.57202-self.goal_loc[0])**2 +
                (-8.75551-self.goal_loc[1])**2 +
                (9.11102-self.goal_loc[2])**2)


if __name__ == '__main__':
    # parse arguments
    my_parser = argparse.ArgumentParser(
        prog='eval_exp.py', usage='%(prog)s [options]',
        description='Computes performance metrics for an experiment.')
    my_parser.add_argument('--exp_names', nargs='+', type=str, default='test')
    my_parser.add_argument('--exp_labels', nargs='+', type=str, default='label_test')
    my_parser.add_argument('--fig_name', type=str, default='test')
    my_parser.add_argument('--task_name', type=str, default='test')
    args = my_parser.parse_args()

    # loads all logfiles
    n_logs = len(args.exp_names)
    logs = []
    for i in range(n_logs):
        # load logfile with experiment data
        if i == 0:  # use n_episodes of the first log
            log = TrajMetrics(exp_name=args.exp_names[i], task_name=args.task_name)
        else:
            log = TrajMetrics(
                exp_name=args.exp_names[i], task_name=args.task_name, n_episodes=logs[0].n_episodes)
        logs.append(log)

    # display trajectories
    fig = plt.figure(figsize=(18,4))
    plt.suptitle(f'Trajectories Comparison: "{args.task_name}"')
    ax = fig.add_subplot(111, projection='3d')
    colors = ['tab:blue', 'tab:red', 'tab:green']
    
    # loop each episode for each log file
    for k in range(logs[0].n_episodes):
        for i in range(n_logs):
            # grab data only for the specific episode
            data_epi = logs[i].data[logs[i].data['epi_num']==k]

            # convert altitude to positive numbers
            data_epi['pos_z'] = data_epi['pos_z'].values*(-1)

            # plot episode traj
            ax.plot(data_epi['pos_x'], data_epi['pos_y'], data_epi['pos_z'],
                color=colors[i], alpha=0.1)

            # store final coordinate for the episode
            logs[i].final_coords[k] = [data_epi['pos_x'].iloc[-1], data_epi['pos_y'].iloc[-1], data_epi['pos_z'].iloc[-1]]

            # compute distance between final coordinates and goal location
            logs[i].dist_to_goal[k] = np.sqrt(
                (logs[i].final_coords[k][0]-logs[i].goal_loc[0])**2 +
                (logs[i].final_coords[k][1]-logs[i].goal_loc[1])**2 +
                (logs[i].final_coords[k][2]-logs[i].goal_loc[2])**2)

            # check if completed goal
            if logs[i].dist_to_goal[k] <= logs[i].goal_radius:
                logs[i].completed_goal[k] = 1
            else:
                logs[i].completed_goal[k] = 0

            # parse trajectory according to initial start side
            if data_epi['pos_x'].iloc[0] > 50.:
                logs[i].trajs_side2.append(data_epi.loc[:, 'pos_x':'pos_z'].values)
            else:
                logs[i].trajs_side1.append(data_epi.loc[:, 'pos_x':'pos_z'].values)

    # compute and plot average trajectory for each side of the map
    for i in range(n_logs):
        # compute average traj
        avg_traj_side1 = np.array(logs[i].trajs_side1).mean(axis=0)
        avg_traj_side2 = np.array(logs[i].trajs_side2).mean(axis=0)

        # plot them (side 1)
        ax.plot(avg_traj_side1[:,0], avg_traj_side1[:,1], avg_traj_side1[:,2],
                    color=colors[i], alpha=0.95, label=args.exp_labels[i])
        # initial: o | final: x
        ax.plot(
            [avg_traj_side1[0,0]],
            [avg_traj_side1[0,1]],
            [avg_traj_side1[0,2]],
            'o', color=colors[i], alpha=0.95)
        ax.plot(
            [avg_traj_side1[-1,0]],
            [avg_traj_side1[-1,1]],
            [avg_traj_side1[-1,2]],
            'x', color=colors[i], alpha=0.95)

        # plot them (side 2)        
        ax.plot(avg_traj_side2[:,0], avg_traj_side2[:,1], avg_traj_side2[:,2],
                    color=colors[i], alpha=0.95)
        # initial: o | final: x
        ax.plot(
            [avg_traj_side2[0,0]],
            [avg_traj_side2[0,1]],
            [avg_traj_side2[0,2]],
            'o', color=colors[i], alpha=0.95)
        ax.plot(
            [avg_traj_side2[-1,0]],
            [avg_traj_side2[-1,1]],
            [avg_traj_side2[-1,2]],
            'x', color=colors[i], alpha=0.95)

        # compute distance metrics
        # use first log as reference
        if i == 0:
            ref_traj1 = avg_traj_side1
            ref_traj2 = avg_traj_side2
            logs[i].hausdorff_dists[0] = directed_hausdorff(ref_traj1, ref_traj1)[0] # side1
            logs[i].hausdorff_dists[1] = directed_hausdorff(ref_traj2, ref_traj2)[0] # side2

        else:
            # compare others to ref data
            data_traj1 = avg_traj_side1
            data_traj2 = avg_traj_side2
            logs[i].hausdorff_dists[0] = directed_hausdorff(ref_traj1, data_traj1)[0]
            logs[i].hausdorff_dists[1] = directed_hausdorff(ref_traj2, data_traj2)[0]

    # plot average final location and compute average metrics for each side of the map
    hausdorff_dists_means = np.zeros(n_logs)
    hausdorff_dists_stds = np.zeros(n_logs)

    for i in range(n_logs):
        # plot average final distance
        avg_final_coords = np.mean(logs[i].final_coords, axis=0)
        ax.plot([avg_final_coords[0]], [avg_final_coords[1]], [avg_final_coords[2]],
                'D', color=colors[i], alpha=0.5, markersize=10)

        # compute average of metrics
        hausdorff_dists_means[i] = np.mean(logs[i].hausdorff_dists)
        hausdorff_dists_stds[i] = np.std(logs[i].hausdorff_dists)

    ## GOAL --------------------------------------------------------------------
    # plot goal location
    ax.plot(
            [logs[0].goal_loc[0]],
            [logs[0].goal_loc[1]],
            [logs[0].goal_loc[2]],
            'D', color='k', alpha=0.5, label='Goal')

    # plot goal radius (sphere)
    r = logs[0].goal_radius
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = r * np.outer(np.cos(u), np.sin(v)) + logs[0].goal_loc[0]
    y = r * np.outer(np.sin(u), np.sin(v)) + logs[0].goal_loc[1]
    z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + logs[0].goal_loc[2]
    ax.plot_surface(x, y, z, color="w", edgecolor="k", alpha=.25, lw=0.1)

    # --------------------------------------------------------------------------


    # set labels and legends
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    plt.legend()

    # save figure
    plt.tight_layout()
    os.makedirs(f'figures/{args.fig_name}', exist_ok=True)
    fig_addr = f'figures/{args.fig_name}_trajs.png'
    plt.savefig(fig_addr, dpi=300, format='png')
    print('Saved', fig_addr)

    ## PLOT DISTANCES
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)  # (Hausdorff plot)
    ax4 = ax3.twinx()  # setup another y-axis on the same plot (MSE)

    x = np.arange(2)  # 2 metrics: hausdorff and mse
    width = 0.35  # the width of the bars

    # add labels
    ax3.set_title(f'Distance Metrics: "{args.task_name}"')
    ax3.set_ylabel('Task Completion (%)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Task Completion', 'Distance to Goal'])
    
    ax4.set_ylabel('Distance to Goal (m)')
    ax4.grid()

    # hausdorff distance in one axis
    # loop for each label
    for i in range(len(args.exp_labels)):
        # bar plot of hausdorff distance
        ax3.bar(
            x[0] + (2*i/3 - 2/3)*width, 100*np.mean(logs[i].completed_goal), width*2/3,
            color=colors[i], alpha=0.75, label=f'{args.exp_labels[i]}')
        
        # bar plot of task completion
        ax4.bar(
            x[1] + (2*i/3 - 2/3)*width, np.mean(logs[i].dist_to_goal), width*2/3,
            yerr=np.std(logs[i].dist_to_goal), color=colors[i], alpha=0.75, label=f'{args.exp_labels[i]}')

    # add bar legends
    ax3.legend(loc='best')

    # # fix aspect ratio of 3D plot    
    # axisEqual3D(ax)

    # save figure
    plt.tight_layout()
    fig_addr = f'figures/{args.fig_name}_metrics.png'
    plt.savefig(fig_addr, dpi=300, format='png')
    print('Saved', fig_addr)

    # display
    plt.show()
