""" data_quality.py
Plots gaze and sampling frequency data for all collected data to check data quality.
"""
import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# address of collected data
n_demonstration_sets = 20
root_addr = './airsim_data/'
dataset_addrs = [
    'truck_mountains',
    'moving_truck_mountains']

# loop for all datasets
for dataset_addr in dataset_addrs:
    # loop for all sets
    for i in range(1, n_demonstration_sets+1):
        # load log file
        log_addr = f'{root_addr}{dataset_addr}{i}/Fly_to_the_nearest_truck/log.csv'
        log_data = pd.read_csv(log_addr)
        log_data_epi_group = log_data.groupby(['epi_num'])

        # separate plots for each episode
        for epi in log_data.epi_num.unique():
            # grab episode data
            epi_data = log_data_epi_group.get_group(epi)

            # plot gaze
            f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
            plt.suptitle(f'Gaze Data: {dataset_addr}{i}, Episode {epi}')

            ax1.plot(epi_data['step_num'], epi_data['gaze_x'])
            ax1.set_ylabel('"X" Gaze Value')
            ax1.set_ylim([0,1])     
            
            ax2.plot(epi_data['step_num'], epi_data['gaze_y'])
            ax2.set_ylabel('"Y" Gaze Value')
            ax2.set_ylim([0,1])

            # plot sampling frequency
            time_values = epi_data['timestamp'].values/1e9
            freq_values = 1/(time_values[1:]-time_values[:-1])
            ax3.plot(epi_data['step_num'][:-1]+1, freq_values)
            ax3.set_ylabel('Sampling Freq. (Hz)')
            ax3.set_xlabel('Time Steps')
            ax3.set_ylim([0,12])     

            ax3.text(0.9, 0.45, f'Max Hz: {np.max(freq_values):.1f}', horizontalalignment='center',verticalalignment='center', transform=ax3.transAxes)
            ax3.text(0.9, 0.3, f'Min Hz: {np.min(freq_values):.1f}', horizontalalignment='center',verticalalignment='center', transform=ax3.transAxes)
            ax3.text(0.9, 0.15, f'Mean Hz: {np.mean(freq_values):.1f}', horizontalalignment='center',verticalalignment='center', transform=ax3.transAxes)

            # save figure to disk
            figname = f'mlruns/{dataset_addr}{i}_epi{epi}.png'
            # plt.show()
            plt.savefig(figname, format='png', dpi=300)
            print('Saved', figname)
            plt.close()