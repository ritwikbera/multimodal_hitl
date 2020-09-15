import pandas as pd 
import numpy as np 

def compute_avg_task_length(csv_file):
    data = pd.read_csv(csv_file)
    data.rename(columns=lambda x: x.strip(), inplace=True)

    gb = data.groupby('epi_num')

    path_lengths = []
    spl = []  # Success Weighted by Path Length (SPL)

    for episode_num in data.epi_num.unique():
        episode = gb.get_group(episode_num)

        if episode['task_done'].iloc[-1] == False:
            continue

        x = episode['pos_x'].values
        y = episode['pos_y'].values
        z = episode['pos_z'].values

        dx = x[1:] - x[:-1]
        dy = y[1:] - y[:-1]
        dz = z[1:] - z[:-1]

        step_size = np.sqrt(dx**2+dy**2+dz**2)

        path_length = np.sum(step_size)
        path_lengths.append(path_length)

        # compute Success Weighted by Path Length (SPL)
        shortest_path_length = np.sqrt(
            (episode['pos_x'].values[0]-episode['pos_x'].values[-1])**2 +
            (episode['pos_y'].values[0]-episode['pos_y'].values[-1])**2 +
            (episode['pos_z'].values[0]-episode['pos_z'].values[-1])**2)
        spl.append(shortest_path_length/path_length)


    # if all tasks were failures, path_lengths will be empty
    if len(path_lengths) == 0:
        avg_task_length = 1e9
        avg_spl = 0.
    else:
        avg_task_length = sum(path_lengths)/len(path_lengths)
        avg_spl = sum(spl)/len(spl)

    return avg_task_length, avg_spl

def num_collisions(csv_file):
    data = pd.read_csv(csv_file)
    data.rename(columns=lambda x: x.strip(), inplace=True)
    return len(data[data['collision_status'] == True])


if __name__=='__main__':
    data = pd.DataFrame({
        'epi_num':[1,1],
        'task_done':[False,True],
        'pos_x':[1,2],
        'pos_y':[2,3],
        'pos_z':[0,1],
        'collision_status':[False,True]
        })
    data.to_csv('metric_test.csv')
    print(avg_task_length('metric_test.csv'))
    print(num_collisions('metric_test.csv'))
