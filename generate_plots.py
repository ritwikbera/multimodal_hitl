import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd 
import mlflow
from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType
import argparse
from scipy.stats import ttest_ind, ttest_rel, ttest_ind_from_stats, wilcoxon


parser = argparse.ArgumentParser(description='Plotting source')
parser.add_argument('-c','--use_csv', action='store_true')
parser.add_argument('--num_runs', type=int, default=20)
parser.add_argument('-f','--filename', type=str, default='runs.csv')
args = parser.parse_args()

sns.set(style='darkgrid')
sns.set_context("paper")
sns.set_palette(sns.color_palette()) # colorblind palette

metrics = ['task_completion_rate', 'avg_spl', 'collision_rate']
metric_names = ['Task Completion Rate', 'Average SPL', 'Collision Rate']

causal_var = 'training_fraction'
# causal_var = 'split_ratio'

if args.use_csv:
    # if csv file available else fetch experiment data directly from tracking server
    df = pd.read_csv(args.filename)

else:
    # fetching data from tracking server
    exp_id = '1'
    num_runs = args.num_runs
    query = "metrics.num_collisions >= 0 and attributes.status = 'FINISHED'"

    mlflow.set_tracking_uri('file:/home/deathstar/Downloads/mlruns')

    runs = mlflow.search_runs(
        experiment_ids=[exp_id],
        filter_string=query,
        run_view_type=ViewType.ALL,
        max_results=num_runs,
        order_by=["attribute.start_time DESC"]
    )

    runs.rename(columns={
        'params.weights':'weights',
        'params.training_fraction':'training_fraction',
        'params.rnd_seed':'rnd_seed',
        'metrics.avg_spl':'avg_spl',
        'metrics.num_collisions':'num_collisions',
        'metrics.task_completion_rate':'task_completion_rate',
        'metrics.avg_task_length':'avg_task_length'
        }, inplace=True)

    df = runs

# clip high avg task length values due to zero completions
df['avg_task_length'].clip(upper=100, inplace=True)

# convert num_collisions to collision rate
total_rollouts = 100
df['collision_rate'] = df['num_collisions']*100/total_rollouts

df['task_completion_rate'] *= 100

# use model names instead of loss weights
df = df.assign(model_type=['Vanilla BC' if str(df['weights'].iloc[i]) == '[0, 1]' else 'Gaze BC' for i in range(len(df))])

fig, axs = plt.subplots(ncols=2, nrows=2, sharex=False)
axs = [axs[0][0], axs[0][1], axs[1][0]]

for i, metric in enumerate(metrics):
    
    ax = sns.barplot(x=causal_var, y=metric, hue='model_type', 
        dodge=True,
        errcolor='k', 
        # capsize=0.2,
        ci='sd', estimator=np.mean,
        data=df, ax=axs[i])

    # set labels
    ax.set(ylabel=metric_names[i], xlabel=None)
    ax.set_xticks([])

# get the last axis from the subplots
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center')

# remove legends from subplots
for ax in axs:
    ax.get_legend().set_visible(False)

fig.tight_layout()
# plt.show()

df_gaze = df.groupby('model_type').get_group('Gaze BC')[metrics].mean()
df_vanilla = df.groupby('model_type').get_group('Vanilla BC')[metrics].mean()

round_off = lambda x: np.around(x, decimals=2)

pct_inc = dict()
for metric in metrics:
    if metric in ['task_completion_rate', 'collision_rate']:
        pct_inc[metric] = round_off((df_gaze[metric] - df_vanilla[metric]))  # /df_vanilla[metric], only pc point increase
    else:
        pct_inc[metric] = round_off((df_gaze[metric] - df_vanilla[metric])*100/df_vanilla[metric])

print('Percentage Point Increments (Gaze BC vs Vanilla BC', pct_inc)

for metric in metrics:
    a = df.groupby('model_type').get_group('Gaze BC').sort_values(by='rnd_seed')[metric].to_numpy()
    b = df.groupby('model_type').get_group('Vanilla BC').sort_values(by='rnd_seed')[metric].to_numpy()
    num_seeds = len(df['rnd_seed'].unique())
    
    print(metric)
    print(f'Gaze BC: mean {round_off(a.mean())}, standard error {round_off(a.std()/np.sqrt(num_seeds))}')
    print(f'Vanilla BC: mean {round_off(b.mean())}, standard error {round_off(b.std()/np.sqrt(num_seeds))}')
    
    if metric in ['task_completion_rate']:
        t2, p2 = ttest_ind_from_stats(a.mean(),a.std(),num_seeds,b.mean(),b.std(),num_seeds,equal_var=False)
        print("t = " + str(t2))
        print("p = " + str(p2))

        t2, p2 = wilcoxon(a,b)
        print("t = " + str(t2))
        print("p = " + str(p2))



