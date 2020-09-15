import pandas as pd 
import numpy as np 
import itertools
import random

df = pd.read_csv('target_pos.csv')

num_episodes = 100
dfnew = pd.DataFrame(index=range(num_episodes), columns=['pos_x','pos_y','pos_z','target_pos_x','target_pos_y','target_pos_z'])

x = range(len(df))

aList =[]
for pair in itertools.combinations(x,2):
    aList.append(list(df[['target_pos_x','target_pos_y','target_pos_z']].iloc[pair[0]]) + list(df[['target_pos_x','target_pos_y','target_pos_z']].iloc[pair[1]]))

print(aList)

indices = np.random.choice(range(len(aList)), num_episodes)

for i, index in enumerate(indices):
    dfnew.iloc[i] = aList[index]

# add random yaw to be used by the drone
dfnew['yaw'] = np.deg2rad(np.random.randint(low=0, high=360, size=num_episodes))

dfnew['epi_id'] = dfnew.index
dfnew.to_csv('locs_config.csv')
print(dfnew)
