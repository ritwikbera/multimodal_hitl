#!/bin/bash
# Make all videos from collected demonstrations
for i in {1..20}
do
    python viz_airsim_log.py --exp_name 'moving_truck_mountains'$i'/Fly_to_the_nearest_truck' --plot_gaze --demo_data
done