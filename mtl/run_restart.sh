#!/bin/bash
# Run experiments restarting AirSim at every time
N_EPOCHS=20

# Repeat for each seed
for i in {0..2}
do
    echo '[***] ---------------'
    echo '[***] Running seed '$i

    # Repeat for each condition: gazebc or vanillabc
    for GAZE_CONDITION in gazebc vanillabc
    do
        echo '[**] Running condition: '$GAZE_CONDITION

        # Repeat for each training_fraction
        # for TRAINING_FRACTION in {0.9,0.8,0.6}
        for TRAINING_FRACTION in {0.9,}
        do
            # Start AirSim (background)
            echo '[*] Starting AirSim in the background'
            airsim_static &

            # Run experiment
            echo '[*] Running train.py for training_fraction '$TRAINING_FRACTION
            # Check if need to resume or start new (run exps with --run_tests)
            if [ $TRAINING_FRACTION = 0.9 ]
            then
                echo '[*] Starting experiments with new episode IDs'
                python train.py --gaze_condition $GAZE_CONDITION --training_fraction $TRAINING_FRACTION --rnd_seed $i --n_epochs $N_EPOCHS --run_tests
            else
                echo '[*] Resume experiment with previous episode IDs'
                python train.py --gaze_condition $GAZE_CONDITION --training_fraction $TRAINING_FRACTION --resume_experiment --rnd_seed $i --n_epochs $N_EPOCHS --run_tests
            fi

            # Find and kill AirSim PIDs
            echo $'[X] Killing AirSim\n'
            for pid in $(pgrep ARL_Test_Small)
            do
                kill -9 $pid
            done
            sleep 3            
        done
    done
done
echo '--- ALL EXPERIMENTS COMPLETED ---'