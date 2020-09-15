#!/bin/bash
# Run experiments for multiple random seeds
for i in {1..3}
do
    python train.py --run_tests
done