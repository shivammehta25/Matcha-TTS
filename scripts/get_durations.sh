#!/bin/bash

echo "Starting script"

echo "Getting LJ Speech durations"
python matcha/utils/get_durations_from_trained_model.py -i ljspeech.yaml -c logs/train/lj_det/runs/2024-01-12_12-05-00/checkpoints/last.ckpt -f

echo "Getting TSG2 durations"
python matcha/utils/get_durations_from_trained_model.py -i tsg2.yaml -c logs/train/tsg2_det_dur/runs/2024-01-05_12-33-35/checkpoints/last.ckpt -f

echo "Getting Joe Spont durations"
python matcha/utils/get_durations_from_trained_model.py -i joe_spont_only.yaml -c logs/train/joe_det_dur/runs/2024-02-20_14-01-01/checkpoints/last.ckpt -f

echo "Getting Ryan durations"
python matcha/utils/get_durations_from_trained_model.py -i ryan.yaml -c logs/train/matcha_ryan_det/runs/2024-02-26_09-28-09/checkpoints/last.ckpt -f