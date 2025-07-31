#!/bin/bash

# Create a new tmux session
tmux new-session -d -s mysession

# Split the first pane horizontally
tmux split-window -h
# Split the left pane vertically
tmux select-pane -t 0
tmux split-window -v
# Split the right pane vertically
tmux select-pane -t 2
tmux split-window -v

# Go back to the first pane and create another horizontal split for a second grid
tmux select-pane -t 0
tmux split-window -h

# Go to the second pane and split it horizontally to create a balanced 6-pane grid
tmux select-pane -t 2
tmux split-window -h

# Run Python scripts in each pane
tmux select-pane -t 0
# tmux send-keys "conda activate LEGS" C-m
# tmux send-keys "python spot_util/pointcloud.py" C-m
# activate image stream
tmux send-keys "conda activate LEGS" C-m

tmux send-keys "python object_detection/update_image_tmp.py" C-m

# tmux send-keys "python ~/Desktop/update_image.py" C-m


tmux select-pane -t 1
# enable audio listener
tmux send-keys "conda activate LEGS" C-m
tmux send-keys "python speech_recog/speech_recog.py" C-m

tmux select-pane -t 2
# run object detection
tmux send-keys "conda activate LEGS" C-m
# tmux send-keys "python3 object_detection/SoM_system.py" C-m
tmux send-keys "python3 receive_instruction.py && python3 receive_observation.py" C-m

tmux select-pane -t 3
# run object detection
tmux send-keys "conda activate LEGS" C-m
# tmux send-keys "python3 object_detection/SoM_system.py" C-m
# tmux send-keys "python3 receive_instruction.py && python3 object_detection/detection_system.py" C-m
# tmux send-keys "python3 receive_observation.py" C-m

tmux select-pane -t 4
# run gesture detection
tmux send-keys "conda activate LEGS" C-m
# tmux send-keys "python3 pose_detection/gesture_system.py" C-m

tmux select-pane -t 5
# 
tmux send-keys "conda activate LEGS" C-m
tmux send-keys "python3 plots/plot_live_prob_update.py" C-m

tmux select-pane -t 6
tmux send-keys "conda activate LEGS" C-m
tmux send-keys "python3 belief_update.py" C-m

# Attach to the tmux session
tmux attach-session -t mysession