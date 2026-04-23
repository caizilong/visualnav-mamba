#!/bin/bash

session_name="topomap_locobot_$(date +%s)"
tmux new-session -d -s $session_name

tmux selectp -t 0
tmux splitw -v -p 50
tmux selectp -t 0
tmux splitw -h -p 50

tmux select-pane -t 0
tmux send-keys "roscore" Enter

tmux select-pane -t 1
tmux send-keys "conda activate vint_deployment" Enter
tmux send-keys "python create_topomap.py --dt 1 --dir $1" Enter

tmux select-pane -t 2
tmux send-keys "mkdir -p ../topomaps/bags" Enter
tmux send-keys "cd ../topomaps/bags" Enter
tmux send-keys "rosbag play -r 1.5 $2"

tmux -2 attach-session -t $session_name
