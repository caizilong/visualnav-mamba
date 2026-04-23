#!/bin/bash

session_name="teleop_locobot_$(date +%s)"
tmux new-session -d -s $session_name

tmux selectp -t 0
tmux splitw -v -p 50

tmux select-pane -t 0
tmux send-keys "roslaunch vint_locobot.launch" Enter

tmux select-pane -t 1
tmux send-keys "conda activate vint_deployment" Enter
tmux send-keys "python joy_teleop.py" Enter

tmux -2 attach-session -t $session_name
