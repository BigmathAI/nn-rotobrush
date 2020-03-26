tmux kill-server
tmux new-session -d -s SESS -n ID0000 \; send-keys "rlaunch --cpu=1 --gpu=8 --memory=10240 -- python3 main.py --num_gpus=8 --batch_size=16 --epoches=4000 --eval_nums=2 --lr=1.0e-04 --weight_ce=5.0e-01 --weight_decay=1.0e-04 --version=v2 --exp_id=ID0000"
tmux new-window -t SESS:1 -n ID0001 \; send-keys "rlaunch --cpu=1 --gpu=8 --memory=10240 -- python3 main.py --num_gpus=8 --batch_size=16 --epoches=4000 --eval_nums=2 --lr=1.0e-04 --weight_ce=5.0e-01 --weight_decay=1.0e-04 --version=v3 --exp_id=ID0001"
tmux new-window -t SESS:2 -n ID0002 \; send-keys "rlaunch --cpu=1 --gpu=8 --memory=10240 -- python3 main.py --num_gpus=8 --batch_size=16 --epoches=4000 --eval_nums=2 --lr=1.0e-04 --weight_ce=5.0e-01 --weight_decay=1.0e-04 --version=v4 --exp_id=ID0002"
