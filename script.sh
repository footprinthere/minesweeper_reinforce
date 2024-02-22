PYTHONPATH=`pwd` python main.py \
    --board_rows 10 \
    --board_cols 10 \
    --n_mines 20 \
    --batch_size 8 \
    --flat_action \
    --n_episodes 50 \
    --print_board \
    --n_channels 128 \
    --model_depth 4
