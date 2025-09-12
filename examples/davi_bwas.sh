### Train a heuristic function with deep approximate value iteration (DAVI) and solve problems with batch weighted A* search

# Rubik's cube
python run_davi.py --env cube3 --step_max 30 --nnet_dir models/cube3/ --batch_size 10000 --up_itrs 5000 --up_procs 48 --up_step_max 30 --rb 10