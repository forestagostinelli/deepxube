### Train a heuristic function with deep approximate value iteration (DAVI) and solve problems with batch weighted A* pathfinding

# Rubik's cube
python run_davi.py --env cube3 --step_max 100 --nnet_dir models/cube3/ --batch_size 10000 --up_itrs 1000 --up_gen_itrs 1000 --up_procs 48 --up_search_itrs 100 --rb 1

python run_bwas.py --env cube3 --insts data/cube3/canon.pkl --heur models/cube3/current.pt --batch_size 1000 --weight 0.4 --results results/cube3/