# db
python train.py --eval -s /home/yixuan/dataset/db/playroom --voxel_size 0.005 --ratio 1 --appearance_dim 0 --update_init_factor 16 -m outputs-ori-A/db/playroom/log/1
python train.py --eval -s /home/yixuan/dataset/db/drjohnson --voxel_size 0.005 --ratio 1 --appearance_dim 0 --update_init_factor 16 -m outputs-ori-A/db/drjohnson/log/1

# tat
python train.py --eval -s /home/yixuan/dataset/tat/train --voxel_size 0.01 --ratio 1 --appearance_dim 0 --update_init_factor 16 -m outputs-ori-A/tat/train/log/1
python train.py --eval -s /home/yixuan/dataset/tat/truck --voxel_size 0.01 --ratio 1 --appearance_dim 0 --update_init_factor 16 -m outputs-ori-A/tat/truck/log/1

# m360in
python train.py --eval -s /home/yixuan/dataset/m360in/bonsai --voxel_size 0.001 --ratio 1 --appearance_dim 0 --update_init_factor 16 -m outputs-ori-A/m360in/bonsai/log/1
python train.py --eval -s /home/yixuan/dataset/m360in/counter --voxel_size 0.001 --ratio 1 --appearance_dim 0 --update_init_factor 16 -m outputs-ori-A/m360in/counter/log/1
python train.py --eval -s /home/yixuan/dataset/m360in/kitchen --voxel_size 0.001 --ratio 1 --appearance_dim 0 --update_init_factor 16 -m outputs-ori-A/m360in/kitchen/log/1
python train.py --eval -s /home/yixuan/dataset/m360in/room --voxel_size 0.001 --ratio 1 --appearance_dim 0 --update_init_factor 16 -m outputs-ori-A/m360in/room/log/1

# m360out
python train.py --eval -s /home/yixuan/dataset/m360out/bicycle --voxel_size 0.001 --ratio 1 --appearance_dim 0 --update_init_factor 16 -m outputs-ori-A/m360out/bicycle/log/1
python train.py --eval -s /home/yixuan/dataset/m360out/flowers --voxel_size 0.001 --ratio 1 --appearance_dim 0 --update_init_factor 16 -m outputs-ori-A/m360out/flowers/log/1
python train.py --eval -s /home/yixuan/dataset/m360out/garden --voxel_size 0.001 --ratio 1 --appearance_dim 0 --update_init_factor 16 -m outputs-ori-A/m360out/garden/log/1
python train.py --eval -s /home/yixuan/dataset/m360out/stump --voxel_size 0.001 --ratio 1 --appearance_dim 0 --update_init_factor 16 -m outputs-ori-A/m360out/stump/log/1
python train.py --eval -s /home/yixuan/dataset/m360out/treehill --voxel_size 0.001 --ratio 1 --appearance_dim 0 --update_init_factor 16 -m outputs-ori-A/m360out/treehill/log/1