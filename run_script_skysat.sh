python main.py \
    --root_path /app/DATASET/Rajasthan_Zones_v4_resampled/Rajasthan_Zones_v4_resampled \
    --dataset skysat \
    --seed 1 \
    --shots 1 \
    --backbone "bio_clip" \
    --n_iters 50 \
    --save_path "/app/DATASET/bioclip_models/"

python main.py \
    --root_path /app/DATASET/Rajasthan_Zones_v4_resampled/Rajasthan_Zones_v4_resampled \
    --dataset skysat \
    --seed 1 \
    --shots 10 \
    --backbone "bio_clip" \
    --n_iters 50 \
    --save_path "/app/DATASET/bioclip_models/"

python main.py \
    --root_path /app/DATASET/Rajasthan_Zones_v4_resampled/Rajasthan_Zones_v4_resampled \
    --dataset skysat \
    --seed 1 \
    --shots 100 \
    --backbone "bio_clip" \
    --n_iters 50 \
    --save_path "/app/DATASET/bioclip_models/"

python main.py \
    --root_path /app/DATASET/Rajasthan_Zones_v4_resampled/Rajasthan_Zones_v4_resampled \
    --dataset skysat \
    --seed 1 \
    --shots 100000000000000 \
    --backbone "bio_clip" \
    --n_iters 50 \
    --save_path "/app/DATASET/bioclip_models/"
