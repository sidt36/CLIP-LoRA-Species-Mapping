python main.py \
    --root_path /app/DATASET/Rajasthan_Zones_v3_resampled/ \
    --dataset gsv \
    --seed 1 \
    --shots 100000000000000 \
    --backbone "bio_clip" \
    --n_iters 50 \
    --save_path "/app/DATASET/bioclip_models_small/"
