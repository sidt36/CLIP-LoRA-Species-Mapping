
#!/bin/bash

# 0.1 subsample
python main.py \
    --root_path /app/DATASET/Rajasthan_Zones_v4_resampled/Rajasthan_Zones_v4_resampled \
    --dataset gsv \
    --seed 1 \
    --shots 100000000000000 \
    --backbone "bio_clip" \
    --n_iters 50 \
    --subsample_ratio 0.1 \
    --save_path "/app/DATASET/bioclip_models/subsample_0.1"

# Run annotation for 0.1 subsample
python image_annotater.py \
  --lora_path /app/DATASET/bioclip_models/subsample_0.1/bio_clip/gsv/100000000000000shots/seed1/lora_weights.pt \
  --image_dir /app/DATASET/Rajasthan_Zones_v4_resampled/images \
  --output_file instances_subsample_0.1.json \
  --encoder both \
  --backbone bio_clip

# 0.2 subsample
python main.py \
    --root_path /app/DATASET/Rajasthan_Zones_v4_resampled/Rajasthan_Zones_v4_resampled \
    --dataset gsv \
    --seed 1 \
    --shots 100000000000000 \
    --backbone "bio_clip" \
    --n_iters 50 \
    --subsample_ratio 0.2 \
    --save_path "/app/DATASET/bioclip_models/subsample_0.2"

# Run annotation for 0.2 subsample
python image_annotater.py \
  --lora_path /app/DATASET/bioclip_models/subsample_0.2/bio_clip/gsv/100000000000000shots/seed1/lora_weights.pt \
  --image_dir /app/DATASET/Rajasthan_Zones_v4_resampled/images \
  --output_file instances_subsample_0.2.json \
  --encoder both \
  --backbone bio_clip

# 0.3 subsample
python main.py \
    --root_path /app/DATASET/Rajasthan_Zones_v4_resampled/Rajasthan_Zones_v4_resampled \
    --dataset gsv \
    --seed 1 \
    --shots 100000000000000 \
    --backbone "bio_clip" \
    --n_iters 50 \
    --subsample_ratio 0.3 \
    --save_path "/app/DATASET/bioclip_models/subsample_0.3"

# Run annotation for 0.3 subsample
python image_annotater.py \
  --lora_path /app/DATASET/bioclip_models/subsample_0.3/bio_clip/gsv/100000000000000shots/seed1/lora_weights.pt \
  --image_dir /app/DATASET/Rajasthan_Zones_v4_resampled/images \
  --output_file instances_subsample_0.3.json \
  --encoder both \
  --backbone bio_clip

# 0.4 subsample
python main.py \
    --root_path /app/DATASET/Rajasthan_Zones_v4_resampled/Rajasthan_Zones_v4_resampled \
    --dataset gsv \
    --seed 1 \
    --shots 100000000000000 \
    --backbone "bio_clip" \
    --n_iters 50 \
    --subsample_ratio 0.4 \
    --save_path "/app/DATASET/bioclip_models/subsample_0.4"

# Run annotation for 0.4 subsample
python image_annotater.py \
  --lora_path /app/DATASET/bioclip_models/subsample_0.4/bio_clip/gsv/100000000000000shots/seed1/lora_weights.pt \
  --image_dir /app/DATASET/Rajasthan_Zones_v4_resampled/images \
  --output_file instances_subsample_0.4.json \
  --encoder both \
  --backbone bio_clip

# 0.5 subsample
python main.py \
    --root_path /app/DATASET/Rajasthan_Zones_v4_resampled/Rajasthan_Zones_v4_resampled \
    --dataset gsv \
    --seed 1 \
    --shots 100000000000000 \
    --backbone "bio_clip" \
    --n_iters 50 \
    --subsample_ratio 0.5 \
    --save_path "/app/DATASET/bioclip_models/subsample_0.5"

# Run annotation for 0.5 subsample
python image_annotater.py \
  --lora_path /app/DATASET/bioclip_models/subsample_0.5/bio_clip/gsv/100000000000000shots/seed1/lora_weights.pt \
  --image_dir /app/DATASET/Rajasthan_Zones_v4_resampled/images \
  --output_file instances_subsample_0.5.json \
  --encoder both \
  --backbone bio_clip

# 0.6 subsample
python main.py \
    --root_path /app/DATASET/Rajasthan_Zones_v4_resampled/Rajasthan_Zones_v4_resampled \
    --dataset gsv \
    --seed 1 \
    --shots 100000000000000 \
    --backbone "bio_clip" \
    --n_iters 50 \
    --subsample_ratio 0.6 \
    --save_path "/app/DATASET/bioclip_models/subsample_0.6"

# Run annotation for 0.6 subsample
python image_annotater.py \
  --lora_path /app/DATASET/bioclip_models/subsample_0.6/bio_clip/gsv/100000000000000shots/seed1/lora_weights.pt \
  --image_dir /app/DATASET/Rajasthan_Zones_v4_resampled/images \
  --output_file instances_subsample_0.6.json \
  --encoder both \
  --backbone bio_clip

# 0.7 subsample
python main.py \
    --root_path /app/DATASET/Rajasthan_Zones_v4_resampled/Rajasthan_Zones_v4_resampled \
    --dataset gsv \
    --seed 1 \
    --shots 100000000000000 \
    --backbone "bio_clip" \
    --n_iters 50 \
    --subsample_ratio 0.7 \
    --save_path "/app/DATASET/bioclip_models/subsample_0.7"

# Run annotation for 0.7 subsample
python image_annotater.py \
  --lora_path /app/DATASET/bioclip_models/subsample_0.7/bio_clip/gsv/100000000000000shots/seed1/lora_weights.pt \
  --image_dir /app/DATASET/Rajasthan_Zones_v4_resampled/images \
  --output_file instances_subsample_0.7.json \
  --encoder both \
  --backbone bio_clip

# 0.8 subsample
python main.py \
    --root_path /app/DATASET/Rajasthan_Zones_v4_resampled/Rajasthan_Zones_v4_resampled \
    --dataset gsv \
    --seed 1 \
    --shots 100000000000000 \
    --backbone "bio_clip" \
    --n_iters 50 \
    --subsample_ratio 0.8 \
    --save_path "/app/DATASET/bioclip_models/subsample_0.8"

# Run annotation for 0.8 subsample
python image_annotater.py \
  --lora_path /app/DATASET/bioclip_models/subsample_0.8/bio_clip/gsv/100000000000000shots/seed1/lora_weights.pt \
  --image_dir /app/DATASET/Rajasthan_Zones_v4_resampled/images \
  --output_file instances_subsample_0.8.json \
  --encoder both \
  --backbone bio_clip

# 0.9 subsample
python main.py \
    --root_path /app/DATASET/Rajasthan_Zones_v4_resampled/Rajasthan_Zones_v4_resampled \
    --dataset gsv \
    --seed 1 \
    --shots 100000000000000 \
    --backbone "bio_clip" \
    --n_iters 50 \
    --subsample_ratio 0.9 \
    --save_path "/app/DATASET/bioclip_models/subsample_0.9"

# Run annotation for 0.9 subsample
python image_annotater.py \
  --lora_path /app/DATASET/bioclip_models/subsample_0.9/bio_clip/gsv/100000000000000shots/seed1/lora_weights.pt \
  --image_dir /app/DATASET/Rajasthan_Zones_v4_resampled/images \
  --output_file instances_subsample_0.9.json \
  --encoder both \
  --backbone bio_clip

# Full dataset (1.0)
python main.py \
    --root_path /app/DATASET/Rajasthan_Zones_v4_resampled/Rajasthan_Zones_v4_resampled \
    --dataset gsv \
    --seed 1 \
    --shots 100000000000000 \
    --backbone "bio_clip" \
    --n_iters 50 \
    --save_path "/app/DATASET/bioclip_models/full_dataset"

# Run annotation for full dataset
python image_annotater.py \
  --lora_path /app/DATASET/bioclip_models/full_dataset/bio_clip/gsv/100000000000000shots/seed1/lora_weights.pt \
  --image_dir /app/DATASET/Rajasthan_Zones_v4_resampled/images \
  --output_file instances_full_dataset.json \
  --encoder both \
  --backbone bio_clip
