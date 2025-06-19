python image_annotater.py \
  --lora_path /app/DATASET/bioclip_models_small/bio_clip/gsv/100000000000000shots/seed1/lora_weights.pt \
  --image_dir /app/DATASET/Rajasthan_Zones_v4_resampled/images \
  --output_file instances_default_new.json \
  --encoder both \
  --backbone bio_clip
