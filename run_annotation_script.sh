python image_annotater.py \
  --lora_path /app/CLIP-LoRA-Species-Mapping/models/bio_clip/google_cc/1000000shots/seed1/lora_weights.pt \
  --image_dir /app/DATASET/Rajasthan_Zones_v4_resampled/images \
  --output_file instances_default_new.json \
  --encoder both \
  --backbone bio_clip
