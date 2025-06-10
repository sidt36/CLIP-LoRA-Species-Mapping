import os
import torch
import clip
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

from utils import *
from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora
'''
python image_annotator.py \
  --lora_path ./path_to_trained_model \
  --image_dir ./your_images_folder \
  --output_file instances_default.json \
  --encoder both \
  --backbone bioclip

'''

def parse_args():
    parser = argparse.ArgumentParser(description='Generate annotations for images using CLIP-LoRA model')
    # Original parameters
    parser.add_argument('--backbone', type=str, default='ViT-B/32', help='CLIP backbone')
    parser.add_argument('--encoder', type=str, default='vision', choices=['vision', 'text', 'both'], help='Which encoder to use LoRA for')
    parser.add_argument('--rank', type=int, default=2, help='Rank for LoRA')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha for LoRA')
    parser.add_argument('--lora_path', type=str, required=True, help='Path to the trained LoRA weights')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images to annotate')
    parser.add_argument('--output_file', type=str, default='instances_default.json', help='Output JSON file path')
    parser.add_argument('--template', type=str, default='a photo of a {}', help='Text template for zero-shot classification')
    
    # Additional parameters needed by apply_lora and load_lora functions
    parser.add_argument('--position', type=str, default='all', help='Which layers to adapt (all, top, bottom, etc.)')
    parser.add_argument('--params', type=str, default='qkvo', help='Which attention parameters to adapt (q,k,v,o)')
    parser.add_argument('--dropout_rate', type=float, default=0.0, help='Dropout rate for LoRA')
    parser.add_argument('--dataset', type=str, default='custom', help='Dataset name (for loading weights)')
    parser.add_argument('--shots', type=int, default=-1, help='Number of shots (for loading weights)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (for loading weights)')
    parser.add_argument('--filename', type=str, default='lora_weights', help='Filename of LoRA weights')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
    
    # Ensure r is set correctly (rank is aliased to r in the apply_lora function)
    args = parser.parse_args()
    args.r = args.rank  # Set r equal to rank for compatibility
    args.save_path = args.lora_path  # Set save_path equal to lora_path for loading
    args.eval_only = True  # We're only doing evaluation/annotation
    
    return args


def get_species_categories():
    """
    Extract species categories from instances_default.json if available,
    or use a predefined list if not.
    """
    # Try to load categories from an existing COCO file
    try:
        with open('instances_default.json', 'r') as f:
            data = json.load(f)
            return data['categories']
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        # Fallback to a minimal default if file doesn't exist or is invalid
        return [
            {"id": 1, "name": "Unknown", "supercategory": ""}
        ]


def create_coco_annotations(images, predictions, confidence_scores, bboxes):
    """
    Create COCO-style annotations JSON file
    """
    categories = get_species_categories()
    
    # Create category name to ID mapping
    category_name_to_id = {category['name']: category['id'] for category in categories}
    
    # Initialize COCO JSON structure
    coco_json = {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {
            "contributor": "",
            "date_created": "",
            "description": "",
            "url": "",
            "version": "",
            "year": ""
        },
        "categories": categories,
        "images": [],
        "annotations": []
    }
    
    annotation_id = 1
    
    # Process each image and its predictions
    for idx, (image_path, pred, score, bbox) in enumerate(zip(images, predictions, confidence_scores, bboxes)):
        image_id = idx + 1
        
        # Get image dimensions
        with Image.open(image_path) as img:
            width, height = img.size
        
        # Add image information
        image_filename = os.path.basename(image_path)
        coco_json["images"].append({
            "id": image_id,
            "file_name": image_filename,
            "width": width,
            "height": height,
            "license": 0,
            "coco_url": "",
            "date_captured": ""
        })
        
        # Add annotation
        species_name = categories[pred]["name"]
        category_id = category_name_to_id.get(species_name, 1)  # Default to ID 1 if not found
        
        x, y, w, h = bbox
        
        coco_json["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [x, y, w, h],
            "area": w * h,
            "segmentation": [],
            "iscrowd": 0,
            "score": float(score)
        })
        
        annotation_id += 1
    
    return coco_json


def generate_center_bbox(image_path, horizontal_crop_ratio=0.5):
    """
    Generate a bounding box for the center 50% horizontal crop of the image
    Returns: [x, y, width, height] in COCO format
    """
    with Image.open(image_path) as img:
        width, height = img.size
    
    # Calculate the center 50% horizontal crop
    crop_width = int(width * horizontal_crop_ratio)
    x = int((width - crop_width) / 2)
    
    # Use the full height
    y = 0
    crop_height = height
    
    return [x, y, crop_width, crop_height]


def process_batch_of_images(clip_model, images, image_paths, text_features, device):
    """
    Process a batch of images with the CLIP-LoRA model
    Returns: predictions, confidence scores
    """
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.float16):
            image_features = clip_model.encode_image(images)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        cosine_similarity = image_features @ text_features.t()
        
        # Get predictions and confidence scores
        predictions = cosine_similarity.argmax(dim=1)
        scores = cosine_similarity.softmax(dim=1)
        confidence_scores = torch.gather(scores, 1, predictions.unsqueeze(1)).squeeze(1)
        
        return predictions.cpu().numpy(), confidence_scores.cpu().numpy()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load CLIP model
    print(f"Loading CLIP model with backbone {args.backbone}...")
    clip_model, preprocess = clip.load(args.backbone, device=device)
    
    try:
        # Apply LoRA to the model
        print(f"Applying LoRA with rank {args.rank}, alpha {args.alpha}, position {args.position}, params {args.params}...")
        list_lora_layers = apply_lora(args, clip_model)
        clip_model = clip_model.to(device)
        
        # Load trained LoRA weights
        print(f"Loading LoRA weights from {args.lora_path}...")
        try:
            load_lora(args, list_lora_layers)
        except (FileNotFoundError, ValueError) as e:
            # If the standard path fails, try a direct path
            print(f"Error loading with standard path: {e}")
            print("Trying direct path loading...")
            # Try to load the weights directly
            if os.path.isfile(args.lora_path):
                loaded_data = torch.load(args.lora_path)
                weights = loaded_data['weights']
                for i, layer in enumerate(list_lora_layers):
                    if i >= len(weights):
                        print(f"Warning: Not enough layers in weights file. Expected at least {i+1}, found {len(weights)}")
                        break
                    layer_weights = weights[f'layer_{i}']
                    if 'q' in args.params and 'q_proj' in layer_weights:
                        layer.q_proj.w_lora_A.data.copy_(layer_weights['q_proj']['w_lora_A'])
                        layer.q_proj.w_lora_B.data.copy_(layer_weights['q_proj']['w_lora_B'])
                    if 'k' in args.params and 'k_proj' in layer_weights:
                        layer.k_proj.w_lora_A.data.copy_(layer_weights['k_proj']['w_lora_A'])
                        layer.k_proj.w_lora_B.data.copy_(layer_weights['k_proj']['w_lora_B'])
                    if 'v' in args.params and 'v_proj' in layer_weights:
                        layer.v_proj.w_lora_A.data.copy_(layer_weights['v_proj']['w_lora_A'])
                        layer.v_proj.w_lora_B.data.copy_(layer_weights['v_proj']['w_lora_B'])
                    if 'o' in args.params and 'proj' in layer_weights:
                        layer.proj.w_lora_A.data.copy_(layer_weights['proj']['w_lora_A'])
                        layer.proj.w_lora_B.data.copy_(layer_weights['proj']['w_lora_B'])
                print(f"LoRA weights loaded directly from {args.lora_path}")
            else:
                print(f"Error: Could not find LoRA weights file at {args.lora_path}")
                return
    except Exception as e:
        print(f"Error during LoRA application: {e}")
        print("Continuing without LoRA weights...")
    
    # Get species categories
    categories = get_species_categories()
    classnames = [category['name'] for category in categories]
    
    # Prepare text features for zero-shot classification
    print("Preparing text features...")
    clip_model.eval()
    with torch.no_grad():
        template = args.template
        texts = [template.format(classname.replace('_', ' ')) for classname in classnames]
        with torch.amp.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.float16):
            texts = clip.tokenize(texts).to(device)
            class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
    
    # Get all images from the specified directory
    image_dir = Path(args.image_dir)
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_paths = [str(p) for p in image_dir.glob('**/*') if p.suffix.lower() in image_extensions]
    
    if not image_paths:
        print(f"No images found in {args.image_dir}")
        return
    
    print(f"Found {len(image_paths)} images to process")
    
    # Process images in batches
    image_data = []
    batch_size = args.batch_size
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        batch_bboxes = []
        valid_paths = []
        
        # Load and preprocess images
        for path in batch_paths:
            try:
                # Generate center crop bounding box
                bbox = generate_center_bbox(path)
                batch_bboxes.append(bbox)
                
                # Preprocess image
                image = Image.open(path).convert("RGB")
                batch_images.append(preprocess(image))
                valid_paths.append(path)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue
        
        if not batch_images:
            continue
            
        # Stack images into a batch tensor
        batch_tensor = torch.stack(batch_images).to(device)
        
        # Get predictions for batch
        predictions, scores = process_batch_of_images(clip_model, batch_tensor, valid_paths, text_features, device)
        
        # Store data for each image
        for j, (path, pred, score, bbox) in enumerate(zip(valid_paths, predictions, scores, batch_bboxes)):
            image_data.append({
                'image_path': path,
                'prediction': int(pred),
                'score': float(score),
                'bbox': bbox
            })
        
        # Print progress
        print(f"Processed {min(i + batch_size, len(image_paths))}/{len(image_paths)} images")
    
    # Create COCO annotations
    print("Creating COCO annotations...")
    coco_json = create_coco_annotations([d['image_path'] for d in image_data], 
                                       [d['prediction'] for d in image_data],
                                       [d['score'] for d in image_data],
                                       [d['bbox'] for d in image_data])
    
    # Save to file
    output_path = args.output_file
    with open(output_path, 'w') as f:
        json.dump(coco_json, f, indent=2)
    
    print(f"Annotations saved to {output_path}")
    print(f"Processed {len(image_data)} images")
    
    # Print statistics
    species_counts = {}
    for data in image_data:
        pred = data['prediction']
        if pred < len(categories):
            species_name = categories[pred]["name"]
            if species_name not in species_counts:
                species_counts[species_name] = 0
            species_counts[species_name] += 1
    
    print("\nSpecies distribution:")
    for species, count in sorted(species_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{species}: {count} images")
    
    if len(species_counts) > 10:
        print(f"... and {len(species_counts) - 10} more species")


if __name__ == "__main__":
    main()