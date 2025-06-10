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
    parser.add_argument('--backbone', type=str, default='ViT-B/32', help='CLIP backbone')
    parser.add_argument('--encoder', type=str, default='vision', choices=['vision', 'text', 'both'], help='Which encoder to use LoRA for')
    parser.add_argument('--rank', type=int, default=2, help='Rank for LoRA')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha for LoRA')
    parser.add_argument('--lora_path', type=str, required=True, help='Path to the trained LoRA weights')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images to annotate')
    parser.add_argument('--output_file', type=str, default='instances_default.json', help='Output JSON file path')
    parser.add_argument('--template', type=str, default='a photo of a {}', help='Text template for zero-shot classification')
    
    return parser.parse_args()


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


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load CLIP model
    print(f"Loading CLIP model with backbone {args.backbone}...")
    clip_model, preprocess = clip.load(args.backbone, device=device)
    
    # Apply LoRA to the model
    print(f"Applying LoRA with rank {args.rank}, alpha {args.alpha}...")
    args.save_path = args.lora_path  # Set save_path for loading
    list_lora_layers = apply_lora(args, clip_model)
    clip_model = clip_model.to(device)
    
    # Load trained LoRA weights
    print(f"Loading LoRA weights from {args.lora_path}...")
    load_lora(args, list_lora_layers)
    
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
    
    # Process images and generate predictions
    all_predictions = []
    all_scores = []
    all_bboxes = []
    
    for image_path in tqdm(image_paths):
        # Generate center crop bounding box
        bbox = generate_center_bbox(image_path)
        
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        preprocessed_image = preprocess(image).unsqueeze(0).to(device)
        
        # Get image features and predictions
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.float16):
                image_features = clip_model.encode_image(preprocessed_image)
            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = image_features @ text_features.t()
            
            # Get prediction and confidence score
            prediction = cosine_similarity.argmax(dim=1)[0].item()
            scores = cosine_similarity[0].softmax(dim=0)
            confidence = scores[prediction].item()
        
        all_predictions.append(prediction)
        all_scores.append(confidence)
        all_bboxes.append(bbox)
    
    # Create COCO annotations
    print("Creating COCO annotations...")
    coco_json = create_coco_annotations(image_paths, all_predictions, all_scores, all_bboxes)
    
    # Save to file
    output_path = args.output_file
    with open(output_path, 'w') as f:
        json.dump(coco_json, f, indent=2)
    
    print(f"Annotations saved to {output_path}")
    print(f"Processed {len(image_paths)} images")


if __name__ == "__main__":
    main()