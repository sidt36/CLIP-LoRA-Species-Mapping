import os
import torch
import argparse
from loralib.utils import apply_lora, load_lora
import clip

def convert_lora_to_full_model(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load base model
    print(f"Loading base {args.backbone} CLIP model...")
    clip_model, _ = clip.load(args.backbone, device=device)
    
    # Apply LoRA layers to prepare for loading weights
    print(f"Applying LoRA architecture with rank={args.r}...")
    list_lora_layers = apply_lora(args, clip_model)
    
    # Load LoRA weights
    print(f"Loading LoRA weights from {args.lora_path}...")
    load_lora(args, list_lora_layers)
    
    # Save full model checkpoint
    output_path = args.output_path
    if not output_path.endswith('.pth'):
        output_path += '.pth'
    
    print(f"Saving full model checkpoint to {output_path}...")
    torch.save(clip_model.state_dict(), output_path)
    print("Conversion complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LoRA checkpoint to full model checkpoint")
    parser.add_argument('--lora_path', type=str, required=True, help='Path to the LoRA .pt file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the full model checkpoint')
    parser.add_argument('--backbone', type=str, default='ViT-B/32', help='CLIP backbone (ViT-B/32, ViT-B/16, etc.)')
    
    # LoRA parameters - these should match those used during training
    parser.add_argument('--r', type=int, default=4, help='LoRA rank')
    parser.add_argument('--alpha', type=float, default=1.0, help='LoRA alpha scaling')
    parser.add_argument('--encoder', type=str, default='vision', 
                        choices=['vision', 'text', 'both'], help='Which encoder to apply LoRA to')
    parser.add_argument('--params', type=str, default='qkvo', help='Which attention params to apply LoRA to (q, k, v, o)')
    parser.add_argument('--position', type=str, default='all', help='Which layers to apply LoRA to')
    
    # Additional parameters needed for the load_lora function
    parser.add_argument('--save_path', type=str, default='./checkpoints', help='Base path for checkpoints')
    parser.add_argument('--dataset', type=str, default='custom', help='Dataset name')
    parser.add_argument('--shots', type=int, default=16, help='Number of shots')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--filename', type=str, default='lora_weights', help='Filename for LoRA weights')
    parser.add_argument('--dropout_rate', type=float, default=0.0, help='LoRA dropout rate')
    
    args = parser.parse_args()
    
    # If args.lora_path is a full path to the .pt file, override other parameters
    if os.path.isfile(args.lora_path):
        # Extract components from the path
        try:
            # Load the LoRA file to get its metadata
            lora_data = torch.load(args.lora_path, map_location="cpu")
            if isinstance(lora_data, dict) and 'metadata' in lora_data:
                metadata = lora_data['metadata']
                print(f"Found metadata in LoRA file, updating parameters...")
                
                # Update args with metadata values
                args.r = metadata.get('r', args.r)
                args.alpha = metadata.get('alpha', args.alpha)
                args.encoder = metadata.get('encoder', args.encoder)
                args.params = metadata.get('params', args.params)
                args.position = metadata.get('position', args.position)
                
                print(f"Using parameters from metadata: r={args.r}, alpha={args.alpha}, "
                      f"encoder={args.encoder}, params={args.params}, position={args.position}")
        except Exception as e:
            print(f"Error reading metadata from LoRA file: {e}")
            print("Will use command line parameters instead.")
    
    convert_lora_to_full_model(args)