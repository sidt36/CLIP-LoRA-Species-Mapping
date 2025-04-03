import os
import torch
import copy
import clip

def convert_lora_to_merged_model(lora_path, output_path, backbone="ViT-B/32"):
    """
    Convert a LoRA checkpoint to a merged model checkpoint compatible with standard CLIP evaluation.
    """
    # Check if input file exists
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA checkpoint not found at {lora_path}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load LoRA checkpoint
    print(f"Loading LoRA checkpoint from {lora_path}")
    lora_checkpoint = torch.load(lora_path, map_location=device)
    
    # Extract metadata
    if not isinstance(lora_checkpoint, dict) or 'metadata' not in lora_checkpoint or 'weights' not in lora_checkpoint:
        raise ValueError("Invalid LoRA checkpoint format. Expected 'metadata' and 'weights' keys.")
    
    metadata = lora_checkpoint['metadata']
    weights = lora_checkpoint['weights']
    
    # Create args class to match the expected interface
    class Args:
        pass
    
    args = Args()
    args.r = metadata.get('r', 4)
    args.alpha = metadata.get('alpha', 1.0)
    args.encoder = metadata.get('encoder', 'vision')
    args.params = metadata.get('params', 'qkvo')
    args.position = metadata.get('position', 'all')
    args.backbone = backbone
    args.dropout_rate = 0.0
    
    print(f"LoRA parameters: r={args.r}, alpha={args.alpha}, encoder={args.encoder}, "
          f"params={args.params}, position={args.position}")
    
    # Load base CLIP model
    print(f"Loading base CLIP model {backbone}")
    clip_model, _ = clip.load(backbone, device=device)
    
    # Inspect the MultiheadAttention structure
    # Find a MultiheadAttention module to check its structure
    found_mha = False
    for name, module in clip_model.named_modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            print(f"Found MultiheadAttention at {name}")
            print(f"MultiheadAttention attributes: {dir(module)}")
            found_mha = True
            break
    
    if not found_mha:
        print("WARNING: No MultiheadAttention module found. Check model structure.")
    
    # Modified LoRA Layer for in_proj style MHA
    class LoRALayer(torch.nn.Module):
        def __init__(self, linear_layer, rank=4, alpha=1.0):
            super().__init__()
            self.in_features = linear_layer.in_features if hasattr(linear_layer, 'in_features') else linear_layer.weight.shape[1]
            self.out_features = linear_layer.out_features if hasattr(linear_layer, 'out_features') else linear_layer.weight.shape[0]
            self.weight = linear_layer.weight
            self.bias = linear_layer.bias if hasattr(linear_layer, 'bias') else None
            
            # LoRA parameters
            self.lora_rank = rank
            self.lora_alpha = alpha
            self.scaling = alpha / rank
            
            # LoRA A and B matrices
            self.w_lora_A = torch.nn.Parameter(torch.zeros((rank, self.in_features)))
            self.w_lora_B = torch.nn.Parameter(torch.zeros((self.out_features, rank)))
    
    # Helper function to merge weights
    def merge_weights(layer, lora_A, lora_B, scaling):
        # Compute the LoRA contribution: B*A scaled by alpha/r
        lora_contribution = (lora_B @ lora_A) * scaling
        # Add to the original weight
        layer.weight.data += lora_contribution
    
    # Define layer indices based on position
    INDEX_POSITIONS_TEXT = {
        'top1': [11], 'top2': [10, 11], 'top3': [9, 10, 11],
        'bottom': [0, 1, 2, 3], 'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11], 'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': list(range(12))
    }
    
    INDEX_POSITIONS_VISION = {
        'ViT-B/16': {
            'top': [11], 'top3': [9, 10, 11], 'bottom': [0, 1, 2, 3],
            'mid': [4, 5, 6, 7], 'up': [8, 9, 10, 11],
            'half-up': [6, 7, 8, 9, 10, 11], 'half-bottom': [0, 1, 2, 3, 4, 5],
            'all': list(range(12))
        },
        'ViT-B/32': {
            'bottom': [0, 1, 2, 3], 'mid': [4, 5, 6, 7], 'up': [8, 9, 10, 11],
            'half-up': [6, 7, 8, 9, 10, 11], 'half-bottom': [0, 1, 2, 3, 4, 5],
            'all': list(range(12))
        },
        'ViT-L/14': {
            'half-up': list(range(12, 24)), 'half-bottom': list(range(12)),
            'all': list(range(24))
        }
    }
    
    # Direct weight merging approach
    print("Applying LoRA weights directly to model weights...")
    
    layer_idx = 0
    
    # Apply to text encoder if specified
    if args.encoder in ['text', 'both']:
        indices = INDEX_POSITIONS_TEXT.get(args.position, [11])  # Default to last layer
        text_encoder = clip_model.transformer
        for i, block in enumerate(text_encoder.resblocks):
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, torch.nn.MultiheadAttention):
                        print(f"Processing text block {i}, {name}")
                        
                        if f'layer_{layer_idx}' in weights:
                            layer_weights = weights[f'layer_{layer_idx}']
                            
                            # Check if we're dealing with in_proj or separate q,k,v projections
                            if hasattr(submodule, 'in_proj_weight'):
                                # This is a PyTorch style MHA with a single in_proj_weight
                                print(f"  Found in_proj_weight style MHA")
                                embed_dim = submodule.embed_dim
                                
                                # Get the weights - in PyTorch MHA, in_proj_weight contains q,k,v stacked
                                q_weight = submodule.in_proj_weight[:embed_dim]
                                k_weight = submodule.in_proj_weight[embed_dim:2*embed_dim]
                                v_weight = submodule.in_proj_weight[2*embed_dim:]
                                out_weight = submodule.out_proj.weight
                                
                                # Apply LoRA contributions
                                if 'q' in args.params and 'q_proj' in layer_weights:
                                    print(f"    Applying q_proj LoRA")
                                    lora_A = layer_weights['q_proj']['w_lora_A']
                                    lora_B = layer_weights['q_proj']['w_lora_B']
                                    scaling = args.alpha / args.r
                                    q_update = (lora_B @ lora_A) * scaling
                                    submodule.in_proj_weight.data[:embed_dim] += q_update
                                
                                if 'k' in args.params and 'k_proj' in layer_weights:
                                    print(f"    Applying k_proj LoRA")
                                    lora_A = layer_weights['k_proj']['w_lora_A']
                                    lora_B = layer_weights['k_proj']['w_lora_B']
                                    scaling = args.alpha / args.r
                                    k_update = (lora_B @ lora_A) * scaling
                                    submodule.in_proj_weight.data[embed_dim:2*embed_dim] += k_update
                                
                                if 'v' in args.params and 'v_proj' in layer_weights:
                                    print(f"    Applying v_proj LoRA")
                                    lora_A = layer_weights['v_proj']['w_lora_A']
                                    lora_B = layer_weights['v_proj']['w_lora_B']
                                    scaling = args.alpha / args.r
                                    v_update = (lora_B @ lora_A) * scaling
                                    submodule.in_proj_weight.data[2*embed_dim:] += v_update
                                
                                if 'o' in args.params and 'proj' in layer_weights:
                                    print(f"    Applying output proj LoRA")
                                    lora_A = layer_weights['proj']['w_lora_A']
                                    lora_B = layer_weights['proj']['w_lora_B']
                                    scaling = args.alpha / args.r
                                    out_update = (lora_B @ lora_A) * scaling
                                    submodule.out_proj.weight.data += out_update
                            
                            else:
                                # This is a different MHA implementation with separate q,k,v projections
                                print(f"  Unable to determine MHA structure. Skipping layer.")
                        
                        layer_idx += 1
    
    # Apply to vision encoder if specified
    if args.encoder in ['vision', 'both']:
        if backbone.startswith("ViT-B/16"):
            model_key = "ViT-B/16"
        elif backbone.startswith("ViT-B/32"):
            model_key = "ViT-B/32"
        elif backbone.startswith("ViT-L/14"):
            model_key = "ViT-L/14"
        else:
            model_key = "ViT-B/32"  # Default
            
        indices = INDEX_POSITIONS_VISION[model_key].get(args.position, [11])  # Default to last layer
        vision_encoder = clip_model.visual.transformer
        for i, block in enumerate(vision_encoder.resblocks):
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, torch.nn.MultiheadAttention):
                        print(f"Processing vision block {i}, {name}")
                        
                        if f'layer_{layer_idx}' in weights:
                            layer_weights = weights[f'layer_{layer_idx}']
                            
                            # Check if we're dealing with in_proj or separate q,k,v projections
                            if hasattr(submodule, 'in_proj_weight'):
                                # This is a PyTorch style MHA with a single in_proj_weight
                                print(f"  Found in_proj_weight style MHA")
                                embed_dim = submodule.embed_dim
                                
                                # Apply LoRA contributions
                                if 'q' in args.params and 'q_proj' in layer_weights:
                                    print(f"    Applying q_proj LoRA")
                                    lora_A = layer_weights['q_proj']['w_lora_A']
                                    lora_B = layer_weights['q_proj']['w_lora_B']
                                    scaling = args.alpha / args.r
                                    q_update = (lora_B @ lora_A) * scaling
                                    submodule.in_proj_weight.data[:embed_dim] += q_update
                                
                                if 'k' in args.params and 'k_proj' in layer_weights:
                                    print(f"    Applying k_proj LoRA")
                                    lora_A = layer_weights['k_proj']['w_lora_A']
                                    lora_B = layer_weights['k_proj']['w_lora_B']
                                    scaling = args.alpha / args.r
                                    k_update = (lora_B @ lora_A) * scaling
                                    submodule.in_proj_weight.data[embed_dim:2*embed_dim] += k_update
                                
                                if 'v' in args.params and 'v_proj' in layer_weights:
                                    print(f"    Applying v_proj LoRA")
                                    lora_A = layer_weights['v_proj']['w_lora_A']
                                    lora_B = layer_weights['v_proj']['w_lora_B']
                                    scaling = args.alpha / args.r
                                    v_update = (lora_B @ lora_A) * scaling
                                    submodule.in_proj_weight.data[2*embed_dim:] += v_update
                                
                                if 'o' in args.params and 'proj' in layer_weights:
                                    print(f"    Applying output proj LoRA")
                                    lora_A = layer_weights['proj']['w_lora_A']
                                    lora_B = layer_weights['proj']['w_lora_B']
                                    scaling = args.alpha / args.r
                                    out_update = (lora_B @ lora_A) * scaling
                                    submodule.out_proj.weight.data += out_update
                            
                            else:
                                # This is a different MHA implementation with separate q,k,v projections
                                print(f"  Unable to determine MHA structure. Skipping layer.")
                        
                        layer_idx += 1
    
    # Save merged model
    print(f"Saving merged model to {output_path}")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(clip_model.state_dict(), output_path)
    print("Conversion complete!")

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert LoRA checkpoint to merged model")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to LoRA checkpoint (.pt)")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save merged model (.pth)")
    parser.add_argument("--backbone", type=str, default="ViT-B/32", help="CLIP backbone architecture")
    
    args = parser.parse_args()
    
    convert_lora_to_merged_model(args.lora_path, args.output_path, args.backbone)