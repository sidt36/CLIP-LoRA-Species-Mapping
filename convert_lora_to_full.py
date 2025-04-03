import os
import torch
import copy
import clip

def convert_lora_to_merged_model(lora_path, output_path, backbone="ViT-B/32"):
    """
    Convert a LoRA checkpoint to a merged model checkpoint compatible with standard CLIP evaluation.
    
    Args:
        lora_path: Path to the LoRA checkpoint (.pt file)
        output_path: Path to save the merged model (.pth file)
        backbone: The CLIP backbone architecture (default: ViT-B/32)
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
    
    # Define MultiheadAttentionLoRA class inline
    class LoRALayer(torch.nn.Module):
        def __init__(self, linear_layer, rank=4, alpha=1.0):
            super().__init__()
            self.in_features = linear_layer.in_features
            self.out_features = linear_layer.out_features
            self.weight = linear_layer.weight
            self.bias = linear_layer.bias
            
            # LoRA parameters
            self.lora_rank = rank
            self.lora_alpha = alpha
            self.scaling = alpha / rank
            
            # LoRA A and B matrices
            self.w_lora_A = torch.nn.Parameter(torch.zeros((rank, self.in_features)))
            self.w_lora_B = torch.nn.Parameter(torch.zeros((self.out_features, rank)))
    
    class PlainMultiheadAttentionLoRA(torch.nn.Module):
        def __init__(self, mha, enable_lora="qkvo", r=4, lora_alpha=1, dropout_rate=0.0):
            super().__init__()
            self.embed_dim = mha.embed_dim
            self.num_heads = mha.num_heads
            self.dropout = mha.dropout
            self.batch_first = mha.batch_first
            self.head_dim = mha.head_dim if hasattr(mha, 'head_dim') else self.embed_dim // self.num_heads
            
            # Apply LoRA to specified projections
            if 'q' in enable_lora:
                self.q_proj = LoRALayer(mha.q_proj, rank=r, alpha=lora_alpha)
            else:
                self.q_proj = mha.q_proj
                
            if 'k' in enable_lora:
                self.k_proj = LoRALayer(mha.k_proj, rank=r, alpha=lora_alpha)
            else:
                self.k_proj = mha.k_proj
                
            if 'v' in enable_lora:
                self.v_proj = LoRALayer(mha.v_proj, rank=r, alpha=lora_alpha)
            else:
                self.v_proj = mha.v_proj
                
            if 'o' in enable_lora:
                self.proj = LoRALayer(mha.out_proj, rank=r, alpha=lora_alpha)
            else:
                self.proj = mha.out_proj
    
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
    
    # Apply LoRA layers
    list_lora_layers = []
    
    # Apply to text encoder if specified
    if args.encoder in ['text', 'both']:
        indices = INDEX_POSITIONS_TEXT.get(args.position, [11])  # Default to last layer
        text_encoder = clip_model.transformer
        for i, block in enumerate(text_encoder.resblocks):
            if i in indices:
                for name, submodule in block.named_children():
                    if isinstance(submodule, torch.nn.MultiheadAttention):
                        print(f"Applying LoRA to text block {i}, {name}")
                        new_module = PlainMultiheadAttentionLoRA(
                            submodule, enable_lora=args.params, r=args.r, 
                            lora_alpha=args.alpha, dropout_rate=args.dropout_rate
                        )
                        setattr(block, name, new_module)
                        list_lora_layers.append(new_module)
    
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
                        print(f"Applying LoRA to vision block {i}, {name}")
                        new_module = PlainMultiheadAttentionLoRA(
                            submodule, enable_lora=args.params, r=args.r, 
                            lora_alpha=args.alpha, dropout_rate=args.dropout_rate
                        )
                        setattr(block, name, new_module)
                        list_lora_layers.append(new_module)
    
    # Load weights
    print(f"Loading LoRA weights into {len(list_lora_layers)} layers")
    for i, layer in enumerate(list_lora_layers):
        if f'layer_{i}' in weights:
            layer_weights = weights[f'layer_{i}']
            
            if 'q' in args.params and hasattr(layer, 'q_proj') and 'q_proj' in layer_weights:
                print(f"  Loading q_proj weights for layer {i}")
                layer.q_proj.w_lora_A.data.copy_(layer_weights['q_proj']['w_lora_A'])
                layer.q_proj.w_lora_B.data.copy_(layer_weights['q_proj']['w_lora_B'])
            
            if 'k' in args.params and hasattr(layer, 'k_proj') and 'k_proj' in layer_weights:
                print(f"  Loading k_proj weights for layer {i}")
                layer.k_proj.w_lora_A.data.copy_(layer_weights['k_proj']['w_lora_A'])
                layer.k_proj.w_lora_B.data.copy_(layer_weights['k_proj']['w_lora_B'])
            
            if 'v' in args.params and hasattr(layer, 'v_proj') and 'v_proj' in layer_weights:
                print(f"  Loading v_proj weights for layer {i}")
                layer.v_proj.w_lora_A.data.copy_(layer_weights['v_proj']['w_lora_A'])
                layer.v_proj.w_lora_B.data.copy_(layer_weights['v_proj']['w_lora_B'])
            
            if 'o' in args.params and hasattr(layer, 'proj') and 'proj' in layer_weights:
                print(f"  Loading proj weights for layer {i}")
                layer.proj.w_lora_A.data.copy_(layer_weights['proj']['w_lora_A'])
                layer.proj.w_lora_B.data.copy_(layer_weights['proj']['w_lora_B'])
    
    # Merge LoRA weights into base weights
    print("Merging LoRA weights into base model weights...")
    for layer in list_lora_layers:
        # Process q_proj
        if hasattr(layer, 'q_proj') and hasattr(layer.q_proj, 'w_lora_A'):
            print(f"Merging q_proj weights")
            lora_contribution = (layer.q_proj.w_lora_B @ layer.q_proj.w_lora_A) * layer.q_proj.scaling
            layer.q_proj.weight.data += lora_contribution
        
        # Process k_proj
        if hasattr(layer, 'k_proj') and hasattr(layer.k_proj, 'w_lora_A'):
            print(f"Merging k_proj weights")
            lora_contribution = (layer.k_proj.w_lora_B @ layer.k_proj.w_lora_A) * layer.k_proj.scaling
            layer.k_proj.weight.data += lora_contribution
        
        # Process v_proj
        if hasattr(layer, 'v_proj') and hasattr(layer.v_proj, 'w_lora_A'):
            print(f"Merging v_proj weights")
            lora_contribution = (layer.v_proj.w_lora_B @ layer.v_proj.w_lora_A) * layer.v_proj.scaling
            layer.v_proj.weight.data += lora_contribution
        
        # Process output projection
        if hasattr(layer, 'proj') and hasattr(layer.proj, 'w_lora_A'):
            print(f"Merging output projection weights")
            lora_contribution = (layer.proj.w_lora_B @ layer.proj.w_lora_A) * layer.proj.scaling
            layer.proj.weight.data += lora_contribution
    
    # Create clean state dict without LoRA parameters
    print("Creating clean state dict...")
    clean_state_dict = {}
    for k, v in clip_model.state_dict().items():
        if 'w_lora_A' not in k and 'w_lora_B' not in k and 'scaling' not in k:
            clean_state_dict[k] = v
    
    # Save merged model
    print(f"Saving merged model to {output_path}")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(clean_state_dict, output_path)
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