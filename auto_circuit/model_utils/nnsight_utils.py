# nnsight_utils.py
import re
from typing import Any, Dict, Optional, Set, Tuple
from itertools import count

import torch as t
from nnsight import LanguageModel

from auto_circuit.types import DestNode, SrcNode

def normalize_module_path(path: str) -> str:
    """Normalize module paths to handle different HuggingFace model structures."""
    # Convert index notation from attribute style to array style
    path = re.sub(r'\.(\d+)\.', r'[\1].', path)
    # Remove any 'model.model.' or 'model.' prefix
    path = re.sub(r'^(model\.model\.|model\.)', '', path)
    return path

def get_module_safely(model: LanguageModel, path: str) -> Any:
    """Safely get a module by path, handling different model structures."""
    # First normalize the path
    path = normalize_module_path(path)
    
    # List of possible model prefixes/structures to try
    model_paths = [
        lambda m: m,  # Direct access
        lambda m: m.model,  # Through model.model
        lambda m: m.model.model,  # Some models have nested .model attributes
    ]
    
    for get_base in model_paths:
        try:
            current = get_base(model)
            parts = path.split('.')
            for part in parts:
                if '[' in part:
                    name, idx = part.split('[')
                    idx = int(idx.rstrip(']'))
                    current = getattr(current, name)[idx]
                else:
                    current = getattr(current, part)
            return current
        except (AttributeError, IndexError):
            continue
            
    print(f"Warning: Could not find module {path}")
    return None

def get_layer_path(layer_idx: int, model_type: str) -> str:
    """Helper to generate correct layer paths for different model types"""
    if model_type in ['llama', 'mistral']:
        return f"layers[{layer_idx}]"
    elif model_type in ['gpt2', 'gpt_neo']:
        return f"h[{layer_idx}]"
    else:
        return f"layers[{layer_idx}]"

def factorized_src_nodes(model: LanguageModel) -> Set[SrcNode]:
    """Create source nodes for any HuggingFace model."""
    nodes = set()
    
    # Add embedding layer first
    nodes.add(
        SrcNode(
            name="Resid Start",
            module_name="embed_tokens",  # This will be found through get_module_safely
            layer=0,
            src_idx=0,
            weight="embed_tokens"
        )
    )

    # Find number of layers by inspecting model structure
    n_layers = None
    try:
        if hasattr(model.model, 'layers'):
            n_layers = len(model.model.layers)
        elif hasattr(model, 'layers'):
            n_layers = len(model.layers)
    except AttributeError:
        raise ValueError("Could not determine number of model layers")

    for layer_idx in range(n_layers):
        # Add attention outputs
        nodes.add(
            SrcNode(
                name=f"A{layer_idx}",
                module_name=f"layers[{layer_idx}].self_attn",
                layer=layer_idx + 1,
                src_idx=layer_idx * 2 + 1,
                weight=f"layers[{layer_idx}].self_attn.o_proj"
            )
        )
        
        # Add MLP outputs 
        nodes.add(
            SrcNode(
                name=f"MLP {layer_idx}",
                module_name=f"layers[{layer_idx}].mlp",
                layer=layer_idx + 1,
                src_idx=layer_idx * 2 + 2,
                weight=f"layers[{layer_idx}].mlp.down_proj"
            )
        )
    
    return nodes

def factorized_dest_nodes(model: LanguageModel, separate_qkv: bool) -> Set[DestNode]:
    """Create destination nodes for any HuggingFace model."""
    nodes = set()
    
    # Find number of layers
    n_layers = None
    try:
        if hasattr(model.model, 'layers'):
            n_layers = len(model.model.layers)
        elif hasattr(model, 'layers'):
            n_layers = len(model.layers)
    except AttributeError:
        raise ValueError("Could not determine number of model layers")

    for layer_idx in range(n_layers):
        # Add attention components
        if separate_qkv:
            for proj, name in [('q_proj', 'Q'), ('k_proj', 'K'), ('v_proj', 'V')]:
                nodes.add(
                    DestNode(
                        name=f"A{layer_idx}.{name}",
                        module_name=f"layers[{layer_idx}].self_attn",
                        layer=layer_idx + 1,
                        head_dim=None,
                        weight=f"layers[{layer_idx}].self_attn.{proj}"
                    )
                )
        else:
            nodes.add(
                DestNode(
                    name=f"A{layer_idx}",
                    module_name=f"layers[{layer_idx}].self_attn",
                    layer=layer_idx + 1,
                    head_dim=None,
                    weight=f"layers[{layer_idx}].self_attn.weight"
                )
            )
        
        # Add MLP node
        nodes.add(
            DestNode(
                name=f"MLP {layer_idx}",
                module_name=f"layers[{layer_idx}].mlp",
                layer=layer_idx + 1,
                weight=f"layers[{layer_idx}].mlp.up_proj"
            )
        )
    
    # Add final layer norm
    nodes.add(
        DestNode(
            name="Norm End",
            module_name="norm",
            layer=n_layers + 1,
            weight="lm_head"
        )
    )
    
    return nodes

def simple_graph_nodes(model: LanguageModel) -> Tuple[Set[SrcNode], Set[DestNode]]:
    """Create simple graph nodes for any HuggingFace model."""
    cfg = get_model_config(model)
    weight_paths = get_weight_paths(cfg['model_type'])
    layers, src_idxs = count(), count()
    src_nodes, dest_nodes = set(), set()
    layer, min_src_idx = next(layers), next(src_idxs)
    
    for layer_idx in range(cfg['n_layers']):
        first_layer = layer_idx == 0
        layer_name = f"{cfg['layer_prefix']}.{layer_idx}"
        prev_layer_name = f"{cfg['layer_prefix']}.{layer_idx - 1}"
        
        # Add residual connections
        src_nodes.add(
            SrcNode(
                name="Resid Start" if first_layer else f"Resid Post {layer_idx - 1}",
                module_name="embed_tokens" if first_layer 
                    else f"{prev_layer_name}.post_attention_layernorm",
                layer=layer,
                src_idx=min_src_idx,
            )
        )
        
        # Add attention outputs
        src_nodes.add(
            SrcNode(
                name=f"A{layer_idx}",
                module_name=f"{layer_name}.{cfg['attn_name']}",
                layer=layer,
                src_idx=next(src_idxs),
                weight=f"{layer_name}.{cfg['attn_name']}.{weight_paths['o_proj']}",
            )
        )
        
        # Rest of the implementation follows the same pattern...
        # [Previous implementation continues with updated paths]
        
    return src_nodes, dest_nodes
