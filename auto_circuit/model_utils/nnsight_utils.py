# nnsight_utils.py
import re
from typing import Any, Dict, Optional, Set, Tuple
from itertools import count

import torch as t
from nnsight import LanguageModel

from auto_circuit.types import DestNode, SrcNode

def normalize_module_path(path: str) -> str:
    """Normalize module paths to handle different HuggingFace model structures."""
    # Convert any remaining dot-notation indices to array notation
    path = re.sub(r'\.(\d+)(?=\.)', r'[\1]', path)
    # Remove any model/model.model prefix
    path = re.sub(r'^(model\.model\.|model\.)', '', path)
    return path

def get_layer_path(layer_idx: int, model_type: str) -> str:
    """Helper to generate correct layer paths for different model types"""
    # Always return array-style indexing
    return f"layers[{layer_idx}]"

def factorized_src_nodes(model: LanguageModel) -> Set[SrcNode]:
    """Create source nodes for any HuggingFace model."""
    nodes = set()
    layers, idxs = count(), count()
    
    # Add embedding layer
    nodes.add(
        SrcNode(
            name="Resid Start",
            module_name="embed_tokens",
            layer=next(layers),
            src_idx=next(idxs),
            weight="embed_tokens",
        )
    )

    # Get number of layers
    n_layers = len(model.model.layers)
    
    # Add attention and MLP nodes for each layer
    for layer_idx in range(n_layers):
        layer = next(layers)
        layer_path = f"layers[{layer_idx}]"
        
        # Add attention outputs
        nodes.add(
            SrcNode(
                name=f"A{layer_idx}",
                module_name=f"{layer_path}.self_attn",
                layer=layer,
                src_idx=next(idxs),
                weight=f"{layer_path}.self_attn.o_proj",
            )
        )
        
        # Add MLP outputs
        nodes.add(
            SrcNode(
                name=f"MLP {layer_idx}",
                module_name=f"{layer_path}.mlp",
                layer=layer,
                src_idx=next(idxs),
                weight=f"{layer_path}.mlp.down_proj",
            )
        )
    
    return nodes

def factorized_dest_nodes(model: LanguageModel, separate_qkv: bool) -> Set[DestNode]:
    """Create destination nodes for any HuggingFace model."""
    nodes = set()
    layers = count(1)
    n_layers = len(model.model.layers)
    
    for layer_idx in range(n_layers):
        layer = next(layers)
        layer_path = f"layers[{layer_idx}]"
        
        # Add attention nodes
        if separate_qkv:
            for proj, name in [('q_proj', 'Q'), ('k_proj', 'K'), ('v_proj', 'V')]:
                nodes.add(
                    DestNode(
                        name=f"A{layer_idx}.{name}",
                        module_name=f"{layer_path}.self_attn",
                        layer=layer,
                        head_dim=None,
                        weight=f"{layer_path}.self_attn.{proj}",
                    )
                )
        else:
            nodes.add(
                DestNode(
                    name=f"A{layer_idx}",
                    module_name=f"{layer_path}.self_attn",
                    layer=layer,
                    head_dim=None,
                    weight=f"{layer_path}.self_attn.weight",
                )
            )
        
        # Add MLP node
        nodes.add(
            DestNode(
                name=f"MLP {layer_idx}",
                module_name=f"{layer_path}.mlp",
                layer=layer,
                weight=f"{layer_path}.mlp.up_proj",
            )
        )
    
    nodes.add(
        DestNode(
            name="Norm End",
            module_name="norm",
            layer=next(layers),
            weight="lm_head",
        )
    )
    
    return nodes

def simple_graph_nodes(model: LanguageModel) -> Tuple[Set[SrcNode], Set[DestNode]]:
    """Create simple graph nodes for any HuggingFace model."""
    src_nodes, dest_nodes = set(), set()
    layers, src_idxs = count(), count()
    layer, min_src_idx = next(layers), next(src_idxs)
    n_layers = len(model.model.layers)
    
    for layer_idx in range(n_layers):
        first_layer = layer_idx == 0
        layer_path = f"layers[{layer_idx}]"
        prev_layer_path = f"layers[{layer_idx - 1}]" if layer_idx > 0 else None
        
        # Add residual connections
        src_nodes.add(
            SrcNode(
                name="Resid Start" if first_layer else f"Resid Post {layer_idx - 1}",
                module_name="embed_tokens" if first_layer 
                    else f"{prev_layer_path}.post_attention_layernorm",
                layer=layer,
                src_idx=min_src_idx,
            )
        )
        
        # Add attention outputs
        src_nodes.add(
            SrcNode(
                name=f"A{layer_idx}",
                module_name=f"{layer_path}.self_attn",
                layer=layer,
                src_idx=next(src_idxs),
                weight=f"{layer_path}.self_attn.o_proj",
            )
        )
        
        # Add MLP components as needed
        src_nodes.add(
            SrcNode(
                name=f"MLP {layer_idx}",
                module_name=f"{layer_path}.mlp",
                layer=layer,
                src_idx=next(src_idxs),
                weight=f"{layer_path}.mlp.down_proj",
            )
        )
        
        # Add corresponding destination nodes
        # [Add your destination node creation here following the same pattern]
        
    return src_nodes, dest_nodes

def get_module_safely(model: LanguageModel, path: str) -> Any:
    """Safely get a module by path, handling different model structures."""
    path = normalize_module_path(path)
    
    try:
        # First try through model.model
        current = model.model
        parts = path.split('.')
        for part in parts:
            if '[' in part and ']' in part:
                # Handle array indexing
                name, idx = part.split('[')
                idx = int(idx.rstrip(']'))
                if hasattr(current, name):
                    current = getattr(current, name)[idx]
                else:
                    return None
            else:
                if hasattr(current, part):
                    current = getattr(current, part)
                else:
                    return None
        return current
    except (AttributeError, IndexError):
        return None
