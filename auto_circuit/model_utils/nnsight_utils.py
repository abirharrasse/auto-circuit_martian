from itertools import count
from typing import Set, Tuple, Optional
from nnsight import LanguageModel

from auto_circuit.types import SrcNode, DestNode

def get_model_config(model: LanguageModel):
    """Extract common configuration elements from any HuggingFace model."""
    config = model.config
    
    # Find the base model (handles different model structures)
    if hasattr(model, 'model'):
        base_model = model.model
    else:
        base_model = model
        
    # Get number of layers
    n_layers = len(base_model.layers)
    
    # Get dimensionality
    hidden_size = config.hidden_size if hasattr(config, 'hidden_size') else config.model_dim
    
    # Get attention heads
    n_heads = config.num_attention_heads if hasattr(config, 'num_attention_heads') else config.n_heads
    
    # Determine model type and paths
    model_type = config.model_type.lower() if hasattr(config, 'model_type') else "generic"
    
    # Build path prefixes based on model structure
    if hasattr(model, 'model'):
        prefix = "model.model."
        weight_prefix = "model."
    else:
        prefix = "model."
        weight_prefix = ""
        
    return {
        'n_layers': n_layers,
        'hidden_size': hidden_size,
        'n_heads': n_heads,
        'model_type': model_type,
        'prefix': prefix,
        'weight_prefix': weight_prefix,
        'base_model': base_model
    }

def factorized_src_nodes(model: LanguageModel) -> Set[SrcNode]:
    """Generic source nodes for HuggingFace models."""
    cfg = get_model_config(model)
    layers, idxs = count(), count()
    nodes = set()
    
    # Add embedding layer
    nodes.add(
        SrcNode(
            name="Resid Start",
            module_name=f"{cfg['prefix']}embed_tokens",
            layer=next(layers),
            src_idx=next(idxs),
            weight=f"{cfg['prefix']}embed_tokens.weight",
        )
    )

    # Add attention and MLP nodes for each layer
    for layer_idx in range(cfg['n_layers']):
        layer = next(layers)
        
        # Add attention outputs
        nodes.add(
            SrcNode(
                name=f"A{layer_idx}",
                module_name=f"{cfg['prefix']}layers.{layer_idx}.self_attn",
                layer=layer,
                src_idx=next(idxs),
                weight=f"{cfg['prefix']}layers.{layer_idx}.self_attn.o_proj.weight",
            )
        )
        
        # Add MLP outputs
        nodes.add(
            SrcNode(
                name=f"MLP {layer_idx}",
                module_name=f"{cfg['prefix']}layers.{layer_idx}.mlp",
                layer=layer,
                src_idx=next(idxs),
                weight=f"{cfg['prefix']}layers.{layer_idx}.mlp.down_proj.weight",
            )
        )
    
    return nodes

def factorized_dest_nodes(model: LanguageModel, separate_qkv: bool) -> Set[DestNode]:
    """Generic destination nodes for HuggingFace models."""
    cfg = get_model_config(model)
    layers = count(1)
    nodes = set()
    
    for layer_idx in range(cfg['n_layers']):
        layer = next(layers)
        
        # Add attention nodes
        if separate_qkv:
            for proj in ['q', 'k', 'v']:
                nodes.add(
                    DestNode(
                        name=f"A{layer_idx}.{proj.upper()}",
                        module_name=f"{cfg['prefix']}layers.{layer_idx}.self_attn",
                        layer=layer,
                        head_dim=None,
                        weight=f"{cfg['prefix']}layers.{layer_idx}.self_attn.{proj}_proj.weight",
                    )
                )
        else:
            nodes.add(
                DestNode(
                    name=f"A{layer_idx}",
                    module_name=f"{cfg['prefix']}layers.{layer_idx}.self_attn",
                    layer=layer,
                    head_dim=None,
                    weight=f"{cfg['prefix']}layers.{layer_idx}.self_attn.weight",
                )
            )
        
        # Add MLP node
        nodes.add(
            DestNode(
                name=f"MLP {layer_idx}",
                module_name=f"{cfg['prefix']}layers.{layer_idx}.mlp",
                layer=layer,
                weight=f"{cfg['prefix']}layers.{layer_idx}.mlp.up_proj.weight",
            )
        )
    
    # Add final layer norm
    nodes.add(
        DestNode(
            name="Norm End",
            module_name=f"{cfg['prefix']}norm",
            layer=next(layers),
            weight=f"{cfg['weight_prefix']}lm_head.weight",
        )
    )
    
    return nodes

def simple_graph_nodes(model: LanguageModel) -> Tuple[Set[SrcNode], Set[DestNode]]:
    """Generic unfactorized graph nodes for HuggingFace models."""
    cfg = get_model_config(model)
    layers, src_idxs = count(), count()
    src_nodes, dest_nodes = set(), set()
    layer, min_src_idx = next(layers), next(src_idxs)
    
    for layer_idx in range(cfg['n_layers']):
        first_layer = layer_idx == 0
        
        # Add residual connections
        src_nodes.add(
            SrcNode(
                name="Resid Start" if first_layer else f"Resid Post {layer_idx - 1}",
                module_name=f"{cfg['prefix']}embed_tokens" if first_layer 
                    else f"{cfg['prefix']}layers.{layer_idx - 1}.post_attention_layernorm",
                layer=layer,
                src_idx=min_src_idx,
            )
        )
        
        # Add attention outputs
        src_nodes.add(
            SrcNode(
                name=f"A{layer_idx}",
                module_name=f"{cfg['prefix']}layers.{layer_idx}.self_attn",
                layer=layer,
                src_idx=next(src_idxs),
                weight=f"{cfg['prefix']}layers.{layer_idx}.self_attn.o_proj.weight",
            )
        )
        
        # Add mid-layer residual
        layer = next(layers)
        dest_nodes.add(
            DestNode(
                name=f"Resid Mid {layer_idx}",
                module_name=f"{cfg['prefix']}layers.{layer_idx}.post_attention_layernorm",
                layer=layer,
                min_src_idx=min_src_idx,
            )
        )
        
        min_src_idx = next(src_idxs)
        src_nodes.add(
            SrcNode(
                name=f"Resid Mid {layer_idx}",
                module_name=f"{cfg['prefix']}layers.{layer_idx}.post_attention_layernorm",
                layer=layer,
                src_idx=min_src_idx,
            )
        )
        
        # Add MLP outputs
        src_nodes.add(
            SrcNode(
                name=f"MLP {layer_idx}",
                module_name=f"{cfg['prefix']}layers.{layer_idx}.mlp",
                layer=layer,
                src_idx=next(src_idxs),
                weight=f"{cfg['prefix']}layers.{layer_idx}.mlp.down_proj.weight",
            )
        )
        
        last_layer = layer_idx + 1 == cfg['n_layers']
        layer = next(layers)
        dest_nodes.add(
            DestNode(
                name="Norm End" if last_layer else f"Resid Post {layer_idx}",
                module_name=f"{cfg['prefix']}norm" if last_layer 
                    else f"{cfg['prefix']}layers.{layer_idx}.post_attention_layernorm",
                layer=layer,
                min_src_idx=min_src_idx,
            )
        )
        min_src_idx = next(src_idxs)
    
    return src_nodes, dest_nodes
