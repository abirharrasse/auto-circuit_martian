from itertools import count
from typing import Set, Tuple
from nnsight import LanguageModel

from auto_circuit.types import SrcNode, DestNode

def factorized_src_nodes(model: LanguageModel) -> Set[SrcNode]:
    """
    Get the source nodes for LLaMA model structure.
    """
    layers, idxs = count(), count()
    nodes = set()
    
    # Add embedding layer as first source
    nodes.add(
        SrcNode(
            name="Resid Start",
            module_name="model.embed_tokens",
            layer=next(layers),
            src_idx=next(idxs),
            weight="model.embed_tokens.weight",
        )
    )

    # Add attention and MLP nodes for each layer
    for layer_idx in range(len(model.model.layers)):
        layer = next(layers)
        
        # Add attention head outputs
        nodes.add(
            SrcNode(
                name=f"A{layer_idx}",
                module_name=f"model.layers.{layer_idx}.self_attn",
                layer=layer,
                src_idx=next(idxs),
                weight=f"model.layers.{layer_idx}.self_attn.o_proj.weight",
            )
        )
        
        # Add MLP outputs
        nodes.add(
            SrcNode(
                name=f"MLP {layer_idx}",
                module_name=f"model.layers.{layer_idx}.mlp",
                layer=layer,
                src_idx=next(idxs),
                weight=f"model.layers.{layer_idx}.mlp.down_proj.weight",
            )
        )
    
    return nodes

def factorized_dest_nodes(model: LanguageModel, separate_qkv: bool) -> Set[DestNode]:
    """
    Get the destination nodes for LLaMA model structure.
    """
    layers = count(1)
    nodes = set()
    
    for layer_idx in range(len(model.model.layers)):
        layer = next(layers)
        
        # Add attention nodes with LLaMA-specific structure
        if separate_qkv:
            # Add Q projection (2048 -> 2048)
            nodes.add(
                DestNode(
                    name=f"A{layer_idx}.Q",
                    module_name=f"model.layers.{layer_idx}.self_attn.q_proj",
                    layer=layer,
                    head_dim=None,  # LLaMA handles heads differently
                    weight=f"model.layers.{layer_idx}.self_attn.q_proj.weight",
                )
            )
            # Add K projection (2048 -> 512)
            nodes.add(
                DestNode(
                    name=f"A{layer_idx}.K",
                    module_name=f"model.layers.{layer_idx}.self_attn.k_proj",
                    layer=layer,
                    head_dim=None,
                    weight=f"model.layers.{layer_idx}.self_attn.k_proj.weight",
                )
            )
            # Add V projection (2048 -> 512)
            nodes.add(
                DestNode(
                    name=f"A{layer_idx}.V",
                    module_name=f"model.layers.{layer_idx}.self_attn.v_proj",
                    layer=layer,
                    head_dim=None,
                    weight=f"model.layers.{layer_idx}.self_attn.v_proj.weight",
                )
            )
        else:
            # Combined attention node
            nodes.add(
                DestNode(
                    name=f"A{layer_idx}",
                    module_name=f"model.layers.{layer_idx}.self_attn",
                    layer=layer,
                    head_dim=None,
                    weight=f"model.layers.{layer_idx}.self_attn.weight",
                )
            )
        
        # Add MLP node
        nodes.add(
            DestNode(
                name=f"MLP {layer_idx}",
                module_name=f"model.layers.{layer_idx}.mlp",
                layer=layer,
                weight=f"model.layers.{layer_idx}.mlp.up_proj.weight",
            )
        )
    
    # Add final layer norm
    nodes.add(
        DestNode(
            name="Norm End",
            module_name="model.norm",
            layer=next(layers),
            weight="lm_head.weight",
        )
    )
    
    return nodes

def simple_graph_nodes(model: LanguageModel) -> Tuple[Set[SrcNode], Set[DestNode]]:
    """
    Get the nodes of the unfactorized graph for LLaMA model.
    """
    layers, src_idxs = count(), count()
    src_nodes, dest_nodes = set(), set()
    layer, min_src_idx = next(layers), next(src_idxs)
    
    for layer_idx in range(len(model.model.layers)):
        first_layer = layer_idx == 0
        
        # Add residual connections
        src_nodes.add(
            SrcNode(
                name="Resid Start" if first_layer else f"Resid Post {layer_idx - 1}",
                module_name="model.embed_tokens" if first_layer else f"model.layers.{layer_idx - 1}.post_attention_layernorm",
                layer=layer,
                src_idx=min_src_idx,
            )
        )
        
        # Add attention outputs
        src_nodes.add(
            SrcNode(
                name=f"A{layer_idx}",
                module_name=f"model.layers.{layer_idx}.self_attn",
                layer=layer,
                src_idx=next(src_idxs),
                weight=f"model.layers.{layer_idx}.self_attn.o_proj.weight",
            )
        )
        
        # Add mid-layer residual
        layer = next(layers)
        dest_nodes.add(
            DestNode(
                name=f"Resid Mid {layer_idx}",
                module_name=f"model.layers.{layer_idx}.post_attention_layernorm",
                layer=layer,
                min_src_idx=min_src_idx,
            )
        )
        
        min_src_idx = next(src_idxs)
        src_nodes.add(
            SrcNode(
                name=f"Resid Mid {layer_idx}",
                module_name=f"model.layers.{layer_idx}.post_attention_layernorm",
                layer=layer,
                src_idx=min_src_idx,
            )
        )
        
        # Add MLP outputs
        src_nodes.add(
            SrcNode(
                name=f"MLP {layer_idx}",
                module_name=f"model.layers.{layer_idx}.mlp",
                layer=layer,
                src_idx=next(src_idxs),
                weight=f"model.layers.{layer_idx}.mlp.down_proj.weight",
            )
        )
        
        # Handle final layer
        last_layer = layer_idx + 1 == len(model.model.layers)
        layer = next(layers)
        dest_nodes.add(
            DestNode(
                name="Norm End" if last_layer else f"Resid Post {layer_idx}",
                module_name="model.norm" if last_layer else f"model.layers.{layer_idx}.post_attention_layernorm",
                layer=layer,
                min_src_idx=min_src_idx,
            )
        )
        min_src_idx = next(src_idxs)
    
    return src_nodes, dest_nodes
