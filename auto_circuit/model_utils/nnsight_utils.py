from itertools import count
from typing import Set, Tuple
from nnsight import LanguageModel

from auto_circuit.types import SrcNode, DestNode

def factorized_src_nodes(model: LanguageModel) -> Set[SrcNode]:
    """
    Get the source part of each edge in the factorized graph for nnsight models.
    Following the same structure as transformer_lens implementation.
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
        for head_idx in range(model.config.num_attention_heads):
            nodes.add(
                SrcNode(
                    name=f"A{layer_idx}.{head_idx}",
                    module_name=f"model.layers.{layer_idx}.self_attn",
                    layer=layer,
                    src_idx=next(idxs),
                    head_dim=2,
                    head_idx=head_idx,
                    weight=f"model.layers.{layer_idx}.self_attn.o_proj.weight",
                    weight_head_dim=0,
                )
            )
        
        # Add MLP outputs
        nodes.add(
            SrcNode(
                name=f"MLP {layer_idx}",
                module_name=f"model.layers.{layer_idx}.mlp",
                layer=layer,  # Note: might need next(layers) depending on parallel_attn_mlp
                src_idx=next(idxs),
                weight=f"model.layers.{layer_idx}.mlp.down_proj.weight",
            )
        )
    
    return nodes

def factorized_dest_nodes(model: LanguageModel, separate_qkv: bool) -> Set[DestNode]:
    """
    Get the destination part of each edge in the factorized graph for nnsight models.
    Following the same structure as transformer_lens implementation.
    """
    layers = count(1)
    nodes = set()
    
    for layer_idx in range(len(model.model.layers)):
        layer = next(layers)
        
        # Add attention nodes
        for head_idx in range(model.config.num_attention_heads):
            if separate_qkv:
                # Add separate Q, K, V nodes for each head
                for letter in ["Q", "K", "V"]:
                    nodes.add(
                        DestNode(
                            name=f"A{layer_idx}.{head_idx}.{letter}",
                            module_name=f"model.layers.{layer_idx}.self_attn.{letter.lower()}_proj",
                            layer=layer,
                            head_dim=2,
                            head_idx=head_idx,
                            weight=f"model.layers.{layer_idx}.self_attn.{letter.lower()}_proj.weight",
                            weight_head_dim=0,
                        )
                    )
            else:
                # Combined QKV node
                nodes.add(
                    DestNode(
                        name=f"A{layer_idx}.{head_idx}",
                        module_name=f"model.layers.{layer_idx}.self_attn",
                        layer=layer,
                        head_dim=2,
                        head_idx=head_idx,
                        weight=f"model.layers.{layer_idx}.self_attn.qkv_proj.weight",
                        weight_head_dim=0,
                    )
                )
        
        # Add MLP nodes
        nodes.add(
            DestNode(
                name=f"MLP {layer_idx}",
                module_name=f"model.layers.{layer_idx}.mlp",
                layer=layer,  # Note: might need next(layers) depending on parallel_attn_mlp
                weight=f"model.layers.{layer_idx}.mlp.up_proj.weight",
            )
        )
    
    # Add final residual output node
    nodes.add(
        DestNode(
            name="Resid End",
            module_name=f"model.layers.{len(model.model.layers) - 1}.final_layernorm",
            layer=next(layers),
            weight="lm_head.weight",
        )
    )
    
    return nodes

def simple_graph_nodes(model: LanguageModel) -> Tuple[Set[SrcNode], Set[DestNode]]:
    """
    Get the nodes of the unfactorized graph for nnsight models.
    Following the same structure as transformer_lens implementation.
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
        
        # Add attention head outputs
        for head_idx in range(model.config.num_attention_heads):
            src_nodes.add(
                SrcNode(
                    name=f"A{layer_idx}.{head_idx}",
                    module_name=f"model.layers.{layer_idx}.self_attn",
                    layer=layer,
                    src_idx=next(src_idxs),
                    head_idx=head_idx,
                    head_dim=2,
                    weight=f"model.layers.{layer_idx}.self_attn.o_proj.weight",
                    weight_head_dim=0,
                )
            )
        
        # Add mid-layer residual connection
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
        
        # Handle final residual connection
        last_layer = layer_idx + 1 == len(model.model.layers)
        layer = next(layers)
        dest_nodes.add(
            DestNode(
                name="Resid End" if last_layer else f"Resid Post {layer_idx}",
                module_name=f"model.layers.{layer_idx}.final_layernorm" if last_layer else f"model.layers.{layer_idx}.post_attention_layernorm",
                layer=layer,
                min_src_idx=min_src_idx,
            )
        )
        min_src_idx = next(src_idxs)
    
    return src_nodes, dest_nodes
