from typing import Set
from auto_circuit.types import SrcNode, DestNode

def factorized_src_nodes(model) -> Set[SrcNode]:
    """Get the set of source nodes for a Gemma model in factorized form."""
    srcs = set()
    
    # Add embedding layer as source
    srcs.add(SrcNode(
        module_name="embed_tokens",
        layer=0,
        src_idx=0,
        head_dim=None
    ))
    
    # Add attention layers as sources
    for layer_idx in range(len(model.layers)):
        # Query sources
        srcs.add(SrcNode(
            module_name=f"layers.{layer_idx}.self_attn.q_proj",
            layer=layer_idx + 1,
            src_idx=0,
            head_dim=32  # Assuming 32 attention heads based on Gemma architecture
        ))
        
        # Key sources
        srcs.add(SrcNode(
            module_name=f"layers.{layer_idx}.self_attn.k_proj",
            layer=layer_idx + 1,
            src_idx=1,
            head_dim=16  # Gemma uses MQA with only 16 key/value heads
        ))
        
        # Value sources
        srcs.add(SrcNode(
            module_name=f"layers.{layer_idx}.self_attn.v_proj",
            layer=layer_idx + 1,
            src_idx=2,
            head_dim=16  # Gemma uses MQA with only 16 key/value heads
        ))
    
    return srcs

def factorized_dest_nodes(model, separate_qkv: bool) -> Set[DestNode]:
    """Get the set of destination nodes for a Gemma model in factorized form."""
    dests = set()
    
    # Add attention layers as destinations
    for layer_idx in range(len(model.layers)):
        if separate_qkv:
            # Separate Q, K, V destinations
            dests.add(DestNode(
                module_name=f"layers.{layer_idx}.self_attn.q_proj",
                layer=layer_idx + 1,
                head_dim=32
            ))
            dests.add(DestNode(
                module_name=f"layers.{layer_idx}.self_attn.k_proj",
                layer=layer_idx + 1,
                head_dim=16
            ))
            dests.add(DestNode(
                module_name=f"layers.{layer_idx}.self_attn.v_proj",
                layer=layer_idx + 1,
                head_dim=16
            ))
        else:
            # Combined attention destination
            dests.add(DestNode(
                module_name=f"layers.{layer_idx}.self_attn",
                layer=layer_idx + 1,
                head_dim=32
            ))
    
    return dests
