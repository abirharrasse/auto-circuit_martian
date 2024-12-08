from typing import Set, Tuple, Dict, Optional, List
from itertools import count
import re
from nnsight import LanguageModel
import torch as t

from auto_circuit.types import DestNode, SrcNode, Node


def get_layer_structure(model: LanguageModel) -> Dict[str, Dict]:
    """
    Analyze model structure to identify attention and MLP layers regardless of naming convention.
    
    Args:
        model: nnsight LanguageModel instance
        
    Returns:
        Dictionary containing layer structure information
    """
    structure = {
        'n_layers': 0,
        'n_heads': None,
        'layers': [],
        'is_parallel': False
    }
    
    # Common patterns for transformer architectures
    layer_patterns = [
        r'.*decoder\.layers\.\d+',
        r'.*transformer\.h\.\d+',
        r'.*model\.layers\.\d+',
        r'.*blocks\.\d+'
    ]
    
    attn_patterns = [
        'attention', 'attn', 'self_attn'
    ]
    
    mlp_patterns = [
        'mlp', 'feed_forward', 'ff', 'FFN'
    ]
    
    # Get all named modules
    named_modules = dict(model.model.named_modules())
    
    # Find transformer layers
    transformer_layers = []
    for name in named_modules:
        if any(re.match(pattern, name) for pattern in layer_patterns):
            layer_num = int(re.search(r'\d+', name).group())
            transformer_layers.append((layer_num, name))
    
    structure['n_layers'] = len(transformer_layers)
    
    # Analyze each transformer layer
    for layer_num, layer_name in sorted(transformer_layers):
        layer_info = {'name': layer_name, 'number': layer_num}
        
        # Find attention module
        for module_name in named_modules:
            if module_name.startswith(layer_name):
                if any(pat in module_name.lower() for pat in attn_patterns):
                    layer_info['attention'] = module_name
                    # Try to determine number of heads if not yet known
                    if structure['n_heads'] is None:
                        module = named_modules[module_name]
                        if hasattr(module, 'num_heads'):
                            structure['n_heads'] = module.num_heads
                        elif hasattr(module, 'n_heads'):
                            structure['n_heads'] = module.n_heads
                
                if any(pat in module_name.lower() for pat in mlp_patterns):
                    layer_info['mlp'] = module_name
        
        structure['layers'].append(layer_info)
    
    # Detect if attention and MLP are parallel
    structure['is_parallel'] = _detect_parallel_architecture(model)
    
    return structure


def _detect_parallel_architecture(model: LanguageModel) -> bool:
    """
    Detect if the model uses parallel attention and MLP computation.
    This is an approximation based on common architectures.
    """
    # Check model class name for common parallel architectures
    parallel_architectures = ['PaLM', 'GPTNeoX', 'MPT']
    return any(arch in model.model.__class__.__name__ for arch in parallel_architectures)


def factorized_src_nodes(model: LanguageModel) -> Set[SrcNode]:
    """
    Get source nodes for the factorized graph representation.
    
    Args:
        model: nnsight LanguageModel instance
        
    Returns:
        Set of SrcNode instances representing source nodes in the factorized graph
    """
    structure = get_layer_structure(model)
    layers, idxs = count(), count()
    nodes = set()
    
    # Add embedding layer as first source
    nodes.add(
        SrcNode(
            name="Resid Start",
            module_name="embeddings",  # Generic name, will be mapped appropriately
            layer=next(layers),
            src_idx=next(idxs),
            weight="word_embeddings",
        )
    )
    
    # Add nodes for each layer
    for layer_info in structure['layers']:
        layer = next(layers)
        
        # Add attention head outputs
        for head_idx in range(structure['n_heads']):
            nodes.add(
                SrcNode(
                    name=f"A{layer_info['number']}.{head_idx}",
                    module_name=f"{layer_info['attention']}.output",
                    layer=layer,
                    src_idx=next(idxs),
                    head_dim=2,
                    head_idx=head_idx,
                    weight=f"{layer_info['attention']}.dense",
                    weight_head_dim=0,
                )
            )
        
        # Add MLP output if present
        if 'mlp' in layer_info:
            nodes.add(
                SrcNode(
                    name=f"MLP {layer_info['number']}",
                    module_name=f"{layer_info['mlp']}.output",
                    layer=layer if structure['is_parallel'] else next(layers),
                    src_idx=next(idxs),
                    weight=f"{layer_info['mlp']}.dense_4h_to_h",  # Common name, will be mapped
                )
            )
    
    return nodes


def factorized_dest_nodes(
    model: LanguageModel, 
    separate_qkv: bool
) -> Set[DestNode]:
    """
    Get destination nodes for the factorized graph representation.
    
    Args:
        model: nnsight LanguageModel instance
        separate_qkv: Whether to separate Q, K, V inputs
        
    Returns:
        Set of DestNode instances representing destination nodes in the factorized graph
    """
    structure = get_layer_structure(model)
    layers = count(1)
    nodes = set()
    
    for layer_info in structure['layers']:
        layer = next(layers)
        
        # Add attention nodes
        for head_idx in range(structure['n_heads']):
            if separate_qkv:
                for letter in ["Q", "K", "V"]:
                    nodes.add(
                        DestNode(
                            name=f"A{layer_info['number']}.{head_idx}.{letter}",
                            module_name=f"{layer_info['attention']}.{letter.lower()}_proj",
                            layer=layer,
                            head_dim=2,
                            head_idx=head_idx,
                            weight=f"{layer_info['attention']}.{letter.lower()}_proj",
                            weight_head_dim=0,
                        )
                    )
            else:
                nodes.add(
                    DestNode(
                        name=f"A{layer_info['number']}.{head_idx}",
                        module_name=f"{layer_info['attention']}.qkv_proj",
                        layer=layer,
                        head_dim=2,
                        head_idx=head_idx,
                        weight=f"{layer_info['attention']}.qkv_proj",
                        weight_head_dim=0,
                    )
                )
        
        # Add MLP input if present
        if 'mlp' in layer_info:
            nodes.add(
                DestNode(
                    name=f"MLP {layer_info['number']}",
                    module_name=f"{layer_info['mlp']}.input",
                    layer=layer if structure['is_parallel'] else next(layers),
                    weight=f"{layer_info['mlp']}.dense_h_to_4h",
                )
            )
    
    # Add final residual output
    nodes.add(
        DestNode(
            name="Resid End",
            module_name=f"final_layernorm",  # Generic name, will be mapped
            layer=next(layers),
            weight="lm_head",
        )
    )
    
    return nodes


def simple_graph_nodes(
    model: LanguageModel,
) -> Tuple[Set[SrcNode], Set[DestNode]]:
    """
    Get nodes for the simple (unfactorized) graph representation.
    
    Args:
        model: nnsight LanguageModel instance
        
    Returns:
        Tuple of (source nodes, destination nodes) for the simple graph
    """
    structure = get_layer_structure(model)
    layers, src_idxs = count(), count()
    src_nodes, dest_nodes = set(), set()
    layer, min_src_idx = next(layers), next(src_idxs)
    
    for layer_info in structure['layers']:
        first_layer = layer_info['number'] == 0
        
        # Add residual connection
        src_nodes.add(
            SrcNode(
                name="Resid Start" if first_layer else f"Resid Post {layer_info['number']-1}",
                module_name="embeddings" if first_layer else f"{structure['layers'][layer_info['number']-1]['name']}.output",
                layer=layer,
                src_idx=min_src_idx,
            )
        )
        
        # Add attention heads
        for head_idx in range(structure['n_heads']):
            src_nodes.add(
                SrcNode(
                    name=f"A{layer_info['number']}.{head_idx}",
                    module_name=f"{layer_info['attention']}.output",
                    layer=layer,
                    src_idx=next(src_idxs),
                    head_idx=head_idx,
                    head_dim=2,
                    weight=f"{layer_info['attention']}.dense",
                    weight_head_dim=0,
                )
            )
        
        # Handle MLP if present
        if 'mlp' in layer_info and not structure['is_parallel']:
            layer = next(layers)
            dest_nodes.add(
                DestNode(
                    name=f"Resid Mid {layer_info['number']}",
                    module_name=f"{layer_info['name']}.intermediate",
                    layer=layer,
                    min_src_idx=min_src_idx,
                )
            )
            min_src_idx = next(src_idxs)
            src_nodes.add(
                SrcNode(
                    name=f"Resid Mid {layer_info['number']}",
                    module_name=f"{layer_info['name']}.intermediate",
                    layer=layer,
                    src_idx=min_src_idx,
                )
            )
            
            src_nodes.add(
                SrcNode(
                    name=f"MLP {layer_info['number']}",
                    module_name=f"{layer_info['mlp']}.output",
                    layer=layer,
                    src_idx=next(src_idxs),
                    weight=f"{layer_info['mlp']}.dense_4h_to_h",
                )
            )
        
        last_layer = layer_info['number'] == structure['n_layers'] - 1
        layer = next(layers)
        dest_nodes.add(
            DestNode(
                name="Resid End" if last_layer else f"Resid Post {layer_info['number']}",
                module_name=f"{layer_info['name']}.output",
                layer=layer,
                min_src_idx=min_src_idx,
            )
        )
        min_src_idx = next(src_idxs)
    
    return src_nodes, dest_nodes
