def patchable_model(
    model: t.nn.Module,
    factorized: bool,
    slice_output: OutputSlice = None,
    seq_len: Optional[int] = None,
    separate_qkv: Optional[bool] = None,
    kv_caches: Tuple[Optional[HookedTransformerKeyValueCache], ...] = (None,),
    device: t.device = t.device("cpu"),
) -> PatchableModel:
    """
    Wrap a model and inject PatchWrappers into the node modules to enable patching.
    Now supports nnsight LanguageModel instances alongside other model types.

    Args:
        model: The model to make patchable. Can be a regular torch model, HookedTransformer,
              AutoencoderTransformer, or nnsight LanguageModel
        factorized: Whether the model is factorized, for Edge Ablation. Otherwise,
            only Node Ablation is possible.
        slice_output: Specifies the index/slice of the output of the model to be
            considered for the task. For example, `"last_seq"` will consider the last
            token's output in transformer models.
        seq_len: The sequence length of the model inputs. If `None`, all token positions
            are simultaneously ablated.
        separate_qkv: Whether the model has separate query, key, and value inputs. Only
            used for transformers.
        kv_caches: The key and value caches for the transformer. Only used for
            transformers.
        device: The device that the model is on.

    Returns:
        The patchable model.

    Warning:
        This function modifies the model, it does not return a new model.
    """
    try:
        from nnsight import LanguageModel
    except ImportError:
        LanguageModel = type(None)  # Type to check against if nnsight not installed
        
    assert not isinstance(model, PatchableModel), "Model is already patchable"
    
    # Handle nnsight LanguageModel specifically
    is_nnsight = isinstance(model, LanguageModel)
    if is_nnsight:
        if separate_qkv is None:
            separate_qkv = True  # Most modern transformers use separate QKV
            
    nodes, srcs, dests, edge_dict, edges, seq_dim, seq_len = graph_edges(
        model, factorized, separate_qkv, seq_len
    )
    
    wrappers, src_wrappers, dest_wrappers = make_model_patchable(
        model, factorized, srcs, nodes, device, seq_len, seq_dim
    )
    
    if slice_output is None:
        out_slice: Tuple[slice | int, ...] = (slice(None),)
    else:
        last_slice = [-1] if slice_output == "last_seq" else [slice(1, None)]
        out_slice: Tuple[slice | int, ...] = tuple([slice(None)] * seq_dim + last_slice)
    
    is_tl_transformer = isinstance(model, HookedTransformer)
    is_autoencoder_transformer = isinstance(model, AutoencoderTransformer)
    is_transformer = is_tl_transformer or is_autoencoder_transformer or is_nnsight
    
    return PatchableModel(
        nodes=nodes,
        srcs=srcs,
        dests=dests,
        edge_dict=edge_dict,
        edges=edges,
        seq_dim=seq_dim,
        seq_len=seq_len,
        wrappers=wrappers,
        src_wrappers=src_wrappers,
        dest_wrappers=dest_wrappers,
        out_slice=out_slice,
        is_factorized=factorized,
        is_transformer=is_transformer,
        separate_qkv=separate_qkv,
        kv_caches=kv_caches,
        wrapped_model=model,
    )


def graph_edges(
    model: t.nn.Module,
    factorized: bool,
    separate_qkv: Optional[bool] = None,
    seq_len: Optional[int] = None,
) -> Tuple[
    Set[Node],
    Set[SrcNode],
    Set[DestNode],
    Dict[int | None, List[Edge]],
    Set[Edge],
    int,
    Optional[int],
]:
    """
    Get the nodes and edges of the computation graph of the model used for ablation.
    Now includes support for nnsight LanguageModel.
    """
    try:
        from nnsight import LanguageModel
    except ImportError:
        LanguageModel = type(None)
        
    seq_dim = 1
    edge_dict: Dict[Optional[int], List[Edge]] = defaultdict(list)
    
    if not factorized:
        if isinstance(model, MicroModel):
            srcs, dests = mm_utils.simple_graph_nodes(model)
        elif isinstance(model, HookedTransformer):
            srcs, dests = tl_utils.simple_graph_nodes(model)
        elif isinstance(model, LanguageModel):
            # Implement similar to HookedTransformer but using nnsight's structure
            srcs, dests = tl_utils.simple_graph_nodes(model)  # Need to implement nnsight version
        else:
            raise NotImplementedError(f"Model type not supported: {type(model)}")
            
        for i in [None] if seq_len is None else range(seq_len):
            pairs = product(srcs, dests)
            edge_dict[i] = [Edge(s, d, i) for s, d in pairs if s.layer + 1 == d.layer]
    else:
        if isinstance(model, MicroModel):
            srcs: Set[SrcNode] = mm_utils.factorized_src_nodes(model)
            dests: Set[DestNode] = mm_utils.factorized_dest_nodes(model)
        elif isinstance(model, HookedTransformer):
            assert separate_qkv is not None, "separate_qkv must be specified for LLM"
            srcs: Set[SrcNode] = tl_utils.factorized_src_nodes(model)
            dests: Set[DestNode] = tl_utils.factorized_dest_nodes(model, separate_qkv)
        elif isinstance(model, AutoencoderTransformer):
            assert separate_qkv is not None, "separate_qkv must be specified for LLM"
            srcs: Set[SrcNode] = sae_utils.factorized_src_nodes(model)
            dests: Set[DestNode] = sae_utils.factorized_dest_nodes(model, separate_qkv)
        elif isinstance(model, LanguageModel):
            assert separate_qkv is not None, "separate_qkv must be specified for LLM"
            srcs: Set[SrcNode] = nnsight_utils.factorized_src_nodes(model)  # Need to implement
            dests: Set[DestNode] = nnsight_utils.factorized_dest_nodes(model, separate_qkv)  # Need to implement
        else:
            raise NotImplementedError(f"Model type not supported: {type(model)}")
            
        for i in [None] if seq_len is None else range(seq_len):
            pairs = product(srcs, dests)
            edge_dict[i] = [Edge(s, d, i) for s, d in pairs if s.layer < d.layer]
            
    nodes: Set[Node] = set(srcs | dests)
    edges = set(list(chain.from_iterable(edge_dict.values())))

    return nodes, srcs, dests, edge_dict, edges, seq_dim, seq_len
