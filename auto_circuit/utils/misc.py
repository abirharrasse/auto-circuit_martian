from contextlib import contextmanager
from datetime import datetime
from functools import reduce
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Set

import torch as t
from einops import einsum
from torch.utils.hooks import RemovableHandle


def repo_path_to_abs_path(path: str) -> Path:
    """
    Convert a path relative to the repository root to an absolute path.

    Args:
        path: A path relative to the repository root.

    Returns:
        The absolute path.
    """
    repo_abs_path = Path(__file__).parent.parent.parent.absolute()
    return repo_abs_path / path


def save_cache(data_dict: Dict[Any, Any], folder_name: str, base_filename: str):
    """
    Save a dictionary to a cache file.

    Args:
        data_dict: The dictionary to save.
        folder_name: The name of the folder to save the cache in.
        base_filename: The base name of the file to save the cache in. The current date
            and time will be appended to the base filename.
    """
    folder = repo_path_to_abs_path(folder_name)
    folder.mkdir(parents=True, exist_ok=True)
    dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    file_path = folder / f"{base_filename}-{dt_string}.pkl"
    print(f"Saving cache to {file_path}")
    t.save(data_dict, file_path)


def load_cache(folder_name: str, filename: str) -> Dict[Any, Any]:
    """
    Load a dictionary from a cache file.

    Args:
        folder_name: The name of the folder to load the cache from.
        filename: The name of the file to load the cache from.

    Returns:
        The loaded dictionary.
    """
    folder = repo_path_to_abs_path(folder_name)
    return t.load(folder / filename)


@contextmanager
def remove_hooks() -> Iterator[Set[RemovableHandle]]:
    """
    Context manager that makes it easier to use temporary PyTorch hooks without
    accidentally leaving them attached.

    Add hooks to the set yielded by this context manager, and they will be removed when
    the context manager exits.

    Yields:
        An empty set that can be used to store the handles of the hooks.
    """
    handles: Set[RemovableHandle] = set()
    try:
        yield handles
    finally:
        for handle in handles:
            handle.remove()


from functools import reduce
from typing import Any, Optional
from transformers import PreTrainedModel

def get_transformer_base(model: PreTrainedModel) -> Optional[str]:
    """
    Get the base attribute name containing transformer layers for different HF architectures.
    
    Args:
        model: HuggingFace model instance
    
    Returns:
        Base attribute name or None if not found
    """
    # Common attribute names for different architectures
    possible_bases = [
        'model',           # LLaMA, Qwen
        'transformer',     # GPT2
        'encoder',        # BERT
        'decoder',        # T5
        'albert',         # ALBERT
        'roberta',        # RoBERTa
    ]
    
    for base in possible_bases:
        if hasattr(model, base):
            # Verify this attribute actually contains the layers
            base_module = getattr(model, base)
            if hasattr(base_module, 'layers') or hasattr(base_module, 'layer'):
                return base
    return None

def module_by_name(model: Any, module_name: str) -> Any:
    """
    Gets a module from a model by name, handling different HF model architectures.
    
    Args:
        model: The model to get the module from
        module_name: The name of the module
        
    Returns:
        The requested module
    """
    init_mod = [model.wrapped_model] if hasattr(model, "wrapped_model") else [model]
    model_instance = init_mod[0]
    
    # Only process if it's a HF model
    if isinstance(model_instance, PreTrainedModel):
        # If the path starts with 'layers' or 'layer', we need to prepend the base
        if module_name.startswith(('layers', 'layer')):
            base = get_transformer_base(model_instance)
            if base:
                module_name = f"{base}.{module_name}"
    
    return reduce(getattr, init_mod + module_name.split("."))

# Helper function to get the correct layer path for a model
def get_layer_path(model: PreTrainedModel, layer_idx: int) -> str:
    """
    Get the correct path to a transformer layer for any HF model.
    
    Args:
        model: HuggingFace model instance
        layer_idx: Index of the layer
        
    Returns:
        Full path to the layer as a string
    """
    base = get_transformer_base(model)
    if not base:
        raise ValueError("Could not determine model architecture")
        
    # Handle different naming conventions
    if hasattr(getattr(model, base), 'layers'):
        return f"{base}.layers.{layer_idx}"
    elif hasattr(getattr(model, base), 'layer'):
        return f"{base}.layer.{layer_idx}"
    else:
        raise ValueError("Could not find layers in model")


def set_module_by_name(model: Any, module_name: str, new_module: t.nn.Module):
    """
    Sets a module in a model by name.

    Args:
        model: The model to set the module in.
        module_name: The name of the module to set.
        new_module: The module to replace the existing module with.

    Warning:
        This function modifies the model in place.
    """
    parent = model
    init_mod = [model.wrapped_model] if hasattr(model, "wrapped_model") else [model]
    if "." in module_name:
        parent = reduce(getattr, init_mod + module_name.split(".")[:-1])  # type: ignore
    setattr(parent, module_name.split(".")[-1], new_module)


def percent_gpu_mem_used(total_gpu_mib: int = 49000) -> str:
    """
    Get the percentage of GPU memory used.

    Args:
        total_gpu_mib: The total amount of GPU memory in MiB.

    Returns:
        The percentage of GPU memory used.
    """
    return (
        "Memory used {:.1f}".format(
            ((t.cuda.memory_allocated() / (2**20)) / total_gpu_mib) * 100
        )
        + "%"
    )


def run_prompt(
    model: t.nn.Module,
    prompt: str,
    answer: Optional[str] = None,
    top_k: int = 10,
    prepend_bos: bool = False,
):
    """
    Helper function to run a string prompt through a TransformerLens `HookedTransformer`
    model and print the top `top_k` output logits.

    Args:
        model: The model to run the prompt through.
        prompt: The prompt to run through the model.
        answer: Token to show the rank of in the output logits.
        top_k: The number of top output logits to show.
        prepend_bos: Whether to prepend the `BOS` token to the prompt.
    """
    print(" ")
    print("Testing prompt", model.to_str_tokens(prompt))
    toks = model.to_tokens(prompt, prepend_bos=prepend_bos)
    logits = model(toks)
    get_most_similar_embeddings(model, logits[0, -1], answer, top_k=top_k)


def get_most_similar_embeddings(
    model: t.nn.Module,
    out: t.Tensor,
    answer: Optional[str] = None,
    top_k: int = 10,
    apply_ln_final: bool = False,
    apply_unembed: bool = False,
    apply_embed: bool = False,
):
    """
    Helper function to print the top `top_k` most similar embeddings to a given vector.
    Can be used for either embeddings or unembeddings.

    Args:
        model: The model to get the embeddings from.
        out: The vector to get the most similar embeddings to.
        answer: Token to show the rank of in the output logits.
        top_k: The number of top output logits to show.
        apply_ln_final: Whether to apply the final layer normalization to the vector
            before getting the most similar embeddings.
        apply_unembed: If `True`, compare to the unembeddings.
        apply_embed: If `True`, compare to the embeddings.
    """
    assert not (apply_embed and apply_unembed), "Can't apply both embed and unembed"
    show_answer_rank = answer is not None
    answer = " cheese" if answer is None else answer
    out = out.unsqueeze(0).unsqueeze(0) if out.ndim == 1 else out
    out = model.ln_final(out) if apply_ln_final else out
    if apply_embed:
        unembeded = einsum(
            out, model.embed.W_E, "batch pos d_model, vocab d_model -> batch pos vocab"
        )
    elif apply_unembed:
        unembeded = model.unembed(out)
    else:
        unembeded = out
    answer_token = model.to_tokens(answer, prepend_bos=False).squeeze()
    answer_str_token = model.to_str_tokens(answer, prepend_bos=False)
    assert len(answer_str_token) == 1
    logits = unembeded.squeeze()  # type: ignore
    probs = logits.softmax(dim=-1)

    sorted_token_probs, sorted_token_values = probs.sort(descending=True)
    # Janky way to get the index of the token in the sorted list
    if answer is not None:
        correct_rank = t.arange(len(sorted_token_values))[
            (sorted_token_values == answer_token).cpu()
        ].item()
    else:
        correct_rank = -1
    if show_answer_rank:
        print(
            f'\n"{answer_str_token[0]}" token rank:',
            f"{correct_rank: <8}",
            f"\nLogit: {logits[answer_token].item():5.2f}",
            f"Prob: {probs[answer_token].item():6.2%}",
        )
    for i in range(top_k):
        print(
            f"Top {i}th token. Logit: {logits[sorted_token_values[i]].item():5.2f}",
            f"Prob: {sorted_token_probs[i].item():6.2%}",
            f'Token: "{model.to_string(sorted_token_values[i])}"',
        )
