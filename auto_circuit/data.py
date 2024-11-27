import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch as t
import torch.utils.data
from attr import dataclass
from torch.utils.data import (
    DataLoader,
    Dataset,
    Subset,
)
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache

BatchKey = int
"""A unique key for a [`PromptPairBatch`][auto_circuit.data.PromptPairBatch]."""


@dataclass(frozen=True)
class PromptPair:
    """
    A pair of clean and corrupt prompts with correct and incorrect answers.

    Args:
        clean: The 'clean' prompt. This is typically an example of the behavior we want
            to isolate where the model performs well.
        corrupt: The 'corrupt' prompt. This is typically similar to the 'clean' prompt,
            but with some crucial difference that changes the model output.
        answers: The correct completions for the clean prompt.
        wrong_answers: The incorrect completions for the clean prompt.
    """

    clean: t.Tensor
    corrupt: t.Tensor
    answers: t.Tensor
    wrong_answers: t.Tensor


@dataclass(frozen=True)
class PromptPairBatch:
    """
    A batch of prompt pairs.

    Args:
        key: A unique integer that identifies the batch.
        batch_diverge_idx: The minimum index over all prompts at which the clean and
            corrupt prompts diverge. This is used to automatically cache the key-value
            activations for the common prefix of the prompts. See
            [`load_datasets_from_json`][auto_circuit.data.load_datasets_from_json] for
            more information.
        clean: The 'clean' prompts in a 2D tensor. These are typically examples of the
            behavior we want to isolate where the model performs well.
        corrupt: The 'corrupt' prompts in a 2D tensor. These are typically similar to
            the 'clean' prompts, but with some crucial difference that changes the model
            output.
        answers: The correct answers completions for the clean prompts.
            If all prompts have the same number of answers, this is a 2D tensor.
            If each prompt has a different number of answers, this is a list of 1D
            tensors. This can make some methods such as
            [`batch_answer_diffs`][auto_circuit.utils.tensor_ops.batch_answer_diffs]
            much slower.
        wrong_answers: The incorrect answers. If each prompt has a different number of
            wrong answers, this is a list of tensors.
            If all prompts have the same number of wrong answers, this is a 2D tensor.
            If each prompt has a different number of wrong answers, this is a list of 1D
            tensors. This can make some methods such as
            [`batch_answer_diffs`][auto_circuit.utils.tensor_ops.batch_answer_diffs]
            much slower.
    """

    key: BatchKey
    batch_diverge_idx: int
    clean: t.Tensor
    corrupt: t.Tensor
    answers: List[t.Tensor] | t.Tensor
    wrong_answers: List[t.Tensor] | t.Tensor


def collate_fn(batch: List[PromptPair]) -> PromptPairBatch:
    clean = t.stack([p.clean for p in batch])
    print(clean.shape)
    corrupt = t.stack([p.corrupt for p in batch])
    print(corrupt.shape)
    if all([p.answers.shape == batch[0].answers.shape for p in batch]):
        answers = t.stack([p.answers for p in batch])
    else:  # Sometimes each prompt has a different number of answers
        answers = [p.answers for p in batch]
    if all([p.wrong_answers.shape == batch[0].wrong_answers.shape for p in batch]):
        wrong_answers = t.stack([p.wrong_answers for p in batch])
    else:  # Sometimes each prompt has a different number of wrong answers
        wrong_answers = [p.wrong_answers for p in batch]
    key = hash((str(clean.tolist()), str(corrupt.tolist())))

    diverge_idxs = (~(clean == corrupt)).int().argmax(dim=1)
    batch_dvrg_idx: int = int(diverge_idxs.min().item())
    return PromptPairBatch(key, batch_dvrg_idx, clean, corrupt, answers, wrong_answers)


class PromptDataset(Dataset):
    def __init__(
        self,
        clean_prompts: List[t.Tensor] | t.Tensor,
        corrupt_prompts: List[t.Tensor] | t.Tensor,
        answers: List[t.Tensor],
        wrong_answers: List[t.Tensor],
    ):
        """
        A dataset of clean/corrupt prompt pairs with correct/incorrect answers.

        Args:
            clean_prompts: The 'clean' prompts. These are typically examples of the
                behavior we want to isolate where the model performs well.
                If a list, each element is a 1D prompt tensor.
                If a tensor, it should be 2D with shape (n_prompts, prompt_length).
            corrupt_prompts: The 'corrupt' prompts. These are typically similar to the
                'clean' prompts, but with some crucial difference that changes the model
                output.
                If a list, each element is a 1D prompt tensor.
                If a tensor, it should be 2D with shape (n_prompts, prompt_length).
            answers: A list of correct answers.
                Each element is a 1D tensor with the answer tokens.
            wrong_answers: A list of incorrect answers.
                Each element is a 1D tensor with the wrong answer tokens.
        """

        self.clean_prompts = clean_prompts
        self.corrupt_prompts = corrupt_prompts
        self.answers = answers
        self.wrong_answers = wrong_answers

    def __len__(self) -> int:
        assert len(self.clean_prompts) == len(self.corrupt_prompts)
        return len(self.clean_prompts)

    def __getitem__(self, idx: int) -> PromptPair:
        return PromptPair(
            self.clean_prompts[idx],
            self.corrupt_prompts[idx],
            self.answers[idx],
            self.wrong_answers[idx],
        )


class PromptDataLoader(DataLoader[PromptPairBatch]):
    def __init__(
        self,
        prompt_dataset: Any,
        seq_len: Optional[int],
        diverge_idx: int,
        kv_cache: Optional[HookedTransformerKeyValueCache] = None,
        seq_labels: Optional[List[str]] = None,
        word_idxs: Dict[str, int] = {},
        **kwargs: Any,
    ):
        """
        A `DataLoader` for clean/corrupt prompt pairs with correct/incorrect answers.

        Args:
            prompt_dataset: A [`PromptDataset`][auto_circuit.data.PromptDataset] with
                clean and corrupt prompts.
            seq_len: The token length of the prompts (if fixed length). This prompt
                length can be passed to `patchable_model` to enable patching specific
                token positions.
            diverge_idx: The index at which the clean and corrupt prompts diverge. (See
                [`load_datasets_from_json`][auto_circuit.data.load_datasets_from_json]
                for more information.)
            kv_cache: A cache of past key-value pairs for the transformer. Only used if
                `diverge_idx` is greater than 0. (See
                [`load_datasets_from_json`][auto_circuit.data.load_datasets_from_json]
                for more information.)
            seq_labels: A list of strings that label each token for fixed length
                prompts. Used by
                [`draw_seq_graph`][auto_circuit.visualize.draw_seq_graph] to label the
                circuit diagram.
            word_idxs: A dictionary with the token indexes of specific words. Used by
                official circuit functions.
            kwargs: Additional arguments to pass to `DataLoader`.

        Note:
            `drop_last=True` is always passed to the parent `DataLoader` constructor. So
            all batches are always the same size. This simplifies the implementation of
            several functions. For example, the `kv_cache` only needs caches for a
            single batch size.
        """
        super().__init__(
            prompt_dataset, **kwargs, drop_last=True, collate_fn=collate_fn
        )
        self.seq_len = seq_len
        """
        The token length of the prompts (if fixed length). This prompt length can be
        passed to `patchable_model` to enable patching specific token positions.
        """
        self.diverge_idx = diverge_idx
        """
        The index at which the clean and corrupt prompts diverge. (See
        [`load_datasets_from_json`][auto_circuit.data.load_datasets_from_json] for more
        information.)
        """
        self.seq_labels = seq_labels
        """
        A list of strings that label each token for fixed length prompts. Used by
        [`draw_seq_graph`][auto_circuit.visualize.draw_seq_graph] to label the circuit
        diagram.
        """
        assert kv_cache is None or diverge_idx > 0
        self.kv_cache = kv_cache
        """
        A cache of past key-value pairs for the transformer. Only used if `diverge_idx`
        is greater than 0. (See
        [`load_datasets_from_json`][auto_circuit.data.load_datasets_from_json] for more
        information.)
        """
        self.word_idxs = word_idxs
        """
        A dictionary with the token indexes of specific words. Used by official circuit
        functions.
        """


def load_datasets_from_json(
    model: Optional[t.nn.Module],
    path: Path | List[Path],
    device: t.device,
    prepend_bos: bool = True,
    batch_size: int | Tuple[int, int] = 32,  # (train, test) if tuple
    train_test_size: Tuple[int, int] = (128, 128),
    return_seq_length: bool = False,
    tail_divergence: bool = False,  # Remove all tokens before divergence
    shuffle: bool = True,
    random_seed: int = 42,
    pad: bool = True,
    max_length: Optional[int] = None,  # Maximum length for padding/truncation
) -> Tuple[PromptDataLoader, PromptDataLoader]:
    """
    Load a dataset from a JSON file with additional handling for sequence length.

    Args:
        model: Model to use for tokenization. If None, data must be pre-tokenized.
        path: Path(s) to the JSON file(s) with the dataset.
        device: Device to load the data on.
        prepend_bos: If True, prepend the `BOS` token to each prompt.
        batch_size: Batch size for train/test loaders. If tuple, specify for both.
        return_seq_length: Whether to return the final sequence length.
        tail_divergence: If True, remove tokens before divergence in clean/corrupt prompts.
        shuffle: Whether to shuffle the dataset before splitting.
        random_seed: Seed for reproducibility.
        pad: Whether to pad the sequences to `max_length`.
        max_length: Maximum length for tokenization. Ensures uniform sequence lengths.

    Returns:
        train_loader: Data loader for the training set.
        test_loader: Data loader for the testing set.
    """
    assert not (prepend_bos and (model is None)), "Need model tokenizer to prepend bos"

    # Load the dataset
    first_path = path if isinstance(path, Path) else path[0]
    assert isinstance(first_path, Path)
    with open(first_path, "r") as f:
        data = json.load(f)
    if isinstance(path, list):
        assert all([isinstance(p, Path) for p in path])
        for p in path[1:]:
            with open(p, "r") as f:
                d = json.load(f)
                data["prompts"].extend(d["prompts"])

    # Shuffle data and split into train and test
    random.seed(random_seed)
    t.random.manual_seed(random_seed)
    random.shuffle(data["prompts"]) if shuffle else None
    n_train_and_test = sum(train_test_size)
    clean_prompts = [d["clean"] for d in data["prompts"]][:n_train_and_test]
    corrupt_prompts = [d["corrupt"] for d in data["prompts"]][:n_train_and_test]
    answer_strs = [d["answers"] for d in data["prompts"]][:n_train_and_test]
    wrong_answer_strs = [d["wrong_answers"] for d in data["prompts"]][:n_train_and_test]
    seq_labels = data.get("seq_labels", None)
    word_idxs = data.get("word_idxs", {})

    if prepend_bos:
        seq_labels = ["<|BOS|>"] + seq_labels if seq_labels else None
        word_idxs = {k: v + 1 for k, v in word_idxs.items()}

    kvs = []
    diverge_idx: int = 0

    if model is None:
        clean_prompts = [t.tensor(p).to(device) for p in clean_prompts]
        corrupt_prompts = [t.tensor(p).to(device) for p in corrupt_prompts]
        answers = [t.tensor(a).to(device) for a in answer_strs]
        wrong_answers = [t.tensor(a).to(device) for a in wrong_answer_strs]
        seq_len = max(len(p) for p in clean_prompts) if pad and max_length is None else max_length
        assert not tail_divergence
    else:
        tokenizer: Any = model.tokenizer
        if prepend_bos:
            clean_prompts = [tokenizer.bos_token + p for p in clean_prompts]
            corrupt_prompts = [tokenizer.bos_token + p for p in corrupt_prompts]
        tokenizer.padding_side = "left"

        clean_prompts = tokenizer(
            clean_prompts,
            padding="max_length" if pad else False,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        corrupt_prompts = tokenizer(
            corrupt_prompts,
            padding="max_length" if pad else False,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        seq_len = max_length if max_length else clean_prompts["input_ids"].shape[1]
        ans_dicts = [tokenizer(a, return_tensors="pt", max_length=max_length, truncation=True) for a in answer_strs]
        wrong_ans_dicts = [tokenizer(a, return_tensors="pt", max_length=max_length, truncation=True) for a in wrong_answer_strs]

        clean_prompts = clean_prompts["input_ids"].to(device)
        corrupt_prompts = corrupt_prompts["input_ids"].to(device)
        answers = [a["input_ids"].squeeze(-1).to(device) for a in ans_dicts]
        wrong_answers = [a["input_ids"].squeeze(-1).to(device) for a in wrong_ans_dicts]

        if tail_divergence:
            diverge_idxs = (~(clean_prompts == corrupt_prompts)).int().argmax(dim=1)
            diverge_idx = int(diverge_idxs.min().item())
        if diverge_idx > 0:
            seq_labels = seq_labels[diverge_idx:] if seq_labels else None
            clean_prompts = clean_prompts[:, diverge_idx:]
            corrupt_prompts = corrupt_prompts[:, diverge_idx:]
            seq_len -= diverge_idx

    dataset = PromptDataset(clean_prompts, corrupt_prompts, answers, wrong_answers)
    train_set = Subset(dataset, list(range(train_test_size[0])))
    test_set = Subset(dataset, list(range(train_test_size[0], n_train_and_test)))
    train_loader = PromptDataLoader(
        train_set,
        seq_len=seq_len,
        diverge_idx=diverge_idx,
        kv_cache=kvs[0] if kvs else None,
        seq_labels=seq_labels,
        word_idxs=word_idxs,
        batch_size=batch_size[0] if isinstance(batch_size, tuple) else batch_size,
        shuffle=False,
    )
    test_loader = PromptDataLoader(
        test_set,
        seq_len=seq_len,
        diverge_idx=diverge_idx,
        kv_cache=kvs[-1] if kvs else None,
        seq_labels=seq_labels,
        word_idxs=word_idxs,
        batch_size=batch_size[1] if isinstance(batch_size, tuple) else batch_size,
        shuffle=False,
    )
    return train_loader, test_loader
